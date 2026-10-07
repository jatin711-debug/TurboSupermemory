#!/usr/bin/env python3
"""Judged end-to-end check of the SHIPPED stack, one mechanism at a time.

The head-to-head runners measure ``TSMAdapter``, an evaluation adapter with
its own settings and its own additions (keyword candidates, date tags). This
runner measures ``tsm.Memory`` as it ships, on the same conversations, facts,
embeddings and judge, and adds one thing per arm so a difference between two
neighbouring arms belongs to one mechanism:

  unbounded, same answer-context token budget
    naive           head-to-head floor: plain vector top-k, truncated      (adapter)
    adapter         head-to-head "tsm" arm: the adapter stack, NLI-verified (adapter)
    sdk_plain       Memory(profile=None): scoped vector top-k, truncated
    sdk_pack        the same pool, packed the way recall() packs it
                    (role prior + MMR under the budget)
    sdk_cognitive   conversational profile, belief detection off: cognitive
                    search (graph expansion, fusion) + the same packing, each
                    fact shown with its turn or date (recall()'s default)
    sdk_belief      + belief detection, no verifier (flagged and ranked lower)
    sdk_belief_llm  + LLMVerifier (superseded facts marked as earlier, the
                    fact that replaced each one served with it)

  bounded storage
    sdk_evict       the engine's cap, max_records: the newest records stay
    sdk_evict_gist  the same with a gist summarizer
    sdk_budget      Memory(max_user_tokens=...): the newest facts stay, the
                    rest is folded into short gists, once, after ingestion
    sdk_budget_rolling  the same store compacted --rolling-passes times as
                    the conversation arrives, each pass folding the last
                    one's gists again: what a live deployment does

  diagnostics (not run by default)
    <recall arm>_untagged   the same store, recall(time_tags=False): the
                            facts without their turn or date
    sdk_cognitive_trunc     cognitive search results, untagged and truncated
                            like sdk_plain (cognitive search without the packer)
    sdk_cognitive_alpha1    sdk_cognitive with cognitive_alpha=1.0: the graph
                            still proposes candidates but ranking is by cosine
    sdk_pack_capped         sdk_pack with the item cap recall() used to have
    sdk_pack_tagged         sdk_pack with recall()'s time tags: what recall()
                            would hand over without cognitive search
    sdk_belief_llm_exclude  sdk_belief_llm with exclude_superseded=True
    sdk_belief_llm_nodemote sdk_belief_llm with supersession_demotion_factor=1.0:
                            a superseded fact keeps its rank and is only marked
    sdk_budget_delete       sdk_budget without a summarizer

Every arm gets the facts of one cached extraction pass and the same cached
embeddings; only the memory system differs. Per-question results are written
to ``--out`` so arms can be compared question by question.

    python benchmarks/cognitive_eval/shipped_stack_eval.py --limit 120 \\
        --token-budget 150 --judge openai --judge-model gpt-4.1-mini \\
        --out results.json
    # plumbing check, no network: cached texts only, proxy judge, fake verdicts
    python benchmarks/cognitive_eval/shipped_stack_eval.py --offline --limit 5
"""

import argparse
import atexit
import hashlib
import json
import logging
import math
import os
import pickle
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from cognitive_eval._secrets import ensure_openai_key, key_file_hint  # noqa: E402
from cognitive_eval.benchmark_datasets.longmemeval import load_longmemeval  # noqa: E402
from cognitive_eval.budgeting import total_tokens, truncate_to_budget  # noqa: E402
from cognitive_eval.run_belief_longmemeval import _msg_content, hit_at  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("shipped_stack_eval")

UNBOUNDED = ("naive", "adapter", "sdk_plain", "sdk_pack", "sdk_cognitive", "sdk_belief",
             "sdk_belief_llm")
BOUNDED = ("sdk_evict", "sdk_evict_gist", "sdk_budget", "sdk_budget_rolling")
# Arms served by recall() as it ships; each also has an _untagged twin.
RECALL_ARMS = ("sdk_cognitive", "sdk_cognitive_alpha1", "sdk_belief", "sdk_belief_llm",
               "sdk_belief_llm_exclude", "sdk_belief_llm_nodemote", "sdk_evict",
               "sdk_evict_gist", "sdk_budget", "sdk_budget_rolling", "sdk_budget_delete")
DIAGNOSTIC = (("sdk_cognitive_trunc", "sdk_cognitive_alpha1", "sdk_pack_capped", "sdk_pack_tagged",
               "sdk_belief_llm_exclude", "sdk_belief_llm_nodemote", "sdk_budget_delete")
              + tuple(a + "_untagged" for a in RECALL_ARMS))
DEFAULT_ARMS = UNBOUNDED + BOUNDED
ARMS = DEFAULT_ARMS + DIAGNOSTIC
# (arm, baseline): each pair differs in one mechanism.
PAIRS = (
    ("sdk_plain", "naive"),
    ("sdk_pack", "sdk_plain"),
    ("sdk_cognitive_untagged", "sdk_pack"),
    ("sdk_cognitive", "sdk_cognitive_untagged"),
    ("sdk_belief", "sdk_cognitive"),
    ("sdk_belief_llm", "sdk_cognitive"),
    ("sdk_belief_llm", "sdk_belief_llm_untagged"),
    ("sdk_belief_llm", "sdk_plain"),
    ("adapter", "naive"),
    ("sdk_belief_llm", "adapter"),
    ("sdk_evict_gist", "sdk_evict"),
    ("sdk_budget", "sdk_evict_gist"),
    ("sdk_budget", "sdk_cognitive"),
    ("sdk_budget", "sdk_budget_untagged"),
    ("sdk_budget_rolling", "sdk_budget"),
    ("sdk_budget", "sdk_budget_delete"),
    ("sdk_budget_untagged", "sdk_budget_delete_untagged"),
    ("sdk_cognitive_trunc", "sdk_plain"),
    ("sdk_pack", "sdk_pack_capped"),
    ("sdk_pack_tagged", "sdk_pack"),
    ("sdk_cognitive", "sdk_pack_tagged"),
    ("sdk_cognitive_alpha1", "sdk_cognitive"),
    ("sdk_cognitive_alpha1", "sdk_pack_tagged"),
    ("sdk_belief_llm", "sdk_pack_tagged"),
    ("sdk_belief_llm", "sdk_belief_llm_exclude"),
    ("sdk_belief_llm_nodemote", "sdk_belief_llm"),
    ("sdk_belief_llm_nodemote", "sdk_cognitive"),
)


def variants(name):
    """A recall() arm and the diagnostics served from the same store."""
    return (name, name + "_trunc", name + "_untagged")


class Embeddings:
    """The harness's OpenAI embedding cache, read-only, plus a small overlay
    file for texts it does not hold. Another run may be writing the shared
    cache, so this one never does."""

    def __init__(self, model, overlay_path, offline):
        from cognitive_eval.openai_embedder import OpenAIEmbedder

        self._base = OpenAIEmbedder(model=model)
        atexit.unregister(self._base.flush)
        self._offline = offline
        self._overlay_path = overlay_path
        self._overlay = {}
        if overlay_path and os.path.exists(overlay_path):
            with open(overlay_path, "rb") as fh:
                self._overlay = pickle.load(fh)
        self.dimension = self._base.get_sentence_embedding_dimension()
        self.hits = self.misses = 0
        atexit.register(self.flush)

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def _lookup(self, text):
        vector = self._base._cache.get(text)
        return self._overlay.get(text) if vector is None else vector

    def encode(self, texts, **_kwargs):
        single = isinstance(texts, str)
        items = [t if (t and t.strip()) else " " for t in ([texts] if single else list(texts))]
        missing = sorted({t for t in items if self._lookup(t) is None})
        self.misses += len(missing)
        self.hits += len(items) - len(missing)
        if missing and self._offline:
            for t in missing:  # a stand-in direction; offline runs only check plumbing
                seed = int(hashlib.sha256(t.encode("utf-8")).hexdigest()[:8], 16)
                v = np.random.default_rng(seed).standard_normal(self.dimension)
                self._overlay[t] = (v / np.linalg.norm(v)).astype(np.float32)
        elif missing:
            for start in range(0, len(missing), self._base.batch):
                chunk = missing[start:start + self._base.batch]
                for t, v in zip(chunk, self._base._embed_batch(chunk)):
                    self._overlay[t] = np.asarray(v, dtype=np.float32)
        out = np.vstack([np.asarray(self._lookup(t), dtype=np.float32) for t in items])
        return out[0] if single else out

    def flush(self):
        if self._overlay and self._overlay_path and not self._offline:
            tmp = self._overlay_path + ".tmp"
            with open(tmp, "wb") as fh:
                pickle.dump(self._overlay, fh)
            os.replace(tmp, self._overlay_path)


class Replay:
    """Extractor that hands back precomputed facts, message by message."""

    def __init__(self, per_message):
        self._facts = list(per_message)
        self._next = 0

    def extract_facts(self, _message, _context=None):
        facts = self._facts[self._next]
        self._next += 1
        return facts


class CachedExtraction:
    """Facts per message from the harness's disk-cached extractor. Offline, a
    message that is not cached yields no facts instead of a request."""

    def __init__(self, model, offline):
        from cognitive_eval.extraction.openai_extractor import (OpenAIExtractor,
                                                                extraction_cache_key)

        self._extractor = OpenAIExtractor(model=model)
        self._key = extraction_cache_key
        self._offline = offline
        self.misses = 0

    def facts(self, message, context):
        if self._offline and self._key(message, context) not in self._extractor._cache:
            self.misses += 1
            return []
        return [f for f in self._extractor.extract_facts(message, context) if f]

    @property
    def instance(self):
        return self._extractor


class CachedGists:
    """A gist summarizer whose answers are kept on disk: a rerun over the same
    stores costs nothing, and two runs compare the same gists instead of two
    samples of the model's wording."""

    def __init__(self, inner, path):
        self._inner = inner
        self._path = path
        self.model = getattr(inner, "model", "?")
        self.calls = self.hits = 0
        self._new = 0
        self._cache = {}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                self._cache = json.load(fh)

    def summarize(self, texts, max_tokens=None):
        texts = list(texts)
        key = hashlib.sha256(json.dumps([self.model, max_tokens, texts]).encode("utf-8")).hexdigest()
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        gist = self._inner.summarize(texts, max_tokens=max_tokens) or ""
        self._cache[key] = gist
        self._new += 1
        return gist

    def __call__(self, texts):
        return self.summarize(texts)

    def flush(self):
        if self._new and self._path:
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._cache, fh)
            os.replace(tmp, self._path)
            self._new = 0


class ProxyJudge:
    """No-cost stand-in: counts a hit when the gold answer's most distinctive
    token is in the context. Only for checking the plumbing."""

    model = "retrieval-proxy"
    calls = input_tokens = output_tokens = 0

    def answer(self, _question, memories):
        return "\n".join(memories)

    def judge(self, _question, gold, prediction):
        return hit_at(gold, [prediction], 1)


class KeepsClient:
    """Offline chat client for LLMVerifier: every pair is judged KEEPS."""

    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    @staticmethod
    def _create(**kwargs):
        pairs = kwargs["messages"][-1]["content"].count("OLDER:")
        text = "\n".join(f"{i + 1}: KEEPS" for i in range(pairs))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def conversation_inputs(conv, extraction):
    """Messages of a conversation and the facts extracted from each."""
    messages, per_message, context = [], [], []
    for m in conv.messages:
        content = _msg_content(m)
        if not content or not content.strip():
            continue
        role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else "user")
        per_message.append(extraction.facts(content, context))
        context.append(content)
        messages.append({"role": role, "content": content,
                         "timestamp": getattr(m, "timestamp", "")})
    return messages, per_message


def sdk_pool(mem, embedder, query, user_id, k):
    """The plain-vector pool recall() would rank: nearest records of the scope."""
    hits = mem.engine.search_ann(np.asarray(embedder.encode(query), dtype=np.float32), k,
                                 scope=user_id)
    records = {r["id"]: r for r in mem.engine.get_records([mid for mid, _ in hits]) if r}
    pool = []
    for mid, score in hits:
        rec = records.get(mid)
        if rec is None:
            continue
        payload = rec.get("payload") or {}
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                payload = {}
        pool.append({"id": mid, "text": rec["text"], "score": float(score),
                     "role": rec.get("source_role") or "", "turn_index": payload.get("turn_index"),
                     "timestamp": payload.get("timestamp")})
    return pool


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--token-budget", type=int, default=150, help="answer-context token cap")
    ap.add_argument("--pool-k", type=int, default=20)
    ap.add_argument("--user-tokens", type=int, default=256,
                    help="storage budget of the sdk_budget arms (tokens per conversation)")
    ap.add_argument("--rolling-passes", type=int, default=4,
                    help="how many times sdk_budget_rolling adds a part of the conversation "
                         "and compacts")
    ap.add_argument("--max-records", type=int, default=16,
                    help="storage cap of the sdk_evict arms (records per conversation)")
    ap.add_argument("--embed-model", default="text-embedding-3-small")
    ap.add_argument("--extractor-model", default="gpt-4.1-nano")
    ap.add_argument("--gist-model", default="gpt-4.1-nano")
    ap.add_argument("--verifier-model", default="gpt-4o-mini")
    ap.add_argument("--judge", choices=["openai", "none"], default="openai")
    ap.add_argument("--judge-model", default="gpt-4.1-mini")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--work-dir", default=None,
                    help="where the verdict cache and the embedding overlay are kept")
    ap.add_argument("--out", default=None, help="write per-question results here (JSON)")
    ap.add_argument("--reuse", default=None, metavar="PATHS",
                    help="comma-separated --out files of earlier runs with the same judge: an "
                         "answer already judged for the same question and context is not "
                         "judged again")
    ap.add_argument("--rejudge", default=None, metavar="PATH",
                    help="skip ingestion: judge the contexts saved in PATH (the --out file of a "
                         "run whose judging did not finish)")
    ap.add_argument("--offline", action="store_true",
                    help="no network: cached texts only, proxy judge, fake verdicts, extractive gists")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        ap.error(f"unknown arms: {unknown}")
    if args.offline:
        args.judge = "none"
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not ensure_openai_key() and not args.offline:
        sys.exit(key_file_hint())
    if args.offline and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "offline"  # clients are built but never called

    work_dir = args.work_dir or tempfile.mkdtemp(prefix="tsm_shipped_eval_")
    os.makedirs(work_dir, exist_ok=True)

    from tsm import Memory
    from tsm.budget import select_under_budget
    from tsm.gist import ExtractiveGistSummarizer, OpenAIGistSummarizer
    from tsm.memory import time_tag
    from tsm.ranking import is_first_person_query, role_prior
    from tsm.verification import LLMVerifier

    extraction = CachedExtraction(args.extractor_model, args.offline)
    embedder = Embeddings(args.embed_model, os.path.join(work_dir, "emb_overlay.pkl"), args.offline)
    if args.judge == "none":
        judge = ProxyJudge()
    else:
        from cognitive_eval.judge import create_judge

        judge = create_judge("openai", openai_model=args.judge_model)
    def asked(*names):
        return any(a in arms for name in names for a in variants(name))

    verifier = None
    if asked("sdk_belief_llm", "sdk_belief_llm_exclude", "sdk_belief_llm_nodemote"):
        verifier = LLMVerifier(model=args.verifier_model,
                               client=KeepsClient() if args.offline else None,
                               cache_dir=None if args.offline else os.path.join(work_dir, "verdicts"))
    summarizer = None
    if asked("sdk_evict_gist", "sdk_budget", "sdk_budget_rolling"):
        summarizer = CachedGists(
            ExtractiveGistSummarizer() if args.offline else OpenAIGistSummarizer(
                model=args.gist_model),
            None if args.offline else os.path.join(
                work_dir, f"gists_{args.gist_model.replace('/', '_')}.json"))
    nli = None

    convs = load_longmemeval(args.data_dir)[:args.limit]
    logger.info("Loaded %d conversations. arms=%s budget=%d judge=%s", len(convs), arms,
                args.token_budget, getattr(judge, "model", "?"))
    for name in ("cognitive_eval.adapters.tsm", "httpx", "httpcore", "urllib3", "tsm.memory",
                 "huggingface_hub", "sentence_transformers", "transformers", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)

    tasks = []
    stats = defaultdict(Counter)

    def sdk_memory(**kwargs):
        db = tempfile.mkdtemp(prefix="tsm_shipped_")
        return Memory(db, embedder=embedder, extractor=Replay(per_message), **kwargs), db

    started = time.time()
    skipped = []
    if args.rejudge:
        with open(args.rejudge, encoding="utf-8") as fh:
            tasks = json.load(fh)["tasks"]
        arms = [a for a in ARMS if any(t["arm"] == a for t in tasks)]
        convs = []

    # One line per finished conversation, so an interrupted run picks up where
    # it stopped instead of paying for the same verdicts and gists again. It
    # may be resumed with fewer arms; a conversation that was not run for
    # every arm asked for now is run again.
    checkpoint = args.out + ".partial.jsonl" if args.out and not args.rejudge else None
    setup = {"arms": arms, "token_budget": args.token_budget, "pool_k": args.pool_k,
             "max_records": args.max_records, "user_tokens": args.user_tokens,
             "rolling_passes": args.rolling_passes, "embed_model": args.embed_model,
             "extractor_model": args.extractor_model, "gist_model": args.gist_model,
             "verifier_model": args.verifier_model, "offline": args.offline}
    done = set()
    if checkpoint and os.path.exists(checkpoint):
        with open(checkpoint, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # a line cut off by the interruption
                was = dict(row.get("setup") or {})
                had = set(was.pop("arms", ()))
                if was != {k: v for k, v in setup.items() if k != "arms"}:
                    sys.exit(f"{checkpoint} was written with different settings; "
                             "remove it or choose another --out")
                if not set(arms) <= had or row["conversation_id"] in done:
                    continue
                done.add(row["conversation_id"])
                tasks.extend(t for t in row["tasks"] if t["arm"] in arms)
                for arm, counts in row["stats"].items():
                    if arm in arms:
                        stats[arm].update(counts)
        logger.info("Resuming: %d conversations already done", len(done))

    for index, conv in enumerate(convs):
        if conv.conv_id in done:
            continue
        messages, per_message = conversation_inputs(conv, extraction)
        questions = [q for q in conv.queries if not q.is_abstention]
        if not questions:
            continue
        user = conv.conv_id
        opened = []  # (close, db) of everything built for this conversation
        retrieve = {}
        conv_tasks = []
        conv_stats = defaultdict(Counter)
        try:
            if "naive" in arms or "adapter" in arms:
                from cognitive_eval.adapters.tsm_adapter import TSMAdapter
                from cognitive_eval.compress_eval import insert_facts
                from cognitive_eval.head_to_head_eval import conv_facts, naive_retrieve
            if "naive" in arms:
                db = tempfile.mkdtemp(prefix="tsm_naive_")
                ad = TSMAdapter(db_path=db, extractor="mock", cognitive_features=False,
                                belief_revision=False, model=embedder)
                opened.append((ad.close, db))
                insert_facts(ad, conv_facts(extraction.instance, conv), user)
                retrieve["naive"] = lambda q, ad=ad: naive_retrieve(
                    ad, q, args.pool_k, args.token_budget)
            if "adapter" in arms:
                if nli is None:
                    from cognitive_eval.verification import get_shared_verifier

                    nli = get_shared_verifier()
                db = tempfile.mkdtemp(prefix="tsm_adapter_")
                ad = TSMAdapter(db_path=db, extractor="openai",
                                extractor_instance=extraction.instance, cognitive_features=True,
                                belief_revision=True, model=embedder, belief_source_roles=["user"],
                                verify_demotions=True, verifier=nli, supersession_mode="exclude")
                opened.append((ad.close, db))
                ad.add(conv.messages, user_id=user)
                ad.trigger_consolidation()
                conv_stats["adapter"]["superseded"] += len(ad.engine.superseded_ids())
                retrieve["adapter"] = lambda q, ad=ad: ad.recall_under_budget(
                    q, user_id=user, token_budget=args.token_budget, method="mmr")

            plain_arms = [a for a in ("sdk_plain", "sdk_pack", "sdk_pack_capped",
                                      "sdk_pack_tagged") if a in arms]
            if plain_arms:
                mem, db = sdk_memory(profile=None)
                opened.append((mem.close, db))
                mem.add(messages, user_id=user)
                if "sdk_plain" in arms:
                    retrieve["sdk_plain"] = lambda q, mem=mem: truncate_to_budget(
                        [p["text"] for p in sdk_pool(mem, embedder, q, user, args.pool_k)],
                        args.token_budget)

                def packed(q, mem=mem, cap=None, tags=False):
                    # recall()'s packing on a plain-vector pool of recall()'s size.
                    pool = sdk_pool(mem, embedder, q, user,
                                    max(args.pool_k, 30, args.token_budget // 8))
                    first_person = is_first_person_query(q)
                    for p in pool:
                        p["score"] *= role_prior(first_person, p["role"])
                        if tags:
                            p["context"] = (time_tag(p["timestamp"], p["turn_index"])
                                            + (p["text"] or ""))
                    pool.sort(key=lambda p: -p["score"])
                    chosen = select_under_budget(pool, args.token_budget, embed=embedder.encode,
                                                 lam=0.7, max_items=cap)
                    return [p.get("context") or p["text"] or "" for p in chosen]
                if "sdk_pack" in arms:
                    retrieve["sdk_pack"] = packed
                if "sdk_pack_tagged" in arms:
                    retrieve["sdk_pack_tagged"] = lambda q: packed(q, tags=True)
                if "sdk_pack_capped" in arms:
                    old_cap = min(10, max(4, args.token_budget // 35))
                    retrieve["sdk_pack_capped"] = lambda q: packed(q, cap=old_cap)

            def shown(results):
                """What goes in the prompt: a result's context (the text, marked
                when the fact has since changed)."""
                return [r.get("context") or r["text"] or "" for r in results]

            def recall_arm(name, feed=None, **kwargs):
                """One Memory serving `name` through recall() as shipped,
                `name`_untagged (the same without time tags) and `name`_trunc,
                which truncates the same search results."""
                wanted = [a for a in variants(name) if a in arms]
                if not wanted:
                    return
                mem, db = sdk_memory(**kwargs)
                opened.append((mem.close, db))
                if feed is None:
                    stored, committed = mem.add(messages, user_id=user), mem.consolidate()
                else:
                    stored, committed = feed(mem)
                store = [r["text"] or "" for r in
                         mem.engine.get_records(mem.engine.scope_ids(user)) if r]
                for arm in wanted:
                    conv_stats[arm]["stored"] += stored
                    conv_stats[arm]["committed"] += committed
                    conv_stats[arm]["superseded"] += len(mem.engine.superseded_ids())
                    conv_stats[arm]["records"] += len(store)
                    conv_stats[arm]["store_tokens"] += total_tokens(store)
                    conv_stats[arm]["conversations"] += 1
                retrieve[name] = lambda q, mem=mem: shown(
                    mem.recall(q, user_id=user, token_budget=args.token_budget,
                               pool_k=args.pool_k))
                retrieve[name + "_untagged"] = lambda q, mem=mem: shown(
                    mem.recall(q, user_id=user, token_budget=args.token_budget,
                               pool_k=args.pool_k, time_tags=False))
                retrieve[name + "_trunc"] = lambda q, mem=mem: truncate_to_budget(
                    shown(mem.recall(q, user_id=user, top_k=args.pool_k, time_tags=False)),
                    args.token_budget)

            def rolling(mem):
                """The conversation in parts, compacting after each."""
                stored = committed = 0
                size = max(1, -(-len(messages) // max(1, args.rolling_passes)))
                for start in range(0, len(messages), size):
                    stored += mem.add(messages[start:start + size], user_id=user)
                    committed += mem.consolidate()
                return stored, committed

            recall_arm("sdk_cognitive", refinement_cosine_threshold=None,
                       contradiction_cosine_threshold=None)
            recall_arm("sdk_cognitive_alpha1", refinement_cosine_threshold=None,
                       contradiction_cosine_threshold=None, cognitive_alpha=1.0)
            recall_arm("sdk_belief")
            recall_arm("sdk_belief_llm", verifier=verifier)
            recall_arm("sdk_belief_llm_exclude", verifier=verifier, exclude_superseded=True)
            recall_arm("sdk_belief_llm_nodemote", verifier=verifier,
                       supersession_demotion_factor=1.0)
            recall_arm("sdk_evict", max_records=args.max_records)
            recall_arm("sdk_evict_gist", max_records=args.max_records,
                       gist_summarizer=summarizer)
            recall_arm("sdk_budget", max_user_tokens=args.user_tokens,
                       gist_summarizer=summarizer)
            recall_arm("sdk_budget_rolling", feed=rolling, max_user_tokens=args.user_tokens,
                       gist_summarizer=summarizer)
            recall_arm("sdk_budget_delete", max_user_tokens=args.user_tokens)

            for q in questions:
                for arm in arms:
                    texts = retrieve[arm](q.query_text)
                    conv_tasks.append({"arm": arm, "conversation_id": user,
                                       "query_id": q.query_id,
                                       "question_type": q.question_type or "?",
                                       "query": q.query_text, "gold": q.answer_text,
                                       "retrieved": texts})
            # Only a conversation every arm handled is scored, so arms stay paired.
            tasks.extend(conv_tasks)
            for arm, counts in conv_stats.items():
                stats[arm].update(counts)
            if checkpoint:
                with open(checkpoint, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"setup": setup, "conversation_id": user,
                                         "tasks": conv_tasks,
                                         "stats": {a: dict(c) for a, c in conv_stats.items()}})
                             + "\n")
            if summarizer is not None:
                summarizer.flush()
        except Exception as e:  # noqa: BLE001 — one bad conversation must not sink the run
            logger.warning("conversation %s skipped for every arm: %s: %s", user,
                           type(e).__name__, e)
            skipped.append(user)
        finally:
            for close, db in opened:
                try:
                    close()
                finally:
                    shutil.rmtree(db, ignore_errors=True)
        if (index + 1) % 20 == 0:
            logger.info("ingested %d/%d conversations (%.0fs)", index + 1, len(convs),
                        time.time() - started)

    for t in tasks:
        stats[t["arm"]]["items"] += len(t["retrieved"])
        stats[t["arm"]]["tokens"] += total_tokens(t["retrieved"])

    judge_model = getattr(judge, "model", "?")

    # The judge sees a question, a context and the gold answer, never the arm:
    # the same three get the same verdict, so each is judged once.
    def what(task):
        return json.dumps([task["query"], task["gold"], task["retrieved"]], sort_keys=True)

    known = {}
    # Verdicts this run already paid for: judging saves them as it goes, so
    # an interrupted run started again does not ask for them twice.
    if args.out and os.path.exists(args.out):
        try:
            with open(args.out, encoding="utf-8") as fh:
                earlier = json.load(fh)
        except (OSError, ValueError):
            earlier = {}
        if earlier.get("judge_model") == judge_model:
            for t in earlier.get("tasks", []):
                if t.get("correct") is not None:
                    known[what(t)] = (t["prediction"], t["correct"])

    def save(summary=None):
        if args.out:
            tmp = args.out + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"summary": summary, "judge_model": judge_model, "tasks": tasks}, fh)
            os.replace(tmp, args.out)

    def settle():
        """Give every task whose context has a verdict that verdict."""
        for t in tasks:
            if t.get("correct") is None and what(t) in known:
                t["prediction"], t["correct"] = known[what(t)]

    own = sum(t.get("correct") is None and what(t) in known for t in tasks)
    if own:
        logger.info("%d answers already have a verdict in %s", own, args.out)
        settle()
    save()  # contexts first: a judging failure can then be resumed with --rejudge
    embedder.flush()

    for path in (args.reuse.split(",") if args.reuse else []):
        with open(path.strip(), encoding="utf-8") as fh:
            earlier = json.load(fh)
        if (earlier.get("summary") or {}).get("judge_model") != judge_model:
            sys.exit(f"{path} was judged by another model; it cannot be reused")
        for t in earlier["tasks"]:
            if t.get("correct") is not None:
                known[what(t)] = (t["prediction"], t["correct"])
    for t in tasks:
        if t.get("correct") is not None:
            known[what(t)] = (t["prediction"], t["correct"])
    distinct = {}
    for t in tasks:
        if t.get("correct") is None and what(t) not in known:
            distinct.setdefault(what(t), t)
    pending = list(distinct.values())
    reused = sum(t.get("correct") is None and what(t) in known for t in tasks)
    logger.info("Judging %d answers with %s (%d workers); %d more reuse an earlier verdict, "
                "%d share a context with another arm", len(pending),
                getattr(judge, "model", "?"), args.workers, reused,
                sum(t.get("correct") is None for t in tasks) - reused - len(pending))

    def score(task):
        try:
            prediction = judge.answer(task["query"], task["retrieved"])
            return prediction, bool(judge.judge(task["query"], task["gold"], prediction))
        except Exception as e:  # noqa: BLE001 — left unjudged, reported below
            logger.warning("judging failed for one answer: %s", type(e).__name__)
            return None, None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for count, (task, (prediction, correct)) in enumerate(
                zip(pending, pool.map(score, pending)), 1):
            if correct is not None:
                known[what(task)] = (prediction, correct)
            if count % 100 == 0:
                settle()
                save()
    settle()
    unjudged = sum(t.get("correct") is None for t in tasks)
    if unjudged:
        save()
        sys.exit(f"{unjudged} answers could not be judged; rerun with --rejudge {args.out}")

    by_arm = defaultdict(dict)  # arm -> (conversation, query) -> correct
    by_type = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for t in tasks:
        by_arm[t["arm"]][(t["conversation_id"], t["query_id"])] = t["correct"]
        for key in ("all", t["question_type"]):
            by_type[t["arm"]][key][0] += 1
            by_type[t["arm"]][key][1] += int(t["correct"])
    n_questions = len(next(iter(by_arm.values()))) if by_arm else 0

    logger.info("=" * 96)
    logger.info("SHIPPED STACK — judged accuracy @ %d-token context, %d questions, judge %s",
                args.token_budget, n_questions, getattr(judge, "model", "?"))
    types = sorted({k for arm in arms for k in by_type[arm] if k != "all"})
    wide = max([15] + [len(a) for a in arms])
    logger.info("  %-*s %7s  %s   ctx items/tokens", wide, "arm", "overall",
                "  ".join(f"{t.replace('single-session-', 'ss-')[:14]:>14}" for t in types))
    overall = {}
    for arm in arms:
        n, c = by_type[arm]["all"]
        overall[arm] = c / n if n else 0.0
        cells = "  ".join(f"{(by_type[arm][t][1] / by_type[arm][t][0]) if by_type[arm][t][0] else 0:>14.2f}"
                          for t in types)
        logger.info("  %-*s %7.3f  %s   %.1f / %.0f", wide, arm, overall[arm], cells,
                    stats[arm]["items"] / max(1, n), stats[arm]["tokens"] / max(1, n))
    logger.info("  %-*s %7s  %s", wide, "(questions)", n_questions,
                "  ".join(f"{by_type[arms[0]][t][0]:>14d}" for t in types))
    sized = [a for a in arms if stats[a]["conversations"]]
    if sized:
        logger.info("-" * 96)
        logger.info("  what each store holds per conversation:")
        for arm in sized:
            n = stats[arm]["conversations"]
            logger.info("  %-*s %6.1f records / %6.0f tokens   (%.1f superseded)", wide, arm,
                        stats[arm]["records"] / n, stats[arm]["store_tokens"] / n,
                        stats[arm]["superseded"] / n)

    def sign_test(wins, losses):
        """Two-sided exact p-value that wins and losses are equally likely."""
        n = wins + losses
        if n == 0:
            return 1.0
        tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) / 2 ** n
        return min(1.0, 2 * tail)

    logger.info("-" * 96)
    logger.info("  question-by-question (only the arm right / only the baseline right):")
    paired = {}
    for arm, base in PAIRS:
        if arm not in by_arm or base not in by_arm:
            continue
        keys = by_arm[arm].keys() & by_arm[base].keys()
        wins = sum(by_arm[arm][k] and not by_arm[base][k] for k in keys)
        losses = sum(by_arm[base][k] and not by_arm[arm][k] for k in keys)
        p = sign_test(wins, losses)
        paired[f"{arm} vs {base}"] = {"wins": wins, "losses": losses, "p": round(p, 4),
                                      "delta": round(overall[arm] - overall[base], 4)}
        logger.info("  %-*s %+.3f   %2d / %2d   p=%.2f", 2 * wide + 4, f"{arm} vs {base}",
                    overall[arm] - overall[base], wins, losses, p)
    logger.info("=" * 96)

    summary = {
        "token_budget": args.token_budget, "pool_k": args.pool_k, "limit": args.limit,
        "max_records": args.max_records, "user_tokens": args.user_tokens,
        "rolling_passes": args.rolling_passes, "questions": n_questions, "arms": arms,
        "overall": {a: round(v, 4) for a, v in overall.items()},
        "by_type": {a: {t: [by_type[a][t][1], by_type[a][t][0]] for t in types} for a in arms},
        "paired": paired, "stats": {a: dict(stats[a]) for a in arms},
        "judge_model": getattr(judge, "model", "?"), "judge_calls": getattr(judge, "calls", 0),
        "judge_input_tokens": getattr(judge, "input_tokens", 0),
        "judge_output_tokens": getattr(judge, "output_tokens", 0),
        "verifier_model": args.verifier_model if verifier else None,
        "verifier_requests": getattr(verifier, "calls", 0) if verifier else 0,
        "gist_model": getattr(summarizer, "model", None) if summarizer else None,
        "gist_calls": getattr(summarizer, "calls", 0) if summarizer else 0,
        "gists_from_cache": getattr(summarizer, "hits", 0) if summarizer else 0,
        "embedding_cache_misses": embedder.misses,
        "extraction_cache_misses": extraction.misses,
        "extractor_calls": getattr(extraction.instance, "calls", 0),
        "skipped_conversations": skipped,
        "offline": args.offline,
    }
    logger.info("GATE_SUMMARY: %s", json.dumps(summary))
    save(summary)
    if args.out:
        logger.info("Wrote per-question results to %s", args.out)
    if checkpoint and os.path.exists(checkpoint):
        os.remove(checkpoint)
    embedder.flush()


if __name__ == "__main__":
    main()
