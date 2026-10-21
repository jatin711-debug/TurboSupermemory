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

  any engine setting, as an arm of its own
    --config NAME:KEY=VALUE,...   adds the arm cfg_NAME: sdk_cognitive (the
                            conversational profile, belief detection off) with
                            these MemoryEngine settings, e.g.
                            --config a90:cognitive_alpha=0.9

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

The facts come from the SDK's own extractor (``tsm.extractors``), so what a
message is stored as is what ``Memory.add`` would store. ``--extraction
legacy`` gives every arm the harness's earlier extraction instead, which the
runs before 2026-10-09 used and which returned nothing for a third of the
user messages. ``--reader notes`` asks the answering model with the SDK's
prompt (``tsm.reading``) instead of the harness's plain one.

What a store is given. Sessions are ingested in the order they took place
(the dataset does not always list them that way; ``--file-order`` restores
the old behaviour), each message with its session's date, so ``recall()``
tags memories with dates (``--time-tags minute`` adds the time of day,
``turn`` drops the dates). The reader is told the question's date when the
dataset has one (``--no-question-date``).

A second, free score. The official files mark the turns that hold each
question's evidence (``download.py --official oracle``). When they are
present, every arm that returns memories is also scored on how much of that
evidence is in its context: no model is asked, so a ranking change can be
compared without paying a judge. ``--judge none`` reports only this and the
retrieval proxy.

Longer histories. ``--variant s`` runs the official LongMemEval-S histories
of the same questions (about 40 sessions each; their facts have to be
extracted first). ``--pool N`` builds a history from N conversations of the
cached oracle data, so each question has N-1 other users' sessions as
distractors at no extraction cost; use it for the evidence score only (the
other conversations can contradict the answer).

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
from cognitive_eval.benchmark_datasets.longmemeval import (  # noqa: E402
    load_longmemeval,
    parse_date,
)
from cognitive_eval.budgeting import estimate_tokens, total_tokens  # noqa: E402
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


# Arms whose results cannot be traced back to the turns they came from.
UNATTRIBUTED = ("naive", "adapter")


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


class LocalEmbeddings:
    """A local ``SentenceTransformerEmbedder`` with its vectors cached on disk.

    ``--local-embedder MODEL`` runs the harness on a model that needs no
    network or key (and, with ``--offline``, still embeds for real: only the
    extraction replay and the judge are stand-ins). Queries and stored texts
    are cached apart, because an asymmetric model gives one text two vectors,
    and the file is named by the embedder's fingerprint (model commit,
    dimension, task prompts) so a change to any of them starts a new cache.
    ``misses`` stays 0: here a cache miss is not a sign of a bad offline run.
    """

    def __init__(self, embedder, cache_dir):
        import re

        from tsm.embedders import _load_cache

        self._embedder = embedder
        self.dimension = embedder.dimension
        self.fingerprint = embedder.fingerprint
        self.misses = self.computed = self.hits = 0
        slug = re.sub(r"[^A-Za-z0-9]+", "_", self.fingerprint)[:80]
        digest = hashlib.sha256(self.fingerprint.encode("utf-8")).hexdigest()[:10]
        os.makedirs(cache_dir, exist_ok=True)
        self._path = os.path.join(cache_dir, f"emb_local_{slug}_{digest}.pkl")
        self._cache = {}
        if os.path.exists(self._path):
            self._cache = _load_cache(self._path)
        self._dirty = False
        atexit.register(self.flush)

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def _encode(self, kind, texts):
        single = isinstance(texts, str)
        items = [t if (t and t.strip()) else " " for t in ([texts] if single else list(texts))]
        missing = sorted({t for t in items if (kind, t) not in self._cache})
        self.hits += len(items) - len(missing)
        if missing:
            encode = (self._embedder.encode_query if kind == "query"
                      else self._embedder.encode_document)
            for start in range(0, len(missing), 256):
                chunk = missing[start:start + 256]
                for t, v in zip(chunk, encode(chunk)):
                    self._cache[(kind, t)] = np.asarray(v, dtype=np.float32)
            self.computed += len(missing)
            self._dirty = True
        out = np.vstack([self._cache[(kind, t)] for t in items])
        return out[0] if single else out

    def encode(self, texts, **_kwargs):
        return self._encode("document", texts)

    def encode_document(self, texts):
        return self._encode("document", texts)

    def encode_query(self, texts):
        return self._encode("query", texts)

    def flush(self):
        if self._dirty:
            tmp = self._path + ".tmp"
            with open(tmp, "wb") as fh:
                pickle.dump(self._cache, fh)
            os.replace(tmp, self._path)
            self._dirty = False


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
    """Facts per message from a disk-cached extractor.

    By default the SDK's own with its defaults (``tsm.extractors
    .OpenAIExtractor``: the first prompt alone), cached under
    ``extraction/_cache/shipped``. ``mode`` switches on its two opt-in steps:
    ``"second"`` asks a second prompt about a message the first finds nothing
    in, ``"all"`` also stores what such a message states when the second
    finds nothing either. ``"legacy"`` is the harness's earlier extractor and
    cache, which every run before 2026-10-09 used. Offline, a message that is
    not cached yields no facts instead of a request.

    The SDK's first prompt is the earlier extractor's, word for word, so a
    reply the earlier cache holds is served instead of being paid for again
    (it covers all 500 conversations of the oracle file; the SDK's own cache
    only those the second prompt was run on).
    """

    def __init__(self, model, offline, mode="shipped"):
        from cognitive_eval.extraction import openai_extractor as earlier

        self._earlier = {}
        if mode == "legacy":
            self._extractor = earlier.OpenAIExtractor(model=model)
            cache = self._extractor._cache
            self._cached = lambda message, context: (
                earlier.extraction_cache_key(message, context) in cache)
        else:
            from tsm import extractors as shipped

            steps = {"second": {"second_look": True},
                     "all": {"second_look": True, "keep_unextracted": True}}.get(mode, {})
            self._extractor = shipped.OpenAIExtractor(
                model=model, cache_dir=os.path.join(HERE, "extraction", "_cache", "shipped"),
                **steps)
            self._cached = self._extractor.is_cached
            path = os.path.join(HERE, "extraction", "_cache",
                                f"extract_{model.replace('/', '_').replace(':', '_')}.json")
            if earlier._SYS == shipped._SYS and os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    self._earlier = json.load(fh)
            self._earlier_key = earlier.extraction_cache_key
        self._offline = offline
        self.misses = 0

    def facts(self, message, context):
        if self._earlier and message and message.strip():
            key = self._extractor.cache_key(message, context)
            if key not in self._extractor._cache:
                paid = self._earlier.get(self._earlier_key(message, context))
                if paid is not None:
                    self._extractor._cache[key] = paid
        if self._offline and not self._cached(message, context):
            self.misses += 1
            return []
        return [f for f in self._extractor.extract_facts(message, context) if f]

    def flush(self):
        self._extractor.flush_cache()

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

    def answer(self, _question, memories, today=None, reader=None, turns=None):
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


def history_inputs(convs, extraction, in_order=True):
    """One history from one or more conversations: its messages and the facts
    extracted from each.

    Facts are extracted conversation by conversation in file order (the
    extraction cache is keyed on a message and the three before it). With
    ``in_order`` the sessions are then put in the order they took place,
    which is the order a memory system would have received them in. Each
    message carries its session's date as ``timestamp``, its position in the
    history as ``turn_index`` (from 1), the conversation it is from, and the
    dataset's evidence flag.
    """
    rows = []
    for c_index, conv in enumerate(convs):
        context = []
        for position, m in enumerate(conv.messages):
            content = _msg_content(m)
            if not content or not content.strip():
                continue
            role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else "user")
            facts = extraction.facts(content, context)
            context.append(content)
            date = getattr(m, "date", "") or ""
            rows.append((
                (parse_date(date), c_index, getattr(m, "session", 0), position),
                {"role": role, "content": content, "timestamp": date,
                 "conversation": conv.conv_id,
                 "has_answer": bool(getattr(m, "has_answer", False))},
                facts))
    if in_order and rows and all(key[0] is not None for key, _, _ in rows):
        rows.sort(key=lambda row: row[0])
    messages = [message for _, message, _ in rows]
    for turn, message in enumerate(messages, 1):
        message["turn_index"] = turn
    return messages, [facts for _, _, facts in rows]


def conversation_inputs(conv, extraction, in_order=True):
    """Messages of a conversation and the facts extracted from each."""
    return history_inputs([conv], extraction, in_order)


def truncated(pairs, token_budget):
    """``truncate_to_budget`` over (text, turn) pairs: packed in order,
    skipping what does not fit."""
    kept, used = [], 0
    for text, turn in pairs:
        tokens = estimate_tokens(text)
        if tokens and used + tokens <= token_budget:
            kept.append((text, turn))
            used += tokens
    return kept


def evidence_scores(tasks, arms):
    """Per arm, how much of each question's evidence reached its context.

    The evidence of a question is the set of turns the dataset marks as
    holding its answer; a turn counts as found when a memory extracted from
    it is in the context. Arms that cannot say which turn a memory is from
    are left out, and a gist is from no one turn, so an arm that serves
    gists is understated.
    """
    scores = {}
    for arm in arms:
        rows = [t for t in tasks
                if t["arm"] == arm and t.get("turns") is not None and t.get("evidence")]
        if not rows:
            continue
        share_sum = some = 0.0
        full = {}
        by_type = defaultdict(lambda: [0, 0])
        for t in rows:
            evidence = set(t["evidence"])
            share = len(evidence & {x for x in t["turns"] if x is not None}) / len(evidence)
            share_sum += share
            some += share > 0
            full[(t["conversation_id"], t["query_id"])] = share == 1.0
            by_type[t["question_type"]][0] += 1
            by_type[t["question_type"]][1] += share == 1.0
        n = len(rows)
        scores[arm] = {"questions": n, "recall": share_sum / n, "any": some / n,
                       "all": sum(full.values()) / n, "full": full,
                       "by_type": {k: [v[1], v[0]] for k, v in by_type.items()}}
    return scores


def sdk_pool(mem, embedder, query, user_id, k):
    """The plain-vector pool recall() would rank: nearest records of the scope."""
    embed_query = getattr(embedder, "encode_query", None) or embedder.encode
    hits = mem.engine.search_ann(np.asarray(embed_query(query), dtype=np.float32), k,
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
    ap.add_argument("--local-embedder", default=None, metavar="MODEL",
                    help="embed with this local sentence-transformers model instead of OpenAI "
                         "(e.g. google/embeddinggemma-2); vectors are cached by the embedder's "
                         "fingerprint under WORK_DIR/embeddings_local")
    ap.add_argument("--local-embedder-opt", action="append", default=[], metavar="KEY=VALUE",
                    help="a SentenceTransformerEmbedder argument for --local-embedder, value read "
                         "as JSON (truncate_dim=256, query_prompt_name=\"QuestionAnswering\"); "
                         "may be given several times")
    ap.add_argument("--extractor-model", default="gpt-4.1-nano")
    ap.add_argument("--gist-model", default="gpt-4.1-nano")
    ap.add_argument("--verifier-model", default="gpt-4o-mini")
    ap.add_argument("--judge", choices=["openai", "none"], default="openai")
    ap.add_argument("--judge-model", default="gpt-4.1-mini")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--work-dir", default=None,
                    help="where the verdict cache and the embedding overlay are kept "
                         "(default: .cache/shipped_stack_eval, which is not checked in)")
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
    ap.add_argument("--config", action="append", default=[], metavar="NAME:KEY=VALUE[,...]",
                    help="an extra arm cfg_NAME: sdk_cognitive with these engine settings "
                         "(values are read as JSON); may be given several times")
    ap.add_argument("--variant", choices=["oracle", "s", "m"], default=None,
                    help="read the official file of this variant for the same questions "
                         "(default: the parquet copy, which is the oracle variant)")
    ap.add_argument("--pool", type=int, default=1, metavar="N",
                    help="build each history from N conversations, so every question has the "
                         "other N-1 as distractors (for the evidence score)")
    ap.add_argument("--time-tags", choices=["date", "minute", "turn"], default="date",
                    help="what recall() is given to tag memories with: the session's date "
                         "(default), date and time of day, or nothing but the turn")
    ap.add_argument("--file-order", action="store_true",
                    help="ingest sessions in the order the file lists them, not the order "
                         "they took place")
    ap.add_argument("--no-question-date", dest="question_date", action="store_false",
                    help="do not tell the reader when the question is asked")
    ap.add_argument("--reader", choices=["plain", "notes"], default="plain",
                    help="how the answering model is asked: the harness's plain prompt, or the "
                         "SDK's (tsm.reading: memories in the order they were said, and the "
                         "model works the answer out before giving it)")
    ap.add_argument("--extraction", choices=["shipped", "second", "all", "legacy"],
                    default="shipped",
                    help="whose facts every arm is given: the SDK's extractor as it ships "
                         "(default: its first prompt alone); with its second prompt for a "
                         "message the first finds nothing in (second); with that and the "
                         "statements of a message both find nothing in (all); or the "
                         "harness's earlier extractor, which holds the same facts as the "
                         "default (legacy)")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    configs = {}
    for spec in args.config:
        name, _, rest = spec.partition(":")
        settings = {}
        for pair in filter(None, (x.strip() for x in rest.split(","))):
            key, _, value = pair.partition("=")
            try:
                settings[key.strip()] = json.loads(value)
            except ValueError:
                settings[key.strip()] = value
        configs["cfg_" + name.strip()] = settings
    arms += [name for name in configs if name not in arms]
    known = set(ARMS) | {v for name in configs for v in variants(name)}
    unknown = [a for a in arms if a not in known]
    if unknown:
        ap.error(f"unknown arms: {unknown}")
    if args.pool > 1 and any(a in arms for a in UNATTRIBUTED):
        ap.error("--pool does not support the adapter arms (naive, adapter)")
    if args.offline:
        args.judge = "none"
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not ensure_openai_key() and not args.offline:
        sys.exit(key_file_hint())
    if args.offline and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "offline"  # clients are built but never called

    # A fixed place by default: the embeddings of texts the shared cache does
    # not hold are kept there, so a later run (an --offline one too) finds
    # them instead of paying for them again or running without them. Not
    # under target/, which `cargo clean` removes.
    work_dir = args.work_dir or os.path.join(os.path.dirname(os.path.dirname(HERE)), ".cache",
                                             "shipped_stack_eval")
    os.makedirs(work_dir, exist_ok=True)

    from tsm import Memory
    from tsm.budget import select_under_budget
    from tsm.compaction import is_gist
    from tsm.gist import ExtractiveGistSummarizer, OpenAIGistSummarizer
    from tsm.memory import time_tag
    from tsm.ranking import is_first_person_query, role_prior
    from tsm.verification import LLMVerifier

    extraction = CachedExtraction(args.extractor_model, args.offline, mode=args.extraction)
    if args.local_embedder:
        from tsm.embedders import SentenceTransformerEmbedder

        options = {}
        for spec in args.local_embedder_opt:
            key, _, value = spec.partition("=")
            options[key.strip()] = json.loads(value)
        embedder = LocalEmbeddings(SentenceTransformerEmbedder(args.local_embedder, **options),
                                   os.path.join(work_dir, "embeddings_local"))
        args.embed_model = embedder.fingerprint
    else:
        embedder = Embeddings(args.embed_model, os.path.join(work_dir, "emb_overlay.pkl"),
                              args.offline)
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
    if args.variant:
        convs = load_longmemeval(args.data_dir, variant=args.variant,
                                 question_ids=[c.queries[0].query_id for c in convs if c.queries])
    size = max(1, args.pool)
    units = [convs[i:i + size] for i in range(0, len(convs), size)]
    # What recall() is asked for: dates need the session date in each message.
    tag_mode = "minute" if args.time_tags == "minute" else True
    logger.info("Loaded %d conversations in %d histories. arms=%s budget=%d judge=%s", len(convs),
                len(units), arms, args.token_budget, getattr(judge, "model", "?"))
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
            saved = json.load(fh)
        tasks = saved["tasks"]
        other_judge = saved.get("judge_model") != getattr(judge, "model", "?")
        for t in tasks:
            if other_judge or (t.get("reader") or "plain") != args.reader:
                # The same contexts read another way, or by another judge (a
                # --judge none run saves the proxy's): the earlier verdicts do
                # not apply.
                t["reader"] = args.reader
                t["prediction"] = t["correct"] = None
        arms = [a for a in ARMS if any(t["arm"] == a for t in tasks)]
        units = []

    # One line per finished history, so an interrupted run picks up where it
    # stopped instead of paying for the same verdicts and gists again. It may
    # be resumed with fewer arms; a history that was not run for every arm
    # asked for now is run again.
    checkpoint = args.out + ".partial.jsonl" if args.out and not args.rejudge else None
    setup = {"arms": arms, "token_budget": args.token_budget, "pool_k": args.pool_k,
             "max_records": args.max_records, "user_tokens": args.user_tokens,
             "rolling_passes": args.rolling_passes, "embed_model": args.embed_model,
             "extractor_model": args.extractor_model, "extraction": args.extraction,
             "gist_model": args.gist_model,
             "verifier_model": args.verifier_model, "offline": args.offline,
             "variant": args.variant, "pool": size, "time_tags": args.time_tags,
             "file_order": args.file_order, "question_date": args.question_date,
             "configs": configs, "reader": args.reader}
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
        logger.info("Resuming: %d histories already done", len(done))

    for index, unit in enumerate(units):
        unit_id = unit[0].conv_id
        if unit_id in done:
            continue
        questions = [(conv, q) for conv in unit for q in conv.queries if not q.is_abstention]
        if not questions:
            continue
        messages, per_message = history_inputs(unit, extraction, in_order=not args.file_order)
        if args.time_tags == "turn":
            for m in messages:
                m["timestamp"] = ""
        evidence = defaultdict(set)  # conversation -> the turns that hold its question's answer
        for m in messages:
            if m["has_answer"]:
                evidence[m["conversation"]].add(m["turn_index"])
        user = unit_id if size == 1 else f"pool_{unit_id}"
        conv = unit[0]
        opened = []  # (close, db) of everything built for this history
        retrieve = {}  # arm -> question -> [(text for the prompt, turn it is from or None)]
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
                retrieve["naive"] = lambda q, ad=ad: [
                    (text, None) for text in naive_retrieve(ad, q, args.pool_k, args.token_budget)]
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
                retrieve["adapter"] = lambda q, ad=ad: [
                    (text, None) for text in ad.recall_under_budget(
                        q, user_id=user, token_budget=args.token_budget, method="mmr")]

            plain_arms = [a for a in ("sdk_plain", "sdk_pack", "sdk_pack_capped",
                                      "sdk_pack_tagged") if a in arms]
            if plain_arms:
                mem, db = sdk_memory(profile=None)
                opened.append((mem.close, db))
                mem.add(messages, user_id=user)
                if "sdk_plain" in arms:
                    retrieve["sdk_plain"] = lambda q, mem=mem: truncated(
                        [(p["text"] or "", p["turn_index"])
                         for p in sdk_pool(mem, embedder, q, user, args.pool_k)],
                        args.token_budget)

                def packed(q, mem=mem, cap=None, tags=False):
                    # recall()'s packing on a plain-vector pool of recall()'s size.
                    pool = sdk_pool(mem, embedder, q, user,
                                    max(args.pool_k, 30, args.token_budget // 8))
                    first_person = is_first_person_query(q)
                    for p in pool:
                        p["score"] *= role_prior(first_person, p["role"])
                        if tags:
                            p["context"] = (time_tag(p["timestamp"], p["turn_index"],
                                                     minutes=tag_mode == "minute")
                                            + (p["text"] or ""))
                    pool.sort(key=lambda p: -p["score"])
                    chosen = select_under_budget(
                        pool, args.token_budget,
                        embed=getattr(embedder, "encode_document", None) or embedder.encode,
                        lam=0.7, max_items=cap)
                    return [(p.get("context") or p["text"] or "", p["turn_index"]) for p in chosen]
                if "sdk_pack" in arms:
                    retrieve["sdk_pack"] = packed
                if "sdk_pack_tagged" in arms:
                    retrieve["sdk_pack_tagged"] = lambda q: packed(q, tags=True)
                if "sdk_pack_capped" in arms:
                    old_cap = min(10, max(4, args.token_budget // 35))
                    retrieve["sdk_pack_capped"] = lambda q: packed(q, cap=old_cap)

            def shown(results):
                """What goes in the prompt (a result's context: the text, dated,
                marked when the fact has since changed) and the turn each memory
                is from. A gist is from no one turn."""
                return [(r.get("context") or r["text"] or "",
                         None if is_gist(r.get("role")) else r.get("turn_index"))
                        for r in results]

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
                               pool_k=args.pool_k, time_tags=tag_mode))
                retrieve[name + "_untagged"] = lambda q, mem=mem: shown(
                    mem.recall(q, user_id=user, token_budget=args.token_budget,
                               pool_k=args.pool_k, time_tags=False))
                retrieve[name + "_trunc"] = lambda q, mem=mem: truncated(
                    shown(mem.recall(q, user_id=user, top_k=args.pool_k, time_tags=False)),
                    args.token_budget)

            def rolling(mem):
                """The history in parts, compacting after each."""
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
            for name, settings in configs.items():
                recall_arm(name, **{"refinement_cosine_threshold": None,
                                    "contradiction_cosine_threshold": None, **settings})
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

            for asked_of, q in questions:
                for arm in arms:
                    pairs = retrieve[arm](q.query_text)
                    conv_tasks.append({
                        "arm": arm, "conversation_id": asked_of.conv_id, "query_id": q.query_id,
                        "question_type": q.question_type or "?",
                        "query": q.query_text, "gold": q.answer_text,
                        "asked": (getattr(q, "question_date", "") or "") if args.question_date else "",
                        "reader": args.reader,
                        "retrieved": [text for text, _ in pairs],
                        "turns": None if arm in UNATTRIBUTED else [turn for _, turn in pairs],
                        "evidence": sorted(evidence[asked_of.conv_id])})
            # Only a history every arm handled is scored, so arms stay paired.
            tasks.extend(conv_tasks)
            for arm, counts in conv_stats.items():
                stats[arm].update(counts)
            if checkpoint:
                with open(checkpoint, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"setup": setup, "conversation_id": unit_id,
                                         "tasks": conv_tasks,
                                         "stats": {a: dict(c) for a, c in conv_stats.items()}})
                             + "\n")
            if summarizer is not None:
                summarizer.flush()
        except Exception as e:  # noqa: BLE001 — one bad history must not sink the run
            logger.warning("history %s skipped for every arm: %s: %s", unit_id,
                           type(e).__name__, e)
            skipped.append(unit_id)
        finally:
            for close, db in opened:
                try:
                    close()
                finally:
                    shutil.rmtree(db, ignore_errors=True)
        if (index + 1) % 20 == 0:
            logger.info("ingested %d/%d histories (%.0fs)", index + 1, len(units),
                        time.time() - started)

    for t in tasks:
        stats[t["arm"]]["items"] += len(t["retrieved"])
        stats[t["arm"]]["tokens"] += total_tokens(t["retrieved"])
    if not args.offline:
        extraction.flush()  # what was extracted in this run is paid for: keep it

    judge_model = getattr(judge, "model", "?")

    # The judge sees a question (with its date, when the reader is told it), a
    # context and the gold answer, never the arm: the same ones get the same
    # verdict, so each is judged once.
    def what(task):
        key = [task["query"], task["gold"], task["retrieved"]]
        if task.get("asked"):
            key.append(task["asked"])
        if (task.get("reader") or "plain") != "plain":
            # The SDK's prompt lists the memories in the order they were said,
            # so the turn each is from is part of what the reader is shown.
            key += ["reader:" + task["reader"], task.get("turns")]
        return json.dumps(key, sort_keys=True)

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
            how = {}
            if task.get("asked"):
                how["today"] = task["asked"]
            if (task.get("reader") or "plain") != "plain":
                how["reader"] = task["reader"]
                if task.get("turns"):
                    how["turns"] = task["turns"]
            prediction = judge.answer(task["query"], task["retrieved"], **how)
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

    # Every --config arm against the arms it is a variation of.
    pairs = list(PAIRS) + [(name, base) for name in configs
                           for base in ("sdk_cognitive", "sdk_cognitive_alpha1", "sdk_pack_tagged")]
    evidence_by_arm = evidence_scores(tasks, arms)
    evidence_pairs = {}
    if evidence_by_arm:
        logger.info("-" * 96)
        logger.info("  evidence in the context (free: the turns the dataset marks, no model asked)")
        logger.info("  %-*s %9s  %12s  %12s  %s", wide, "arm", "questions", "all of it",
                    "share of it", "all of it, by type")
        for arm, e in evidence_by_arm.items():
            cells = "  ".join(
                f"{t.replace('single-session-', 'ss-')[:12]} {e['by_type'][t][0]}/{e['by_type'][t][1]}"
                for t in types if t in e["by_type"])
            logger.info("  %-*s %9d  %12.3f  %12.3f  %s", wide, arm, e["questions"], e["all"],
                        e["recall"], cells)
        for arm, base in pairs:
            if arm not in evidence_by_arm or base not in evidence_by_arm:
                continue
            a, b = evidence_by_arm[arm]["full"], evidence_by_arm[base]["full"]
            keys = a.keys() & b.keys()
            wins = sum(a[k] and not b[k] for k in keys)
            losses = sum(b[k] and not a[k] for k in keys)
            evidence_pairs[f"{arm} vs {base}"] = {"wins": wins, "losses": losses,
                                                 "p": round(sign_test(wins, losses), 4)}
            logger.info("  %-*s all of the evidence: %2d / %2d   p=%.2f", 2 * wide + 4,
                        f"{arm} vs {base}", wins, losses, sign_test(wins, losses))

    logger.info("-" * 96)
    logger.info("  question-by-question (only the arm right / only the baseline right):")
    paired = {}
    for arm, base in pairs:
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
        "evidence": {a: {k: v for k, v in e.items() if k != "full"}
                     for a, e in evidence_by_arm.items()},
        "evidence_paired": evidence_pairs,
        "variant": args.variant, "pool": size, "time_tags": args.time_tags,
        "file_order": args.file_order, "question_date": args.question_date,
        "configs": configs, "extraction": args.extraction, "reader": args.reader,
        "judge_model": getattr(judge, "model", "?"), "judge_calls": getattr(judge, "calls", 0),
        "judge_input_tokens": getattr(judge, "input_tokens", 0),
        "judge_output_tokens": getattr(judge, "output_tokens", 0),
        "verifier_model": args.verifier_model if verifier else None,
        "verifier_requests": getattr(verifier, "calls", 0) if verifier else 0,
        "gist_model": getattr(summarizer, "model", None) if summarizer else None,
        "gist_calls": getattr(summarizer, "calls", 0) if summarizer else 0,
        "gists_from_cache": getattr(summarizer, "hits", 0) if summarizer else 0,
        "embedding_cache_misses": embedder.misses,
        "local_embedder": getattr(embedder, "fingerprint", None),
        "local_embeddings_computed": getattr(embedder, "computed", 0),
        "extraction_cache_misses": extraction.misses,
        "extractor_calls": getattr(extraction.instance, "calls", 0),
        "skipped_conversations": skipped,
        "offline": args.offline,
    }
    if args.offline and (embedder.misses or extraction.misses):
        logger.warning(
            "offline: %d texts had no cached embedding and %d messages no cached extraction, so "
            "the numbers above are not what an online run would give. Run once without --offline "
            "(--judge none asks no judge) and with the same --work-dir to fetch them.",
            embedder.misses, extraction.misses)
    logger.info("GATE_SUMMARY: %s", json.dumps(summary))
    save(summary)
    if args.out:
        logger.info("Wrote per-question results to %s", args.out)
    if checkpoint and os.path.exists(checkpoint):
        os.remove(checkpoint)
    embedder.flush()


if __name__ == "__main__":
    main()
