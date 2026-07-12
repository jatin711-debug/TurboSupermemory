#!/usr/bin/env python3
"""A4 — the market number: TSM full stack vs naive-RAG vs Mem0.

The only number outsiders care about: under an identical token budget, on the
same conversations and queries, graded by the same gold-standard judge, does the
TSM cognitive stack answer better than the baselines everyone already has?

Three pluggable backends, MemDelta-conformant (vary ONLY the memory system):
  - naive-RAG : our engine, cognition OFF. Store the same extracted user facts,
                plain vector top-k, truncate to the token budget. The floor.
  - tsm       : the winning stack — role-scoped belief revision + NLI-verified
                supersession with EXCLUDE-from-context (B1) + MMR budget packing
                (B2). Its full pipeline ingests the raw messages.
  - mem0      : Mem0 1.0 with its own extraction/consolidation (gpt-4.1-nano LLM
                + text-embedding-3-small + chroma). Its full pipeline too.

Every system hands the SAME judge (default gpt-4.1-mini) a context capped at the
same token budget. naive + tsm share one disk-cached gpt-4.1-nano extractor so
the fact SUPPLY is identical and only the memory logic differs; Mem0 extracts
for itself because extraction is part of the system being compared.

Usage:
    python benchmarks/cognitive_eval/head_to_head_eval.py --limit 120 \
        --token-budget 150 --pool-k 20 \
        --extractor openai --extractor-model gpt-4.1-nano \
        --mem0-model gpt-4.1-nano --embed-model text-embedding-3-small \
        --judge openai --judge-model gpt-4.1-mini --workers 10
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cognitive_eval.adapters.tsm_adapter import TSMAdapter
from cognitive_eval.benchmark_datasets.longmemeval import load_longmemeval
from cognitive_eval.compress_eval import insert_facts
from cognitive_eval.run_belief_longmemeval import _msg_content, prewarm_extraction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("head_to_head_eval")

SYSTEMS = ("naive", "tsm", "mem0")


def conv_facts(extractor, conv, roles=None):
    """Ordered, de-duplicated atomic facts from messages whose role is in `roles`
    (None = all roles). Cache-backed (key is message content), so with a warmed
    extractor this is free. `roles={"user"}` reproduces conv_user_facts; the
    default all-role supply is the fair naive-RAG floor — the same fact SUPPLY the
    TSM stack ingests (TSM stores every role; belief detection is user-scoped)."""
    facts, ctx = [], []
    for m in conv.messages:
        role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else "user")
        c = _msg_content(m)
        if not c or not c.strip():
            continue
        if roles is None or role in roles:
            facts.extend(extractor.extract_facts(c, ctx))
        ctx.append(c)
    seen, out = set(), []
    for f in facts:
        if f and f not in seen:
            seen.add(f); out.append(f)
    return out


def truncate_to_budget(texts, token_budget):
    """Greedy pack in the given order until the token budget is spent (a text is
    ~len//4 tokens). Shared by naive-RAG and Mem0 so budget accounting matches."""
    out, used = [], 0
    for t in texts:
        if not t or not t.strip():
            continue
        n = max(1, len(t) // 4)
        if used + n <= token_budget:
            out.append(t)
            used += n
    return out


def to_mem0_messages(conv):
    """Conversation -> Mem0 [{role, content}] list (drops empty turns)."""
    out = []
    for m in conv.messages:
        role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else "user")
        c = _msg_content(m)
        if c and c.strip():
            out.append({"role": role if role in ("user", "assistant", "system") else "user",
                        "content": c})
    return out


def make_mem0(model_name, embed_model):
    """Build one Mem0 Memory instance (per-conv scoping via user_id)."""
    from mem0 import Memory
    cfg = {
        "llm": {"provider": "openai", "config": {"model": model_name, "temperature": 0.0}},
        "embedder": {"provider": "openai", "config": {"model": embed_model}},
        "vector_store": {"provider": "chroma", "config": {
            "path": tempfile.mkdtemp(prefix="mem0_h2h_"), "collection_name": "mem0eval"}},
    }
    return Memory.from_config(cfg)


def mem0_add(mem0, messages, user_id, max_retries=4):
    """Robust Mem0 ingestion (its extraction is an LLM call that can rate-limit)."""
    for attempt in range(max_retries):
        try:
            mem0.add(messages, user_id=user_id)
            return True
        except Exception as e:  # noqa: BLE001
            wait = min(5.0 * (2 ** attempt), 60.0)
            logger.warning("mem0.add failed (attempt %d/%d): %s; retry %.0fs",
                           attempt + 1, max_retries, e, wait)
            time.sleep(wait)
    return False


def mem0_ingest(mem0, conv, mode):
    """Feed a conversation to Mem0. `incremental` adds per user(+assistant reply)
    exchange — the way Mem0 is designed to be driven; `oneshot` dumps the whole
    flattened history in a single add() (which we found under-stores badly and is
    NOT how Mem0 is used). Returns True if Mem0 holds at least a partial memory."""
    msgs = to_mem0_messages(conv)
    if not msgs:
        return False
    if mode == "oneshot":
        return mem0_add(mem0, msgs, conv.conv_id)
    i, ok_any = 0, False
    while i < len(msgs):
        chunk = [msgs[i]]
        if i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant":
            chunk.append(msgs[i + 1]); i += 2
        else:
            i += 1
        if mem0_add(mem0, chunk, conv.conv_id):
            ok_any = True
    return ok_any


def mem0_retrieve(mem0, query, user_id, pool_k, token_budget):
    try:
        res = mem0.search(query, user_id=user_id, limit=pool_k)
    except Exception as e:  # noqa: BLE001
        logger.warning("mem0.search failed: %s", e)
        return []
    results = res.get("results", res) if isinstance(res, dict) else res
    texts = [(r.get("memory") if isinstance(r, dict) else str(r)) for r in (results or [])]
    return truncate_to_budget([t for t in texts if t], token_budget)


def naive_retrieve(adapter, query, pool_k, token_budget):
    """Plain vector top-k over the same facts, truncate to budget (no cognition)."""
    results = adapter.search(query, top_k=pool_k, use_cognitive=False)
    return truncate_to_budget([r["text"] for r in results], token_budget)


def main():
    ap = argparse.ArgumentParser(description="A4 head-to-head: TSM vs naive-RAG vs Mem0 (judged)")
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--token-budget", type=int, default=150, help="answer-context token budget")
    ap.add_argument("--pool-k", type=int, default=20, help="retrieval pool size before packing")
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                    help="local embedding model for naive/tsm")
    ap.add_argument("--extractor", type=str, default="openai")
    ap.add_argument("--extractor-model", type=str, default="gpt-4.1-nano")
    ap.add_argument("--mem0-model", type=str, default="gpt-4.1-nano", help="Mem0 LLM")
    ap.add_argument("--embed-model", type=str, default="text-embedding-3-small", help="Mem0 embedder")
    ap.add_argument("--mem0-ingest", choices=["incremental", "oneshot"], default="incremental",
                    help="how to feed Mem0: per-exchange (fair, default) vs single bulk add")
    ap.add_argument("--systems", type=str, default="naive,tsm,mem0")
    ap.add_argument("--naive-facts", choices=["all", "user"], default="all",
                    help="fact supply for the naive-RAG floor: all roles (fair, matches "
                         "TSM's supply) or user-only")
    ap.add_argument("--judge", choices=["auto", "ollama", "openai"], default="openai")
    ap.add_argument("--judge-model", type=str, default=None)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    systems = [s for s in args.systems.split(",") if s.strip() in SYSTEMS]

    from cognitive_eval.judge import create_judge
    jkw = {"ollama_model": args.judge_model, "openai_model": args.judge_model} if args.judge_model else {}
    judge = create_judge(args.judge, **jkw)
    from cognitive_eval.extraction import create_extractor
    ekw = {"openai_model": args.extractor_model} if args.extractor_model else {}
    shared_extractor = create_extractor(args.extractor, **ekw)

    convs = load_longmemeval(args.data_dir)[:args.limit]
    logger.info("Loaded %d conversations. systems=%s token_budget=%d pool_k=%d judge=%s",
                len(convs), systems, args.token_budget, args.pool_k, type(judge).__name__)
    prewarm_extraction(shared_extractor, convs, workers=max(args.workers, 8))
    for name in ("cognitive_eval.adapters.tsm", "httpx", "httpcore", "urllib3", "chromadb",
                 "mem0", "huggingface_hub", "sentence_transformers", "transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)

    mem0 = make_mem0(args.mem0_model, args.embed_model) if "mem0" in systems else None

    # (system, qtype, query, texts, gold) — judged concurrently at the end.
    tasks = []
    model, verifier = None, None
    n_conv = 0
    naive_roles = None if args.naive_facts == "all" else {"user"}
    for ci, conv in enumerate(convs):
        # ---- ingest ----
        facts = conv_facts(shared_extractor, conv, roles=naive_roles)  # cached supply

        # If Mem0 can't ingest this conv, drop it from ALL systems so n stays
        # matched (a fair paired comparison).
        if "mem0" in systems:
            if not mem0_ingest(mem0, conv, args.mem0_ingest):
                logger.warning("skipping conv %s (mem0 ingest failed)", conv.conv_id)
                continue

        naive_ad = tsm_ad = None
        if "naive" in systems:
            db = tempfile.mkdtemp(prefix="tsm_naive_")
            naive_ad = TSMAdapter(db_path=db, embedding_model=args.model, extractor="mock",
                                  cognitive_features=False, belief_revision=False, model=model)
            model = naive_ad.model
            insert_facts(naive_ad, facts, conv.conv_id)
        if "tsm" in systems:
            db = tempfile.mkdtemp(prefix="tsm_full_")
            tsm_ad = TSMAdapter(db_path=db, embedding_model=args.model, extractor=args.extractor,
                                extractor_instance=shared_extractor, cognitive_features=True,
                                belief_revision=True, model=model, belief_source_roles=["user"],
                                verify_demotions=True, verifier=verifier, supersession_mode="exclude")
            model = tsm_ad.model
            verifier = tsm_ad.verifier
            tsm_ad.add(conv.messages, user_id=conv.conv_id)
            tsm_ad.trigger_consolidation()

        # ---- retrieve + build judge tasks ----
        try:
            for q in conv.queries:
                if q.is_abstention:
                    continue
                qt = q.question_type or "?"
                if "naive" in systems:
                    tasks.append(("naive", qt, q.query_text,
                                  naive_retrieve(naive_ad, q.query_text, args.pool_k, args.token_budget),
                                  q.answer_text))
                if "tsm" in systems:
                    tasks.append(("tsm", qt, q.query_text,
                                  tsm_ad.recall_under_budget(q.query_text, token_budget=args.token_budget,
                                                             method="mmr"),
                                  q.answer_text))
                if "mem0" in systems:
                    tasks.append(("mem0", qt, q.query_text,
                                  mem0_retrieve(mem0, q.query_text, conv.conv_id, args.pool_k, args.token_budget),
                                  q.answer_text))
        finally:
            if naive_ad is not None:
                naive_ad.close()
            if tsm_ad is not None:
                tsm_ad.close()
        n_conv += 1
        if (ci + 1) % 20 == 0:
            logger.info("ingested %d/%d convs (%d usable)", ci + 1, len(convs), n_conv)

    logger.info("Judging %d answers across %d systems (%d workers)...",
                len(tasks), len(systems), args.workers)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        verdicts = list(ex.map(lambda t: judge.answer_and_judge(t[2], t[3], t[4]), tasks))

    # agg[system][type] = [n, correct]
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for (system, qt, _q, _t, _g), correct in zip(tasks, verdicts):
        for key in ("__all__", qt):
            agg[system][key][0] += 1
            agg[system][key][1] += int(correct)

    def acc(system, key="__all__"):
        n, c = agg[system][key]
        return (c / n if n else 0.0), n

    logger.info("=" * 92)
    logger.info("A4 HEAD-TO-HEAD — judged answer accuracy @ %d-token budget (n_conv=%d)",
                args.token_budget, n_conv)
    logger.info("-" * 92)
    overall = {}
    for system in systems:
        a, n = acc(system)
        overall[system] = a
        logger.info("  %-6s | judged accuracy %.3f  (n=%d answers)", system, a, n)
    logger.info("-" * 92)
    if "tsm" in overall and "mem0" in overall:
        logger.info("  TSM vs Mem0      : %+.3f", overall["tsm"] - overall["mem0"])
    if "tsm" in overall and "naive" in overall:
        logger.info("  TSM vs naive-RAG : %+.3f", overall["tsm"] - overall["naive"])

    # per-type breakdown
    all_types = sorted({t for system in systems for t in agg[system] if t != "__all__"})
    logger.info("-" * 92)
    logger.info("  %-28s %s", "question_type", "  ".join(f"{s:>7}" for s in systems))
    for t in all_types:
        cells = []
        for system in systems:
            a, n = acc(system, t)
            cells.append(f"{a:>7.2f}")
        n_ref = max(agg[system][t][0] for system in systems)
        logger.info("  %-28s %s  (n=%d)", t[:28], "  ".join(cells), n_ref)
    logger.info("=" * 92)

    best = max(overall, key=overall.get) if overall else None
    if best == "tsm" and len(overall) > 1:
        margin = overall["tsm"] - max(v for k, v in overall.items() if k != "tsm")
        logger.info("A4 VERDICT: TSM stack WINS the head-to-head (+%.3f over the best baseline) — "
                    "the cognitive layer beats the market baseline at equal budget.", margin)
    elif best == "tsm":
        logger.info("A4 VERDICT: TSM measured alone at %.3f (add baselines to compare).", overall["tsm"])
    else:
        logger.info("A4 VERDICT: TSM does NOT lead (%s leads). Honest result — the market baseline "
                    "matches or beats the stack at this budget; report and diagnose.", best)
    logger.info("GATE_SUMMARY: %s", json.dumps({
        "token_budget": args.token_budget, "pool_k": args.pool_k, "n_conv": n_conv,
        "systems": systems, "overall": {s: round(overall[s], 4) for s in systems},
        "judge_calls": judge.calls,
        "mem0_model": args.mem0_model, "mem0_ingest": args.mem0_ingest,
        "naive_facts": args.naive_facts, "extractor_model": args.extractor_model,
        "judge_model": getattr(judge, "model", "?")}))


if __name__ == "__main__":
    main()
