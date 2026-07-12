#!/usr/bin/env python3
"""Bounded-storage head-to-head: TSM-compress vs naive-delete vs Mem0 (judged).

The moat test. Every prior head-to-head (A4) let naive/TSM keep EVERY fact — no
storage pressure, so the "what to keep" question never bound. Real long-lived
memory is bounded: you cannot keep everything, and the system must decide what to
drop. This measures whose compaction strategy answers best under ONE shared
memory budget, graded by the gold-standard judge.

Built on the CLEAN, oracle-free lever (B4 compression), NOT the oracle-tainted
smart-eviction (A2 depended on rehearsing the eval queries). Survivors are chosen
by recency in BOTH the naive and TSM arms — identical — so the ONLY difference is
what happens to the EVICTED overflow:
  - naive-delete : drop it (store = the B most-recent facts).
  - tsm-compress : keep B-1 recent facts + ONE gist of the evicted tail (B4).
  - mem0         : its native incremental extraction + LLM consolidation, which
                   self-compacts the whole history (its own answer to bounded
                   memory). We report its actual retained memory count so budgets
                   are visibly comparable.

All three retrieve under the same 150-token context (plain top-k — ranking is
commodity on strong embeddings, see A4 leveling) and answer via the same judge.
naive/TSM embed with the SAME OpenAI model Mem0 uses (no embedder asymmetry).

Usage:
    python benchmarks/cognitive_eval/bounded_head_to_head.py --limit 120 --budget 8 \
        --token-budget 150 --embed-model text-embedding-3-small \
        --extractor openai --extractor-model gpt-4.1-nano --gist-model gpt-4.1-nano \
        --judge openai --judge-model gpt-4.1-mini --workers 10
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cognitive_eval.adapters.tsm_adapter import TSMAdapter
from cognitive_eval.benchmark_datasets.longmemeval import load_longmemeval
from cognitive_eval.compress_eval import insert_facts
from cognitive_eval.head_to_head_eval import (conv_facts, truncate_to_budget, make_mem0,
                                              mem0_ingest)
from cognitive_eval.run_belief_longmemeval import key_tokens, prewarm_extraction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("bounded_head_to_head")

SYSTEMS = ("naive", "tsm", "mem0")


def _mem0_all(mem0, user_id):
    try:
        r = mem0.get_all(user_id=user_id)
        return r.get("results", r) if isinstance(r, dict) else (r or [])
    except Exception:  # noqa: BLE001
        return []


def mem0_count(mem0, user_id):
    return len(_mem0_all(mem0, user_id))


def _recent_texts(memories, budget):
    """The `budget` most-recent memory texts (by created_at if present, else list
    order) — the SAME recency policy naive/TSM use, so Mem0 competes at equal slots."""
    def created(m):
        return (m.get("created_at") or m.get("updated_at") or "") if isinstance(m, dict) else ""
    if memories and isinstance(memories[0], dict) and (memories[0].get("created_at")):
        memories = sorted(memories, key=created)
    recent = memories[-budget:]
    return {(m.get("memory") if isinstance(m, dict) else str(m)) for m in recent}


def mem0_bounded_retrieve(mem0, query, user_id, budget, pool_k, token_budget):
    """Retrieve from ONLY Mem0's `budget` most-recent memories, so it operates at
    the same shared storage budget as the naive/TSM arms (Mem0 self-compacts to a
    larger set otherwise, which would be an unfair slot advantage)."""
    allowed = _recent_texts(_mem0_all(mem0, user_id), budget)
    try:
        res = mem0.search(query, user_id=user_id, limit=pool_k * 3)
    except Exception as e:  # noqa: BLE001
        logger.warning("mem0.search failed: %s", e)
        return []
    results = res.get("results", res) if isinstance(res, dict) else res
    texts = [(r.get("memory") if isinstance(r, dict) else str(r)) for r in (results or [])]
    texts = [t for t in texts if t and t in allowed]
    return truncate_to_budget(texts, token_budget)


def plain_retrieve(adapter, query, pool_k, token_budget):
    """Plain top-k over the stored facts, truncate to budget (no cognition/MMR)."""
    results = adapter.search(query, top_k=pool_k, use_cognitive=False)
    return truncate_to_budget([r["text"] for r in results], token_budget)


def main():
    ap = argparse.ArgumentParser(description="Bounded-storage head-to-head (judged)")
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--budget", type=int, default=8, help="shared storage budget (memory slots)")
    ap.add_argument("--token-budget", type=int, default=150, help="answer-context token budget")
    ap.add_argument("--pool-k", type=int, default=20)
    ap.add_argument("--local-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--embed-model", type=str, default="text-embedding-3-small",
                    help="OpenAI embedder for naive/TSM (matches Mem0) — levels the field")
    ap.add_argument("--extractor", type=str, default="openai")
    ap.add_argument("--extractor-model", type=str, default="gpt-4.1-nano")
    ap.add_argument("--gist-model", type=str, default="gpt-4.1-nano")
    ap.add_argument("--mem0-model", type=str, default="gpt-4.1-nano")
    ap.add_argument("--mem0-path", type=str, default=None,
                    help="persistent chroma dir for Mem0 so a killed run RESUMES (skips "
                         "already-ingested convs) instead of re-ingesting from scratch")
    ap.add_argument("--systems", type=str, default="naive,tsm,mem0")
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
    extractor = create_extractor(args.extractor, **ekw)
    from cognitive_eval.gist import create_gister
    gister = create_gister("openai", model=args.gist_model)
    from cognitive_eval.openai_embedder import OpenAIEmbedder
    embed_model = OpenAIEmbedder(model=args.embed_model)  # shared by naive + TSM

    convs = load_longmemeval(args.data_dir)[:args.limit]
    logger.info("Loaded %d convs. budget=%d token_budget=%d systems=%s embed=%s judge=%s",
                len(convs), args.budget, args.token_budget, systems, args.embed_model, type(judge).__name__)
    prewarm_extraction(extractor, convs, workers=max(args.workers, 8))
    for name in ("cognitive_eval.adapters.tsm", "httpx", "httpcore", "urllib3", "chromadb",
                 "mem0", "huggingface_hub", "sentence_transformers", "transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)

    mem0 = done_ids = done_path = None
    if "mem0" in systems:
        mem0 = make_mem0(args.mem0_model, args.embed_model, path=args.mem0_path)
        done_ids = set()
        if args.mem0_path:
            done_path = os.path.join(args.mem0_path, "_completed.json")
            if os.path.exists(done_path):
                try:
                    done_ids = set(json.load(open(done_path)))
                    logger.info("Mem0 RESUME: %d convs already ingested at %s", len(done_ids), args.mem0_path)
                except Exception:  # noqa: BLE001
                    done_ids = set()

    tasks = []          # (system, in_evicted, qtype, query, texts, gold)
    mem_sizes = defaultdict(list)   # system -> [stored count per conv]
    pressured = 0
    for ci, conv in enumerate(convs):
        facts = conv_facts(extractor, conv)          # all-role, cached
        if len(facts) <= args.budget:
            continue                                  # no storage pressure — skip
        pressured += 1

        # Shared recency survivors; arms differ ONLY on the evicted overflow (B4).
        survivors = facts[-(args.budget - 1):]
        tail = facts[:-(args.budget - 1)]
        delete_store = facts[-args.budget:]                        # naive: B recent facts
        gist = gister.summarize(tail)
        compress_store = list(survivors) + ([gist] if gist else [])  # tsm: B-1 recent + 1 gist

        def in_evicted(ans):
            kts = key_tokens(ans)
            if not kts:
                return False
            tok = kts[0]
            return any(tok in (e or "").lower() for e in tail) and \
                not any(tok in (e or "").lower() for e in delete_store)

        # Mem0: native compaction of the whole history, then capped to `budget`
        # most-recent memories at retrieval so it competes at the same slot count.
        if "mem0" in systems:
            if conv.conv_id not in done_ids:            # resume: skip if already ingested
                if not mem0_ingest(mem0, conv, "incremental"):
                    logger.warning("mem0 ingest failed for %s — skipping conv from ALL arms", conv.conv_id)
                    pressured -= 1
                    continue
                done_ids.add(conv.conv_id)
                if done_path:
                    try:
                        json.dump(sorted(done_ids), open(done_path, "w"))
                    except Exception:  # noqa: BLE001
                        pass
            native = mem0_count(mem0, conv.conv_id)
            mem_sizes["mem0_native"].append(native)
            mem_sizes["mem0"].append(min(native, args.budget))

        stores = {"naive": delete_store, "tsm": compress_store}
        adapters = {}
        for name in ("naive", "tsm"):
            if name not in systems:
                continue
            db = tempfile.mkdtemp(prefix=f"tsm_bnd_{name}_")
            ad = TSMAdapter(db_path=db, embedding_model=args.local_model, extractor="mock",
                            cognitive_features=False, belief_revision=False, model=embed_model)
            insert_facts(ad, stores[name], conv.conv_id)
            adapters[name] = (ad, db)
            mem_sizes[name].append(len(stores[name]))

        try:
            for q in conv.queries:
                if q.is_abstention:
                    continue
                ev = in_evicted(q.answer_text)
                qt = q.question_type or "?"
                for name in ("naive", "tsm"):
                    if name in adapters:
                        texts = plain_retrieve(adapters[name][0], q.query_text, args.pool_k, args.token_budget)
                        tasks.append((name, ev, qt, q.query_text, texts, q.answer_text))
                if "mem0" in systems:
                    texts = mem0_bounded_retrieve(mem0, q.query_text, conv.conv_id, args.budget,
                                                  args.pool_k, args.token_budget)
                    tasks.append(("mem0", ev, qt, q.query_text, texts, q.answer_text))
        finally:
            for ad, db in adapters.values():
                ad.close()
                shutil.rmtree(db, ignore_errors=True)
        if (ci + 1) % 20 == 0:
            logger.info("processed %d/%d convs (%d pressured)", ci + 1, len(convs), pressured)

    logger.info("Judging %d answers (%d workers)...", len(tasks), args.workers)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        verdicts = list(ex.map(lambda t: judge.answer_and_judge(t[3], t[4], t[5]), tasks))

    # agg[subset][system] = [n, correct]
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for (system, ev, _qt, _q, _t, _g), correct in zip(tasks, verdicts):
        for subset in ("all", "answer_in_evicted" if ev else "answer_in_survivors"):
            agg[subset][system][0] += 1
            agg[subset][system][1] += int(correct)

    def acc(subset, system):
        n, c = agg[subset][system]
        return (c / n if n else 0.0), n

    logger.info("=" * 94)
    logger.info("BOUNDED HEAD-TO-HEAD — judged accuracy @ budget=%d slots, %d-tok context (pressured=%d)",
                args.budget, args.token_budget, pressured)
    for system in systems:
        sizes = mem_sizes.get(system, [])
        avg = sum(sizes) / len(sizes) if sizes else 0.0
        extra = ""
        if system == "mem0" and mem_sizes.get("mem0_native"):
            nat = mem_sizes["mem0_native"]
            extra = f"  (native avg={sum(nat) / len(nat):.1f}, capped to budget)"
        logger.info("  mem size  %-6s avg=%.1f slots%s", system, avg, extra)
    logger.info("-" * 94)
    overall = {}
    for subset in ("all", "answer_in_evicted", "answer_in_survivors"):
        cells = []
        for system in systems:
            a, n = acc(subset, system)
            if subset == "all":
                overall[system] = a
            cells.append(f"{system}={a:.3f}")
        n_ref = max((agg[subset][s][0] for s in systems), default=0)
        logger.info("  %-20s | %s  (n=%d)", subset, "  ".join(cells), n_ref)
    logger.info("-" * 94)
    if "tsm" in overall and "mem0" in overall:
        logger.info("  TSM-compress vs Mem0   : %+.3f", overall["tsm"] - overall["mem0"])
    if "tsm" in overall and "naive" in overall:
        logger.info("  TSM-compress vs naive  : %+.3f", overall["tsm"] - overall["naive"])
    logger.info("=" * 94)
    best = max(overall, key=overall.get) if overall else None
    if best == "tsm" and len(overall) > 1:
        margin = overall["tsm"] - max(v for k, v in overall.items() if k != "tsm")
        logger.info("VERDICT: under bounded storage, TSM's gist-compression WINS (+%.3f over best baseline) "
                    "— compressing the overflow beats deleting it AND beats Mem0's consolidation.", margin)
    else:
        logger.info("VERDICT: TSM-compress does NOT lead under bounded storage (%s leads) — honest result.", best)
    logger.info("GATE_SUMMARY: %s", json.dumps({
        "budget": args.budget, "token_budget": args.token_budget, "pressured": pressured,
        "systems": systems, "embedder": args.embed_model,
        "overall": {s: round(overall.get(s, 0.0), 4) for s in systems},
        "mem_size_avg": {s: round(sum(mem_sizes[s]) / len(mem_sizes[s]), 2) if mem_sizes.get(s) else 0
                         for s in systems},
        "evicted": {s: round(acc("answer_in_evicted", s)[0], 4) for s in systems},
        "judge_calls": judge.calls, "gist_calls": gister.calls,
        "judge_model": getattr(judge, "model", "?")}))


if __name__ == "__main__":
    main()
