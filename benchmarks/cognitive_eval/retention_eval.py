#!/usr/bin/env python3
"""LongMemEval — retention / forgetting under a memory budget (W5).

Reinforcement showed no DIRECT ranking lift (Phase 4). Its real claim is the
retain/forget axis: under a bounded store, a memory that gets USED (retrieved /
rehearsed) should survive eviction over one that doesn't — so the facts a user
keeps coming back to stay available, and dead weight is forgotten. Nobody had
tested that. This eval does, with the same disciplined ON-vs-OFF isolation.

Both arms are IDENTICAL in every operation — same corpus, same insertion order,
same rehearsals of the query-relevant facts, same budget (`max_records` forces
eviction). They differ in ONE engine switch:

  - ON  (`access_aware_eviction=True`):  evict by cognitive salience — a
    rehearsed/retrieved memory (high access_count, recently accessed) survives.
  - OFF (`access_aware_eviction=False`): naive FIFO — evict oldest-inserted
    first, ignoring access entirely (the "database, not memory" baseline).

Procedure per conversation (only those with > budget user-facts, so eviction
actually bites): insert all user facts → rehearse each query's text a few times
(bumps access on the facts that query needs, in BOTH arms) → consolidate (evict)
→ measure. Metrics:

  - survival: is the gold-answer fact still ALIVE after eviction?
  - hit@k:    is it still RETRIEVED (end-to-end retain + recall)?

Lift (ON - OFF) isolates whether access-aware retention keeps the right memories.

Usage:
    python benchmarks/cognitive_eval/retention_eval.py --budget 10 --limit 200
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cognitive_eval.adapters.tsm_adapter import TSMAdapter
from cognitive_eval.benchmark_datasets.longmemeval import load_longmemeval
from cognitive_eval.run_belief_longmemeval import hit_at, key_tokens

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("retention_eval")


def gold_memory_ids(answer, id_to_text):
    """Ids of inserted facts whose text carries the gold answer's distinctive
    token — the memories that must survive for the query to be answerable."""
    kts = key_tokens(answer)
    if not kts:
        return []
    tok = kts[0]
    return [mid for mid, text in id_to_text.items() if tok in (text or "").lower()]


def run_arm(access_aware, conversations, budget, model_name, top_k, extractor, shared_model=None):
    """Identical operations in both arms; only `access_aware_eviction` differs."""
    agg = {"n": 0, "survived": 0, "h1": 0, "h3": 0, "hk": 0}
    model = shared_model
    pressured_convs = 0
    for conv in conversations:
        db_path = tempfile.mkdtemp(prefix="tsm_ret_")
        adapter = TSMAdapter(db_path=db_path, embedding_model=model_name, extractor=extractor,
                             cognitive_features=True, belief_revision=True, model=model,
                             belief_source_roles=["user"], max_records=budget,
                             access_aware_eviction=access_aware, recency_half_life_secs=3600)
        model = adapter.model
        try:
            adapter.add(conv.messages, user_id=conv.conv_id)
            n_facts = len(adapter._id_to_text)
            if n_facts <= budget:
                continue  # no budget pressure — eviction wouldn't fire
            pressured_convs += 1
            id_to_text = dict(adapter._id_to_text)

            # Rehearse: the facts a user keeps needing get accessed. Search each
            # query's text a few times to bump access on its gold fact(s).
            for q in conv.queries:
                if q.is_abstention:
                    continue
                for _ in range(3):
                    adapter.search(q.query_text, user_id=conv.conv_id, top_k=top_k, use_cognitive=False)

            adapter.trigger_consolidation()  # eviction fires under the budget

            for q in conv.queries:
                if q.is_abstention:
                    continue
                agg["n"] += 1
                golds = gold_memory_ids(q.answer_text, id_to_text)
                # survival: did any gold-bearing memory stay alive post-eviction?
                survived = any(adapter.engine.contains_id(mid) for mid in golds) if golds else False
                agg["survived"] += int(survived)
                res = adapter.search(q.query_text, user_id=conv.conv_id, top_k=top_k, use_cognitive=True)
                texts = [r.get("text", "") for r in res]
                agg["h1"] += int(hit_at(q.answer_text, texts, 1))
                agg["h3"] += int(hit_at(q.answer_text, texts, 3))
                agg["hk"] += int(hit_at(q.answer_text, texts, top_k))
        finally:
            adapter.close()
            shutil.rmtree(db_path, ignore_errors=True)
    return agg, pressured_convs, model


def main():
    ap = argparse.ArgumentParser(description="LongMemEval retention isolation (access-aware vs FIFO eviction)")
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--limit", type=int, default=200, help="cap #conversations (full set OOMs, see W7)")
    ap.add_argument("--budget", type=int, default=10, help="max_records cap (facts above this are evicted)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--extractor", type=str, default="mock", choices=["mock", "ollama"])
    args = ap.parse_args()

    convs = load_longmemeval(args.data_dir)
    if args.limit:
        convs = convs[:args.limit]
    logger.info("Loaded %d conversations. budget(max_records)=%d model=%s top_k=%d",
                len(convs), args.budget, args.model, args.top_k)
    for name in ("cognitive_eval.adapters.tsm", "httpx", "httpcore", "urllib3",
                 "huggingface_hub", "sentence_transformers", "transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)

    logger.info("Running OFF arm (access_aware_eviction=False — naive FIFO)...")
    off, off_p, model = run_arm(False, convs, args.budget, args.model, args.top_k, args.extractor)
    logger.info("Running ON arm (access_aware_eviction=True — retain-what-is-used)...")
    on, on_p, _ = run_arm(True, convs, args.budget, args.model, args.top_k, args.extractor, shared_model=model)

    n = max(on["n"], off["n"]) or 1
    logger.info("=" * 92)
    logger.info("Budget-pressured conversations: OFF %d / ON %d | scored queries: %d", off_p, on_p, n)
    logger.info("%-16s | %-10s | %-10s | %-10s", "metric", "OFF", "ON", "lift")
    logger.info("-" * 92)

    def row(label, key):
        of = off[key] / (off["n"] or 1)
        oo = on[key] / (on["n"] or 1)
        logger.info("%-16s | %-10.2f | %-10.2f | %+.2f", label, of, oo, oo - of)
        return oo - of

    surv_lift = row("gold survival", "survived")
    row("hit@1", "h1")
    row("hit@3", "h3")
    hk_lift = row("hit@k", "hk")
    logger.info("=" * 92)
    if surv_lift > 0.05 or hk_lift > 0.05:
        logger.info("VERDICT: access-aware eviction MEASURABLY retains the right memories under budget "
                    "pressure — the retain/forget mechanism works.")
    else:
        logger.info("VERDICT: access-aware eviction shows ~no retention lift over FIFO (honest negative).")
    logger.info("GATE_SUMMARY: %s", json.dumps({
        "budget": args.budget,
        "pressured_convs_on": on_p,
        "scored_queries": n,
        "survival_lift": round(surv_lift, 4),
        "hitk_lift": round(hk_lift, 4),
        "survival_off": round(off["survived"] / (off["n"] or 1), 4),
        "survival_on": round(on["survived"] / (on["n"] or 1), 4),
    }))


if __name__ == "__main__":
    main()
