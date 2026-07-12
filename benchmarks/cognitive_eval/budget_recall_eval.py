#!/usr/bin/env python3
"""B2 — budget-aware recall: submodular (MMR) selection vs naive truncation.

Both our proven wins (A2 retention, B1 verified-exclude) live at SMALL context
budgets. B2 asks the next question: given a fixed token budget, does choosing a
DIVERSE, high-coverage set (submodular Maximal-Marginal-Relevance, PACMS 2026)
beat taking the top-scoring memories until the budget is spent (truncation)?

Isolation: identical corpus, identical retrieval pool per query, identical
budget — the ONLY difference is the selection objective (relevance-only vs
relevance-minus-redundancy). Answers are graded by the gold-standard LLM judge.
Uses the best config from Phase A/B1: role-filtered belief + NLI-verified
exclusion of stale facts.

Usage:
    python benchmarks/cognitive_eval/budget_recall_eval.py --limit 120 \
        --budgets 50,100,150 --extractor openai --extractor-model gpt-4.1-nano \
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
from cognitive_eval.run_belief_longmemeval import prewarm_extraction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("budget_recall_eval")

METHODS = ("truncate", "mmr")


def main():
    ap = argparse.ArgumentParser(description="B2 budget-aware recall: MMR vs truncation (judged)")
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--budgets", type=str, default="50,100,150", help="token budgets, comma-separated")
    ap.add_argument("--pool-k", type=int, default=20, help="candidate pool size to select from")
    ap.add_argument("--lam", type=float, default=0.7, help="MMR relevance vs diversity tradeoff")
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--extractor", type=str, default="openai", choices=["mock", "ollama", "openai", "auto"])
    ap.add_argument("--extractor-model", type=str, default=None)
    ap.add_argument("--judge", choices=["auto", "ollama", "openai"], default="openai")
    ap.add_argument("--judge-model", type=str, default=None)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    budgets = sorted({int(x) for x in args.budgets.split(",") if x.strip()})

    from cognitive_eval.judge import create_judge
    jkw = {"ollama_model": args.judge_model, "openai_model": args.judge_model} if args.judge_model else {}
    judge = create_judge(args.judge, **jkw)
    from cognitive_eval.extraction import create_extractor
    ekw = {"openai_model": args.extractor_model} if args.extractor_model else {}
    shared_extractor = create_extractor(args.extractor, **ekw)

    convs = load_longmemeval(args.data_dir)[:args.limit]
    logger.info("Loaded %d conversations. budgets=%s pool_k=%d lam=%.2f judge=%s",
                len(convs), budgets, args.pool_k, args.lam, type(judge).__name__)
    prewarm_extraction(shared_extractor, convs, workers=max(args.workers, 8))
    for name in ("cognitive_eval.adapters.tsm", "httpx", "httpcore", "urllib3",
                 "huggingface_hub", "sentence_transformers", "transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # (type, budget, method, texts, gold) — judged concurrently at the end.
    tasks = []
    model = None
    verifier = None
    for conv in convs:
        db = tempfile.mkdtemp(prefix="tsm_b2_")
        adapter = TSMAdapter(db_path=db, embedding_model=args.model, extractor=args.extractor,
                             extractor_instance=shared_extractor, cognitive_features=True,
                             belief_revision=True, model=model, belief_source_roles=["user"],
                             verify_demotions=True, verifier=verifier, supersession_mode="exclude")
        model = adapter.model
        verifier = adapter.verifier
        try:
            adapter.add(conv.messages, user_id=conv.conv_id)
            adapter.trigger_consolidation()
            for q in conv.queries:
                if q.is_abstention:
                    continue
                pool = adapter.search(q.query_text, user_id=conv.conv_id, top_k=args.pool_k, use_cognitive=True)
                if not pool:
                    continue
                for b in budgets:
                    for meth in METHODS:
                        texts = adapter.recall_under_budget(q.query_text, token_budget=b,
                                                            method=meth, lam=args.lam, pool=pool)
                        tasks.append((q.question_type or "?", b, meth, q.query_text, texts, q.answer_text))
        finally:
            adapter.close()
            shutil.rmtree(db, ignore_errors=True)

    logger.info("Judging %d (query x budget x method) answers concurrently (%d workers)...",
                len(tasks), args.workers)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        verdicts = list(ex.map(lambda t: judge.answer_and_judge(t[3], t[4], t[5]), tasks))

    # agg[budget][method] = {type: [n, c]}
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    for (t, b, meth, _q, _texts, _gold), correct in zip(tasks, verdicts):
        cell = agg[b][meth][t]
        cell[0] += 1
        cell[1] += int(correct)

    logger.info("=" * 88)
    summary = {}
    for b in budgets:
        types = sorted(set(agg[b]["truncate"]) | set(agg[b]["mmr"]))
        n_tot = sum(agg[b]["mmr"][t][0] for t in types) or 1
        acc = {m: sum(agg[b][m][t][1] for t in types) / (sum(agg[b][m][t][0] for t in types) or 1)
               for m in METHODS}
        lift = acc["mmr"] - acc["truncate"]
        logger.info("budget=%-4d tok | truncate acc %.2f | MMR acc %.2f | lift %+.2f  (n=%d)",
                    b, acc["truncate"], acc["mmr"], lift, n_tot)
        summary[str(b)] = {"truncate": round(acc["truncate"], 4), "mmr": round(acc["mmr"], 4),
                           "lift": round(lift, 4), "n": n_tot}
    logger.info("=" * 88)
    best = max((summary[str(b)]["lift"] for b in budgets), default=0.0)
    if best > 0.05:
        logger.info("B2 VERDICT: submodular (MMR) budget-aware recall MEASURABLY beats truncation "
                    "(+%.2f judged at best budget) — diversity-aware selection is real value.", best)
    else:
        logger.info("B2 VERDICT: MMR shows ~no judged gain over truncation at these budgets "
                    "(honest negative — the pool may lack redundancy, or budgets too tight).")
    logger.info("GATE_SUMMARY: %s", json.dumps({"budgets": budgets, "pool_k": args.pool_k,
                "lam": args.lam, "by_budget": summary, "judge_calls": judge.calls}))


if __name__ == "__main__":
    main()
