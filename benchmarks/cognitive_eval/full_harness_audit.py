"""Full Harness Audit: TSM vs Mem0 across token budgets (150, 300, 600, Unconstrained).

Validates Mem0 official usage against TSM under identical embeddings, identical
questions, and identical judge models.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cognitive_eval.adapters.tsm_adapter import TSMAdapter
from cognitive_eval.benchmark_datasets.longmemeval import load_longmemeval
from cognitive_eval.judge import create_judge
from cognitive_eval.openai_embedder import OpenAIEmbedder
from cognitive_eval.extraction.mock import MockExtractor
from cognitive_eval.head_to_head_eval import (
    conv_facts, make_mem0, mem0_ingest, mem0_retrieve, naive_retrieve, to_mem0_messages, truncate_to_budget
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("full_harness_audit")


def main():
    ap = argparse.ArgumentParser(description="Full Harness Audit: TSM vs Mem0 across token budgets")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--budgets", type=str, default="150,300,600", help="Comma-separated budgets")
    ap.add_argument("--mem0-path", type=str, default="./mem0_eval_db")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    budgets = [int(b.strip()) for b in args.budgets.split(",") if b.strip()]

    # Load dataset
    convs = load_longmemeval(None)[:args.limit]
    logger.info("Loaded %d conversations for audit.", len(convs))

    # Embedder & Judge
    model = OpenAIEmbedder(model="text-embedding-3-small")
    judge = create_judge("openai", openai_model="gpt-4o-mini")
    shared_extractor = MockExtractor(split_sentences=True)

    # Initialize Mem0
    mem0 = make_mem0(
        model_name="gpt-4.1-nano",
        embed_model="text-embedding-3-small",
        path=args.mem0_path,
        llm_provider="openai",
        embed_provider="openai",
    )

    # Ensure all convs are ingested in mem0
    done_ids = set()
    for fname in ("_completed.json", "ingested_conv_ids.json"):
        p = os.path.join(args.mem0_path, fname)
        if os.path.exists(p):
            try:
                done_ids.update(json.load(open(p)))
            except Exception:
                pass

    for conv in convs:
        if conv.conv_id not in done_ids:
            mem0_ingest(mem0, conv, "incremental")
            done_ids.add(conv.conv_id)

    # Pre-ingest TSM and Naive for all conversations
    tsm_adapters = {}
    naive_adapters = {}
    logger.info("Pre-ingesting TSM and Naive adapters across %d conversations...", len(convs))
    
    for conv in convs:
        # TSM Adapter
        db_tsm = tempfile.mkdtemp(prefix="tsm_audit_full_")
        tsm_ad = TSMAdapter(
            db_path=db_tsm,
            embedding_model="text-embedding-3-small",
            extractor="mock",
            extractor_instance=shared_extractor,
            cognitive_features=True,
            belief_revision=True,
            model=model,
            dimension=1536,
            belief_source_roles=["user"],
            verify_demotions=False,
            supersession_mode="exclude",
        )
        tsm_ad.add(conv.messages, user_id=conv.conv_id)
        tsm_ad.trigger_consolidation()
        tsm_adapters[conv.conv_id] = tsm_ad

        # Naive Adapter
        db_naive = tempfile.mkdtemp(prefix="tsm_audit_naive_")
        naive_ad = TSMAdapter(
            db_path=db_naive,
            embedding_model="text-embedding-3-small",
            extractor="mock",
            extractor_instance=shared_extractor,
            cognitive_features=False,
            belief_revision=False,
            model=model,
            dimension=1536,
        )
        facts = conv_facts(shared_extractor, conv, roles=None)
        from cognitive_eval.head_to_head_eval import insert_facts
        insert_facts(naive_ad, facts, conv.conv_id)
        naive_adapters[conv.conv_id] = naive_ad

    # Evaluate across each budget
    results_by_budget = {}

    for budget in budgets:
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATING TOKEN BUDGET: %d TOKENS", budget)
        logger.info("=" * 80)

        tasks = []
        for conv in convs:
            tsm_ad = tsm_adapters[conv.conv_id]
            naive_ad = naive_adapters[conv.conv_id]

            for q in conv.queries:
                if q.is_abstention:
                    continue
                qt = q.question_type or "?"

                # 1. TSM Retrieval (MMR budget packed)
                tsm_memories = tsm_ad.recall_under_budget(
                    q.query_text, user_id=conv.conv_id, token_budget=budget, method="mmr"
                )
                tasks.append(("tsm", qt, q.query_text, tsm_memories, q.answer_text))

                # 2. Mem0 Retrieval (Top-k truncated to budget)
                mem0_memories = mem0_retrieve(mem0, q.query_text, conv.conv_id, 20, budget)
                tasks.append(("mem0", qt, q.query_text, mem0_memories, q.answer_text))

                # 3. Naive-RAG Retrieval
                naive_memories = naive_retrieve(naive_ad, q.query_text, 20, budget)
                tasks.append(("naive", qt, q.query_text, naive_memories, q.answer_text))

        logger.info("Judging %d answers across 3 systems (budget=%d tokens)...", len(tasks), budget)

        def eval_task(t):
            sys_name, qt, query, memories, gold = t
            is_corr = judge.answer_and_judge(query, memories, gold)
            return sys_name, qt, is_corr

        scored = defaultdict(lambda: {"correct": 0, "total": 0})
        scored_by_type = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for sys_name, qt, is_corr in pool.map(eval_task, tasks):
                scored[sys_name]["total"] += 1
                if is_corr:
                    scored[sys_name]["correct"] += 1
                scored_by_type[qt][sys_name]["total"] += 1
                if is_corr:
                    scored_by_type[qt][sys_name]["correct"] += 1

        acc_summary = {}
        for s in ("tsm", "mem0", "naive"):
            t = scored[s]["total"]
            c = scored[s]["correct"]
            acc_summary[s] = (c / t) if t else 0.0
            logger.info("  %-6s @ %4d tok | judged accuracy: %.3f (%d/%d)", s, budget, acc_summary[s], c, t)

        results_by_budget[budget] = {
            "overall": acc_summary,
            "by_type": {
                qt: {s: scored_by_type[qt][s]["correct"] / max(1, scored_by_type[qt][s]["total"]) for s in ("tsm", "mem0", "naive")}
                for qt in scored_by_type
            }
        }

    # Clean up adapters
    for ad in list(tsm_adapters.values()) + list(naive_adapters.values()):
        try:
            ad.close()
        except Exception:
            pass

    # Print Final Comparative Matrix
    logger.info("\n" + "=" * 90)
    logger.info("FINAL HARNESS AUDIT SUMMARY: TSM vs MEM0 vs NAIVE-RAG ACROSS BUDGETS (n_conv=%d)", args.limit)
    logger.info("=" * 90)
    logger.info("%-16s | %-16s | %-16s | %-16s | %-16s", "Token Budget", "TSM (CUDA)", "Mem0 (Official)", "Naive-RAG", "TSM vs Mem0 Delta")
    logger.info("-" * 90)
    for b in budgets:
        r = results_by_budget[b]["overall"]
        delta = r["tsm"] - r["mem0"]
        logger.info("%-16s | %-16.3f | %-16.3f | %-16.3f | %-+16.3f", f"{b} tokens", r["tsm"], r["mem0"], r["naive"], delta)
    logger.info("=" * 90)


if __name__ == "__main__":
    main()
