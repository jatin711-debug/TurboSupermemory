#!/usr/bin/env python3
"""
Drilldown into the 4 Multi-Session Questions in LongMemEval
=========================================================
"""

import os
import sys
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "benchmarks", "cognitive_eval"))

from benchmark_datasets.longmemeval import load_longmemeval
from head_to_head_eval import conv_facts
from cognitive_eval.judge import OpenAIJudge
from tsm.embedders import SentenceTransformerEmbedder
from tsm.extractors import OpenAIExtractor
from tsm.memory import Memory


def main():
    convs = load_longmemeval()[:10]
    embedder = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    extractor = OpenAIExtractor(cache_dir=os.path.join(os.path.expanduser("~"), ".cache", "tsm"))
    judge = OpenAIJudge()

    multi_count = 0
    for ci, conv in enumerate(convs):
        multi_q = [q for q in conv.queries if q.question_type == "multi-session"]
        if not multi_q:
            continue

        db_path = os.path.join(project_root, f"test_ms_eval_{ci}")
        mem = Memory(
            db_path=db_path,
            embedder=embedder,
            extractor="passthrough",
        )
        raw_facts = conv_facts(extractor, conv, roles=["user"])
        # Semantic deduplication of raw extracted facts
        dedup_facts = []
        for f in raw_facts:
            f_norm = " ".join(f.lower().split())
            if not any(f_norm in d.lower() or d.lower() in f_norm for d in dedup_facts):
                dedup_facts.append(f)

        for f in dedup_facts:
            mem.add(f, user_id="user")

        for q in multi_q:
            multi_count += 1
            print("=" * 80)
            print(f"MULTI-SESSION QUESTION #{multi_count} (Conversation {ci+1}, ID: {conv.conv_id})")
            print(f"Query:        {q.query_text}")
            print(f"Ground Truth: {q.answer_text}")
            
            results_150 = mem.recall(q.query_text, user_id="user", token_budget=150, pool_k=40, lam=0.45)
            texts = [r["text"] for r in results_150]
            print("\nRetrieved Facts (150 tokens):")
            for t in texts:
                print(f"  • {t}")

            correct = judge.answer_and_judge(q.query_text, texts, q.answer_text)
            print(f"\nTSM Judged Score: {'PASS (1.0)' if correct else 'FAIL (0.0)'}")
            print("-" * 80)

        mem.close()
        import shutil
        shutil.rmtree(db_path, ignore_errors=True)


if __name__ == "__main__":
    main()
