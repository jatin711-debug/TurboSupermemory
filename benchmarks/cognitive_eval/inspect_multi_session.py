#!/usr/bin/env python3
"""
Diagnostic tool to inspect why Multi-Session questions miss in LongMemEval
========================================================================
"""

import os
import sys
import json
import logging

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark_datasets.longmemeval import load_longmemeval
from head_to_head_eval import conv_facts
from tsm.embedders import SentenceTransformerEmbedder
from tsm.extractors import OpenAIExtractor
from tsm.memory import Memory


def inspect():
    convs = load_longmemeval()[:10]
    embedder = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    extractor = OpenAIExtractor(cache_dir=os.path.join(os.path.expanduser("~"), ".cache", "tsm"))

    for ci, conv in enumerate(convs):
        multi_q = [q for q in conv.queries if q.question_type == "multi-session"]
        if not multi_q:
            continue

        print("=" * 80)
        print(f"Conversation {ci+1} (ID: {conv.conv_id}) has {len(multi_q)} multi-session query(s):")
        
        # Ingest into a fresh TSM DB
        db_path = os.path.join(project_root, f"test_inspect_conv_{ci}")
        mem = Memory(
            db_path=db_path,
            embedder=embedder,
            extractor="passthrough",
        )
        
        facts = conv_facts(extractor, conv, roles=None)
        for f in facts:
            mem.add(f, user_id="user")

        for q in multi_q:
            print(f"\nQuestion: {q.query_text}")
            print(f"Ground Truth Answer: {q.answer_text}")
            
            # Retrieve with budget=150
            results_150 = mem.recall(q.query_text, user_id="user", token_budget=150, pool_k=30)
            print("\nRetrieved (150-token budget):")
            for r in results_150:
                print(f"  • [Score: {r['score']:.3f}] {r['text']}")
                
            # Retrieve with top_k=5 (no budget)
            results_top5 = mem.recall(q.query_text, user_id="user", top_k=5)
            print("\nRetrieved Top-5 (unbudgeted):")
            for r in results_top5:
                print(f"  • [Score: {r['score']:.3f}] {r['text']}")

        mem.close()
        import shutil
        shutil.rmtree(db_path, ignore_errors=True)


if __name__ == "__main__":
    inspect()
