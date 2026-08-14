"""Inspect the exact queries, retrieved contexts, and judge verdicts for failed categories."""

import json
import os
import sys
import tempfile
import numpy as np

# Ensure project root is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "benchmarks"))

from cognitive_eval.adapters.tsm_adapter import TSMAdapter
from cognitive_eval.benchmark_datasets.longmemeval import load_longmemeval
from cognitive_eval.judge import create_judge
from cognitive_eval.openai_embedder import OpenAIEmbedder


def debug():
    openai_key_file = os.path.join(project_root, "openai_key.txt")
    if os.path.exists(openai_key_file):
        with open(openai_key_file, "r") as f:
            os.environ["OPENAI_API_KEY"] = f.read().strip()

    convs = load_longmemeval(None)[:10]
    model = OpenAIEmbedder(model="text-embedding-3-small")
    judge = create_judge("openai", openai_model="gpt-4o-mini")

    print(f"Loaded {len(convs)} conversations. Inspecting queries by type...\n")

    for ci, conv in enumerate(convs):
        queries_to_inspect = [
            q for q in conv.queries 
            if not q.is_abstention and q.question_type in ("temporal-reasoning", "single-session-preference")
        ]
        if not queries_to_inspect:
            continue

        print("=" * 80)
        print(f"CONVERSATION #{ci+1} (ID: {conv.conv_id})")
        print("Total messages in conv:", len(conv.messages))
        print("Sample messages:")
        for m in conv.messages[:6]:
            ts = getattr(m, 'timestamp', None) or ''
            print(f"  [{m.role}] ({ts}) {m.content[:100]}...")

        # Ingest into TSM
        db = tempfile.mkdtemp(prefix="tsm_debug_")
        tsm_ad = TSMAdapter(
            db_path=db,
            embedding_model="text-embedding-3-small",
            extractor="mock",
            cognitive_features=True,
            belief_revision=True,
            model=model,
            dimension=1536,
            supersession_mode="exclude",
        )
        tsm_ad.add(conv.messages, user_id=conv.conv_id)
        tsm_ad.trigger_consolidation()

        for q in queries_to_inspect:
            print("\n" + "-" * 60)
            print(f"QUESTION TYPE: {q.question_type}")
            print(f"QUERY TEXT:    {q.query_text}")
            print(f"GOLD ANSWER:   {q.answer_text}")

            tsm_recalled = tsm_ad.recall_under_budget(q.query_text, user_id=conv.conv_id, token_budget=150, method="mmr")
            print(f"\nTSM RECALLED ({len(tsm_recalled)} items, token budget 150):")
            for idx, item in enumerate(tsm_recalled, 1):
                print(f"  [{idx}] {item}")

            pred = judge.answer(q.query_text, tsm_recalled)
            is_correct = judge.judge(q.query_text, q.answer_text, pred)
            print(f"\nTSM PREDICTED ANSWER: '{pred}'")
            print(f"GOLD ANSWER:         '{q.answer_text}'")
            print(f"TSM JUDGE RESULT:    {'CORRECT' if is_correct else 'INCORRECT'}")

        tsm_ad.close()


if __name__ == "__main__":
    debug()
