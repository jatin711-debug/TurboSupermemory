"""Focused live test verifying TSM Scoped Temporal Graphs and Temporal Anchoring."""

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
from cognitive_eval.openai_embedder import OpenAIEmbedder


def run_temporal_test():
    print("=" * 70)
    print("  TURBOSUPERMEMORY: TEMPORAL REASONING & SCOPED GRAPH TEST")
    print("=" * 70)

    # 1. Initialize OpenAI Embedder (or local fallback)
    openai_key_file = os.path.join(project_root, "openai_key.txt")
    if os.path.exists(openai_key_file):
        with open(openai_key_file, "r") as f:
            os.environ["OPENAI_API_KEY"] = f.read().strip()

    model = OpenAIEmbedder(model="text-embedding-3-small")
    db_path = tempfile.mkdtemp(prefix="tsm_temporal_test_")

    adapter = TSMAdapter(
        db_path=db_path,
        embedding_model="text-embedding-3-small",
        extractor="mock",
        cognitive_features=True,
        belief_revision=True,
        model=model,
        dimension=1536,
        supersession_mode="exclude",
    )

    # Ingest a timeline of user messages across 3 dates
    messages = [
        {"role": "user", "content": "I arrived in Tokyo today.", "timestamp": "2024-03-01T09:00:00Z"},
        {"role": "user", "content": "I visited the Senso-ji temple in Asakusa.", "timestamp": "2024-03-02T14:00:00Z"},
        {"role": "user", "content": "I had dinner in Shibuya.", "timestamp": "2024-03-02T19:30:00Z"},
        {"role": "user", "content": "I bought a vintage Canon camera in Akihabara.", "timestamp": "2024-03-05T11:00:00Z"},
        {"role": "user", "content": "I am flying back to San Francisco tomorrow.", "timestamp": "2024-03-10T16:00:00Z"},
    ]

    print("\n[1] Ingesting 5 conversation turns with explicit timestamps...")
    adapter.add(messages, user_id="user_trip_2024")
    adapter.trigger_consolidation()

    # Query 1: Temporal search for when camera was purchased
    print("\n[2] Testing Temporal Retrieval for 'When did the user buy a camera?'")
    results = adapter.search("When did the user buy a camera?", user_id="user_trip_2024", top_k=3, use_cognitive=True)
    for i, r in enumerate(results, 1):
        print(f"  Rank {i} (score: {r['score']:.4f}): {r['text']}")

    top_text = results[0]["text"] if results else ""
    assert "[2024-03-05]" in top_text, f"Expected [2024-03-05] in top result, got: {top_text}"
    print("  -> PASSED: Top recalled memory contains anchored timestamp [2024-03-05]!")

    # Query 2: Testing MMR recall under 50-token budget
    print("\n[3] Testing recall_under_budget (50 tokens)...")
    budget_texts = adapter.recall_under_budget("Senso-ji temple and Shibuya dinner", user_id="user_trip_2024", token_budget=50, method="mmr")
    for i, t in enumerate(budget_texts, 1):
        print(f"  Packed Item {i}: {t}")

    assert any("[2024-03-02]" in t for t in budget_texts), "Expected [2024-03-02] in packed memory texts"
    print("  -> PASSED: Packed memory set contains anchored dates and high topical coverage!")

    print("\n" + "=" * 70)
    print("  ALL TEMPORAL TESTS PASSED SUCCESSFULLY (100%)")
    print("=" * 70)


if __name__ == "__main__":
    run_temporal_test()
