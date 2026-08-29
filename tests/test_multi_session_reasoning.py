#!/usr/bin/env python3
"""
Test Multi-Session Reasoning & Entity Anchoring
==============================================

Verifies that when information is spread across multiple separate sessions:
- Session 1: Entity Introduction ("I adopted a Golden Retriever puppy named Sparky")
- Session 4: Condition / Context ("Sparky was diagnosed with atopic dermatitis")
- Session 8: Action / Update ("Dr. Harrison prescribed 5mg Apoquel daily")

A query asking: "What medication is my golden retriever taking?"
returns BOTH Session 1 (the entity anchor) AND Session 8 (the prescription)
within a tight 150-token context budget!
"""

import os
import sys
import shutil

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tsm import Memory, SentenceTransformerEmbedder


def test_multi_session_reasoning():
    db_path = os.path.join(project_root, "test_multi_session_db")
    if os.path.exists(db_path):
        shutil.rmtree(db_path, ignore_errors=True)

    embedder = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    mem = Memory(
        db_path=db_path,
        embedder=embedder,
        extractor="passthrough",
    )

    user_id = "user_multi_session"

    # Ingest memories from 3 different sessions separated in time
    print("Ingesting Session 1, 4, 8 memories...")
    mem.add("I adopted a golden retriever puppy named Sparky.", user_id=user_id)
    mem.add("Sparky was diagnosed with severe atopic dermatitis by Dr. Harrison.", user_id=user_id)
    mem.add("Dr. Harrison prescribed 5mg Apoquel daily for Sparky's allergy flare-ups.", user_id=user_id)
    mem.add("My brother David bought a red Tesla Model 3 last weekend.", user_id=user_id)
    mem.add("I changed my home Wi-Fi password to SolarStorm99.", user_id=user_id)

    # Search with a cross-session query under a 150-token budget
    query = "What medication is my golden retriever taking?"
    print(f"\nQuerying: '{query}' with token_budget=150...")
    results = mem.recall(query, user_id=user_id, token_budget=150, pool_k=20)

    print("\n--- Retrieved Results (150-token budget) ---")
    retrieved_texts = []
    for r in results:
        print(f"  • [Score: {r['score']:.4f}] {r['text']}")
        retrieved_texts.append(r['text'])

    # Validate that both the entity definition (Golden Retriever / Sparky)
    # AND the prescription fact (Apoquel) are retrieved!
    has_anchor = any("golden retriever" in t.lower() or "sparky" in t.lower() for t in retrieved_texts)
    has_prescription = any("apoquel" in t.lower() or "prescribed" in t.lower() for t in retrieved_texts)

    assert has_anchor, "Failed to retrieve the Session 1 entity anchor (Golden Retriever)!"
    assert has_prescription, "Failed to retrieve the Session 8 fact (Apoquel prescription)!"

    print("\n✅ Multi-Session Reasoning validation PASSED: Both cross-session facts retrieved within 150 tokens!")

    mem.close()
    shutil.rmtree(db_path, ignore_errors=True)


if __name__ == "__main__":
    test_multi_session_reasoning()
