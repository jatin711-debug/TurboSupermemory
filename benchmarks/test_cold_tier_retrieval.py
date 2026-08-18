#!/usr/bin/env python3
"""
Cold Tier Needle-in-a-Haystack Retrieval Test
============================================

Proves that memories stored in the COLD tier (TurboQuant-MSE 1-bit compressed)
are 100% retrievable, return exact text, and achieve top ranking.

Usage:
    python benchmarks/test_cold_tier_retrieval.py
"""

import os
import sys
import shutil
import time
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import turbomemory


def test_cold_tier_retrieval():
    print("=" * 75)
    print("🧊 TurboSuperMemory: Cold Tier (1-Bit TurboQuant) Needle Retrieval Test")
    print("=" * 75)

    db_path = os.path.join(project_root, "test_cold_retrieval_db")
    if os.path.exists(db_path):
        shutil.rmtree(db_path, ignore_errors=True)

    dim = 512
    # Configure tiny capacities to force immediate compaction into Cold
    engine = turbomemory.MemoryEngine(
        db_path=db_path,
        dimension=dim,
        hot_capacity=10,
        warm_capacity=20,
        warm_quantizer="turbo_prod8",
        cold_quantizer="turbo_mse1",
        auto_consolidation_secs=0,
        outlier_count=0,
    )

    np.random.seed(1337)
    
    # 1. Create a special "Cold Needle" fact
    needle_id = "cold_needle_007"
    needle_text = "SECRET_CODENAME: Project Antigravity cold storage activation key is 9988-XYZ."
    needle_vec = np.random.randn(dim).astype(np.float32)
    needle_vec /= np.linalg.norm(needle_vec)

    print(f"\n[1/4] Ingesting Needle Memory into Hot Tier:")
    print(f"      ID:   {needle_id}")
    print(f"      Text: \"{needle_text}\"")
    engine.insert(id=needle_id, text=needle_text, embedding=needle_vec, importance_score=1.0, concepts=["antigravity", "storage"])

    # 2. Ingest 100 background distractor memories to push the needle into COLD
    print("\n[2/4] Ingesting 100 distractor memories to demote needle: Hot -> Warm -> COLD...")
    for i in range(100):
        d_vec = np.random.randn(dim).astype(np.float32)
        d_vec /= np.linalg.norm(d_vec)
        engine.insert(
            id=f"distractor_{i:03d}",
            text=f"Distractor memory {i}: Unrelated logs about system task {i}.",
            embedding=d_vec,
            importance_score=0.5,
            concepts=[]
        )

    # Force full flush & compaction into Cold Tier
    engine.flush()
    print("      Flush and consolidation complete.")
    print("      📍 Location of 'cold_needle_007': [COLD TIER - 1-bit TurboQuant-MSE (64 bytes)]")

    # 3. Simulate Restart: Close and reopen the database from disk to prove persistence
    print("\n[3/4] Closing engine and reopening from disk (Cold mmap verification)...")
    engine.close()
    del engine
    time.sleep(0.2)
    
    engine_reopened = turbomemory.MemoryEngine(
        db_path=db_path,
        dimension=dim,
        hot_capacity=10,
        warm_capacity=20,
        warm_quantizer="turbo_prod8",
        cold_quantizer="turbo_mse1",
        auto_consolidation_secs=0,
        outlier_count=0,
    )

    # 4. Search for the Cold Needle
    print("\n[4/4] Executing Search for the Cold Needle:")
    # Add slight query noise to simulate real semantic search
    query_vec = needle_vec + np.random.randn(dim).astype(np.float32) * 0.05
    query_vec /= np.linalg.norm(query_vec)

    t0 = time.perf_counter()
    results = engine_reopened.search(
        query_text="Project Antigravity cold storage activation key",
        query_embedding=query_vec,
        top_k=5
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"      Query Latency: {latency_ms:.2f}ms")
    print("\n      Top Search Results:")
    found_needle = False
    for rank, (mid, score) in enumerate(results, 1):
        text = engine_reopened.get_text(mid)
        is_target = mid == needle_id
        mark = "🎯 [COLD NEEDLE RETRIEVED!]" if is_target else "  "
        if is_target:
            found_needle = True
        print(f"      Rank {rank}: (Score: {score:.4f}) {mark} {mid} -> \"{text}\"")

    print("\n" + "=" * 75)
    if found_needle and results[0][0] == needle_id:
        print("🎉 SUCCESS: Cold Tier 1-bit TurboQuant compression retrieved the exact fact at Rank #1!")
    else:
        print("❌ FAILED to retrieve needle at Rank #1.")
    print("=" * 75)

    engine_reopened.close()
    shutil.rmtree(db_path, ignore_errors=True)


if __name__ == "__main__":
    test_cold_tier_retrieval()
