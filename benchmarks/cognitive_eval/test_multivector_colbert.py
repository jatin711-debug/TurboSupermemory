#!/usr/bin/env python3
"""
TSM + MultiVectorEncoder (ColBERT Late Interaction) Pilot Benchmark
==================================================================

Evaluates the retrieval precision of TurboSuperMemory paired with a Stage-2
MultiVector late-interaction MaxSim reranker on fine-grained entity, numeric,
and constraint-matching memory queries.

Usage:
    python benchmarks/cognitive_eval/test_multivector_colbert.py
"""

import sys
import os
import time
import shutil
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
import turbomemory
from sentence_transformers import MultiVectorEncoder, SentenceTransformer

# 1. Define Hard Information Extraction & Entity Memory Scenarios
TEST_SCENARIOS = [
    {
        "category": "Exact Part Number & Dosage",
        "query": "What exact dosage of medication does Sparky take daily?",
        "gold_keyword": "5mg of Apoquel",
        "memories": [
            {"id": "mem_1", "text": "Sparky is a 3-year old golden retriever who loves running in the park and eating chicken treats."},
            {"id": "mem_2", "text": "Sparky had an allergy flare up last week and the vet prescribed 5mg of Apoquel daily with meals."},
            {"id": "mem_3", "text": "Remember to buy dog food and a new leash for Sparky this weekend."},
            {"id": "mem_4", "text": "The vet appointment for general checkup is scheduled for next month on Tuesday."},
            {"id": "mem_5", "text": "Sparky weighs 32kg and needs regular flea medication once every three months."},
        ]
    },
    {
        "category": "Server Port & IP Configuration",
        "query": "What port is the production telemetry collector running on?",
        "gold_keyword": "port 9443",
        "memories": [
            {"id": "mem_6", "text": "The main production web server is deployed on AWS EC2 at 10.0.4.12 listening on port 443."},
            {"id": "mem_7", "text": "Database read-replica is running on port 5432 with max connections set to 500."},
            {"id": "mem_8", "text": "Internal telemetry and metrics collector is configured on port 9443 with TLS mutual auth."},
            {"id": "mem_9", "text": "Staging telemetry collector runs on port 9090 for Prometheus scraping."},
            {"id": "mem_10", "text": "Redis caching cluster is accessible at redis-cluster.internal on default port 6379."},
        ]
    },
    {
        "category": "Temporal Timeline / Version Assertion",
        "query": "What is the latest compiler version we upgraded to for the storage engine?",
        "gold_keyword": "Rust 1.96.2",
        "memories": [
            {"id": "mem_11", "text": "We initially built the storage engine on Rust 1.82 back in early 2024."},
            {"id": "mem_12", "text": "Upgraded dependencies to support Rust 1.90 with SIMD feature flags."},
            {"id": "mem_13", "text": "Last week we finalized the toolchain migration and upgraded the storage engine to Rust 1.96.2."},
            {"id": "mem_14", "text": "The Python extension is targeting ABI3 with Python 3.12 compatibility."},
            {"id": "mem_15", "text": "Storage optimizer was refactored to support async seal queues in Rust."},
        ]
    }
]


def run_benchmark():
    print("=" * 70)
    print("🚀 TurboSuperMemory + MultiVectorEncoder (ColBERT Late Interaction)")
    print("=" * 70)

    # Load bi-encoder for TSM Stage 1
    print("\n[1/3] Initializing TSM Vector + Cognitive Engine...")
    bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    dim = 384
    
    # Initialize ColBERT MultiVectorEncoder for Stage 2
    print("\n[2/3] Initializing MultiVectorEncoder (ColBERT / LFM2.5)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"      Compute Device: {device.upper()}")
    
    try:
        colbert_model_id = "LiquidAI/LFM2.5-ColBERT-350M"
        print(f"      Loading {colbert_model_id}...")
        colbert_model = MultiVectorEncoder(colbert_model_id, trust_remote_code=True, device=device)
        print("      Successfully loaded LFM2.5-ColBERT-350M!")
    except Exception as e:
        print(f"      Notice: Could not load {colbert_model_id} ({e}).")
        print("      Loading ColBERTv2 fallback for MultiVector late-interaction...")
        colbert_model_id = "colbert-ir/colbertv2.0"
        colbert_model = MultiVectorEncoder(colbert_model_id, device=device)

    db_path = os.path.join(project_root, "test_db_colbert")
    if os.path.exists(db_path):
        shutil.rmtree(db_path, ignore_errors=True)

    print("\n[3/3] Running Head-to-Head Precision Benchmarks across Test Scenarios...")
    print("-" * 70)

    for scenario_idx, scenario in enumerate(TEST_SCENARIOS, 1):
        print(f"\n📌 Scenario {scenario_idx}: [{scenario['category']}]")
        print(f"   Query: \"{scenario['query']}\"")
        print(f"   Target Fact: \"{scenario['gold_keyword']}\"")

        # Ingest into fresh TSM instance
        scenario_db = f"{db_path}_{scenario_idx}"
        engine = turbomemory.MemoryEngine(
            db_path=scenario_db,
            dimension=dim,
            outlier_count=0,
            auto_consolidation_secs=0,
            cognitive_alpha=0.5,
        )

        texts = [m["text"] for m in scenario["memories"]]
        ids = [m["id"] for m in scenario["memories"]]
        embeddings = bi_encoder.encode(texts, normalize_embeddings=True).astype(np.float32)

        for mid, text, emb in zip(ids, texts, embeddings):
            engine.insert(id=mid, text=text, embedding=emb, importance_score=1.0, concepts=[])
        
        # Stage 1: TSM Cognitive Retrieval
        t0 = time.perf_counter()
        q_emb = bi_encoder.encode([scenario["query"]], normalize_embeddings=True)[0].astype(np.float32)
        tsm_results = engine.search(query_text=scenario["query"], query_embedding=q_emb, top_k=5)
        stage1_time_ms = (time.perf_counter() - t0) * 1000

        print(f"\n   [Stage 1: TSM Single-Vector + Graph Retrieval] ({stage1_time_ms:.2f}ms):")
        for rank, (mid, score) in enumerate(tsm_results, 1):
            text = next(m["text"] for m in scenario["memories"] if m["id"] == mid)
            is_gold = scenario["gold_keyword"] in text
            mark = "🎯 [TARGET]" if is_gold else "  "
            print(f"     Rank {rank}: (Score: {score:.4f}) {mark} {text[:75]}...")

        # Stage 2: ColBERT MultiVector Late-Interaction MaxSim Reranking
        t0 = time.perf_counter()
        candidate_texts = [next(m["text"] for m in scenario["memories"] if m["id"] == mid) for mid, _ in tsm_results]
        candidate_ids = [mid for mid, _ in tsm_results]

        q_multi = colbert_model.encode_query([scenario["query"]])
        doc_multi = colbert_model.encode_document(candidate_texts)
        colbert_scores = colbert_model.similarity(q_multi, doc_multi)[0].cpu().numpy()
        stage2_time_ms = (time.perf_counter() - t0) * 1000

        # Sort by ColBERT MaxSim score
        reranked_indices = np.argsort(colbert_scores)[::-1]
        
        print(f"\n   [Stage 2: ColBERT MultiVector MaxSim Rerank] (+{stage2_time_ms:.2f}ms):")
        for rank, idx in enumerate(reranked_indices, 1):
            mid = candidate_ids[idx]
            text = candidate_texts[idx]
            score = colbert_scores[idx]
            is_gold = scenario["gold_keyword"] in text
            mark = "🎯 [TARGET]" if is_gold else "  "
            print(f"     Rank {rank}: (MaxSim: {score:.2f}) {mark} {text[:75]}...")

        engine.close()
        shutil.rmtree(scenario_db, ignore_errors=True)

    print("\n" + "=" * 70)
    print("✅ Benchmark Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
