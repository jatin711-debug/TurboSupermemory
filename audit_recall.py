#!/usr/bin/env python
"""
Retrieval-correctness audit for TurboSuperMemory.

Measures recall at the ANN stage and after the cognitive graph gate,
and compares both against a flat NumPy L2 ground truth.
"""
import os
import sys
import shutil
import time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DLL = os.path.join(ROOT, "target", "release", "turbomemory.dll")
PYD = os.path.join(ROOT, "turbomemory.pyd")

if not os.path.exists(DLL):
    print(f"Release DLL not found: {DLL}")
    sys.exit(1)
shutil.copy(DLL, PYD)

import turbomemory


def recall_at_k(ground_truth, results, k=5):
    """Recall = |intersection| / k (ground truth size is k)."""
    if not results or len(results) == 0:
        return 0.0
    gt_ids = {t[0] for t in ground_truth[:k]}
    res_ids = {t[0] for t in results[:k]}
    return len(gt_ids & res_ids) / float(len(gt_ids))


def run_audit(num_items, dimension, num_queries=50, top_k=5, ef=256):
    print("=" * 70)
    print(f"Audit config: N={num_items}, D={dimension}, queries={num_queries}, top_k={top_k}, ef={ef}")
    print("=" * 70)

    np.random.seed(42)
    raw = np.random.randn(num_items, dimension).astype(np.float32)
    embeddings = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    raw_q = np.random.randn(num_queries, dimension).astype(np.float32)
    queries = raw_q / np.linalg.norm(raw_q, axis=1, keepdims=True)

    db_dir = os.path.join(ROOT, f"audit_test_db_d{dimension}_n{num_items}")
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

    engine = turbomemory.MemoryEngine(
        db_path=db_dir,
        dimension=dimension,
        max_edges=max(16, dimension // 16),  # Scale M with dimension
        search_list_size=ef,
        outlier_count=0,
        initial_capacity=num_items,
    )

    t0 = time.perf_counter()
    for i in range(num_items):
        engine.insert(
            f"mem_{i}",
            f"Sample memory text content {i}",
            embeddings[i],
            1.0,
            [f"concept_{i % 5}"],
        )
    ingest_ms = (time.perf_counter() - t0) * 1000.0 / num_items
    print(f"Ingest: {ingest_ms:.3f} ms/item")

    # Flat NumPy ground truth (L2).
    ann_recalls = []
    candidate_recalls = []
    cognitive_empty_recalls = []
    cognitive_text_recalls = []
    none_count_empty = 0
    none_count_text = 0

    t0 = time.perf_counter()
    for q in queries:
        dists = np.linalg.norm(embeddings - q, axis=1)
        top_idx = np.argsort(dists)[:top_k]
        gt = [(f"mem_{int(idx)}", float(dists[idx])) for idx in top_idx]

        candidates = engine.search_ann_candidates(q, top_k)
        ann = engine.search_ann(q, top_k)
        cog_empty = engine.search("", q, top_k)
        cog_text = engine.search("memory", q, top_k)

        candidate_ids = {item[0] for item in candidates}
        gt_ids = {item[0] for item in gt}
        candidate_recalls.append(len(candidate_ids & gt_ids) / float(top_k))
        ann_recalls.append(recall_at_k(gt, ann, top_k))
        if cog_empty is None:
            none_count_empty += 1
            cognitive_empty_recalls.append(0.0)
        else:
            cognitive_empty_recalls.append(recall_at_k(gt, cog_empty, top_k))

        if cog_text is None:
            none_count_text += 1
            cognitive_text_recalls.append(0.0)
        else:
            cognitive_text_recalls.append(recall_at_k(gt, cog_text, top_k))

    search_ms = (time.perf_counter() - t0) * 1000.0 / num_queries
    print(f"Search: {search_ms:.3f} ms/query (includes all three retrieval paths)")

    print(f"ANN candidate Recall@{top_k}: {np.mean(candidate_recalls) * 100:.1f}%")
    print(f"ANN reranked Recall@{top_k}:  {np.mean(ann_recalls) * 100:.1f}%")
    print(f"Cognitive '' Recall@{top_k}:    {np.mean(cognitive_empty_recalls) * 100:.1f}% ({none_count_empty}/{num_queries} queries returned None)")
    print(f"Cognitive 'memory' Recall@{top_k}: {np.mean(cognitive_text_recalls) * 100:.1f}% ({none_count_text}/{num_queries} queries returned None)")

    # Print a few example queries.
    print("\nSample query details:")
    for qi in [0, 1, 2]:
        q = queries[qi]
        dists = np.linalg.norm(embeddings - q, axis=1)
        top_idx = np.argsort(dists)[:top_k]
        gt = [f"mem_{int(idx)}" for idx in top_idx]
        ann_ids = [r[0] for r in engine.search_ann(q, top_k)]
        cog_empty = engine.search("", q, top_k)
        cog_text = engine.search("memory", q, top_k)
        print(f"  query {qi}: GT={gt}")
        print(f"           ANN={ann_ids}")
        print(f"           cog('')={([r[0] for r in cog_empty] if cog_empty else None)}")
        print(f"           cog('memory')={([r[0] for r in cog_text] if cog_text else None)}")

    # Restart correctness: close engine, reopen, compare ANN results.
    # NOTE: Skipped due to redb lock issue - engine holds file lock until process exits
    print("\nRestart correctness check: SKIPPED (redb lock held by process)")
    
    engine.close()
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Recall audit for TurboSuperMemory")
    parser.add_argument("--num-items", type=int, default=200, help="Number of items to ingest")
    parser.add_argument("--dimension", type=int, default=8, help="Vector dimension")
    parser.add_argument("--num-queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k for recall")
    parser.add_argument("--ef", type=int, default=256, help="HNSW ef parameter")
    args = parser.parse_args()
    
    run_audit(args.num_items, args.dimension, num_queries=args.num_queries, top_k=args.top_k, ef=args.ef)
