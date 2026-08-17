#!/usr/bin/env python3
"""
TurboSuperMemory GPU Benchmark — 100k×768 Scale Test.

Measures build time, search latency, and recall with/without GPU acceleration.
Requires: `make build-python FEATURES=cuda` for GPU mode.

Usage:
    python benchmark_gpu.py --scale 100k --gpu
    python benchmark_gpu.py --scale 100k --cpu
    python benchmark_gpu.py --scale 50k  --gpu --ef 256
"""

import os
import sys
import time
import shutil
import argparse
import numpy as np

# Setup environment (same as benchmark.py)
def setup_environment():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    is_windows = sys.platform.startswith("win")
    ext_suffix = ".pyd" if is_windows else ".so"
    pyd_path = os.path.join(project_root, f"turbomemory{ext_suffix}")
    lib_prefix = "" if is_windows else "lib"
    lib_suffix = ".dll" if is_windows else (".dylib" if sys.platform.startswith("darwin") else ".so")
    lib_filename = f"{lib_prefix}turbomemory{lib_suffix}"
    dll_candidates = [
        os.path.join(project_root, "target", "release", lib_filename),
        os.path.join(project_root, "target", "debug", lib_filename),
    ]
    resolved = None
    for c in dll_candidates:
        if os.path.exists(c):
            resolved = c
            break
    if not resolved:
        print(f"Error: Could not find {lib_filename}")
        sys.exit(1)
    shutil.copy(resolved, pyd_path)
    sys.path.insert(0, project_root)

setup_environment()
import turbomemory


def parse_args():
    parser = argparse.ArgumentParser(description="TSM GPU Benchmark")
    parser.add_argument("--scale", choices=["10k", "50k", "100k"], default="50k",
                        help="Dataset scale")
    parser.add_argument("--gpu", action="store_true", help="Expect GPU acceleration")
    parser.add_argument("--cpu", action="store_true", help="Force CPU fallback (disable GPU)")
    parser.add_argument("--dimension", type=int, default=768)
    parser.add_argument("--ef", type=int, default=256, help="HNSW search beam width")
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hot-capacity", type=int, default=1024)
    parser.add_argument("--warm-capacity", type=int, default=8192)
    parser.add_argument("--hnsw-threshold", type=int, default=4096)
    parser.add_argument("--trigger-consolidation", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def generate_clustered_data(n, dim, num_clusters=64, seed=42):
    """Generate clustered embeddings similar to benchmark.py."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(num_clusters, dim).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8
    labels = rng.randint(0, num_clusters, size=n)
    embeddings = centers[labels] + rng.randn(n, dim).astype(np.float32) * 0.15
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    return embeddings


def flat_search(query, embeddings, top_k):
    """NumPy flat exact search for recall ground truth."""
    scores = np.dot(embeddings, query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_indices]


def recall_at_k(results, ground_truth, k):
    """Compute recall@k."""
    result_ids = {r[0] for r in results[:k]}
    gt_ids = {gt[0] for gt in ground_truth[:k]}
    if not gt_ids:
        return 1.0
    return len(result_ids & gt_ids) / len(gt_ids)


def main():
    args = parse_args()
    n = {"10k": 10_000, "50k": 50_000, "100k": 100_000}[args.scale]
    dim = args.dimension

    print(f"=" * 60)
    print(f"TurboSuperMemory GPU Benchmark")
    print(f"Scale: {n:,} vectors × {dim} dim")
    print(f"GPU expected: {args.gpu}")
    print(f"=" * 60)

    # Generate data
    print("\n[1/5] Generating clustered embeddings...")
    t0 = time.time()
    embeddings = generate_clustered_data(n, dim, seed=args.seed)
    texts = [f"memory_{i:06d}" for i in range(n)]
    ids = [f"id_{i:06d}" for i in range(n)]
    scores = [1.0] * n
    concepts = [[] for _ in range(n)]
    print(f"      Done in {time.time() - t0:.2f}s")

    # Setup engine
    db_path = f"./benchmark_gpu_{args.scale}_{dim}"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    print("\n[2/5] Opening MemoryEngine...")
    t0 = time.time()
    engine = turbomemory.MemoryEngine(
        db_path=db_path,
        dimension=dim,
        hot_capacity=args.hot_capacity,
        warm_capacity=args.warm_capacity,
        hnsw_threshold=args.hnsw_threshold,
        auto_consolidation_secs=0,
    )
    print(f"      Opened in {time.time() - t0:.2f}s")
    print(f"      GPU accelerated: {engine.gpu_accelerated}")

    if args.gpu and not engine.gpu_accelerated:
        print("WARNING: GPU was requested but CUDA is not available!")
        print("         Build with: make build-python FEATURES=cuda")
    if args.cpu and engine.gpu_accelerated:
        print("WARNING: --cpu flag set but GPU is active (no runtime disable yet)")

    # Ingestion
    print(f"\n[3/5] Ingesting {n:,} records...")
    t0 = time.time()
    batch_size = 1000
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        engine.insert_batch(
            ids[i:end],
            texts[i:end],
            embeddings[i:end].tolist(),
            scores[i:end],
            concepts[i:end],
        )
    ingest_time = time.time() - t0
    print(f"      Ingested in {ingest_time:.2f}s ({n / ingest_time:,.0f} records/s)")

    # Trigger consolidation to build HNSW
    if args.trigger_consolidation:
        print("\n[4/5] Triggering background consolidation (HNSW build)...")
        t0 = time.time()
        engine.trigger_consolidation()
        consolidate_time = time.time() - t0
        print(f"      Consolidated in {consolidate_time:.2f}s")
    else:
        consolidate_time = 0.0

    # Generate query embeddings
    print(f"\n[5/5] Running {args.num_queries} search queries...")
    query_embeddings = generate_clustered_data(args.num_queries, dim, seed=args.seed + 1)

    # Warmup
    print("      Warming up...")
    for i in range(min(5, args.num_queries)):
        engine.search_ann(query_embeddings[i].tolist(), args.top_k, args.ef)

    # Benchmark search
    latencies = []
    recalls = []
    t0 = time.time()
    for i in range(args.num_queries):
        q = query_embeddings[i]

        # Ground truth from flat NumPy
        gt = flat_search(q, embeddings, args.top_k)

        # TSM search
        t1 = time.time()
        results = engine.search_ann(q.tolist(), args.top_k, args.ef)
        t2 = time.time()
        latencies.append((t2 - t1) * 1000.0)  # ms

        # Map result IDs to indices for recall
        result_indices = []
        for rid, score in results:
            idx = int(rid.split("_")[1])
            result_indices.append((idx, score))

        r = recall_at_k(result_indices, gt, args.top_k)
        recalls.append(r)

    total_search_time = time.time() - t0

    # Report
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Ingestion time:        {ingest_time:>10.2f}s")
    if args.trigger_consolidation:
        print(f"Consolidation time:    {consolidate_time:>10.2f}s")
    print(f"Total search time:     {total_search_time:>10.2f}s")
    print(f"Mean latency:          {np.mean(latencies):>10.2f}ms")
    print(f"P50 latency:           {np.percentile(latencies, 50):>10.2f}ms")
    print(f"P95 latency:           {np.percentile(latencies, 95):>10.2f}ms")
    print(f"P99 latency:           {np.percentile(latencies, 99):>10.2f}ms")
    print(f"Recall@{args.top_k}:            {np.mean(recalls):>10.3f}")
    print(f"Min recall@{args.top_k}:        {np.min(recalls):>10.3f}")
    print(f"GPU accelerated:       {str(engine.gpu_accelerated):>10}")
    print("=" * 60)

    # Cleanup
    del engine
    if os.path.exists(db_path):
        shutil.rmtree(db_path)


if __name__ == "__main__":
    main()
