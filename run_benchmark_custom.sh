#!/bin/bash
# Run TSM benchmark with configurable parameters
# Usage: ./run_benchmark_custom.sh [NUM_ITEMS] [DIMENSION] [NUM_QUERIES] [TOP_K] [EF]
# Defaults: 1M items, 1024 dim, 1000 queries, top_k=5, ef=256

set -e

# Parse arguments with defaults
NUM_ITEMS=${1:-1000000}
DIM=${2:-1024}
NUM_QUERIES=${3:-1000}
TOP_K=${4:-5}
EF=${5:-256}

echo "=========================================="
echo "  TSM Custom Benchmark"
echo "  Items:     ${NUM_ITEMS}"
echo "  Dimension: ${DIM}"
echo "  Queries:   ${NUM_QUERIES}"
echo "  Top-K:     ${TOP_K}"
echo "  EF:        ${EF}"
echo "  CUDA:      ENABLED"
echo "  Recall:    ENABLED (vs flat NumPy)"
echo "=========================================="

source .venv/bin/activate

# Build if needed
if [ ! -f "turbomemory.so" ]; then
    echo "Building TSM with CUDA..."
    make FEATURES=cuda build-python
fi

# Run benchmark
PYTHONPATH="." python -c "
import sys
import time
import numpy as np
import tempfile
import shutil

# Import TSM
import turbomemory

# Configuration from args
NUM_ITEMS = ${NUM_ITEMS}
DIM = ${DIM}
NUM_QUERIES = ${NUM_QUERIES}
TOP_K = ${TOP_K}
EF = ${EF}

print(f'Configuration:')
print(f'  Items:     {NUM_ITEMS:,}')
print(f'  Dimension: {DIM}')
print(f'  Queries:   {NUM_QUERIES}')
print(f'  Top-K:     {TOP_K}')
print(f'  EF:        {EF}')
print()

# Generate clustered data (more realistic)
print('Generating clustered data...')
np.random.seed(42)
n_clusters = min(64, NUM_ITEMS // 100)
cluster_centers = np.random.randn(n_clusters, DIM).astype(np.float32)
cluster_centers /= np.linalg.norm(cluster_centers, axis=1, keepdims=True)

embeddings = []
for i in range(NUM_ITEMS):
    center = cluster_centers[i % n_clusters]
    noise = np.random.randn(DIM).astype(np.float32) * 0.1
    vec = center + noise
    vec /= np.linalg.norm(vec)
    embeddings.append(vec)

embeddings = np.array(embeddings)

# Generate queries
queries = []
for i in range(NUM_QUERIES):
    center = cluster_centers[i % n_clusters]
    noise = np.random.randn(DIM).astype(np.float32) * 0.15
    vec = center + noise
    vec /= np.linalg.norm(vec)
    queries.append(vec)

queries = np.array(queries)
print(f'  Data generated: {embeddings.shape}')
print()

# Create TSM engine
db_path = tempfile.mkdtemp(prefix='tsm_benchmark_')
print(f'Creating TSM engine at {db_path}...')
engine = turbomemory.MemoryEngine(
    db_path=db_path,
    dimension=DIM,
)

# Ingest
print(f'Ingesting {NUM_ITEMS:,} items...')
t0 = time.perf_counter()
for i in range(NUM_ITEMS):
    engine.insert(
        f'mem_{i}',
        f'Sample text {i}',
        embeddings[i],
        1.0,
        [],
    )
    if (i + 1) % 10000 == 0:
        print(f'  Progress: {(i+1)/NUM_ITEMS*100:.1f}% ({i+1:,}/{NUM_ITEMS:,})')
ingest_time = (time.perf_counter() - t0) * 1000.0
print(f'Ingest complete: {ingest_time:.1f}ms total, {ingest_time/NUM_ITEMS:.3f}ms/item')
print()

# Flat NumPy ground truth
print('Computing flat NumPy ground truth...')
t0 = time.perf_counter()
ground_truth = []
for q in queries:
    dists = np.linalg.norm(embeddings - q, axis=1)
    top_idx = np.argsort(dists)[:TOP_K]
    gt = [(f'mem_{int(idx)}', float(dists[idx])) for idx in top_idx]
    ground_truth.append(gt)
flat_time = (time.perf_counter() - t0) * 1000.0
print(f'Flat search: {flat_time:.1f}ms total, {flat_time/NUM_QUERIES:.3f}ms/query')
print()

# TSM ANN search
print(f'Running TSM ANN search ({NUM_QUERIES} queries)...')
t0 = time.perf_counter()
tsm_results = []
for q in queries:
    results = engine.search_ann(q, TOP_K, EF)
    tsm_results.append(results)
tsm_time = (time.perf_counter() - t0) * 1000.0
print(f'TSM search: {tsm_time:.1f}ms total, {tsm_time/NUM_QUERIES:.3f}ms/query')
print()

# Calculate recall
print('Calculating recall...')
recalls = []
for gt, tsm_res in zip(ground_truth, tsm_results):
    gt_ids = {r[0] for r in gt}
    tsm_ids = {r[0] for r in tsm_res} if tsm_res else set()
    if gt_ids:
        recall = len(gt_ids & tsm_ids) / len(gt_ids)
        recalls.append(recall)

avg_recall = np.mean(recalls) * 100
print(f'Recall@{TOP_K}: {avg_recall:.1f}%')
print()

# Summary
print('=' * 70)
print('BENCHMARK RESULTS')
print('=' * 70)
print(f'Items:              {NUM_ITEMS:,}')
print(f'Dimension:          {DIM}')
print(f'Ingestion time:     {ingest_time:.1f}ms ({ingest_time/NUM_ITEMS:.3f}ms/item)')
print(f'TSM search time:    {tsm_time:.1f}ms ({tsm_time/NUM_QUERIES:.3f}ms/query)')
print(f'Flat search time:   {flat_time:.1f}ms ({flat_time/NUM_QUERIES:.3f}ms/query)')
print(f'Speedup vs flat:    {flat_time/tsm_time:.1f}x')
print(f'Recall@{TOP_K}:          {avg_recall:.1f}%')
print('=' * 70)

# Cleanup
engine.close()
shutil.rmtree(db_path, ignore_errors=True)
print()
print('Benchmark complete!')
"

echo ""
echo "Benchmark complete!"
