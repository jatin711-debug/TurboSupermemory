#!/usr/bin/env python3
"""Diagnose ANN recall vs ef and segment structure."""
import os
import sys
import shutil
import time
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
is_windows = sys.platform.startswith("win")
ext_suffix = ".pyd" if is_windows else ".so"
pyd_path = os.path.join(current_dir, f"turbomemory{ext_suffix}")
lib_prefix = "" if is_windows else "lib"
lib_suffix = ".dll" if is_windows else ".so"
lib_filename = f"{lib_prefix}turbomemory{lib_suffix}"
source = os.path.join(current_dir, "target", "release", lib_filename)
if os.path.exists(source):
    shutil.copy(source, pyd_path)

import turbomemory


def clustered_embeddings(n, dim, n_clusters=64, seed=42):
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, dim).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    assign = rng.randint(0, n_clusters, size=n)
    jitter = 0.15 * rng.randn(n, dim).astype(np.float32)
    embs = centers[assign] + jitter
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return embs


def exact_topk(queries, embeddings, k=5):
    scores = queries @ embeddings.T
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    return [set(f"mem_{idx}" for idx in top[i]) for i in range(len(queries))]


def measure(n, dim, ef_values):
    db_dir = os.path.join(current_dir, f"diag_recall_{n}_{dim}")
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

    embeddings = clustered_embeddings(n, dim)
    queries = clustered_embeddings(100, dim, seed=43)
    gt = exact_topk(queries, embeddings, k=5)

    engine = turbomemory.MemoryEngine(
        db_path=db_dir,
        dimension=dim,
        auto_consolidation_secs=0,
        initial_capacity=n,
    )

    batch_size = 512
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        ids = [f"mem_{i}" for i in range(start, end)]
        texts = [f"text {i}" for i in range(start, end)]
        engine.insert_batch(ids, texts, embeddings[start:end], [1.0] * (end - start), [[f"c{i%5}"] for i in range(start, end)])

    print(f"\n=== n={n}, dim={dim} ===")
    print(f"Record count before consolidation: {engine.record_count()}")

    # Recall before consolidation (exact/Hot scan)
    pre_recalls = []
    for q in queries:
        res = engine.search_ann(q, 5)
        pre_recalls.append({r[0] for r in res})
    pre_match = sum(len(a & b) for a, b in zip(pre_recalls, gt))
    print(f"Pre-consolidation recall@5 (exact scan): {pre_match / (len(queries)*5):.1%}")

    t0 = time.perf_counter()
    engine.trigger_consolidation()
    print(f"Consolidation time: {(time.perf_counter()-t0)*1000:.0f} ms")
    print(f"Record count after consolidation: {engine.record_count()}")

    for ef in ef_values:
        recalls = []
        latencies = []
        for q in queries:
            t0 = time.perf_counter()
            res = engine.search_ann(q, 5, search_list_size=ef)
            latencies.append(time.perf_counter() - t0)
            recalls.append({r[0] for r in res})
        match = sum(len(a & b) for a, b in zip(recalls, gt))
        print(f"  ef={ef:4d}: recall@5={match / (len(queries)*5):.1%}, latency={sum(latencies)/len(latencies)*1000:.2f} ms")

    engine.close()
    shutil.rmtree(db_dir, ignore_errors=True)


if __name__ == "__main__":
    measure(10000, 768, [64, 128, 256, 512, 1024])
    measure(50000, 768, [128, 256, 512, 1024, 2048])
