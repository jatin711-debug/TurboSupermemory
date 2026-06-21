#!/usr/bin/env python3
"""Focused diagnostic: measure TSM ingest, search, memory, and recall
across collection sizes and dimensions."""
import os
import sys
import shutil
import time
import json
import numpy as np

# Import compiled extension (same logic as verify.py)
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

import turbomemory  # noqa: E402

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def peak_rss_mb():
    if not HAS_PSUTIL:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


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
    results = []
    for i, q in enumerate(queries):
        ids = [f"mem_{idx}" for idx in top[i]]
        results.append(set(ids))
    return results


def run_trial(n, dim, ef=128, seed=42):
    db_dir = os.path.join(current_dir, f"diag_db_{n}_{dim}")
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

    rng = np.random.RandomState(seed)
    embeddings = clustered_embeddings(n, dim, n_clusters=min(64, n), seed=seed)
    queries = clustered_embeddings(100, dim, n_clusters=min(64, n), seed=seed + 1)

    engine = turbomemory.MemoryEngine(
        db_path=db_dir,
        dimension=dim,
        max_edges=16,
        search_list_size=100,
        outlier_count=0,
        initial_capacity=n,
        auto_consolidation_secs=0,
    )

    # Ingest in batches
    batch_size = 512
    t0 = time.perf_counter()
    peak_mem = 0.0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        ids = [f"mem_{i}" for i in range(start, end)]
        texts = [f"text {i}" for i in range(start, end)]
        scores = [1.0] * (end - start)
        concepts = [[f"c{i % 5}"] for i in range(start, end)]
        engine.insert_batch(ids, texts, embeddings[start:end], scores, concepts)
        if HAS_PSUTIL:
            peak_mem = max(peak_mem, peak_rss_mb())
    ingest_ms_per_item = (time.perf_counter() - t0) * 1000.0 / n

    # Search before consolidation
    t0 = time.perf_counter()
    for q in queries:
        engine.search_ann(q, 5, ef)
    search_pre_ms = (time.perf_counter() - t0) * 1000.0 / len(queries)
    if HAS_PSUTIL:
        peak_mem = max(peak_mem, peak_rss_mb())

    # Consolidate
    t0 = time.perf_counter()
    engine.trigger_consolidation()
    consolidate_ms = (time.perf_counter() - t0) * 1000.0
    if HAS_PSUTIL:
        peak_mem = max(peak_mem, peak_rss_mb())

    # Search after consolidation
    t0 = time.perf_counter()
    results = []
    for q in queries:
        results.append(engine.search_ann(q, 5, ef))
    search_post_ms = (time.perf_counter() - t0) * 1000.0 / len(queries)
    if HAS_PSUTIL:
        peak_mem = max(peak_mem, peak_rss_mb())

    # Recall vs exact
    gt = exact_topk(queries, embeddings, k=5)
    matched = 0
    for i, res in enumerate(results):
        ids = {r[0] for r in res}
        matched += len(ids & gt[i])
    recall = matched / (len(queries) * 5.0)

    # Disk size
    disk_kb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fnames in os.walk(db_dir)
        for f in fnames
    ) / 1024.0

    engine.close()
    shutil.rmtree(db_dir, ignore_errors=True)

    return {
        "n": n,
        "dim": dim,
        "ingest_ms_per_item": ingest_ms_per_item,
        "search_pre_ms": search_pre_ms,
        "search_post_ms": search_post_ms,
        "consolidate_ms": consolidate_ms,
        "recall_at_5": recall,
        "peak_rss_mb": peak_mem,
        "disk_kb": disk_kb,
        "raw_vectors_mb": n * dim * 4 / (1024 * 1024),
    }


def main():
    configs = [
        (10000, 768),
        (50000, 768),
        (100000, 768),
        (50000, 1024),
    ]
    results = []
    for n, dim in configs:
        print(f"Running n={n}, dim={dim}...", flush=True)
        try:
            r = run_trial(n, dim, ef=128)
            results.append(r)
            print(json.dumps(r, indent=2))
        except Exception as e:
            print(f"FAILED n={n}, dim={dim}: {e}")
            import traceback
            traceback.print_exc()

    print("\n=== Summary ===")
    print(f"{'N':>8} {'Dim':>6} {'Ingest(ms)':>12} {'SearchPre':>10} {'SearchPost':>11} {'Consol(s)':>10} {'Recall':>8} {'PeakRSS':>10} {'DiskMB':>10} {'Overhead':>10}")
    for r in results:
        ov = r["peak_rss_mb"] / r["raw_vectors_mb"] if r["raw_vectors_mb"] else 0
        print(
            f"{r['n']:8} {r['dim']:6} {r['ingest_ms_per_item']:12.3f} "
            f"{r['search_pre_ms']:10.3f} {r['search_post_ms']:11.3f} "
            f"{r['consolidate_ms']/1000:10.1f} {r['recall_at_5']:8.1%} "
            f"{r['peak_rss_mb']:10.1f} {r['disk_kb']/1024:10.1f} {ov:10.1f}x"
        )


if __name__ == "__main__":
    main()
