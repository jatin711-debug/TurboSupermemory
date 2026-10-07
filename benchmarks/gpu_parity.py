#!/usr/bin/env python3
"""CPU-vs-CUDA parity and timing check for the turbomemory extension.

Runs one deterministic workload against whichever build lives in ``--ext`` and
prints what it measured. Run it once per build and compare:

    make build-python                     # CPU build -> target/release
    cargo build --release -p turbomemory_python --features cuda --target-dir target/cuda
    mkdir -p target/cuda_ext
    cp target/cuda/release/turbomemory.dll target/cuda_ext/turbomemory.pyd   # .so on Linux

    python benchmarks/gpu_parity.py --ext .                 # repo-root CPU extension
    python benchmarks/gpu_parity.py --ext target/cuda_ext   # CUDA extension

What must hold on every build (the script exits non-zero otherwise):
  - single-query and batch search return the same ids, and those match an
    exact numpy search;
  - every returned score is the cosine of that record (also for a query that
    is not unit length);
  - concurrent searches from many threads all succeed.

The timings are for comparison between builds on one machine; alternate the
builds a few times, a busy machine easily moves them by 2x.
"""

import argparse
import shutil
import sys
import tempfile
import threading
import time

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--ext", required=True, help="directory holding turbomemory.pyd / .so")
parser.add_argument("--n", type=int, default=16000, help="records (must exceed 4,096)")
parser.add_argument("--dim", type=int, default=768)
parser.add_argument("--hot-capacity", type=int, default=4000)
parser.add_argument("--queries", type=int, default=128)
args = parser.parse_args()

sys.path.insert(0, args.ext)
import numpy as np  # noqa: E402
import turbomemory  # noqa: E402

N, D, Q = args.n, args.dim, args.queries
rng = np.random.default_rng(7)
centers = rng.standard_normal((60, D)).astype(np.float32)
centers /= np.linalg.norm(centers, axis=1, keepdims=True)


def clustered(count):
    """Clustered unit vectors, so neighbouring queries share candidates."""
    v = centers[rng.integers(0, 60, size=count)]
    v = v + 1.4 * rng.standard_normal((count, D)).astype(np.float32) / np.sqrt(D)
    v = v.astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


data, queries = clustered(N), clustered(Q)
ids = [f"r{i}" for i in range(N)]
exact = queries @ data.T
exact_ids = [[ids[j] for j in row] for row in np.argsort(-exact, axis=1)[:, :10]]

db = tempfile.mkdtemp(prefix="tsm_gpu_parity_")
engine = turbomemory.MemoryEngine(db, D, hot_capacity=args.hot_capacity, auto_consolidation_secs=0)
print(f"extension : {turbomemory.__file__}")
print(f"gpu       : {engine.gpu_accelerated}   ({N} x {D}-d, {Q} queries)")
for s in range(0, N, 1000):
    e = min(N, s + 1000)
    engine.insert_batch(ids[s:e], ["t"] * (e - s), data[s:e], [0.5] * (e - s),
                        [[] for _ in range(e - s)])
t0 = time.perf_counter()
engine.flush()
print(f"seal      : {time.perf_counter() - t0:.1f} s")

failures = []


def recall(results):
    hits = sum(len({i for i, _ in got} & set(want)) for got, want in zip(results, exact_ids))
    return hits / (10 * Q)


def worst_score_error(results):
    return max((abs(float(exact[qi, int(rid[1:])]) - score)
                for qi, r in enumerate(results) for rid, score in r), default=0.0)


# Untimed warm-up: the first batch on a CUDA build pays for creating the
# context and the cuBLAS handle (seconds), which is not a per-query cost.
engine.search_ann(queries[0], 10)
engine.search_ann_batch(queries[:4], 10, 400)

for label, ef in (("default ef", None), ("ef=400", 400)):
    t0 = time.perf_counter()
    single = [engine.search_ann(q, 10, ef) for q in queries]
    single_ms = (time.perf_counter() - t0) * 1e3 / Q
    t0 = time.perf_counter()
    batch = []
    for s in range(0, Q, 32):
        batch.extend(engine.search_ann_batch(queries[s:s + 32], 10, ef))
    batch_ms = (time.perf_counter() - t0) * 1e3 / Q
    same = sum([i for i, _ in a] == [i for i, _ in b] for a, b in zip(single, batch))
    err = max(worst_score_error(single), worst_score_error(batch))
    print(f"{label:<10}: recall@10 single {recall(single):.3f} batch {recall(batch):.3f} | "
          f"batch == single {same}/{Q} | max score error {err:.1e} | "
          f"ms/query single {single_ms:.2f} batch {batch_ms:.2f}")
    if same != Q:
        failures.append(f"{label}: batch and single results differ for {Q - same} queries")
    if err > 1e-4:
        failures.append(f"{label}: a returned score is off by {err:.3g} from the true cosine")

unit = engine.search_ann_batch(queries[:8], 5, 400)
scaled = engine.search_ann_batch(queries[:8] * 3.0, 5, 400)
drift = max(abs(a[0][1] - b[0][1]) for a, b in zip(unit, scaled))
print(f"scaled x3 : top score moves by {drift:.1e} (must be ~0: scores are cosines)")
if drift > 1e-4:
    failures.append(f"a non-unit query changes the score by {drift:.3g}")

errors = []


def worker():
    try:
        for q in queries[:40]:
            engine.search_ann(q, 10, 400)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{type(exc).__name__}: {exc}")


threads = [threading.Thread(target=worker) for _ in range(32)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(f"32 threads: {len(errors)} failed")
if errors:
    failures.append(f"concurrent searches failed: {errors[0]}")

engine.close()
shutil.rmtree(db, ignore_errors=True)
if failures:
    print("FAILED:\n  " + "\n  ".join(failures))
    sys.exit(1)
print("OK")
