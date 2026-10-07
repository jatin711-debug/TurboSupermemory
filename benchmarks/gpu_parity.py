#!/usr/bin/env python3
"""CPU-vs-CUDA parity and timing check for the turbomemory extension.

Runs one deterministic workload against whichever build lives in ``--ext`` and
prints what it measured. Run it once per build and compare:

    make build-python                     # CPU build -> repo-root turbomemory.pyd/.so
    cargo build --release -p turbomemory_python --features cuda --target-dir target/cuda
    mkdir -p target/cuda_ext
    cp target/cuda/release/turbomemory.dll target/cuda_ext/turbomemory.pyd   # .so on Linux

    python benchmarks/gpu_parity.py --ext .                                # CPU build
    python benchmarks/gpu_parity.py --ext target/cuda_ext                  # CUDA, GPU exact search
    python benchmarks/gpu_parity.py --ext target/cuda_ext --gpu-exact off  # CUDA, rerank only

``--db DIR`` keeps the store between runs, so the index segments are built
once and every build searches the same files.

What must hold on every build (the script exits non-zero otherwise):
  - single-query and batch search agree, and with GPU exact search both are
    the exact numpy answer (recall 1.000);
  - every returned score is the cosine of that record (also for a query that
    is not unit length);
  - concurrent searches from many threads all succeed.

The timings are for comparison between builds on one machine; alternate the
builds a few times, a busy machine easily moves them by 2x.
"""

import argparse
import shutil
import statistics
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
parser.add_argument("--gpu-exact", choices=("on", "off"), default="on",
                    help="GPU exact search over device-resident vectors (CUDA builds only)")
parser.add_argument("--db", help="keep the store here and reuse it on the next run")
parser.add_argument("--threads", type=int, default=8, help="threads for the throughput run")
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

# What an exact search costs on this CPU with a tuned BLAS, for scale.
t0 = time.perf_counter()
for q in queries[:32]:
    np.argpartition(-(data @ q), 10)[:10]
numpy_ms = (time.perf_counter() - t0) * 1e3 / 32

db = args.db or tempfile.mkdtemp(prefix="tsm_gpu_parity_")
engine = turbomemory.MemoryEngine(
    db, D, hot_capacity=args.hot_capacity, auto_consolidation_secs=0,
    gpu_exact_search=args.gpu_exact == "on",
)
print(f"extension : {turbomemory.__file__}")
print(f"gpu       : {engine.gpu_accelerated}, exact search {args.gpu_exact}"
      f"   ({N} x {D}-d, {Q} queries)")
if engine.record_count() == 0:
    t0 = time.perf_counter()
    for s in range(0, N, 1000):
        e = min(N, s + 1000)
        engine.insert_batch(ids[s:e], ["t"] * (e - s), data[s:e], [0.5] * (e - s),
                            [[] for _ in range(e - s)])
    print(f"insert    : {time.perf_counter() - t0:.1f} s")
    t0 = time.perf_counter()
    engine.flush()
    print(f"seal      : {time.perf_counter() - t0:.1f} s")
elif engine.record_count() != N:
    sys.exit(f"{db} holds {engine.record_count()} records, expected {N}: use a fresh --db")
else:
    print("store     : reused")

failures = []


def recall(results):
    hits = sum(len({i for i, _ in got} & set(want)) for got, want in zip(results, exact_ids))
    return hits / (10 * Q)


def worst_score_error(results):
    return max((abs(float(exact[qi, int(rid[1:])]) - score)
                for qi, r in enumerate(results) for rid, score in r), default=0.0)


def agree(a, b):
    """Same records in the same order; two records whose scores differ only
    in the last bits may swap (a batch and a single query round differently
    on a GPU)."""
    return len(a) == len(b) and all(
        ia == ib or abs(sa - sb) < 1e-5 for (ia, sa), (ib, sb) in zip(a, b))


# Untimed warm-up: the first search on a CUDA build pays for creating the
# context and the cuBLAS handle and for uploading the vectors (see below),
# which is not a per-query cost.
t0 = time.perf_counter()
engine.search_ann(queries[0], 10)
first_ms = (time.perf_counter() - t0) * 1e3
engine.search_ann_batch(queries[:4], 10, 400)
stats = engine.gpu_search_stats()
if stats:
    print(f"mirror    : {stats['rows']} rows, {stats['memory_bytes'] / 2**20:.0f} MiB of "
          f"{stats['budget_bytes'] / 2**20:.0f} MiB budget, active {stats['active']}, "
          f"first search {first_ms:.0f} ms")
exact_path = bool(stats and stats["active"])
print(f"numpy     : {numpy_ms:.2f} ms/query for an exact scan (data @ q + argpartition)")

for label, ef in (("default ef", None), ("ef=400", 400)):
    times = []
    single = []
    for q in queries:
        t0 = time.perf_counter()
        single.append(engine.search_ann(q, 10, ef))
        times.append((time.perf_counter() - t0) * 1e3)
    t0 = time.perf_counter()
    batch = []
    for s in range(0, Q, 32):
        batch.extend(engine.search_ann_batch(queries[s:s + 32], 10, ef))
    batch_ms = (time.perf_counter() - t0) * 1e3 / Q
    same = sum(agree(a, b) for a, b in zip(single, batch))
    err = max(worst_score_error(single), worst_score_error(batch))
    print(f"{label:<10}: recall@10 single {recall(single):.3f} batch {recall(batch):.3f} | "
          f"batch == single {same}/{Q} | max score error {err:.1e} | "
          f"ms/query single {statistics.median(times):.2f} (p95 "
          f"{sorted(times)[int(0.95 * Q)]:.2f}) batch {batch_ms:.3f}")
    if same != Q:
        failures.append(f"{label}: batch and single results differ for {Q - same} queries")
    if err > 1e-4:
        failures.append(f"{label}: a returned score is off by {err:.3g} from the true cosine")
    if exact_path and min(recall(single), recall(batch)) < 0.9995:
        failures.append(f"{label}: GPU exact search returned an inexact result")
    if ef is None and exact_path:
        break  # ef does not apply to an exact search

unit = engine.search_ann_batch(queries[:8], 5, 400)
scaled = engine.search_ann_batch(queries[:8] * 3.0, 5, 400)
drift = max(abs(a[0][1] - b[0][1]) for a, b in zip(unit, scaled))
print(f"scaled x3 : top score moves by {drift:.1e} (must be ~0: scores are cosines)")
if drift > 1e-4:
    failures.append(f"a non-unit query changes the score by {drift:.3g}")

errors = []
ROUNDS = 4


def worker():
    try:
        for _ in range(ROUNDS):
            for q in queries:
                engine.search_ann(q, 10)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{type(exc).__name__}: {exc}")


before = engine.gpu_search_stats() or {}
threads = [threading.Thread(target=worker) for _ in range(args.threads)]
t0 = time.perf_counter()
for t in threads:
    t.start()
for t in threads:
    t.join()
elapsed = time.perf_counter() - t0
after = engine.gpu_search_stats() or {}
shared = ""
if after.get("active"):
    shared = (f" ({after['queries'] - before['queries']} queries in "
              f"{after['device_calls'] - before['device_calls']} device calls)")
print(f"{args.threads} threads : {args.threads * ROUNDS * Q / elapsed:,.0f} queries/s, "
      f"{len(errors)} failed{shared}")
if errors:
    failures.append(f"concurrent searches failed: {errors[0]}")

engine.close()
if not args.db:
    shutil.rmtree(db, ignore_errors=True)
if failures:
    print("FAILED:\n  " + "\n  ".join(failures))
    sys.exit(1)
print("OK")
