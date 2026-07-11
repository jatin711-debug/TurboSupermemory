#!/usr/bin/env python3
"""W7 — profile cognitive-layer consolidation cost at scale ("measure first").

The cognitive consolidation cycle runs several O(live-records) passes each time:
supersession detection (mutual-nearest-neighbour: ~2 ANN queries per candidate),
importance recomputation, dedup, eviction, and a full graph JSON snapshot on
flush. Before making any of these incremental, we measure which one actually
dominates at 10k / 50k / 100k so the fix targets the real bottleneck.

Uses ONE long-lived engine per scale (not one-per-record), so it does not hit
the per-engine-lifecycle native-memory leak flagged separately.

Usage:
    python benchmarks/profile_consolidation.py --sizes 5000 20000 50000 --dim 768
"""

import argparse
import gc
import os
import shutil
import sys
import time
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # benchmarks/ (for turbomemory)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load_ext():
    ext = ".pyd" if sys.platform.startswith("win") else ".so"
    pyd = os.path.join(ROOT, f"turbomemory{ext}")
    if not os.path.exists(pyd):
        dll = os.path.join(ROOT, "target", "release", f"turbomemory.dll")
        if os.path.exists(dll):
            shutil.copy2(dll, pyd)
    import turbomemory
    return turbomemory


def unit_rows(n, dim, rng):
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v


def timed(label, fn):
    gc.collect()
    t = time.perf_counter()
    out = fn()
    ms = (time.perf_counter() - t) * 1000.0
    return label, ms, out


def graph_snapshot_bytes(db_path):
    """Total bytes of the persisted cognitive-graph snapshot (JSON in redb/meta).
    We approximate by the on-disk size of the metadata store files, which hold
    the graph JSON blob."""
    total = 0
    for root, _dirs, files in os.walk(db_path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def profile(tsm, n, dim, concepts_per, vocab, rng):
    db = tempfile.mkdtemp(prefix="tsm_prof_")
    # Cognitive features ON; big max_records so eviction does not delete during
    # the read-cost measurements (evict is timed separately as a no-op scan).
    engine = tsm.MemoryEngine(
        db_path=db, dimension=dim, max_concepts=5,
        auto_consolidation_secs=0,
        refinement_cosine_threshold=0.5, contradiction_cosine_threshold=0.5,
        importance_auto_scoring=True, concept_evolution_enabled=True,
        abstraction_co_occurrence_threshold=3,
        belief_source_roles=["user"], max_records=n * 10,
    )
    rows = []
    try:
        ids = [f"m{i}" for i in range(n)]
        texts = [f"memory record number {i} about topic {i % 97}" for i in range(n)]
        embs = unit_rows(n, dim, rng)
        cps = [[vocab[(i + j) % len(vocab)] for j in range(concepts_per)] for i in range(n)]
        scores = [1.0] * n

        # Ingest in batches to bound peak memory.
        t = time.perf_counter()
        B = 5000
        for s in range(0, n, B):
            e = min(s + B, n)
            engine.insert_batch(ids[s:e], texts[s:e], embs[s:e], scores[s:e], cps[s:e],
                                source_roles=["user"] * (e - s))
        ingest_ms = (time.perf_counter() - t) * 1000.0
        rows.append(("ingest (insert_batch)", ingest_ms))

        # Individual consolidation passes (propose_supersessions is PURE — no
        # mutation — so it is a clean read-cost measurement of MNN detection).
        for label, fn in [
            ("propose_supersessions (MNN detect)", lambda: engine.propose_supersessions()),
            ("recompute_importance (O(N))", lambda: engine.recompute_importance()),
            ("deduplicate (O(N)+ANN)", lambda: engine.deduplicate()),
            ("evict (O(N) scan, no-op cap)", lambda: engine.evict()),
        ]:
            _, ms, _ = timed(label, fn)
            rows.append((label, ms))

        # Whole cycle + durable flush (flush persists the graph JSON snapshot).
        _, cons_ms, _ = timed("trigger_consolidation (whole)", lambda: engine.trigger_consolidation())
        rows.append(("trigger_consolidation (whole)", cons_ms))
        _, flush_ms, _ = timed("flush (persist + graph JSON)", lambda: engine.flush())
        rows.append(("flush (persist + graph JSON)", flush_ms))

        snap = graph_snapshot_bytes(db)
        return rows, snap
    finally:
        engine.close()
        shutil.rmtree(db, ignore_errors=True)


def steady_state(tsm, base, delta, dim, concepts_per, vocab, rng, incremental):
    """Build `base` records + consolidate (warms the watermark), add `delta`
    more, then TIME the second consolidation. This is the steady-state pattern
    incremental detection targets: a cycle should cost O(delta), not O(base)."""
    db = tempfile.mkdtemp(prefix="tsm_ss_")
    engine = tsm.MemoryEngine(
        db_path=db, dimension=dim, max_concepts=5, auto_consolidation_secs=0,
        refinement_cosine_threshold=0.5, contradiction_cosine_threshold=0.5,
        importance_auto_scoring=True, belief_source_roles=["user"], max_records=(base + delta) * 10,
        incremental_supersession_detection=incremental,
    )
    try:
        def add(start, count):
            ids = [f"m{i}" for i in range(start, start + count)]
            texts = [f"memory record number {i} about topic {i % 97}" for i in range(start, start + count)]
            embs = unit_rows(count, dim, rng)
            cps = [[vocab[(i + j) % len(vocab)] for j in range(concepts_per)] for i in range(start, start + count)]
            for s in range(0, count, 5000):
                e = min(s + 5000, count)
                engine.insert_batch(ids[s:e], texts[s:e], embs[s:e], [1.0] * (e - s), cps[s:e],
                                    source_roles=["user"] * (e - s))
        add(0, base)
        engine.trigger_consolidation()  # cycle 1 (warms watermark when incremental)
        add(base, delta)
        _, ms, _ = timed("cycle2", lambda: engine.trigger_consolidation())
        return ms
    finally:
        engine.close()
        shutil.rmtree(db, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="Profile cognitive consolidation at scale (W7)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[5000, 20000, 50000])
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--concepts-per", type=int, default=3)
    ap.add_argument("--vocab", type=int, default=500, help="#distinct concepts")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steady-state", action="store_true",
                    help="compare incremental ON vs OFF for a 2nd consolidation after a bulk load")
    ap.add_argument("--base", type=int, default=20000)
    ap.add_argument("--delta", type=int, default=200)
    args = ap.parse_args()

    tsm = _load_ext()
    rng = np.random.default_rng(args.seed)
    vocab = [f"concept_{i}" for i in range(args.vocab)]

    if args.steady_state:
        print("=" * 92)
        print(f"Steady-state consolidation: base={args.base:,} + delta={args.delta} more, "
              f"time the 2nd cycle (dim={args.dim})")
        print("=" * 92)
        off = steady_state(tsm, args.base, args.delta, args.dim, args.concepts_per, vocab, rng, False)
        on = steady_state(tsm, args.base, args.delta, args.dim, args.concepts_per, vocab, rng, True)
        print(f"  2nd consolidation, incremental OFF (full re-scan): {off:>12.1f} ms")
        print(f"  2nd consolidation, incremental ON  (delta only):   {on:>12.1f} ms")
        print(f"  speedup: {off / on:>.1f}x   (base={args.base:,}, delta={args.delta})")
        return

    print("=" * 92)
    print(f"Consolidation profile  dim={args.dim}  concepts/rec={args.concepts_per}  vocab={args.vocab}")
    print("=" * 92)
    for n in args.sizes:
        rows, snap = profile(tsm, n, args.dim, args.concepts_per, vocab, rng)
        print(f"\n--- N = {n:,} records ---")
        print(f"{'phase':<40} {'total ms':>12} {'us/record':>12}")
        print("-" * 68)
        for label, ms in rows:
            print(f"{label:<40} {ms:>12.1f} {ms * 1000.0 / n:>12.1f}")
        print(f"{'graph snapshot + store on disk':<40} {snap / 1024 / 1024:>10.1f} MB")


if __name__ == "__main__":
    main()
