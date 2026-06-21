#!/usr/bin/env python3
"""Inspect segment structure after consolidation."""
import os
import sys
import shutil
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


def clustered_embeddings(n, dim, seed=42):
    rng = np.random.RandomState(seed)
    centers = rng.randn(64, dim).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    assign = rng.randint(0, 64, size=n)
    jitter = 0.15 * rng.randn(n, dim).astype(np.float32)
    embs = centers[assign] + jitter
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return embs


def main():
    n, dim = 50000, 768
    db_dir = os.path.join(current_dir, "diag_segments_db")
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

    embeddings = clustered_embeddings(n, dim)
    engine = turbomemory.MemoryEngine(
        db_path=db_dir,
        dimension=dim,
        auto_consolidation_secs=0,
        initial_capacity=n,
    )
    for start in range(0, n, 512):
        end = min(start + 512, n)
        ids = [f"mem_{i}" for i in range(start, end)]
        engine.insert_batch(ids, ["x"] * (end - start), embeddings[start:end], [1.0] * (end - start), [["c"]] * (end - start))

    engine.trigger_consolidation()

    segments_dir = os.path.join(db_dir, "segments")
    print("\nSegment structure:")
    for tier in ["sealed_hot", "warm", "cold"]:
        tier_dir = os.path.join(segments_dir, tier)
        if not os.path.exists(tier_dir):
            continue
        segs = [d for d in os.listdir(tier_dir) if os.path.isdir(os.path.join(tier_dir, d))]
        print(f"  {tier}: {len(segs)} segments")
        for seg in segs:
            seg_path = os.path.join(tier_dir, seg)
            size_mb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fnames in os.walk(seg_path) for f in fnames) / (1024 * 1024)
            print(f"    - {seg}: {size_mb:.1f} MB")

    engine.close()
    shutil.rmtree(db_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
