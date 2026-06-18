#!/usr/bin/env python
"""Timing repro: does idle time (background consolidation/merge) drop recall?

Ingest 50k, measure recall immediately, then sleep past consolidation cycles
and re-measure. If recall drops only after the sleep, the background optimizer
demotion/merge is the cause of the benchmark's 27.6%.
"""
import os, shutil, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
shutil.copy(os.path.join(ROOT, "target", "release", "turbomemory.dll"),
            os.path.join(ROOT, "turbomemory.pyd"))
import turbomemory

N, D, NQ, TOP_K = 50000, 1024, 50, 5
np.random.seed(42)
raw = np.random.randn(N, D).astype(np.float32)
emb = raw / np.linalg.norm(raw, axis=1, keepdims=True)
raw_q = np.random.randn(NQ, D).astype(np.float32)
queries = raw_q / np.linalg.norm(raw_q, axis=1, keepdims=True)
gt = [{f"mem_{int(i)}" for i in np.argsort(emb @ q)[::-1][:TOP_K]} for q in queries]

db = os.path.join(ROOT, "repro_timing_db")
if os.path.exists(db):
    shutil.rmtree(db)
eng = turbomemory.MemoryEngine(db_path=db, dimension=D, max_edges=16,
                               search_list_size=100, outlier_count=0,
                               initial_capacity=N)
bs = 512
for s in range(0, N, bs):
    e = min(s + bs, N)
    eng.insert_batch([f"mem_{i}" for i in range(s, e)],
                     [f"t{i}" for i in range(s, e)], emb[s:e],
                     [1.0] * (e - s), [[f"c{i%5}"] for i in range(s, e)])


def recall(label):
    r = []
    for q, g in zip(queries, gt):
        ann = eng.search_ann(q, TOP_K)
        r.append(len({a[0] for a in ann} & g) / TOP_K)
    print(f"{label:28s} ann R@5 = {np.mean(r)*100:5.1f}%", flush=True)


recall("t=0 (right after ingest)")
for wait in (65, 65, 65):
    time.sleep(wait)
    recall(f"t=+{wait}s slept (cumulative)")

eng.close()
shutil.rmtree(db, ignore_errors=True)
