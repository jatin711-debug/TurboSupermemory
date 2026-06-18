#!/usr/bin/env python
"""Localize the 50k recall drop: sweep tier/ef configs against NumPy ground truth.

Reuses benchmark.py's exact data (seed 42, random Gaussian unit vectors) so the
numbers line up with the extreme benchmark. No Chroma/Qdrant — pure NumPy GT.
"""
import os
import sys
import shutil
import time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DLL = os.path.join(ROOT, "target", "release", "turbomemory.dll")
PYD = os.path.join(ROOT, "turbomemory.pyd")
shutil.copy(DLL, PYD)
import turbomemory

N = int(os.environ.get("DIAG_N", "50000"))
D = int(os.environ.get("DIAG_D", "1024"))
NQ = int(os.environ.get("DIAG_NQ", "50"))
TOP_K = 5

np.random.seed(42)
raw = np.random.randn(N, D).astype(np.float32)
emb = raw / np.where(np.linalg.norm(raw, axis=1, keepdims=True) == 0, 1.0,
                     np.linalg.norm(raw, axis=1, keepdims=True))
raw_q = np.random.randn(NQ, D).astype(np.float32)
queries = raw_q / np.where(np.linalg.norm(raw_q, axis=1, keepdims=True) == 0, 1.0,
                           np.linalg.norm(raw_q, axis=1, keepdims=True))

# Cosine ground truth (unit vectors -> dot == cosine), matches benchmark.py.
gt = []
for q in queries:
    scores = emb @ q
    gt.append({f"mem_{int(i)}" for i in np.argsort(scores)[::-1][:TOP_K]})

# Score-gap diagnostic: how separable is true #5 from #6 in raw cosine?
gaps = []
for q in queries:
    s = np.sort(emb @ q)[::-1]
    gaps.append(float(s[TOP_K - 1] - s[TOP_K]))
print(f"data: N={N} D={D} NQ={NQ}")
print(f"cosine gap between rank-{TOP_K} and rank-{TOP_K+1}: "
      f"mean={np.mean(gaps):.4f} min={np.min(gaps):.4f} max={np.max(gaps):.4f}")
print()


def build(**overrides):
    db = os.path.join(ROOT, f"diag_db_{overrides.get('tag','def')}")
    if os.path.exists(db):
        shutil.rmtree(db)
    kw = dict(db_path=db, dimension=D, max_edges=16, search_list_size=100,
              outlier_count=0, initial_capacity=N)
    for k in ("hot_capacity", "warm_capacity", "hnsw_threshold", "ef_construction"):
        if k in overrides:
            kw[k] = overrides[k]
    eng = turbomemory.MemoryEngine(**kw)
    bs = 512
    for s in range(0, N, bs):
        e = min(s + bs, N)
        eng.insert_batch([f"mem_{i}" for i in range(s, e)],
                         [f"t{i}" for i in range(s, e)],
                         emb[s:e], [1.0] * (e - s),
                         [[f"c{i%5}"] for i in range(s, e)])
    return eng, db


def measure(eng, sls=None):
    cand_r, ann_r = [], []
    t0 = time.perf_counter()
    for q, g in zip(queries, gt):
        cand = eng.search_ann_candidates(q, TOP_K, sls) if sls else eng.search_ann_candidates(q, TOP_K)
        ann = eng.search_ann(q, TOP_K, sls) if sls else eng.search_ann(q, TOP_K)
        cand_r.append(len({c[0] for c in cand} & g) / TOP_K)
        ann_r.append(len({a[0] for a in ann} & g) / TOP_K)
    ms = (time.perf_counter() - t0) * 1000.0 / NQ
    return np.mean(cand_r) * 100, np.mean(ann_r) * 100, ms


def run(label, sls=None, **ov):
    eng, db = build(**ov)
    cand, ann, ms = measure(eng, sls)
    print(f"{label:42s} | cand R@5 {cand:5.1f}% | ann R@5 {ann:5.1f}% | {ms:7.2f} ms/q")
    eng.close()
    shutil.rmtree(db, ignore_errors=True)


print(f"{'config':42s} | {'candidate':>13s} | {'reranked':>13s} | latency")
print("-" * 90)
run("default tiering (ef=100)", tag="def")
run("default tiering, ef=400", tag="ef400", sls=400)
run("default tiering, ef=800", tag="ef800", sls=800)
run("force exact (hot_capacity=N)", tag="exact", hot_capacity=N)
run("no quant: huge warm, low hnsw_thr", tag="warmonly",
    hnsw_threshold=10, warm_capacity=N)
