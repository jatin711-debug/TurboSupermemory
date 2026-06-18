#!/usr/bin/env python
"""Verify the Cold-demotion + ef fixes.

Ingest 50k, force full consolidation, then measure recall at several ef values
on BOTH adversarial (random) and realistic (clustered) data. Proves:
  - 8-bit Cold (Fix 1) no longer floors recall at ~27%.
  - a sane ef (Fix 3) recovers HNSW recall on adversarial data.
  - clustered data (Fix 3) is robust even at default ef.
"""
import os, shutil, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
shutil.copy(os.path.join(ROOT, "target", "release", "turbomemory.dll"),
            os.path.join(ROOT, "turbomemory.pyd"))
import turbomemory

N, D, NQ, TOP_K = 50000, 1024, 50, 5


def make_data(kind):
    np.random.seed(42)
    if kind == "clustered":
        nc = 64
        centers = np.random.randn(nc, D).astype(np.float32)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        a = np.random.randint(0, nc, N)
        emb = centers[a] + 0.15 * np.random.randn(N, D).astype(np.float32)
        qa = np.random.randint(0, nc, NQ)
        q = centers[qa] + 0.15 * np.random.randn(NQ, D).astype(np.float32)
    else:
        emb = np.random.randn(N, D).astype(np.float32)
        q = np.random.randn(NQ, D).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    gt = [{f"mem_{int(i)}" for i in np.argsort(emb @ qq)[::-1][:TOP_K]} for qq in q]
    return emb, q, gt


def run(kind):
    emb, queries, gt = make_data(kind)
    db = os.path.join(ROOT, f"verify_fix_db_{kind}")
    if os.path.exists(db):
        shutil.rmtree(db)
    # auto_consolidation disabled; we trigger it manually so timing is deterministic.
    eng = turbomemory.MemoryEngine(db_path=db, dimension=D, max_edges=16,
                                   search_list_size=100, outlier_count=0,
                                   initial_capacity=N, auto_consolidation_secs=0)
    bs = 512
    for s in range(0, N, bs):
        e = min(s + bs, N)
        eng.insert_batch([f"mem_{i}" for i in range(s, e)],
                         [f"t{i}" for i in range(s, e)], emb[s:e],
                         [1.0] * (e - s), [[f"c{i%5}"] for i in range(s, e)])
    eng.trigger_consolidation()
    time.sleep(2)
    eng.trigger_consolidation()
    time.sleep(2)

    print(f"\n=== {kind} data (after full consolidation) ===", flush=True)
    for ef in (100, 200, 400, 800):
        r = []
        for q, g in zip(queries, gt):
            ann = eng.search_ann(q, TOP_K, ef)
            r.append(len({a[0] for a in ann} & g) / TOP_K)
        print(f"  ef={ef:4d}  ann R@5 = {np.mean(r)*100:5.1f}%", flush=True)
    eng.close()
    shutil.rmtree(db, ignore_errors=True)


run("random")
run("clustered")
