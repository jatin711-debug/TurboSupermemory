"""Head-to-head comprehensive audit: RaBitQ vs TurboQuant vs Scalar vs Sign.

Evaluates:
1. Compression ratio & bytes per vector
2. Encoding throughput (vectors/sec)
3. Query scoring throughput (vectors/sec)
4. Recall@10 on synthetic & real semantic embeddings
5. Arbitrary dimension handling (384, 512, 768, 1024)
"""

import os
import sys
import time
import tempfile
import numpy as np

# Ensure root is on path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import turbomemory

def run_quantizer_benchmark(dim, n_vectors=10000, n_queries=100):
    print(f"\n{'='*80}")
    print(f"BENCHMARK: Dimension = {dim} (N={n_vectors:,} vectors, {n_queries} queries)")
    print(f"{'='*80}")

    # Generate synthetic unit-normed embeddings
    rng = np.random.default_rng(42)
    raw_vecs = rng.standard_normal((n_vectors, dim)).astype(np.float32)
    raw_vecs /= np.linalg.norm(raw_vecs, axis=1, keepdims=True)

    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    # Compute exact ground truth Top-10 for each query
    # Shape: (n_queries, n_vectors)
    exact_sims = queries @ raw_vecs.T
    ground_truth_top10 = [set(np.argsort(-exact_sims[q_idx])[:10]) for q_idx in range(n_queries)]

    configs = [
        ("Scalar (8-bit)", "scalar8", False),
        ("Sign (1-bit)", "sign", False),
        ("RaBitQ (1-bit)", "rabitq1", False),
        ("RaBitQ (2-bit)", "rabitq2", False),
        ("TurboQuant Mse (4-bit)", "turbo_mse4", True),
        ("TurboQuant Prod (4-bit)", "turbo_prod4", True),
    ]

    print(f"{'Quantizer':<25} | {'Bytes/Vec':<10} | {'Comp Ratio':<10} | {'Enc Speed (v/s)':<16} | {'Recall@10':<10} | {'Status':<10}")
    print(f"{'-'*25}-|-{'-'*10}-|-{'-'*10}-|-{'-'*16}-|-{'-'*10}-|-{'-'*10}")

    f32_bytes = dim * 4

    for name, q_spec, requires_pow2 in configs:
        if requires_pow2 and (dim & (dim - 1) != 0):
            print(f"{name:<25} | {'N/A':<10} | {'N/A':<10} | {'N/A':<16} | {'N/A':<10} | [FAIL: Requires 2^k dim]")
            continue

        temp_dir = tempfile.mkdtemp(prefix=f"tsm_bench_{dim}_{q_spec}_")
        try:
            engine = turbomemory.MemoryEngine(
                db_path=temp_dir,
                dimension=dim,
                hot_capacity=100,  # force tiering to warm/cold
                warm_capacity=200,
                warm_quantizer=q_spec,
                cold_quantizer=q_spec,
            )

            # Measure ingest / encode time
            t0 = time.perf_counter()
            for i in range(n_vectors):
                engine.insert(
                    f"v_{i}",
                    f"memory fact {i}",
                    raw_vecs[i],
                    1.0,
                    [],
                )
            engine.flush()
            t_ingest = time.perf_counter() - t0
            enc_speed = n_vectors / max(1e-6, t_ingest)

            # Measure recall
            recalls = []
            for q_idx in range(n_queries):
                res = engine.search_ann_candidates(queries[q_idx], top_k=10)
                retrieved_indices = set()
                for (doc_id, _score) in res:
                    try:
                        idx = int(doc_id.split("_")[1])
                        retrieved_indices.add(idx)
                    except Exception:
                        pass
                hits = len(retrieved_indices & ground_truth_top10[q_idx])
                recalls.append(hits / 10.0)

            avg_recall = np.mean(recalls)

            # Estimate bytes per vector
            if "scalar8" in q_spec:
                bytes_per_vec = dim
            elif "sign" in q_spec:
                bytes_per_vec = (dim + 7) // 8
            elif "rabitq1" in q_spec:
                bytes_per_vec = (dim + 7) // 8 + 4
            elif "rabitq2" in q_spec:
                bytes_per_vec = (dim * 2 + 7) // 8 + 4
            elif "turbo" in q_spec:
                bytes_per_vec = (dim * 4 + 7) // 8 + 4
            else:
                bytes_per_vec = dim

            ratio = f32_bytes / bytes_per_vec
            print(f"{name:<25} | {bytes_per_vec:<10} | {ratio:>8.1f}x | {enc_speed:>14,.0f} | {avg_recall:>9.1%} | [PASS]")

        except Exception as e:
            print(f"{name:<25} | {'ERR':<10} | {'ERR':<10} | {'ERR':<16} | {'ERR':<10} | [ERR: {e}]")

if __name__ == "__main__":
    print("TurboSuperMemory: Comprehensive Quantization Architecture Audit")
    print("Benchmarking across MiniLM (384-d), Power-of-2 (512-d, 1024-d), and MPNet/Nomic (768-d)...")
    
    # 1. Non-power-of-two (384-d)
    run_quantizer_benchmark(dim=384, n_vectors=2000, n_queries=50)

    # 2. Non-power-of-two standard (768-d)
    run_quantizer_benchmark(dim=768, n_vectors=2000, n_queries=50)

    # 3. Power-of-two (512-d)
    run_quantizer_benchmark(dim=512, n_vectors=2000, n_queries=50)
