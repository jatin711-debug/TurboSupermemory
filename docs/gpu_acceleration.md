# GPU Acceleration Subsystem

`turbomemory_gpu` ([crates/turbomemory_gpu](../crates/turbomemory_gpu)) is an
optional crate behind the `cuda` cargo feature. This page says what it does
today, what it does not do, and what was measured.

> **Rewritten 2026-10-06.** The previous version of this page described a GPU
> HNSW build (random-projection bucketing, gateway connections, a search
> algorithm) and build-time speedups of 3–5×. The code never contained that
> algorithm. What it had was a brute-force neighbour pass that produced a
> graph nothing searched, on top of the normal CPU index build, and it made
> sealing about twice as slow. That path has been removed.

## 1. What the `cuda` feature does

| Operation | Where it runs |
|---|---|
| HNSW construction (sealed Hot segments) | CPU (`usearch`) |
| HNSW traversal, per query | CPU (`usearch`) |
| Quantized Warm / Cold scan | CPU (SIMD) |
| Full-f32 rerank, single query (`search_ann`, `search`) | CPU (SIMD) |
| Full-f32 rerank, **batch** (`search_ann_batch`, two or more queries, 256+ candidates in total) | **GPU**: one cuBLAS `gemm`, CPU fallback |

So the engine calls exactly one GPU path. Everything else is identical in a
CUDA build and a CPU build.

The ColBERT reranker in the Python SDK (`reranker="colbert"`) also uses the
GPU, but through PyTorch. It is unrelated to this crate and works without the
`cuda` cargo feature.

## 2. How the batched rerank works

`SegmentSnapshot::search_gpu_batch` (in
`crates/turbomemory_storage/src/segment_holder.rs`):

1. Each query gets its candidate list from the CPU segment search.
2. The union of all candidate offsets is built; every offset gets one index.
3. The union's vectors are gathered into one buffer and uploaded once.
4. The queries are scaled to unit length and uploaded as one matrix. Stored
   vectors are unit length, so the dot products `gemm` computes are cosines,
   exactly what the CPU path returns.
5. One `gemm` scores every query against every union vector; each query keeps
   the scores of its own candidates, sorts, and truncates.

Any CUDA error (no device, out of memory, kernel failure) is logged and the
CPU rerank runs instead. `MemoryEngine.gpu_accelerated` reports whether a CUDA
device was initialised.

## 3. The `GpuBackend` trait

```rust
pub trait GpuBackend: Send + Sync {
    fn name(&self) -> &str;
    fn total_memory(&self) -> usize;
    fn available_memory(&self) -> usize;
    fn upload_vectors(&self, vectors: &[f32], dim: usize) -> Result<DeviceBuffer>;
    fn upload_quantized(&self, quantized: &[u8], n: usize, dim: usize) -> Result<DeviceBuffer>;
    fn batch_cosine_similarity(&self, query: &[f32], vectors: &DeviceBuffer) -> Result<Vec<f32>>;
    fn batch_dot_product(&self, query: &[f32], vectors: &DeviceBuffer) -> Result<Vec<f32>>;
    fn batch_cosine_similarity_matrix(&self, queries: &[f32], m: usize, vectors: &DeviceBuffer)
        -> Result<Vec<f32>>;
    fn quantized_scan(/* 8-bit scalar scan */) -> Result<Vec<f32>>;
    fn spreading_activation_spmv(/* CSR step */) -> Result<Vec<f32>>;
}
```

Two implementations: `CudaBackend` (cudarc: cuBLAS + NVRTC) and `CpuFallback`.
`init_backend()` returns the CUDA backend when the feature is compiled in and
a device initialises, otherwise the fallback.

`batch_cosine_similarity_matrix` is the method the engine uses. The other
kernels (`batch_cosine_similarity` / `gemv`, `quantized_scan`,
`spreading_activation_spmv`) compile and have unit tests, but nothing in the
engine calls them.

Known limits of the CUDA backend:

- `available_memory()` returns the card's total memory, so the pre-upload
  check does not account for memory already in use; an oversized upload is
  caught by the allocation failing, which falls back to the CPU.
- `batch_cosine_similarity` (the single-query `gemv`) computes a plain dot
  product and assumes a unit-length query. It is not on any engine path.

## 4. Measurements (RTX 3050 Laptop 4 GB, CUDA toolkit 12.6, 16-thread CPU)

Taken 2026-10-06 on a machine that was also running other work; run-to-run
noise was large (the same build's sealing time ranged from 26 s to 54 s).
Treat these as order-of-magnitude. `benchmarks/gpu_parity.py` re-runs the
correctness checks and prints the same kind of timings for any build.

**Before the fixes** (CUDA build as it was):

| Check | CPU build | CUDA build |
|---|---|---|
| Sealing 16,000 × 768-d into 4 segments | 26–32 s | 61–62 s |
| Single query, rerank pool 512 | 4.3 ms | 7.2 ms |
| Single query, `top_k` 256, pool 2,000 | 8.5–9.6 ms | 17.6–19.3 ms |
| Batch search, recall@10 vs exact | 1.00 | 0.86, then an out-of-bounds crash |
| Score for a query scaled ×3 | 0.43 | 1.29 (dot product, not cosine) |

**After the fixes** (single-query rerank on CPU, the extra graph build gone,
batch rerank corrected):

| Check | CPU build | CUDA build |
|---|---|---|
| Sealing 16,000 × 768-d | 26–54 s | 28–63 s (same code; the spread is noise) |
| Batch search recall@10 vs exact (12,000 × 256-d) | 1.00 | 1.00 |
| Largest batch score error vs true cosine | 2e-7 | 4e-7 |
| Batch search, ms per query (16,000 × 768-d, batches of 32–256) | 3.5–5.0 | 3.5–6.8 |
| Score for a query scaled ×3 | 0.43 | 0.43 |

Conclusion for this card: the CUDA build is now correct and no longer slower
to build indexes, and it is **not faster** than the CPU build at anything
measured. HNSW traversal dominates a query; the rerank the GPU takes over is
a small part of it and is offset by the host-to-device copy. A larger GPU, or
much larger batches and candidate pools, may change that; it has not been
measured.

## 5. Build and test

```bash
make build-python FEATURES=cuda          # extension with the CUDA backend
cargo test -p turbomemory_gpu -p turbomemory_storage --features cuda --target-dir target/cuda
```

The second command needs a GPU. With the feature on,
`batch_search_matches_single_search_on_a_tiered_store`
(`crates/turbomemory_storage/tests/robustness.rs`) runs the batched rerank on
the GPU and checks it against the single-query path, including queries that
share candidates and a query that is not unit length. The CUDA build is not
part of `make gate`.

To compare a CUDA build with a CPU build from Python (results must agree;
timings are printed for both):

```bash
cargo build --release -p turbomemory_python --features cuda --target-dir target/cuda
mkdir -p target/cuda_ext && cp target/cuda/release/turbomemory.dll target/cuda_ext/turbomemory.pyd
python benchmarks/gpu_parity.py --ext .                 # CPU build at the repo root
python benchmarks/gpu_parity.py --ext target/cuda_ext   # CUDA build
```

A separate target directory keeps the CUDA build from replacing the CPU
extension the gate and the SDK tests use.

Requirements: an NVIDIA driver and the CUDA runtime libraries (cuBLAS, NVRTC)
on the library path. The workspace pins cudarc's `cuda-12080` bindings; they
load against the 12.6 toolkit used above.

## 6. Troubleshooting

- `gpu_accelerated` is `False` in a CUDA build: the device could not be
  initialised (no driver, `CUDA_VISIBLE_DEVICES` hiding it, missing runtime
  DLLs). The engine works normally on the CPU.
- The Python extension does not initialise a logger, so the engine's
  "falling back to CPU" warnings are not visible from Python; check
  `gpu_accelerated`.
- About 65 MiB of GPU memory stays allocated to the process after `close()`
  (the CUDA context); it is released when the process exits.
