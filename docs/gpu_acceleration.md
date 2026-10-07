# GPU Acceleration Subsystem

`turbomemory_gpu` ([crates/turbomemory_gpu](../crates/turbomemory_gpu)) is an
optional crate behind the `cuda` cargo feature. This page says what it does,
what it does not do, and what was measured.

> **History.** An earlier version of this page described a GPU HNSW build
> with 3–5× build speedups. The code never contained that algorithm, and the
> path it did have made sealing about twice as slow; it was removed on
> 2026-10-06. The first corrected CUDA build was merely *as fast* as the CPU
> build, because it re-uploaded candidate vectors on every query. Section 2
> describes what replaced it and section 5 what it measures.

## 1. What the `cuda` feature does

| Operation | Where it runs |
|---|---|
| **Vector search** (`search_ann`, `search_ann_batch`, and the vector stage of the cognitive `search`), once the store holds more than 4,096 records and fits the device budget | **GPU**: exact search over device-resident vectors (section 2) |
| The same searches at 4,096 records or fewer | CPU exact scan (already faster than a device round trip) |
| Vector search when the store does not fit the budget, or after any device error | CPU tiered search; a batch then reranks its candidates with one GPU `gemm` (section 3) |
| HNSW construction, quantized Warm / Cold segments | CPU. They are still built, so the CPU path is ready if the GPU path switches off |
| Graph expansion, BM25, fusion, filtering, id lookup | CPU |

The SDK's models use the GPU as well, through PyTorch and independently of
this feature: the local embedder (`SentenceTransformerEmbedder`), the GLiNER
extractor, the NLI verifier and the ColBERT reranker all pick `cuda` when
`torch.cuda.is_available()`.

## 2. GPU exact search over resident vectors

`crates/turbomemory_storage/src/gpu_exact.rs` and the `resident_*` methods of
`GpuBackend`.

**Idea.** The store's full-precision vectors are copied to device memory once
and stay there, row `i` holding the vector at offset `i`. A search uploads
only the query, computes its dot product with every row in one cuBLAS call
(`gemv` for one query, `gemm` for several), and downloads one score per row.
The host then picks the best `top_k` among the rows that are live records and
that the scope / payload filter allows.

**It is exact.** Stored vectors are unit length and the query is scaled to
unit length, so the scores are the cosines an exact CPU scan computes. There
is no index and no `ef` to tune: recall is 1.0 by construction, and
`search_list_size` is ignored on this path.

**Keeping it current.** Offsets are handed out in order and a vector is never
rewritten in place, so the rows already on the device stay valid. Each search
first uploads whatever was written since the previous one (nothing, usually).
Deletes and updates leave their old row on the device; it is masked on the
host because it is no longer a live offset. The first search after opening a
store creates the CUDA context and uploads everything: between 0.4 s and 5 s
for the stores in section 5, once per process.

**Concurrent searches share a device call.** One product runs on the device
at a time. Searches that arrive meanwhile queue up, and the next thread to
get the device answers all of them (up to 64) with a single `gemm`, which
reads the vectors once instead of once per query. `gpu_search_stats()` reports
`queries` and `device_calls`.

**Memory.** The mirror takes `rows × dimension × 4` bytes plus room to grow by
half. By default it may use half of the device memory that is free when the
first search runs (`gpu_memory_budget_mb` sets it explicitly). A store that
outgrows the budget, or any device error, switches the mirror off for the
life of the engine and frees its memory; searches continue on the CPU path
and return the same kind of results.

**Controls** (`TierConfig` fields, also `MemoryEngine` keyword arguments):

| Option | Default | Meaning |
|---|---|---|
| `gpu_exact_search` | `True` | Use the path at all. Only has an effect in a CUDA build with a usable device |
| `gpu_exact_min_records` | 4,097 | Store size from which it is used |
| `gpu_memory_budget_mb` | 0 = half of free device memory | Device memory the mirror may take |

`MemoryEngine.gpu_search_stats()` returns `None` until a search has used the
mirror, then a dict: `backend`, `active`, `rows`, `capacity_rows`,
`budget_bytes`, `memory_bytes`, `queries`, `device_calls`.

**What bounds its speed.** A single search reads the whole mirror once, so its
latency is the mirror's size divided by the card's memory bandwidth: about
1.1 ms per 150 MB on the RTX 3050 used here, plus ~0.3 ms per query. On this
card that beats the CPU paths up to a few hundred thousand vectors; it grows
linearly with the store, where a graph index grows much more slowly. Past the
budget the engine is back on the CPU path anyway. Keeping half-precision or
8-bit rows on the device would halve or quarter both the memory and the time;
that is not implemented.

## 3. The batched rerank (fallback)

`SegmentSnapshot::search_gpu_batch` in
`crates/turbomemory_storage/src/segment_holder.rs`. Used by `search_ann_batch`
when the resident path is off: each query gets its candidates from the CPU
segment search, the union of the candidates' vectors is uploaded once, and one
`gemm` scores every query against it. Queries are scaled to unit length so the
scores are cosines. Measured, it is no faster than the CPU rerank on this
card (the upload offsets the gain); it is kept because it is correct and
costs nothing when unused.

## 4. The `GpuBackend` trait

```rust
pub trait GpuBackend: Send + Sync {
    fn name(&self) -> &str;
    fn total_memory(&self) -> usize;
    fn available_memory(&self) -> usize;           // free device memory right now

    // resident exact search
    fn resident_create(&self, dim: usize, capacity_rows: usize) -> Result<ResidentMatrix>;
    fn resident_append(&self, matrix: &mut ResidentMatrix, rows: &[f32]) -> Result<()>;
    fn resident_scores(&self, matrix: &mut ResidentMatrix, query: &[f32]) -> Result<Vec<f32>>;
    fn resident_scores_batch(&self, matrix: &mut ResidentMatrix, queries: &[f32], count: usize)
        -> Result<Vec<f32>>;                       // scores[q * rows + r]

    // candidate rerank
    fn upload_vectors(&self, vectors: &[f32], dim: usize) -> Result<DeviceBuffer>;
    fn batch_cosine_similarity_matrix(&self, queries: &[f32], m: usize, vectors: &DeviceBuffer)
        -> Result<Vec<f32>>;

    // compiled and unit-tested, not called by the engine
    fn upload_quantized(/* ... */) -> Result<DeviceBuffer>;
    fn batch_cosine_similarity(/* single-query gemv */) -> Result<Vec<f32>>;
    fn batch_dot_product(/* ... */) -> Result<Vec<f32>>;
    fn quantized_scan(/* 8-bit scalar scan */) -> Result<Vec<f32>>;
    fn spreading_activation_spmv(/* CSR step */) -> Result<Vec<f32>>;
}
```

Two implementations: `CudaBackend` (cudarc: cuBLAS + NVRTC) and `CpuFallback`,
which implements the same calls on the host so the engine-side logic is
tested on machines without a GPU. `init_backend()` returns the CUDA backend
when the feature is compiled in and a device initialises, otherwise the
fallback. The resident path is only *used* with a real device.

## 5. Measurements (RTX 3050 Laptop 4 GB, CUDA toolkit 12.6, 16-thread CPU)

Taken 2026-10-06 with `benchmarks/gpu_parity.py`: clustered unit vectors,
128 queries, `top_k` 10, segments of 4,000 records, the CPU build and the
CUDA build searching the same store files, runs alternated. The machine was
also doing other work, so read the CPU figures as ±30% (runs of the same
configuration are shown as a range). The 100,000 × 384 row includes a run of
the final build on 2026-10-07.

| Store | Path | One query (median) | Batch of 32, per query | 8 threads | recall@10 |
|---|---|---|---|---|---|
| 20,000 × 384-d | CPU tiered | 2.2–3.0 ms | 2.3 ms | 600–780 /s | 1.000 |
| | **GPU exact** | **0.40 ms** | **0.27 ms** | **4,034 /s** | 1.000 |
| 100,000 × 384-d | CPU tiered | 5.9–7.5 ms | 6.0–9.2 ms | 128–174 /s | 0.999 |
| | **GPU exact** | **1.35–1.55 ms** | **0.34–0.39 ms** | **1,436–1,754 /s** | 1.000 |
| 100,000 × 768-d | CPU tiered | 13.2–15.8 ms | 13.6–15.8 ms | 69–71 /s | 1.000 |
| | **GPU exact** | **2.3–2.4 ms** | **0.34–0.39 ms** | **1,088–1,190 /s** | 1.000 |

For scale, an exact scan with NumPy (`data @ q` on a multithreaded BLAS, then
`argpartition`) took 1.1–1.5 ms, 5.0–6.4 ms and 8.8–11.2 ms on the three
stores.

So on this card the GPU path is 4–7× faster for a single query, 8–45× per
query in a batch, and 5–17× in throughput under concurrent load, and it
returns the exact result. The mirror took 44, 220 and 439 MiB.

Things these numbers do not show:

- **The CPU baseline is this engine's tiered search with 4,000-record
  segments**, each searched in turn. It is slower here than a plain BLAS scan
  of the same data, which says the CPU path has room to improve; a
  better-tuned CPU index would narrow the gap.
- **Larger stores.** Nothing above 100,000 records was measured. Single-query
  latency grows linearly with the store on the GPU path.
- **End-to-end `recall()`.** Embedding the query (a model call) usually costs
  more than the vector search, so the SDK's latency improves by less than the
  search does.
- **Other cards.** A card with more memory bandwidth will be proportionally
  faster; none was tested.

The CUDA build with `gpu_exact_search=False` (tiered search, batch rerank on
the GPU) measured the same as the CPU build: 2.2 / 7.2–7.6 / 13.2 ms per
query on the three stores.

## 6. Build and test

```bash
make build-python FEATURES=cuda          # extension with the CUDA backend
cargo test -p turbomemory_gpu -p turbomemory_storage --features cuda --target-dir target/cuda
```

The second command needs a GPU. The `resident_search_*` tests in
`crates/turbomemory_storage/tests/robustness.rs` check the path against a
brute-force ranking through inserts, a reallocation, deletes, updates, scope
and payload filters, a store that outgrows its budget, and concurrent
searches and writers. Without the feature they run the same engine code on
the host backend; with it they run on the device, as does
`batch_search_matches_single_search_on_a_tiered_store` (the batched rerank).
The CUDA build is not part of `make gate` or CI.

To compare a CUDA build with a CPU build from Python (results must agree;
timings are printed for both):

```bash
cargo build --release -p turbomemory_python --features cuda --target-dir target/cuda
mkdir -p target/cuda_ext && cp target/cuda/release/turbomemory.dll target/cuda_ext/turbomemory.pyd
python benchmarks/gpu_parity.py --ext . --db target/bench/db                         # CPU build
python benchmarks/gpu_parity.py --ext target/cuda_ext --db target/bench/db           # GPU exact search
python benchmarks/gpu_parity.py --ext target/cuda_ext --db target/bench/db --gpu-exact off
```

`--db` keeps the store so every build searches the same files. A separate
target directory keeps the CUDA build from replacing the CPU extension the
gate and the SDK tests use.

Requirements: an NVIDIA driver and the CUDA runtime libraries (cuBLAS, NVRTC)
on the library path. The workspace pins cudarc's `cuda-12080` bindings; they
load against the 12.6 toolkit used above.

## 7. Troubleshooting

- `gpu_accelerated` is `False` in a CUDA build: the device could not be
  initialised (no driver, `CUDA_VISIBLE_DEVICES` hiding it, missing runtime
  DLLs). The engine works normally on the CPU.
- `gpu_search_stats()` is `None`: no search has qualified yet (the store is
  at or below 4,096 records), or the path is off. `active: False` means it
  was switched off: the store outgrew `budget_bytes`, or a device call
  failed. Raise `gpu_memory_budget_mb` if the card has room.
- Another process took the device memory (a local LLM, a training job): the
  mirror fails to allocate or grow, switches off, and searches run on the
  CPU. It is not retried until the store is reopened.
- A batch and a single query can differ in the last bit of a score on the
  GPU (two different kernels); two records whose scores tie that closely may
  swap places.
- The Python extension does not initialise a logger, so the engine's
  "switched off" warnings are not visible from Python; check the stats.
- About 65 MiB of GPU memory stays allocated to the process after `close()`
  (the CUDA context); it is released when the process exits. The mirror
  itself is given back when the engine closes or the mirror switches off:
  with a 220 MiB mirror the process went from 299 MiB to 65 MiB on `close()`,
  and a store reopened in the same process got the full budget again.
