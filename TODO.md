# TurboSuperMemory — Engineering TODO

Status key: **Done** | **In Progress** | **Pending**

---

## 1. HNSW / Vector Index Engine

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 1.1 | Replace `vector-index` with `hnswlib-rs`, `usearch`, or a custom C++ HNSW | `crates/turbomemory_storage/src/segments/hot.rs` | Done | Replaced with `usearch` v2.25; all tests pass |
| 1.2 | Make the HNSW backend swappable behind a trait | New: `crates/turbomemory_storage/src/index/` | Pending | Lets you benchmark CPU vs GPU backends later |
| 1.3 | Add bulk insert / bulk build path for sealed segments | `crates/turbomemory_storage/src/segments/hot.rs` | Pending | Qdrant/Chroma build HNSW in bulk, not one-by-one |
| 1.4 | Add `VisitedPool` to reuse visited bitsets across searches | `crates/turbomemory_storage/src/index/visited_pool.rs` | Pending | Avoids per-search allocation |
| 1.5 | Support index save/load/mmap for sealed segments | `crates/turbomemory_storage/src/segments/hot.rs` | Pending | Needed for segment lifecycle |
| 1.6 | Add deletion / tombstone support in HNSW | `crates/turbomemory_storage/src/segments/hot.rs`, graph layer | Pending | Currently no real delete path |
| 1.7 | Tune `M`, `ef_construct`, `ef_search` per dimension and dataset size | `crates/turbomemory_storage/src/segments/hot.rs`, config | Pending | `D=8` and `D=1024` need different params |
| 1.8 | Add `full_scan_threshold` fallback like Qdrant | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Brute force is fine for tiny segments |
| 1.9 | Current Hot HNSW based on `vector-index` crate | `crates/turbomemory_storage/src/segments/hot.rs` | Done | Replaced by `usearch` |

---

## 2. Distance Metrics & SIMD

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 2.1 | Introduce a `Metric<T>` trait with `similarity`, `query_similarity`, `preprocess` | `crates/turbomemory_core/src/metrics.rs` (new) | Done | Added `Metric`, `CosineMetric`, `DotProductMetric`, `EuclideanMetric` |
| 2.2 | Implement AVX2/FMA f32 dot/cosine/euclidean kernels | `crates/turbomemory_core/src/metrics.rs` | Done | AVX2/FMA dot, L2, dot+norms |
| 2.3 | Implement SSE f32 kernels | `crates/turbomemory_core/src/metrics.rs` | Done | SSE dot, L2, dot+norms |
| 2.4 | Implement AArch64 NEON kernels | `crates/turbomemory_core/src/metrics.rs` | Done | NEON dot, L2, dot+norms |
| 2.5 | Add runtime CPU feature detection | `crates/turbomemory_core/src/metrics.rs` | Done | `is_x86_feature_detected!` + aarch64 NEON |
| 2.6 | Add quantized distance kernels (u8, binary, i8 RaBitQ) | `crates/turbomemory_core/src/metrics_quantized.rs` | Pending | Needed for Warm/Cold search |
| 2.7 | Add batched distance computation for re-ranking | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Helps exact re-ranking throughput |
| 2.8 | (Future) cuBLAS / CUDA batched dot-product path | New: `crates/turbomemory_gpu/` | Pending | Only after CPU path is competitive |

---

## 3. Storage Architecture & Durability

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 3.1 | Make WAL the primary durability source of truth | `crates/turbomemory_storage/src/wal.rs`, `engine.rs` | Done | Engine appends WAL first; `redb` is snapshot |
| 3.2 | Append to WAL first, then update in-memory state | `crates/turbomemory_storage/src/engine.rs` | Done | `insert`/`insert_batch` write WAL before cache/segments |
| 3.3 | Stop committing `redb` on every insert | `crates/turbomemory_storage/src/engine.rs` | Done | `meta.put` only updates cache; redb flushed lazily |
| 3.4 | Use `redb` only for metadata / ID maps, or replace it | `crates/turbomemory_storage/src/engine.rs` | Done | `redb` is now metadata snapshot + sequences |
| 3.5 | Add atomic segment swap during optimization | `crates/turbomemory_storage/src/segment_holder.rs` | Done | `seal_hot` swaps via `std::mem::replace` under write lock |
| 3.6 | Implement segment versioning / sequence numbers | `crates/turbomemory_storage/src/segment_holder.rs` | In Progress | WAL seq used; per-segment version not yet added |
| 3.7 | Add `Flusher` closure pattern per component | All `*Segment`, `MetadataStore`, `Wal` | Done | Segment flusher pattern already in place |
| 3.8 | Add async flush worker | `crates/turbomemory_storage/src/update_handler.rs` | In Progress | Periodic fsync without blocking queries |
| 3.9 | Implement crash recovery: replay WAL into segments | `crates/turbomemory_storage/src/engine.rs` | Done | `StorageEngine::open` replays WAL, snapshots, clears |
| 3.10 | Add `sync_threshold` / auto-persist for sealed indices | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Depends on sealed segment persistence |
| 3.11 | Framed append-only WAL with CRC32-C | `crates/turbomemory_storage/src/wal.rs` | Done | Foundation for 3.1 |

---

## 4. Segment Lifecycle & Optimizers

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 4.1 | Define sealed/immutable segment type | `crates/turbomemory_storage/src/segments/` | Pending | Hot is appendable; Warm/Cold should be immutable |
| 4.2 | Implement Hot → Warm sealing when Hot hits size threshold | `crates/turbomemory_storage/src/segment_holder.rs` | In Progress | Manual consolidation exists; make threshold-driven |
| 4.3 | Implement Warm → Cold demotion | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Use access scoring |
| 4.4 | Add indexing optimizer: build HNSW once segment is large enough | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Avoids indexing tiny segments |
| 4.5 | Add merge optimizer to reduce segment count | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Too many segments hurts search |
| 4.6 | Add vacuum optimizer to reclaim deleted points | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Tombstones accumulate over time |
| 4.7 | Add config-mismatch optimizer | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Rebuilds segments when config changes |
| 4.8 | Add optimizer scheduling / backpressure | `crates/turbomemory_storage/src/update_handler.rs` | Pending | Don't optimize while queries are heavy |

---

## 5. Tiered Storage (Hot / Warm / Cold)

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 5.1 | Actually route vectors to Warm/Cold tiers by access frequency | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Currently everything stays Hot |
| 5.2 | Add access scoring / recency tracking per record | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Needed for promotion/demotion |
| 5.3 | Implement scalar quantization with per-segment calibration | `crates/turbomemory_storage/src/segments/warm.rs` | Done | Exists; ensure it is used |
| 5.4 | Implement product quantization (PQ) option | `crates/turbomemory_storage/src/segments/warm.rs` | Pending | Better recall/compression than scalar |
| 5.5 | Implement binary / 1-bit quantization for Cold tier | `crates/turbomemory_storage/src/segments/cold.rs` | Done | Exists; ensure it is used |
| 5.6 | Precompute query LUT for quantized scoring | `crates/turbomemory_storage/src/segments/warm.rs` | In Progress | Exists; verify SIMD/fast path |
| 5.7 | Add mmap-backed quantized vector storage | `crates/turbomemory_storage/src/segments/{warm,cold}.rs` | In Progress | Chunked mmap exists for Warm |
| 5.8 | Add promotion from Warm/Cold back to Hot on access | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Cognitive / ANN hot data should stay fast |
| 5.9 | Hot/Warm/Cold segment scaffolding | `crates/turbomemory_storage/src/segments/` | Done | Architecture in place |

---

## 6. Memory Layout & Zero-Copy

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 6.1 | Replace `Vec<f32>` clones with mmap-backed storage | `crates/turbomemory_storage/src/segments/hot.rs` | Pending | Needs custom mmap vector store around usearch / dense storage |
| 6.2 | Use `bytemuck` for zero-copy `&[f32]`/`&[u8]` views | `crates/turbomemory_storage/src/segments/mmap_array.rs` | Done | Added `as_typed_slice<T>` / `as_typed_slice_mut<T>` |
| 6.3 | Introduce `CowVector` / `BorrowedVector` abstractions | `crates/turbomemory_core/src/` | Pending | Defer until dense vector store refactor |
| 6.4 | Store vectors in contiguous aligned arrays | `crates/turbomemory_storage/src/segments/hot.rs` | Pending | Depends on 6.1 mmap vector store |
| 6.5 | Avoid `String` clones in `id_index`; use `Arc<str>` or string pool | `crates/turbomemory_storage/src/engine.rs` | Done | `id_index` is now `ahash::HashMap<Arc<str>, PointOffset>` |
| 6.6 | Use `smallvec` for short candidate / link lists | `crates/turbomemory_storage/src/segments/mod.rs` | Done | `merge_candidates` uses `SmallVec<[ScoredPoint; 64]>` |
| 6.7 | Use `ahash` for `id_index` and graph maps | `crates/turbomemory_storage/src/engine.rs` | Done | `id_index` uses `ahash` |
| 6.8 | O(1) duplicate-ID index | `crates/turbomemory_storage/src/engine.rs` | Done | Recently added |

---

## 7. Concurrency & Locking

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 7.1 | Remove `Mutex<StorageEngine>` from Python binding | `crates/turbomemory_python/src/lib.rs` | Done | `PyMemoryEngine` holds `Arc<StorageEngine>` directly |
| 7.2 | Use `Arc<StorageEngine>` + `RwLock` for segment holder | `crates/turbomemory_storage/src/engine.rs` | Done | StorageEngine already uses `Arc<RwLock>` |
| 7.3 | Add `parking_lot::upgradable_read` for segment mutation | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Better than plain RwLock |
| 7.4 | Separate read/write locks per segment | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Fine-grained concurrency |
| 7.5 | Use lock-free object pools where possible | Search / visited lists | Pending | Reduces contention |
| 7.6 | Ensure background optimizer doesn't block readers | `crates/turbomemory_storage/src/update_handler.rs` | Pending | Copy-on-write segment swap |
| 7.7 | Add graceful shutdown: stop workers, flush WAL, close mmap | `crates/turbomemory_storage/src/engine.rs`, `main.rs` | Pending | Prevents corruption |

---

## 8. Python Bindings

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 8.1 | Expose batch insert returning async future / background handle | `crates/turbomemory_python/src/lib.rs` | Pending | Better Python UX |
| 8.2 | Accept numpy arrays without copying where possible | `crates/turbomemory_python/src/lib.rs` | Pending | Currently likely copies via PyAny |
| 8.3 | Release the GIL during Rust search/indexing | `crates/turbomemory_python/src/lib.rs` | Pending | Otherwise Python is blocked |
| 8.4 | Add proper Python exceptions mapping | `crates/turbomemory_python/src/lib.rs` | Pending | Better error messages |
| 8.5 | Add `__del__` / context manager for clean engine shutdown | `crates/turbomemory_python/src/lib.rs` | Pending | Resource cleanup |
| 8.6 | PyO3 bindings with original Python API preserved | `crates/turbomemory_python/src/lib.rs` | Done | Existing baseline |

---

## 9. API / gRPC / REST

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 9.1 | Add batch insert endpoint to gRPC/REST | `crates/turbomemory_api/src/grpc.rs`, `rest.rs` | Pending | Clients expect batch APIs |
| 9.2 | Add delete / update endpoints | `crates/turbomemory_api/src/grpc.rs`, `rest.rs` | Pending | Basic CRUD missing |
| 9.3 | Add health metrics endpoint (beyond `/health`) | `crates/turbomemory_api/src/rest.rs` | Pending | Prometheus-style metrics |
| 9.4 | Add request timeouts and payload size limits | `crates/turbomemory_api/src/main.rs` | Pending | Production hardening |
| 9.5 | Add CORS / auth middleware stubs | `crates/turbomemory_api/src/rest.rs` | Pending | Deployment readiness |
| 9.6 | gRPC + REST server scaffold | `crates/turbomemory_api/src/` | Done | Existing baseline |

---

## 10. Cognitive Graph Layer

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 10.1 | Persist graph edges durably | `crates/turbomemory_graph/src/` | Pending | Currently rebuilt from metadata |
| 10.2 | Use quantized or sparse edge weights | `crates/turbomemory_graph/src/` | Pending | Memory blowup with dense graph |
| 10.3 | Add access-time decay to edge weights | `crates/turbomemory_graph/src/` | Pending | Forgetting / recency |
| 10.4 | Make spreading activation concurrent-safe | `crates/turbomemory_graph/src/` | Pending | Currently behind RwLock |
| 10.5 | Add graph pruning / consolidation | `crates/turbomemory_graph/src/` | Pending | Remove stale edges |
| 10.6 | Deterministic sorted graph edges | `crates/turbomemory_graph/src/` | Done | Existing baseline |

---

## 11. Payload, Filtering & Metadata

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 11.1 | Add payload storage (JSON / key-value) per record | `crates/turbomemory_storage/src/` | Pending | Needed for real use cases |
| 11.2 | Add payload index (roaring bitmaps) | `crates/turbomemory_storage/src/` | Pending | Filtered ANN |
| 11.3 | Integrate filtered scorer with HNSW search | `crates/turbomemory_storage/src/segments/hot.rs` | Pending | Qdrant-style ACORN/pre-filter |
| 11.4 | Add full-text index for `text` field | `crates/turbomemory_storage/src/` | Pending | Chroma uses `tantivy` |
| 11.5 | Support sparse vectors | `crates/turbomemory_core/src/` | Pending | Nice-to-have later |

---

## 12. Correctness & Edge Cases

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 12.1 | Validate embedding dimension matches config on every insert | `crates/turbomemory_storage/src/engine.rs` | Pending | Silent corruption risk |
| 12.2 | Handle duplicate IDs deterministically (update vs reject) | `crates/turbomemory_storage/src/engine.rs` | In Progress | O(1) check exists; semantics need docs/tests |
| 12.3 | Add idempotent batch insert | `crates/turbomemory_storage/src/engine.rs` | Pending | Replay safety |
| 12.4 | Handle out-of-disk and mmap failures gracefully | All storage files | Pending | Currently likely panics |
| 12.5 | Add deletion semantics and tombstone cleanup | `crates/turbomemory_storage/src/engine.rs` | Pending | Needed for production |
| 12.6 | Add corruption detection on WAL replay / segment load | `crates/turbomemory_storage/src/wal.rs` | Pending | CRC is there, but verify all paths |
| 12.7 | Make `trigger_consolidation` idempotent and observable | `crates/turbomemory_storage/src/engine.rs` | Pending | Return work done / errors clearly |

---

## 13. Observability & Tooling

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 13.1 | Replace `println!` / ad-hoc logging with `tracing` spans | Whole workspace | Pending | Needed for production debugging |
| 13.2 | Add metrics: ingest latency, search latency, segment sizes, tier counts | `crates/turbomemory_storage/src/engine.rs` | Pending | Use `metrics` crate |
| 13.3 | Add WAL lag / optimizer queue depth metrics | `crates/turbomemory_storage/src/update_handler.rs` | Pending | Operational visibility |
| 13.4 | Add structured logging configuration | `crates/turbomemory_api/src/main.rs` | Pending | Env-filter etc. |

---

## 14. Build, C++ / CUDA Integration & Deployment

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 14.1 | Add `build.rs` for C++ HNSW backend | New crate `crates/turbomemory_hnsw_cpp/` | Pending | If going custom C++ |
| 14.2 | Use `cxx` or `bindgen` for safe C++ interop | `crates/turbomemory_hnsw_cpp/` | Pending | Avoid raw FFI |
| 14.3 | Add CUDA/cuBLAS feature flag and crate | `crates/turbomemory_gpu/` | Pending | Future path |
| 14.4 | Cross-platform build support (Windows MSVC, Linux, macOS) | `Cargo.toml`, build scripts | Pending | SIMD detection differs |
| 14.5 | Add `.cargo/config.toml` for target-specific flags | `.cargo/config.toml` | Pending | e.g., `-C target-cpu=native` |
| 14.6 | Add container / Docker build for the server | `deployments/` or new `docker/` | Pending | Deployment readiness |
| 14.7 | Release profile with LTO + codegen-units = 1 | `Cargo.toml` | Done | Already configured |

---

## 15. Testing & Benchmarking

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 15.1 | Add recall vs latency benchmarks across dimensions | `benchmark.py` / `benches/` | Pending | Prove improvements |
| 15.2 | Add ingest throughput benchmark with concurrent writers | `benchmark.py` / `benches/` | Pending | Catch Mutex regressions |
| 15.3 | Add crash-recovery tests (kill -9, replay WAL) | `crates/turbomemory_storage/tests/` | Pending | Durability correctness |
| 15.4 | Add property-based tests for segment lifecycle | `crates/turbomemory_storage/tests/` | Pending | Seal/merge correctness |
| 15.5 | Add comparison harness vs Qdrant/Chroma | `benchmark.py` | In Progress | Exists; needs to be run regularly |
| 15.6 | Add continuous benchmark tracking | CI / `benches/` | Pending | Detect regressions |
| 15.7 | `cargo test --workspace`, `make verify`, `make audit`, `make benchmark --tsm-only`, `cargo clippy -D warnings` all pass | Workspace | Done | Existing baseline |

---

## Suggested Execution Order

1. **2.1–2.5** — SIMD metrics (quick win, ~5–10×)
2. **1.1–1.2** — Replace HNSW engine (biggest win, ~10–50×)
3. **3.1–3.5** — WAL-first + segment sealing (fixes ingest latency)
4. **6.1–6.4** — Mmap/zero-copy vector storage (fixes memory + copies)
5. **7.1–7.3** — Remove Python `Mutex`, add proper concurrency
6. **5.1–5.7** — Actually use Warm/Cold tiers
7. **10.1–10.4** — Harden cognitive graph persistence
8. **11+** — Features like payload, filtering, full-text
