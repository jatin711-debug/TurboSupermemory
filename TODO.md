# TurboSuperMemory — Engineering TODO

Status key: **Done** | **In Progress** | **Pending**

---

## 1. HNSW / Vector Index Engine

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 1.1 | Replace `vector-index` with `hnswlib-rs`, `usearch`, or a custom C++ HNSW | `crates/turbomemory_storage/src/segments/hot.rs` | Done | Replaced with `usearch` v2.25; all tests pass |
| 1.2 | Make the HNSW backend swappable behind a trait | New: `crates/turbomemory_storage/src/index/` | Pending | Lets you benchmark CPU vs GPU backends later (Deferred to Phase E; `usearch` is sufficient for now) |
| 1.3 | Add bulk insert / bulk build path for sealed segments | `crates/turbomemory_storage/src/segments/sealed_hot.rs` | Done | `from_vectors` reads from `VectorStore` mmap without `Record` clones; still one-by-one `usearch` adds |
| 1.4 | Add `VisitedPool` to reuse visited bitsets across searches | `crates/turbomemory_storage/src/index/visited_pool.rs` | Pending | Avoids per-search allocation (Phase C) |
| 1.5 | Support index save/load/mmap for sealed segments | `crates/turbomemory_storage/src/segments/sealed_hot.rs`, `warm.rs`, `cold.rs` | Done | SealedHot + Warm + Cold segments persist and reload via manifest + mmap data file |
| 1.6 | Add deletion / tombstone support in HNSW | `crates/turbomemory_storage/src/segments/hot.rs`, graph layer | In Progress | Currently no real delete path (Phase A: implement delete/update API and tombstones) (Phase A: delete implemented; tombstone cleanup in vacuum) |
| 1.7 | Tune `M`, `ef_construct`, `ef_search` per dimension and dataset size | `crates/turbomemory_storage/src/segments/hot.rs`, config | In Progress | Added `hnsw_threshold`; still need dimension-aware `M`/`ef` defaults (Phase C) |
| 1.8 | Add `full_scan_threshold` fallback like Qdrant | `crates/turbomemory_storage/src/segment_holder.rs` | Done | `hnsw_threshold` routes small segments to quantized/plain scan; engine exact fallback remains at 4,096 records |
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
| 2.6 | Add quantized distance kernels (u8, binary, i8 RaBitQ) | `crates/turbomemory_core/src/metrics_quantized.rs` | Done | AVX2 u8 scalar + 1-bit sign dot-product kernels; i8/PQ future |
| 2.7 | Add batched distance computation for re-ranking | `crates/turbomemory_core/src/metrics.rs`, `segment_holder.rs` | Done | `cosine_similarity_batch` + chunked rerank in `SegmentHolder::search` |
| 2.8 | (Future) cuBLAS / CUDA batched dot-product path | New: `crates/turbomemory_gpu/` | Pending | Only after CPU path is competitive (Future / Phase E) |

---

## 3. Storage Architecture & Durability

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 3.1 | Make WAL the primary durability source of truth | `crates/turbomemory_storage/src/wal.rs`, `engine.rs` | Done | Embeddings now durable in `VectorStore`; WAL is metadata-only |
| 3.2 | Append to WAL first, then update in-memory state | `crates/turbomemory_storage/src/engine.rs` | Done | `VectorStore.put` first, then metadata WAL, then metadata cache/segments |
| 3.3 | Stop committing `redb` on every insert | `crates/turbomemory_storage/src/engine.rs` | Done | `meta.put` only updates cache; redb flushed lazily |
| 3.4 | Use `redb` only for metadata / ID maps, or replace it | `crates/turbomemory_storage/src/engine.rs` | Done | `redb` is now metadata snapshot + sequences |
| 3.5 | Add atomic segment swap during optimization | `crates/turbomemory_storage/src/segment_holder.rs` | Done | `seal_hot` swaps via `std::mem::replace` under write lock |
| 3.6 | Implement segment versioning / sequence numbers | `crates/turbomemory_storage/src/segment_holder.rs` | In Progress | WAL seq used; per-segment version not yet added (In Progress; finalize in Phase C) |
| 3.7 | Add `Flusher` closure pattern per component | All `*Segment`, `MetadataStore`, `Wal` | Done | Segment flusher pattern already in place |
| 3.8 | Add async flush worker | `crates/turbomemory_storage/src/update_handler.rs` | In Progress | Periodic fsync without blocking queries (In Progress; finalize in Phase D) |
| 3.9 | Implement crash recovery: replay WAL into segments | `crates/turbomemory_storage/src/engine.rs` | Done | `StorageEngine::open` replays WAL, snapshots, clears |
| 3.10 | Add `sync_threshold` / auto-persist for sealed indices | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Depends on sealed segment persistence (Phase C) |
| 3.11 | Framed append-only WAL with CRC32-C | `crates/turbomemory_storage/src/wal.rs` | Done | Foundation for 3.1 |

---

## 4. Segment Lifecycle & Optimizers

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 4.1 | Define sealed/immutable segment type | `crates/turbomemory_storage/src/segments/` | Pending | Hot is appendable; Warm/Cold should be immutable (Phase C) |
| 4.2 | Implement Hot → Warm sealing when Hot hits size threshold | `crates/turbomemory_storage/src/segment_holder.rs` | Done | First seal → SealedHot; subsequent seals → Warm; Warm → Cold when over capacity |
| 4.3 | Implement Warm → Cold demotion | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Use access scoring (Phase C) |
| 4.4 | Add indexing optimizer: build HNSW once segment is large enough | `crates/turbomemory_storage/src/optimizer.rs` | Done | `hnsw_threshold` in `TierConfig`; optimizer builds `SealedHotSegment` for large seals and `WarmSegment` for small ones |
| 4.5 | Add merge optimizer to reduce segment count | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Too many segments hurts search (Phase C) |
| 4.6 | Add vacuum optimizer to reclaim deleted points | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Tombstones accumulate over time (Phase A/B: vacuum deleted points) |
| 4.7 | Add config-mismatch optimizer | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Rebuilds segments when config changes (Phase C) |
| 4.8 | Add optimizer scheduling / backpressure | `crates/turbomemory_storage/src/update_handler.rs` | Pending | Don't optimize while queries are heavy (Phase D) |

---

## 5. Tiered Storage (Hot / Warm / Cold)

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 5.1 | Actually route vectors to Warm/Cold tiers by access frequency | `crates/turbomemory_storage/src/segment_holder.rs` | Done | Hot → SealedHot → Warm → Cold lifecycle now active; promotion back to Hot by access score |
| 5.2 | Add access scoring / recency tracking per record | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Needed for promotion/demotion (Phase C) |
| 5.3 | Implement scalar quantization with per-segment calibration | `crates/turbomemory_storage/src/segments/warm.rs` | Done | Exists; ensure it is used |
| 5.4 | Implement product quantization (PQ) option | `crates/turbomemory_storage/src/segments/warm.rs` | Pending | Better recall/compression than scalar (Phase C) |
| 5.5 | Implement binary / 1-bit quantization for Cold tier | `crates/turbomemory_storage/src/segments/cold.rs` | Done | Exists; ensure it is used |
| 5.6 | Precompute query LUT for quantized scoring | `crates/turbomemory_storage/src/segments/warm.rs` | In Progress | Exists; verify SIMD/fast path (In Progress; finalize in Phase C) |
| 5.7 | Add mmap-backed quantized vector storage | `crates/turbomemory_storage/src/segments/{warm,cold}.rs` | Done | Warm/Cold segments write manifest + mmap data file and reload on open |
| 5.8 | Add promotion from Warm/Cold back to Hot on access | `crates/turbomemory_storage/src/segment_holder.rs` | Done | `promote_hot` uses access score; now reachable because Warm/Cold tiers are populated |
| 5.9 | Hot/Warm/Cold segment scaffolding | `crates/turbomemory_storage/src/segments/` | Done | Architecture in place |

---

## 6. Memory Layout & Zero-Copy

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 6.1 | Replace `Vec<f32>` clones with mmap-backed storage | `crates/turbomemory_storage/src/vector_store.rs` | Done | `VectorStore` mmap-backed dense f32 file keyed by `PointOffset`; header CRC + `VectorReadView` for lock-free reads |
| 6.2 | Use `bytemuck` for zero-copy `&[f32]`/`&[u8]` views | `crates/turbomemory_storage/src/segments/mmap_array.rs` | Done | Added `as_typed_slice<T>` / `as_typed_slice_mut<T>` |
| 6.3 | Introduce `CowVector` / `BorrowedVector` abstractions | `crates/turbomemory_core/src/` | Pending | Defer until dense vector store refactor (Deferred; mmap + VectorReadView already zero-copy) |
| 6.4 | Store vectors in contiguous aligned arrays | `crates/turbomemory_storage/src/vector_store.rs` | Done | `VectorStore` uses contiguous `f32` array; search/rerank use `read_view` to avoid per-get locks |
| 6.5 | Avoid `String` clones in `id_index`; use `Arc<str>` or string pool | `crates/turbomemory_storage/src/engine.rs` | Done | `id_index` is now `ahash::HashMap<Arc<str>, PointOffset>` |
| 6.6 | Use `smallvec` for short candidate / link lists | `crates/turbomemory_storage/src/segments/mod.rs` | Done | `merge_candidates` uses `SmallVec<[ScoredPoint; 64]>` |
| 6.7 | Use `ahash` for `id_index` and graph maps | `crates/turbomemory_storage/src/engine.rs` | Done | `id_index` uses `ahash` |
| 6.8 | O(1) duplicate-ID index | `crates/turbomemory_storage/src/engine.rs` | Done | Recently added |
| 6.9 | O(1) `record_count` and avoid metadata `HashMap` clone on hot paths | `crates/turbomemory_storage/src/metadata_store.rs`, `engine.rs` | Done | `record_count` maintained atomically; `exact_top_k` and engine open iterate without cloning the map |

---

## 7. Concurrency & Locking

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 7.1 | Remove `Mutex<StorageEngine>` from Python binding | `crates/turbomemory_python/src/lib.rs` | Done | `PyMemoryEngine` holds `Arc<StorageEngine>` directly |
| 7.2 | Use `Arc<StorageEngine>` + `RwLock` for segment holder | `crates/turbomemory_storage/src/engine.rs` | Done | StorageEngine already uses `Arc<RwLock>` |
| 7.3 | Add `parking_lot::upgradable_read` for segment mutation | `crates/turbomemory_storage/src/segment_holder.rs` | Done | `insert` uses holder read lock + per-segment write lock; seal takes holder write lock only when needed |
| 7.4 | Separate read/write locks per segment | `crates/turbomemory_storage/src/segment_holder.rs` | Done | Each segment is `Arc<RwLock<dyn VectorSegment>>`; searches lock per-segment |
| 7.5 | Use lock-free object pools where possible | Search / visited lists | Pending | Reduces contention (Phase C) |
| 7.6 | Ensure background optimizer doesn't block readers | `crates/turbomemory_storage/src/update_handler.rs` | Done | Sealing builds the new segment under holder write lock briefly; readers use per-segment locks |
| 7.7 | Add graceful shutdown: stop workers, flush WAL, close mmap | `crates/turbomemory_storage/src/engine.rs`, `main.rs` | Done | Prevents corruption (Phase A: stop workers, flush WAL/vectors/segments, close mmap) (Phase A: `StorageEngine::shutdown` + Python context manager) |

---

## 8. Python Bindings

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 8.1 | Expose batch insert returning async future / background handle | `crates/turbomemory_python/src/lib.rs` | Done | GIL released; zero-copy numpy; async handle deferred to Phase D if needed (Phase B: async batch insert handle) |
| 8.2 | Accept numpy arrays without copying where possible | `crates/turbomemory_python/src/lib.rs` | Done | `numpy::PyReadonlyArray1/2` zero-copy views (Phase B: zero-copy numpy extraction) |
| 8.3 | Release the GIL during Rust search/indexing | `crates/turbomemory_python/src/lib.rs` | Done | `py.allow_threads` on all heavy calls (Phase B: release GIL during search/indexing) |
| 8.4 | Add proper Python exceptions mapping | `crates/turbomemory_python/src/lib.rs` | Done | `DuplicateId`/`DimensionMismatch`/`InvalidArgument` -> `ValueError`; `NotFound` -> `KeyError`; rest -> `RuntimeError` (Phase B: map StorageError to Python exceptions) |
| 8.5 | Add `__del__` / context manager for clean engine shutdown | `crates/turbomemory_python/src/lib.rs` | Done | Resource cleanup (Phase A/B: `close()` + context manager) (Phase A: close() + context manager) |
| 8.6 | PyO3 bindings with original Python API preserved | `crates/turbomemory_python/src/lib.rs` | Done | Existing baseline |

---

## 9. API / gRPC / REST

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 9.1 | Add batch insert endpoint to gRPC/REST | `crates/turbomemory_api/src/grpc.rs`, `rest.rs` | Done | Batch insert supports payloads; `/insert_batch` and `InsertBatch` RPC updated (Phase B: batch insert gRPC/REST endpoint) |
| 9.2 | Add delete / update endpoints | `crates/turbomemory_api/src/grpc.rs`, `rest.rs` | Done | `/delete`, `/update`, `Delete`, `Update`, plus `/get_payload` and `GetPayload` (Phase B: delete/update gRPC/REST endpoints) |
| 9.3 | Add health metrics endpoint (beyond `/health`) | `crates/turbomemory_api/src/rest.rs` | Pending | Prometheus-style metrics (Phase D) |
| 9.4 | Add request timeouts and payload size limits | `crates/turbomemory_api/src/main.rs` | Pending | Production hardening (Phase D) |
| 9.5 | Add CORS / auth middleware stubs | `crates/turbomemory_api/src/rest.rs` | Pending | Deployment readiness (Phase D) |
| 9.6 | gRPC + REST server scaffold | `crates/turbomemory_api/src/` | Done | Existing baseline |

---

## 10. Cognitive Graph Layer

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 10.1 | Persist graph edges durably | `crates/turbomemory_graph/src/` | Pending | Currently rebuilt from metadata (Phase E / Future) |
| 10.2 | Use quantized or sparse edge weights | `crates/turbomemory_graph/src/` | Pending | Memory blowup with dense graph (Phase E / Future) |
| 10.3 | Add access-time decay to edge weights | `crates/turbomemory_graph/src/` | Pending | Forgetting / recency (Phase E / Future) |
| 10.4 | Make spreading activation concurrent-safe | `crates/turbomemory_graph/src/` | Pending | Currently behind RwLock (Phase E / Future) |
| 10.5 | Add graph pruning / consolidation | `crates/turbomemory_graph/src/` | Pending | Remove stale edges (Phase E / Future) |
| 10.6 | Deterministic sorted graph edges | `crates/turbomemory_graph/src/` | Done | Existing baseline |

---

## 11. Payload, Filtering & Metadata

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 11.1 | Add payload storage (JSON / key-value) per record | `crates/turbomemory_storage/src/` | Done | `Record`/`MetaRecord` carry `payload: Option<String>`; WAL replay is bincode-safe because the JSON is stored as raw text rather than `serde_json::Value` |
| 11.2 | Add payload index (roaring bitmaps) | `crates/turbomemory_storage/src/` | Done | `PayloadIndex` with keyword + numeric range; filtered ANN via bitmap post-filter/over-fetch (Phase B: Roaring bitmap indexes for keyword/int/range filters) |
| 11.3 | Integrate filtered scorer with HNSW search | `crates/turbomemory_storage/src/segments/hot.rs` | Done | `VectorSegment::search` accepts `allowed_offsets`; Hot/SealedHot over-fetch + post-filter; Warm/Cold intersect bitmap (Phase B: pre-filter candidates before HNSW search) |
| 11.4 | Add full-text index for `text` field | `crates/turbomemory_storage/src/` | Done | `TextIndex` backed by `tantivy`; `Filter::FullText` supported in filtered ANN/cognitive search (Phase B: integrate tantivy or trigram full-text index) |
| 11.5 | Support sparse vectors | `crates/turbomemory_core/src/` | Pending | Nice-to-have later (Future / Phase E) |

---

## 12. Correctness & Edge Cases

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 12.1 | Validate embedding dimension matches config on every insert | `crates/turbomemory_storage/src/engine.rs` | Done | Silent corruption risk (Phase A: `validate_dimension` on insert and batch insert) |
| 12.2 | Handle duplicate IDs deterministically (update vs reject) | `crates/turbomemory_storage/src/engine.rs` | Done | O(1) check exists; semantics need docs/tests (Phase A: insert rejects duplicates, update replaces existing) |
| 12.3 | Add idempotent batch insert | `crates/turbomemory_storage/src/engine.rs` | Done | Replay safety (Phase A: skip existing/duplicate ids in batch insert) |
| 12.4 | Handle out-of-disk and mmap failures gracefully | All storage files | Done | Currently likely panics (Phase A: io errors from file set_len/mmap propagate as StorageError) |
| 12.5 | Add deletion semantics and tombstone cleanup | `crates/turbomemory_storage/src/engine.rs` | In Progress | Needed for production (Phase A: delete/update API implemented; vacuum cleanup pending) (Phase B/C: vacuum cleanup after delete/update) |
| 12.6 | Add corruption detection on WAL replay / segment load | `crates/turbomemory_storage/src/wal.rs` | Done | CRC is there, but verify all paths (Phase A: WAL CRC check, vector store header magic/version/CRC validation) |
| 12.7 | Make `trigger_consolidation` idempotent and observable | `crates/turbomemory_storage/src/engine.rs` | Done | Return work done / errors clearly (Phase A: returns sealed/compacted/promoted tuple) |

---

## 13. Observability & Tooling

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 13.1 | Replace `println!` / ad-hoc logging with `tracing` spans | Whole workspace | Pending | Needed for production debugging (Phase D) |
| 13.2 | Add metrics: ingest latency, search latency, segment sizes, tier counts | `crates/turbomemory_storage/src/engine.rs` | Pending | Use `metrics` crate (Phase D) |
| 13.3 | Add WAL lag / optimizer queue depth metrics | `crates/turbomemory_storage/src/update_handler.rs` | Pending | Operational visibility (Phase D) |
| 13.4 | Add structured logging configuration | `crates/turbomemory_api/src/main.rs` | Pending | Env-filter etc. (Phase D) |

---

## 14. Build, C++ / CUDA Integration & Deployment

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 14.1 | Add `build.rs` for C++ HNSW backend | New crate `crates/turbomemory_hnsw_cpp/` | Pending | If going custom C++ (Future / Phase E) |
| 14.2 | Use `cxx` or `bindgen` for safe C++ interop | `crates/turbomemory_hnsw_cpp/` | Pending | Avoid raw FFI (Future / Phase E) |
| 14.3 | Add CUDA/cuBLAS feature flag and crate | `crates/turbomemory_gpu/` | Pending | Future path (Future / Phase E) |
| 14.4 | Cross-platform build support (Windows MSVC, Linux, macOS) | `Cargo.toml`, build scripts | Pending | SIMD detection differs (Phase D) |
| 14.5 | Add `.cargo/config.toml` for target-specific flags | `.cargo/config.toml` | Pending | e.g., `-C target-cpu=native` (Phase D) |
| 14.6 | Add container / Docker build for the server | `deployments/` or new `docker/` | Pending | Deployment readiness (Phase D) |
| 14.7 | Release profile with LTO + codegen-units = 1 | `Cargo.toml` | Done | Already configured |

---

## 15. Testing & Benchmarking

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 15.1 | Add recall vs latency benchmarks across dimensions | `benchmark.py` / `benches/` | In Progress | `benches/vector_search.rs` covers exact/HNSW/Warm/Cold at D=128 |
| 15.2 | Add ingest throughput benchmark with concurrent writers | `benchmark.py` / `benches/` | Pending | Catch Mutex regressions (Phase B/C) |
| 15.3 | Add crash-recovery tests (kill -9, replay WAL) | `crates/turbomemory_storage/tests/` | Done | `tests/crash_recovery.rs` covers WAL replay, tier reload, truncated WAL |
| 15.4 | Add property-based tests for segment lifecycle | `crates/turbomemory_storage/tests/` | Pending | Seal/merge correctness (Phase B/C) |
| 15.5 | Add comparison harness vs Qdrant/Chroma | `benchmark.py` | In Progress | Exists; needs to be run regularly |
| 15.6 | Add continuous benchmark tracking | CI / `benches/` | Pending | Detect regressions (Future) |
| 15.7 | `cargo test --workspace`, `make verify`, `make audit`, `make benchmark --tsm-only`, `cargo clippy -D warnings` all pass | Workspace | Done | Existing baseline |

---

## 16. Imported Architecture Optimizations (Qdrant + Chroma)

Lessons from `docs/qdrant_architecture.md` and `docs/chroma_architecture.md` that are now queued for TSM implementation.

### 16.1 Ingestion speed

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 16.1.1 | Add appendable plain / brute-force segment for hot writes | `crates/turbomemory_storage/src/segments/hot.rs` | Done | `HotSegment` is now a plain offset list; HNSW built offline by optimizer |
| 16.1.2 | Add in-memory write buffer + periodic batch flush to HNSW | `crates/turbomemory_storage/src/segments/hot.rs`, `engine.rs` | Pending | Chroma-style fallback: buffer ≤100 items, flush asynchronously (deferred; plain segment already removes HNSW insert cost) |
| 16.1.3 | Move Hot-seal / HNSW rebuild / compaction to background optimizer | `crates/turbomemory_storage/src/optimizer.rs` (new), `segment_holder.rs` | Done | `BackgroundOptimizer` builds `SealedHot`/`Warm` from `sealing_plain`; `flush()` drains pending seals |
| 16.1.4 | Batch WAL appends and make flush policy configurable | `crates/turbomemory_storage/src/wal.rs`, `engine.rs` | Pending | `EveryWrite` / `EveryBatch` / `Periodic`; Qdrant/Chroma both amortize fsync |
| 16.1.5 | Move Tantivy text-index commit out of query path | `crates/turbomemory_storage/src/text_index.rs`, `engine.rs` | Done | `TextIndex` tracks pending docs; `commit_if_pending()` in `evaluate_filter`; periodic flush commits |
| 16.1.6 | Apply metadata / payload / text updates in log-compaction batches | `crates/turbomemory_storage/src/metadata_store.rs`, `payload_index.rs`, `text_index.rs` | Pending | Chroma-style: batch-apply a chunk of WAL records |

### 16.2 Search speed

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 16.2.1 | Use `ef = max(search_list_size, top_k)` at query time | `crates/turbomemory_storage/src/segment_holder.rs`, `segments/hot.rs`, `segments/sealed_hot.rs` | Done | `pool_k = max(search_list_size, top_k * multiplier)`; segment search uses caller's `pool_k` directly |
| 16.2.2 | Query segments in parallel with Rayon / thread pool | `crates/turbomemory_storage/src/segment_holder.rs` | Done | `rayon` added; segments searched in parallel; sequential fallback for single segment |
| 16.2.3 | Push payload bitmap into HNSW traversal as `FilteredScorer` | `crates/turbomemory_storage/src/segments/sealed_hot.rs` | Done | Selective-filter fallback to exact scan per `SealedHotSegment`; post-filter path uses larger `pool_k` |
| 16.2.4 | Add small-segment exact / brute-force fallback threshold | `crates/turbomemory_storage/src/segment_holder.rs` | Done | `hnsw_threshold` decides HNSW vs quantized/plain per segment; engine still exact-falls-back below 4,096 records |
| 16.2.5 | Accelerate Warm/Cold scans with SIMD + early-exit top-k | `crates/turbomemory_core/src/metrics_quantized.rs`, `segments/warm.rs`, `segments/cold.rs` | In Progress | Min-heap top-k implemented; SIMD sign dot and wider scalar SSE/NEON paths still pending |

### 16.3 Recall

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 16.3.1 | Expose `ef` / `search_list_size` in Python `search_ann` API | `crates/turbomemory_python/src/lib.rs`, `crates/turbomemory_api/` | Pending | Lets callers trade latency for recall |
| 16.3.2 | Floor filtered-search candidate pool at `ef` | `crates/turbomemory_storage/src/segment_holder.rs` | Done | `pool_k = max(search_list_size, top_k * 8)` for filtered queries |
| 16.3.3 | Add recall audit auto-tune: sample + raise `ef` if below target | `audit_recall.py`, `crates/turbomemory_storage/src/engine.rs` | Pending | Chroma/Qdrant both keep recall high via ef tuning |

---

## Suggested Execution Order

### Phase A — Production correctness ✅
1. **12.1–12.7** — Dimension validation, duplicate-ID semantics, idempotent batch, out-of-disk handling, deletion/tombstones, corruption detection, observable consolidation.
2. **7.7** — Graceful shutdown (stop workers, flush WAL/vectors/segments, close mmap).

### Phase B — Core DB features for users ✅
3. **11.1–11.4** — Payload storage, payload index (Roaring bitmaps), filtered ANN, full-text index.
4. **8.1–8.4** — Python binding hardening (async batch handle, zero-copy numpy, GIL release, exception mapping).
5. **9.1–9.2** — Batch insert and delete/update REST/gRPC endpoints.

### Phase C — Scalability / optimizer hardening (current)
6. **16.1–16.3** — Imported architecture optimizations (done):
   - **16.2.1** ✅ Query-time `ef` fix (recall@5 now 100% at N=5k D=128/768).
   - **16.1.5** ✅ Batch Tantivy commits / remove commit-on-query.
   - **16.1.1** ✅ Appendable plain segment for hot writes.
   - **16.1.3** ✅ Background optimizer for seal/merge/compaction.
   - **16.2.2 + 16.2.3** ✅ Parallel segment search + filtered HNSW traversal.
7. **Remaining Phase C items** — `16.1.2`, `16.1.4`, `16.1.6`, `16.2.5` (SIMD part), `16.3.1`, `16.3.3`, plus the original `4.5`, `4.6`, `5.x`, `1.7`, `3.x` backlog.
7. **4.1–4.7 + 5.2 + 5.4** — Immutable segments, access scoring, background builders, merge/vacuum/config-mismatch optimizers, product quantization.
8. **1.4 + 1.7 + 1.8 + 3.6 + 3.10** — VisitedPool, HNSW tuning, full-scan threshold, segment versioning, sync thresholds.

### Phase D — Operations / deployment
8. **13.1–13.4** — `tracing` + `metrics` + structured logging.
9. **9.3–9.5 + 14.4–14.6** — Health metrics, timeouts/limits, CORS/auth, cross-platform builds, Docker.
10. **3.8** — Async flush worker hardening.

### Phase E — Future / research
11. **10.1–10.5** — Durable graph persistence, then decay/pruning/consolidation.
12. **1.2 + 11.5 + 14.1–14.3** — Swappable HNSW trait, sparse vectors, optional custom C++/CUDA backends only after CPU path is benchmarked at million-scale.
13. **15.6** — Continuous benchmark tracking / CI.
