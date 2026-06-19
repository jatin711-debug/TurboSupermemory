# TurboSuperMemory — Million-Scale Optimization Roadmap

> Target: **1M+ vectors at 512–4096 dimensions**, sustained ingest, and sub-10 ms P99 search latency on capable hardware.
> Reference: Qdrant v1.18.2 (`D:/personal-projects/TurboSuperMemory/qdrant`), Chroma, Faiss, cuVS/RAFT.
> Status key: **Done** | **In Progress** | **Pending** | **Blocked**

---

## Recently Completed — 2026-06-19 (cognitive layer + TurboQuant hardening)

Shipped: the first installment of the *memory-as-cognition* layer (learnable
graph edges, abstraction hierarchy, durable graph reload) plus a latent-crash
fix in the TurboQuant config path. This is the work that differentiates TSM
from a plain vector DB and directly serves the "retain what matters, adapt as
new information arrives, build a coherent knowledge base" vision.

- **Learnable edge weights (`graph.rs`).** Edges are now weighted by
  `importance.sqrt()` at insert time instead of the constant `1.0`/`0.5`.
  `MemoryGraph::reinforce(id, now)` strengthens a memory's edges on retrieval
  (first recall gives a 1.5× boost, subsequent recalls grow as
  `1 + 0.1/(1+w)`, clamped at 8.0). `decay_edges(now, half_life)` erodes the
  *learned* portion of reinforced edges with an exponential half-life, floored
  at the baseline weight so decay never drops an edge below its birth strength.
  Unreinforced edges are untouched. This is the "retain what matters / forget
  what doesn't" loop: retrieval itself is the learning signal.
- **Abstraction hierarchy activated (`graph.rs`).** `EdgeKind::Abstraction`
  was defined but never constructed — now `build_abstractions(threshold)`
  creates parent concept nodes (e.g. `rust+safety`) from accumulated concept
  co-occurrence, with bidirectional `Abstraction` edges. Idempotent; co
  -occurrence counters reset after each call so only *new* co-occurrences
  trigger subsequent abstractions. Spreading activation now traverses
  `Abstraction` edges (exempt from hub suppression) so a query hitting one
  concept can reach memories of a sibling concept through the parent.
- **Durable graph reload (`engine.rs`).** `rebuild_graph` now loads the
  persisted graph JSON on open and merges in only records not already present,
  preserving learned weights, reinforcement timestamps, and abstraction nodes
  across restarts. Previously `MemoryGraph::from_json` was dead code and the
  graph was always rebuilt from records, losing all learned state.
- **Engine hooks.** `reinforce` is called on every cognitive-search result
  (`search` / `search_filtered`); `decay_edges` + `build_abstractions` run on
  `trigger_consolidation`; `add_memory_with_importance` is used on the insert
  and dedup-merge paths. All opt-in via config (disabled by default preserves
  pre-learning behavior).
- **Config + Python exposure.** `SpreadingConfig` added to `StoreConfig`; new
  `TierConfig` fields `abstraction_co_occurrence_threshold` (default 0 = off)
  and `edge_decay_half_life_secs` (default 0 = off); new Python kwargs
  `fok_threshold`, `spreading_decay`, `spreading_iterations`,
  `abstraction_co_occurrence_threshold`, `edge_decay_half_life_secs`.
- **TurboQuant panic fix (`config.rs`).** `QuantizerKind::build` now returns
  `Result` instead of `.expect()`, and `StorageEngine::open` validates
  TurboQuant tiers against non-power-of-two dimensions up front with an
  actionable `InvalidArgument` error. Previously selecting `turbo_mse`/
  `turbo_prod` with the default `dimension = 768` (not a power of two) would
  panic inside the quantizer constructor.
- **Table corrections.** 4.1 (batched SIMD kernel) and 4.6 (TurboQuant
  quantizer) were already implemented in code but still marked Pending in the
  table below — corrected to Done. 4.1 was noted Done in the 2026-06-19 audit
  header; 4.6 is fully implemented in `turbo_quant.rs` (1150 LOC, 12 tests,
  both MSE and Prod variants with LUT scoring).

Validated: `cargo fmt --check` clean; `cargo clippy -- -D warnings` clean;
`cargo test --workspace --exclude turbomemory_python` = 97 passed / 0 failed
(core 29 + graph 14 + storage 51 + crash_recovery 3); `make build-python` +
`python verify.py` E2E all pass.

---

## Recently Completed — 2026-06-19 (audit + memory lifecycle)

Two updates: (1) a deep codebase audit that found several roadmap items were
**already implemented but still marked Pending**, now corrected below; (2) the
memory-lifecycle feature (bounded-storage eviction + semantic dedup) shipped.

- **Bounded-storage eviction + semantic consolidation (new, opt-in, default OFF).**
  `StorageEngine::evict()` and `StorageEngine::deduplicate()` (`engine.rs`), wired
  into the consolidation cycle. Eviction selects victims by capacity cap
  (`max_records`) and/or `access_score` floor (`evict_score_floor`), with a grace
  period (`now - last_accessed < recency_half_life_secs / 8`) so freshly inserted,
  never-queried records survive. Dedup reuses the ANN index for candidate pairs +
  exact cosine above `dedup_cosine_threshold`, keeps the higher-salience record,
  transfers the victim's concept edges to the survivor, then `delete_by_id`.
  Config fields added to `TierConfig` (all `Option`, default `None` = current
  unbounded/no-dedup behavior) and surfaced in the Python constructor plus
  `evict()` / `deduplicate()` methods. Smoke-tested: eviction capped 60→20; dedup
  merged 2 of 4 near-duplicates.

- **Audit corrections — these were Pending in the roadmap but are DONE in code:**
  - **0.2 / 6.1 lock-free segment list** — `ArcSwap<SegmentSnapshot>` in
    `segment_holder.rs`; searches read a published snapshot, never block on swap.
  - **0.5 single batched update worker** — `update_worker.rs` (`IndexApplier` +
    crossbeam channel); all writes serialize through one `apply_batch`.
  - **3.1 visited-set pool** — `visited_pool.rs` (`VisitedSet` token array +
    generation wrap), parking_lot-guarded.
  - **3.5 parallel multi-segment search** — `into_par_iter` over segments in
    `segment_holder.rs` with top-k merge.
  - **4.1 batched SIMD distance kernel** — `cosine_similarity_batch` in
    `turbomemory_core/src/metrics.rs` with AVX2/FMA + SSE paths and a 4-vector
    unrolled kernel (`dot_and_nb_x4_avx2`).
  - **6.3 parking_lot** — in use across the storage crate (segment holder, update
    worker, metadata cache, access counters).

---

## Recently Completed — 2026-06-18 (session 2)

Shipped in commit `9fd68f5` ("feat: implement oversampling in reranking for improved neighbor recovery in vector segments"). Three improvement areas targeting consolidation speed, high-dimensional recall, and ingest cost.

- **Multi-threaded HNSW build (resolves 2.11).** `usearch_index.rs::build` now seeds the graph with the first `PARALLEL_SEED_POINTS = 256` points single-threaded, then inserts the remainder via a bounded rayon pool. Per-build thread count is `clamp(num_cpus / max_concurrent_builds, 1, 16)` so concurrent segment builds don't oversubscribe. Measured at 10k: consolidation time roughly halved — 768-dim 28.5 s → 14.2 s, 1024-dim 34.8 s → 17.9 s — with no recall regression (a self-recall guard test, `usearch_parallel_build_preserves_recall`, asserts recall@1 ≥ 0.95).
- **Oversampling margin in quantize-and-rerank (recall fix).** `warm.rs`/`cold.rs` `search` previously shortlisted *exactly* `top_k` quantized candidates before f32 rerank, so a true neighbor that quantizes to rank `top_k+1` was dropped before rerank could recover it. Now shortlists `max(top_k * RERANK_OVERSAMPLE, MIN_RERANK_SHORTLIST)` (8× / 64-floor) then reranks and truncates to `top_k`. Effect is muted at 10k (everything merges into one HNSW segment, so recall is HNSW-bound); the payoff is at 100k+ where Warm/Cold dominate. Guard test `warm_segment_oversampling_recovers_exact_nearest`.
- **Zero-copy numpy ingest (resolves 1.9 / 8.1).** PyO3 bindings now borrow contiguous `float32` numpy arrays directly (`F32Input`/`F32Matrix` View variants) instead of `slice.to_vec()` / `row.to_vec()`; lists, non-contiguous arrays, and wrong-dtype inputs fall back to owned copies. Engine batch signature changed to `&[&[f32]]` to carry the borrow through. The borrowed view is held on-stack across `py.allow_threads`; only the derived `&[f32]` is captured by the GIL-released closure.

Validated: full workspace test suite passes (storage incl. 2 new guard tests, core, graph, api); PyO3 build clean.

---

## Recently Completed — 2026-06-18

Shipped in commit `35cc6c7` ("fix: improve ANN recall with 8-bit Cold tier and configurable consolidation"). These resolve part of Phase 3/4 ahead of the structural work and fix a recall regression surfaced by high-dimensional benchmarks.

- **8-bit Cold tier (resolves Exec-Summary #7).** Cold tier default quantizer changed from 1-bit `Sign` to 8-bit `Scalar` (`config.rs`). Sign-only quantization could not resolve the tiny cosine gaps in high-dim data and floored recall at ~27% after full consolidation. `warm_capacity` budget also raised (16→64 MiB, clamp 200k→500k) so vectors stay in 8-bit Warm longer before demotion.
- **Caller-specified `ef` (partial 3.10).** `search_ann(query, top_k, ef)` is now wired through Python; benchmark exposes `--ef`. Adaptive auto-raise still pending.
- **Configurable background consolidation (partial 8.4).** `auto_consolidation_secs` Python kwarg (0 disables) replaces the hardcoded 60 s interval, so benchmarks and manual-consolidation workloads aren't silently penalized.
- **Realistic benchmark data (partial 10.5).** `benchmark.py` now defaults to clustered embeddings (64 centers, 0.15 jitter) instead of near-orthogonal Gaussian; old behavior available via `--data-distribution random`.

Validated at 50k × 1024 after full consolidation: adversarial-data recall floor lifted ~27% → ~71% at default `ef`; clustered data reaches 65–78% and scales with `ef`.

---

## Executive Summary — Are We Ready for 1M × 4k?

**No.** The codebase is solid for 100k × 128, but 1M+ vectors at high dimension introduces bottlenecks that are structural, not incremental:

1. **Single-node / single-shard design** — one `VectorStore`, one `SegmentHolder`, one `MetadataStore`, one graph.
2. **Single-threaded update worker** — all writes serialize through one channel/worker.
3. **~~Global `RwLock<SegmentHolder>`~~** — **Fixed (2026-06-19 audit).** Replaced with a lock-free `ArcSwap<SegmentSnapshot>`; searches read a published snapshot and never block on seal/merge swap (0.2 / 6.1).
4. **In-memory metadata cache** — every record's metadata lives in a `HashMap`; won't fit RAM at 1M+ text records. (redb is a lazy snapshot, not the primary cache; the cache itself is still a single in-memory `HashMap`.)
5. **No vacuum / physical deletion** — deleted vectors and old segments are never reclaimed.
6. **~~Single-threaded HNSW builds~~** — **Fixed (2026-06-18).** Builds now seed 256 points single-threaded then insert the remainder in parallel; see [Recently Completed](#recently-completed--2026-06-18-session-2). The 512 MiB default budget still applies.
7. **~~1-bit Cold tier collapse at 4k~~** — **Fixed (2026-06-18).** Cold tier now defaults to 8-bit scalar quantization; see [Recently Completed](#recently-completed--2026-06-18).
8. **~~No visited-set pool, no segment-level parallelism~~** — **Fixed (2026-06-19 audit).** Visited pool (3.1) and parallel multi-segment search (3.5) are implemented. Adaptive/cardinality-aware filtering is still pending.
9. **No GPU acceleration** — CPU-only HNSW build and distance compute.

This roadmap is the complete set of optimizations needed to reach 1M+ nodes × high dimensions. Items are grouped by phase, with Qdrant-derived defaults and concrete file targets.

---

## Phase 0 — Structural Foundations for Million-Scale

These are prerequisites. Without them, later optimizations are local fixes.

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 0.1 | **Shard the collection** into N independent `Shard` instances | `crates/turbomemory_storage/src/shard.rs` (new) | Pending | Each shard owns its own `VectorStore`, `SegmentHolder`, `MetadataStore`, WAL, optimizers, graph shard. Default `num_shards = clamp(num_cpus / 4, 2, 16)`. Route by `id` hash or explicit partition key. Qdrant model: `lib/collection/src/shards/`. |
| 0.2 | **Replace global `RwLock<SegmentHolder>` with lock-free segment list** | `crates/turbomemory_storage/src/segment_holder.rs` | Done | Done (2026-06-19 audit). `ArcSwap<SegmentSnapshot>` (`segment_holder.rs:208`); searches read the published snapshot via `snapshot_handle()`, mutations publish a new `Arc`. Single-node, never blocks on swap. |
| 0.3 | **Add collection abstraction above `StorageEngine`** | `crates/turbomemory_storage/src/collection.rs` (new) | Pending | One `Collection` = many `Shard`s. Python `MemoryEngine` opens a collection directory. Required for sharding and config per collection. |
| 0.4 | **Move metadata out of single in-memory `HashMap`** | `crates/turbomemory_storage/src/metadata_store.rs` | Pending | Use per-segment metadata files (Qdrant: `segment.json` + mmap id_tracker) or a paged metadata store. Keep hot cache, spill cold records to mmap. Target: < 20% of working set in RAM. |
| 0.5 | **Introduce per-shard update worker pool** | `crates/turbomemory_storage/src/update_worker.rs` | In Progress | A single batched update worker exists (`update_worker.rs`, `IndexApplier` + crossbeam channel; all writes serialize through `apply_batch`). The **per-shard pool** (one worker per shard) still depends on sharding (0.1) and is pending. |
| 0.6 | **Separate read / write / optimize / flush thread pools** | `crates/turbomemory_storage/src/runtime.rs` (new) | Pending | CPU-bound search pool, IO-bound pool, optimizer pool, flush pool. Adaptive switching when CPU > 90% (Qdrant `AdaptiveSearchHandle`). |
| 0.7 | **Add NUMA / huge-page awareness stubs** | `crates/turbomemory_storage/src/memory_policy.rs` (new) | Pending | Pin optimizer threads, advise `MEM_LARGE_PAGES` on Windows, `madvise(MADV_HUGEPAGE)` on Linux for vector mmap. |

---

## Phase 1 — Ingestion Pipeline (WAL + Update Workers)

Qdrant's ingestion is fast because inserts are **appends to plain segments + WAL**, not HNSW graph mutations. Copy that model exactly.

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 1.1 | **Make mutable Hot segment plain / brute-force only** | `crates/turbomemory_storage/src/segments/hot.rs` | Pending | Never mutate HNSW on insert. Hot segment = append-only offset list + optional exact index. Qdrant: `lib/segment/src/segment_constructor/segment_constructor_base.rs:160`. |
| 1.2 | **Chunk upserts into 32-point batches** | `crates/turbomemory_storage/src/update_worker.rs`, `engine.rs` | Pending | Qdrant `UPDATE_OP_CHUNK_SIZE = 32` (`lib/shard/src/update.rs:176`). Prevents long write locks. Deletions batch to 512 (`lib/shard/src/update.rs:342`). |
| 1.3 | **WAL segment size = 32 MiB with CRC32-C framing** | `crates/turbomemory_storage/src/wal.rs` | Pending | Qdrant default (`lib/wal/src/lib.rs:40`). Current TSM WAL is single-file; split into rotating 32 MB segments. |
| 1.4 | **Add `first-index` / acknowledged-offset file** | `crates/turbomemory_storage/src/wal.rs` | Pending | Qdrant `lib/shard/src/wal.rs:28`. Bounded replay on restart; prefix-truncate old segments after flush. |
| 1.5 | **Configurable flush policy: EveryWrite / EveryBatch / Periodic / None** | `crates/turbomemory_storage/src/wal.rs`, `config.rs` | Pending | Default Periodic 5 s (Qdrant `flush_interval_sec = 5`). `wait=true` requests force flush. |
| 1.6 | **Group commit / async WAL fsync** | `crates/turbomemory_storage/src/wal.rs` | Pending | Batch multiple in-flight inserts into one fsync. Separate committer thread. |
| 1.7 | **Smallest-appendable-segment selection** | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | If multiple Hot segments exist, write to the one with most free capacity. Qdrant: `lib/shard/src/segment_holder/mod.rs:313`. |
| 1.8 | **Batch payload / text / graph updates** | `crates/turbomemory_storage/src/update_worker.rs` | Pending | Apply metadata, payload index, text index, and graph edges in chunks, not per-point. Qdrant chunks payload ops at 32 (`lib/shard/src/update.rs:714`). |
| 1.9 | **Zero-copy Python ingest** | `crates/turbomemory_python/src/lib.rs` | Done | Done (2026-06-18). Borrows contiguous `float32` numpy arrays via `F32Input`/`F32Matrix`; copies only for non-contiguous / wrong-dtype / list inputs. Engine batch API now takes `&[&[f32]]`. |
| 1.10 | **Streaming bulk insert API** | `crates/turbomemory_python/src/lib.rs` | Pending | Accept iterator/reader or chunked callback API for datasets larger than RAM. |
| 1.11 | **Pipeline insert: validate → WAL append → segment append → ack** | `crates/turbomemory_storage/src/engine.rs` | Pending | Decouple durability ack from index visibility. Qdrant acks after WAL write; segment update is async. |

---

## Phase 2 — Segment Lifecycle & Optimizers

Qdrant uses four optimizers: **Indexing**, **Merge**, **Vacuum**, **ConfigMismatch**. We need all four, plus tier-aware versions.

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 2.1 | **Byte-threshold sealing: indexing_threshold_kb = 10,000 KB** | `crates/turbomemory_storage/src/config.rs`, `segment_holder.rs` | Pending | Seal Hot → SealedHot when vector storage reaches 10 MB. Convert to point count at runtime: `threshold_kb * 1024 / (dim * 4)`. Qdrant: `lib/shard/src/optimizers/config.rs:15`. |
| 2.2 | **Max segment size: 256,000 KB per indexing thread** | `crates/turbomemory_storage/src/config.rs` | Pending | Cap segment size to avoid giant unmanageable segments. `max_segment_size_kb = num_indexing_threads * 256_000`. Qdrant: `lib/collection/src/optimizers_builder.rs:184`. |
| 2.3 | **Target segment count: clamp(num_cpus / 2, 2, 8)** | `crates/turbomemory_storage/src/optimizer.rs`, `config.rs` | Pending | Merge optimizer tries to converge to this count. Qdrant: `lib/collection/src/optimizers_builder.rs:155`. |
| 2.4 | **IndexingOptimizer: build HNSW only when segment crosses threshold** | `crates/turbomemory_storage/src/optimizer.rs` | Pending | Background build from plain segment to SealedHot. Do not build HNSW for segments below `full_scan_threshold`. |
| 2.5 | **MergeOptimizer: merge small segments up to max size** | `crates/turbomemory_storage/src/optimizer.rs` | Pending | Greedily merge smallest segments. Require ≥3 segments in first batch or two batches of ≥2 to guarantee count reduction. Qdrant: `lib/shard/src/optimizers/merge_optimizer.rs`. |
| 2.6 | **VacuumOptimizer: reclaim deleted vectors at 20% / 1,000 vectors** | `crates/turbomemory_storage/src/optimizer.rs` | Pending | Trigger rebuild when deleted_ratio > 0.2 and deleted_count ≥ 1,000. Qdrant: `lib/shard/src/optimizers/config.rs:16-17`. |
| 2.7 | **ConfigMismatchOptimizer: rebuild when HNSW/quantization/tier config changes** | `crates/turbomemory_storage/src/optimizer.rs` | Pending | Detect desired vs actual segment config and rebuild. Qdrant: `lib/shard/src/optimizers/config_mismatch_optimizer.rs`. |
| 2.8 | **Build new segments in temp directory + atomic rename** | `crates/turbomemory_storage/src/optimizer.rs`, `segment_builder.rs` (new) | Pending | Qdrant `temp_segments/` → atomic rename into `segments/` (`lib/segment/src/segment_constructor/segment_builder.rs:761`). Current TSM builds in-place under lock; fix. |
| 2.9 | **Disk-space guard before optimization** | `crates/turbomemory_storage/src/optimizer.rs` | Pending | Require ≥2× source segment size free in temp path. Qdrant: `lib/shard/src/optimize.rs:601`. |
| 2.10 | **Resource budget: IO permit → CPU permit** | `crates/turbomemory_storage/src/optimizer.rs`, `resource_budget.rs` | Pending | Acquire IO permit for data copying, replace with CPU permit for HNSW indexing. Qdrant: `lib/shard/src/optimize.rs:330`. |
| 2.11 | **Multi-threaded HNSW construction** | `crates/turbomemory_storage/src/segments/usearch_index.rs` | Done | Done (2026-06-18). First 256 points single-threaded (`PARALLEL_SEED_POINTS`), remainder via bounded rayon pool; threads = `clamp(num_cpus / max_concurrent_builds, 1, 16)`. ~2× faster consolidation at 10k, no recall regression. |
| 2.12 | **Old-index reuse / heal on merge** | `crates/turbomemory_storage/src/segments/sealed_hot.rs` | Pending | When merging already-indexed segments, reuse/heal existing HNSW graph instead of rebuilding from scratch. Qdrant: `OldIndexCandidate` in `lib/segment/src/index/hnsw_index/hnsw/build.rs:150`. |
| 2.13 | **Cache hygiene: prefault before build, drop after build** | `crates/turbomemory_storage/src/optimizer.rs` | Pending | `populate()` vectors before HNSW build; `clear_cache()` after sealing to avoid page-cache pollution. Qdrant: `lib/segment/src/segment_constructor/segment_builder.rs:834`, `:675-733`. |
| 2.14 | **Deferred points / `prevent_unoptimized`** | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Hide new writes in over-capacity segments until optimization completes, preventing unbounded exact-scan segments. Qdrant pattern. |

---

## Phase 3 — Search Architecture

Search at 1M × 4k is dominated by HNSW traversal, multi-segment aggregation, and filtering. These are the highest-ROI CPU wins.

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 3.1 | **Visited-set pool** | `crates/turbomemory_storage/src/visited_pool.rs` | Done | Done (2026-06-19 audit). `visited_pool.rs` — `VisitedSet { tokens: Vec<u8>, generation }`, token wrap → refill, parking_lot-guarded pool. |
| 3.2 | **HNSW defaults aligned with Qdrant** | `crates/turbomemory_storage/src/config.rs` | Pending | `M = 16`, `M0 = 32`, `ef_construct = 100`, `ef_search = max(ef, top_k)`, `full_scan_threshold = 10_000` KB converted to vector count. |
| 3.3 | **Compressed / inline-vector graph links for sealed segments** | `crates/turbomemory_storage/src/segments/sealed_hot.rs` | Pending | Reduce random seeks by inlining quantized vectors or packing neighbor lists. Qdrant graph formats: `Plain`, `Compressed`, `CompressedWithVectors`. |
| 3.4 | **Heuristic neighbor selection (HNSW)** | `crates/turbomemory_storage/src/segments/sealed_hot.rs` | Pending | Use diverse neighbor heuristic during build, not just top-M by distance. |
| 3.5 | **Parallel search across segments with result aggregation** | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Spawn per-segment search on thread pool; merge with k-way top-k. Use `BatchResultAggregator`. Cancel slow/abandoned tasks on drop. Qdrant: `lib/collection/src/collection_manager/segments_searcher.rs:211`. |
| 3.6 | **Probabilistic sampling for multi-segment search** | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Poisson-derived per-segment limit; rerun if boundary crosses global top-k. Qdrant: `lib/collection/src/collection_manager/segments_searcher.rs:571`. |
| 3.7 | **Cardinality-aware filter routing** | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Estimate `{min, exp, max}` cardinality. `max < full_scan_threshold` → plain; `min > threshold` → HNSW; else sample ≤1,000 points with Agresti-Coull confidence interval. Qdrant: `lib/segment/src/index/hnsw_index/hnsw/vector_index_impl.rs:55-173`. |
| 3.8 | **ACORN-1 adaptive filtered search** | `crates/turbomemory_storage/src/segments/sealed_hot.rs` | Pending | When selectivity ≤ 0.4, expand 1-hop and conditional 2-hop neighbors during HNSW traversal. Use two pooled visited lists. Qdrant: `lib/segment/src/index/hnsw_index/hnsw/search.rs:36-85`. |
| 3.9 | **Plain search with SIMD + early-exit top-k** | `crates/turbomemory_storage/src/segments/hot.rs` | Pending | Hot segment exact scan must be competitive. Use batched SIMD distance + min-heap top-k. |
| 3.10 | **Per-query latency budget / adaptive `ef`** | `crates/turbomemory_storage/src/segment_holder.rs` | In Progress | Caller-specified `ef` done (2026-06-18) — `search_ann(q, top_k, ef)` + benchmark `--ef`. Auto-raise on low recall audit still pending. |
| 3.11 | **Exact reranking after quantization** | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Rescore top-k approximate results with full f32 vectors. Default on for binary/TurboQuant, off for scalar/PQ. Qdrant: `lib/segment/src/index/vector_index_search_common.rs:48-87`. |
| 3.12 | **Abort-on-drop for long searches** | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Wrap blocking search tasks so dropping the request cancels the thread work. Qdrant: `AbortOnDropHandle`. |

---

## Phase 4 — Quantization & Distance Compute (CPU)

At 4k dimensions, distance compute is the bottleneck. We need SIMD, batched kernels, and richer quantizers.

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 4.1 | **Batched SIMD distance kernel (matrix × query)** | `crates/turbomemory_core/src/metrics.rs` | Done | Done (2026-06-19 audit). `cosine_similarity_batch` with `dot_and_nb_x4` 4-vector unrolled kernel, AVX2/FMA + SSE + NEON paths. |
| 4.2 | **Pre-normalize cosine vectors** | `crates/turbomemory_core/src/metrics.rs` | Done | Already implemented. Vectors are L2-normalized on insert/update (`engine.rs` `normalize`); usearch index uses `MetricKind::Cos`; rerank uses self-normalizing `cosine_similarity_batch`. |
| 4.3 | **Scalar int8 quantizer** | `crates/turbomemory_core/src/quantization/` (new) | Pending | `alpha=(max-min)/127`, offset=min, per-metric multiplier, vector offset prefix. SIMD i8 dot/L1. Qdrant: `lib/quantization/src/encoded_vectors_u8.rs`. |
| 4.4 | **Product Quantization (PQ)** | `crates/turbomemory_core/src/quantization/` (new) | Pending | 256 centroids/subspace, kmeans sample 10k, max 100 iter, tol 1e-5. Query builds LUT; SIMD LUT gather. Qdrant: `lib/quantization/src/encoded_vectors_pq.rs`. |
| 4.5 | **Binary / 1-bit + 1.5-bit / 2-bit quantizers** | `crates/turbomemory_core/src/quantization/` (new) | Pending | XOR-popcount via SSE4.2/AVX-512/NEON; optional scalar query encoding. Default rescoring=true. Qdrant: `lib/quantization/src/encoded_vectors_binary.rs`. |
| 4.6 | **TurboQuant-style 1/2/4-bit quantizer** | `crates/turbomemory_core/src/turbo_quant.rs` | Done | Done. Both MSE and Prod variants fully implemented in `turbo_quant.rs` (1150 LOC, 12 tests): FWHT rotation, Lloyd-Max codebooks (bits 1–4 true optimum; 5–8 uniform approximation — a known gap), QJL residual 1-bit transform, LUT scoring with AVX2 gather + byte-weight fast paths. Config panic on non-pow2 dim fixed (2026-06-19). |
| 4.7 | **OPQ / learned rotation before PQ** | `crates/turbomemory_core/src/quantization/` (new) | Pending | Reduce PQ distortion for high-dimensional embeddings. |
| 4.8 | **Quantization auto-selection by dimension and recall target** | `crates/turbomemory_storage/src/config.rs` | In Progress | Cold tier no longer defaults to 1-bit sign at high dim — now 8-bit scalar by default (2026-06-18). Full auto-selection (scalar/PQ/TurboQuant by dim + recall target) still pending. |
| 4.9 | **Zero-copy quantized scans** | `crates/turbomemory_storage/src/segments/warm.rs`, `cold.rs` | Pending | Read mmap slices directly; remove `chunk_bytes.extend_from_slice` copies. |
| 4.10 | **Query LUT precomputation for quantized tiers** | `crates/turbomemory_storage/src/segments/warm.rs`, `cold.rs` | Pending | Build lookup table once per query, reuse across chunks. |
| 4.11 | **Distance compute thread pool with work stealing** | `crates/turbomemory_storage/src/runtime.rs` | Pending | Use `rayon` or custom pool for batch distance jobs; separate from search threads. |

---

## Phase 5 — Memory, Storage & I/O

1M × 4k = ~16 GB of vectors alone. We must be mmap-first, shard-first, and I/O-aware.

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 5.1 | **Shard `VectorStore` by point-offset range** | `crates/turbomemory_storage/src/vector_store.rs` | Pending | Split vectors into multiple `vectors-*.bin` files (e.g., per shard or per 256k offsets). Avoid 16 GB+ single file remap. |
| 5.2 | **Recycle deleted vector slots** | `crates/turbomemory_storage/src/vector_store.rs` | Pending | Maintain a free-list of deleted offsets; new inserts reuse slots instead of always appending. |
| 5.3 | **Async / prefetch mmap I/O** | `crates/turbomemory_storage/src/vector_store.rs`, `segments/warm.rs`, `cold.rs` | Pending | `madvise` / `PrefetchVirtualMemory` (Windows) for cold-start and sequential build reads. Add `MmapOptions::populate()` option. |
| 5.4 | **Separate hot and cold mmap policies** | `crates/turbomemory_storage/src/vector_store.rs` | Pending | Hot vectors locked/prefaulted; Warm/Cold left for OS cache. |
| 5.5 | **Paged metadata store with cache eviction** | `crates/turbomemory_storage/src/metadata_store.rs` | Pending | Replace single `HashMap` with LRU cache + mmap-backed pages. Critical for 1M+ text records. |
| 5.6 | **Per-segment metadata files** | `crates/turbomemory_storage/src/segments/` | Pending | Each segment owns its id→offset map and payload indexes, not a global redb. Qdrant: `segment.json` + mmap id_tracker. |
| 5.7 | **Replace redb with per-segment metadata + WAL** | `crates/turbomemory_storage/src/metadata_store.rs` | Pending | redb becomes a bottleneck at high metadata throughput. Use append-only segment manifests + WAL replay. |
| 5.8 | **Offset-mapped payload storage (mmap)** | `crates/turbomemory_storage/src/payload_storage.rs` (new) | Pending | Store payloads in per-segment mmap files keyed by local offset, like Qdrant `MmapPayloadStorage`. |
| 5.9 | **Bitmap payload indexes with mmap backing** | `crates/turbomemory_storage/src/payload_index.rs` | Pending | Keyword/int/range indexes as Roaring bitmaps persisted to mmap; not fully in-memory. |
| 5.10 | **Text index segmentation and batch commits** | `crates/turbomemory_storage/src/text_index.rs` | Pending | Separate text index per segment; periodic commit; avoid `TopDocs::with_limit(num_docs)` full materialization. |
| 5.11 | **Write-ahead snapshot / checkpointing** | `crates/turbomemory_storage/src/engine.rs` | Pending | Take periodic checkpoints (segments + manifest) without blocking foreground; truncate WAL after checkpoint. |
| 5.12 | **mmap growth strategy: pre-allocate in power-of-two chunks** | `crates/turbomemory_storage/src/vector_store.rs` | Pending | Avoid full-file remap on every growth; double capacity and zero-fill lazily. |
| 5.13 | **DiskANN / SPANN-style on-disk ANN index option** | `crates/turbomemory_storage/src/index/diskann.rs` (new) | Pending | For >RAM collections, provide disk-based approximate index as alternative to in-RAM HNSW. |

---

## Phase 6 — Concurrency & Scheduling

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 6.1 | **Per-shard RwLock-free segment list** | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | See 0.2. Publish `Arc<Vec<Arc<Segment>>>` on mutation; readers snapshot cheaply. |
| 6.2 | **Lock-free access counters** | `crates/turbomemory_storage/src/access_counters.rs` | Pending | Current `Mutex<AHashMap>` contends. Use sharded atomic counters or thread-local buffers with periodic drain. |
| 6.3 | **parking_lot everywhere** | Whole workspace | Pending | Replace `std::sync::{Mutex,RwLock}` with `parking_lot` variants for lower overhead. |
| 6.4 | **Per-point link locks during mutable HNSW build** | `crates/turbomemory_storage/src/segments/sealed_hot.rs` | Pending | If we ever support incremental HNSW, use `Vec<RwLock<LinksContainer>>` per point (Qdrant pattern). For now, avoid incremental build. |
| 6.5 | **Resource isolation: search vs ingest vs optimize** | `crates/turbomemory_storage/src/resource_budget.rs` | Pending | CPU/IO/memory permits per operation class. Background optimizer throttled when foreground CPU > threshold. |
| 6.6 | **Adaptive search thread pool** | `crates/turbomemory_storage/src/runtime.rs` | Pending | Switch between HighIo and HighCpu pools based on process CPU ratio (Qdrant `AdaptiveSearchHandle`). |
| 6.7 | **Cancel slow/abandoned queries** | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | See 3.12. |
| 6.8 | **Concurrent graph search / spreading activation** | `crates/turbomemory_graph/src/` | Pending | Remove single `RwLock<SpreadingActivation>`; shard graph or use RCU. |
| 6.9 | **Batch cognitive search rehydration** | `crates/turbomemory_storage/src/engine.rs` | Pending | Load result payloads in batches, not one-by-one. |

---

## Phase 7 — GPU / CUDA / cuBLAS / Vulkan Acceleration

GPU is **not a magic bullet**. Qdrant uses GPU only for HNSW **build**, not search, via Vulkan compute. Distance compute can be GPU-accelerated for batch queries. We should support both CUDA (NVIDIA) and Vulkan (portable) paths behind feature flags.

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 7.1 | **Add `turbomemory_gpu` crate with backend trait** | `crates/turbomemory_gpu/` (new) | Pending | `GpuBackend` trait: `init()`, `upload_vectors()`, `upload_query_batch()`, `batch_dot()`, `build_hnsw_approx()`, `shutdown()`. Implementations: `CudaBackend`, `VulkanBackend`. |
| 7.2 | **cuBLAS batched distance compute** | `crates/turbomemory_gpu/src/cuda/dot.rs` | Pending | For batch queries (e.g., 64 queries × 1M vectors × 4k), cuBLAS `S GEMM` or custom CUDA kernel is much faster than CPU. Use for exact/rerank batches. |
| 7.3 | **CUDA kernel for top-k reduction** | `crates/turbomemory_gpu/src/cuda/topk.rs` | Pending | Fuse distance + top-k on GPU to avoid host↔device round trips. NVIDIA cuVS / RAFT provide `cuvs::neighbors::cagra` and `raft::spatial::knn`. |
| 7.4 | **CUDA HNSW build path** | `crates/turbomemory_gpu/src/cuda/hnsw_build.rs` | Pending | Parallel level-by-level HNSW insertion on GPU. Fallback to CPU on OOM/error. Only for sealed segments, not incremental. |
| 7.5 | **Vulkan compute HNSW build path (Qdrant model)** | `crates/turbomemory_gpu/src/vulkan/` | Pending | Use `ash` + compute shaders. Default 512 parallel insertion groups. Cross-platform, no NVIDIA dependency. Qdrant: `lib/segment/src/index/hnsw_index/gpu/`. |
| 7.6 | **GPU device manager** | `crates/turbomemory_gpu/src/device_manager.rs` | Pending | Global manager with per-optimization device lock. Select discrete > integrated; allow explicit device filter. Qdrant: `GPU_DEVICES_MANAGER` in `lib/segment/src/index/hnsw_index/gpu/mod.rs:23`. |
| 7.7 | **Feature flags: `cuda`, `vulkan`, `gpu`** | `Cargo.toml`, `crates/turbomemory_gpu/Cargo.toml` | Pending | `gpu` enables whichever backend is available; `cuda`/`vulkan` force specific path. |
| 7.8 | **cuVS / RAFT CAGRA index integration** | `crates/turbomemory_gpu/src/cuda/cagra.rs` | Pending | NVIDIA CAGRA is currently fastest GPU ANN for batch search. Use as alternative HNSW backend for sealed segments. |
| 7.9 | **GPU quantization scoring kernels** | `crates/turbomemory_gpu/src/cuda/quantized.rs` | Pending | PQ LUT lookup, binary popcount, scalar i8 dot on GPU for Warm/Cold tiers when batch is large enough to amortize upload. |
| 7.10 | **Host↔device memory pool** | `crates/turbomemory_gpu/src/memory_pool.rs` | Pending | Avoid `cudaMalloc` per query; keep pinned host buffers and device arenas. |
| 7.11 | **CUDA/C++ build.rs with `cc` crate** | `crates/turbomemory_gpu/build.rs` | Pending | Compile `.cu` files with `nvcc`; fallback to prebuilt stubs if CUDA unavailable. |
| 7.12 | **Benchmark GPU vs CPU per operation** | `benchmark.py`, `benches/` | Pending | Auto-select backend based on batch size, dimension, and measured latency. |

### GPU Strategy Notes

- **Search**: keep CPU HNSW as default. GPU search wins only for large batch queries; single-query latency is usually worse due to upload overhead.
- **Build**: GPU HNSW/CAGRA build is a clear win for 1M × 4k segments.
- **Distance**: cuBLAS batch GEMM for exact reranking is a clear win for batch search.
- **Fallback**: every GPU path must silently fall back to CPU on error (OOM, driver issue, no device).

---

## Phase 8 — Python Bindings & API

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 8.1 | **Zero-copy numpy ingest** | `crates/turbomemory_python/src/lib.rs` | Done | Done (2026-06-18). See 1.9. |
| 8.2 | **Streaming / chunked bulk insert** | `crates/turbomemory_python/src/lib.rs` | Pending | See 1.10. |
| 8.3 | **Async Python API (`asyncio`)** | `crates/turbomemory_python/src/lib.rs` | Pending | Return awaitable futures for insert/search; release GIL. |
| 8.4 | **Collection / shard config in Python** | `crates/turbomemory_python/src/lib.rs` | Pending | Expose `num_shards`, `indexing_threshold_kb`, `max_segment_size_kb`, quantization config. |
| 8.5 | **Batch search API** | `crates/turbomemory_python/src/lib.rs` | Pending | Accept matrix of queries; return list of result lists. Enables GPU batching. |
| 8.6 | **Progress callbacks for build/optimize** | `crates/turbomemory_python/src/lib.rs` | Pending | Long seals/merges report progress. |
| 8.7 | **Recall audit auto-tune helper** | `audit_recall.py` | Pending | Python helper to sample, measure recall, and raise `ef` until target is met. |

---

## Phase 9 — Observability & Operations

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 9.1 | **Replace `println!` with `tracing` spans** | Whole workspace | Pending | Structured, level-filtered logging. |
| 9.2 | **Metrics: ingest latency, search latency, segment sizes, tier counts** | `crates/turbomemory_storage/src/metrics.rs` (new) | Pending | Use `metrics` crate with Prometheus exporter in API server. |
| 9.3 | **WAL lag / optimizer queue depth metrics** | `crates/turbomemory_storage/src/update_worker.rs`, `optimizer.rs` | Pending | Operational visibility. |
| 9.4 | **Health metrics endpoint** | `crates/turbomemory_api/src/rest.rs` | Pending | Prometheus `/metrics`. |
| 9.5 | **Request timeouts, payload size limits, CORS, auth** | `crates/turbomemory_api/src/main.rs`, `rest.rs` | Pending | Production hardening. |
| 9.6 | **Docker / container build** | `deployments/docker/` (new) | Pending | Multi-stage build with optional CUDA base image. |
| 9.7 | **Cross-platform build support** | `Cargo.toml`, `.cargo/config.toml` | Pending | Windows MSVC, Linux, macOS; target-cpu=native optional. |
| 9.8 | **Release profile with LTO + codegen-units=1** | `Cargo.toml` | Done | Already configured. |

---

## Phase 10 — Testing & Benchmarking at Million-Scale

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 10.1 | **Add 1M × 4k synthetic benchmark** | `benchmark.py`, `benches/` | Pending | Measure ingest time, search P50/P99, recall@10, peak RSS, disk usage. |
| 10.2 | **Add concurrent ingest + search benchmark** | `benchmark.py` | Pending | Catch lock contention and tail latency. |
| 10.3 | **Add crash-recovery tests for sharded WAL** | `crates/turbomemory_storage/tests/` | Pending | Kill mid-seal, verify no data loss. |
| 10.4 | **Add property-based tests for segment lifecycle** | `crates/turbomemory_storage/tests/` | Pending | Seal/merge/vacuum correctness. |
| 10.5 | **Add comparison harness vs Qdrant/Chroma/Faiss** | `benchmark.py` | In Progress | TSM vs NumPy/Chroma/Qdrant harness with clustered + random data and configurable `ef`/dim/N (2026-06-18). Faiss and fixed recall-target alignment still pending. |
| 10.6 | **Continuous benchmark tracking** | CI / `benches/` | Pending | Detect regressions on PRs. |
| 10.7 | **GPU correctness tests** | `crates/turbomemory_gpu/tests/` | Pending | Compare GPU exact top-k vs CPU brute force bit-exact. |

---

## Qdrant Default Cheat Sheet (Reference)

| Parameter | Qdrant Default | TSM Target |
|---|---|---|
| HNSW `m` | 16 | 16 |
| HNSW `m0` | 32 | 32 |
| `ef_construct` | 100 | 100 |
| `ef` search | `max(ef, top_k)` | `max(ef, top_k)` |
| `full_scan_threshold` | 10,000 KB → vector count | 10,000 KB → vector count |
| `indexing_threshold_kb` | 10,000 | 10,000 |
| `max_segment_size_kb` | 256,000 × indexing_threads | 256,000 × indexing_threads |
| Target segment count | `clamp(num_cpus/2, 2, 8)` | `clamp(num_cpus/2, 2, 8)` |
| Deleted threshold | 0.2 | 0.2 |
| Vacuum min vectors | 1,000 | 1,000 |
| HNSW build threads | `clamp(num_cpus, 1, 16)` | `clamp(num_cpus, 1, 16)` |
| Single-threaded HNSW prefix | 256 points | 256 points |
| Upsert chunk size | 32 | 32 |
| Deletion batch size | 512 | 512 |
| WAL segment size | 32 MiB | 32 MiB |
| Flush interval | 5 s | 5 s |
| Visited pool size | `clamp(num_cpus, 16, 128)` | `clamp(num_cpus, 16, 128)` |
| GPU groups (Vulkan) | 512 | 512 |

---

## Suggested Execution Order

### Stage 1 — Unlock 1M × 4k structurally (P0)
1. **0.1–0.3** — Shard collection, collection abstraction, lock-free segment list.
2. **0.4–0.5** — Paged metadata store, per-shard update worker.
3. **1.1–1.3** — Plain Hot segment, 32-point chunks, 32 MiB WAL segments.
4. **2.1–2.2, 2.4–2.6** — Byte-threshold sealing, max segment size, Indexing/Merge/Vacuum optimizers.
5. **5.1–5.3** — Sharded VectorStore, slot recycling, prefetch.

### Stage 2 — Make search fast at high dimension
6. **3.1, 3.5, 3.7** — Visited pool, parallel multi-segment search, cardinality-aware filtering.
7. **4.1–4.3** — Batched SIMD distance, scalar int8 quantizer, cosine pre-normalization.
8. **3.2, 3.11** — Qdrant-aligned HNSW defaults, exact reranking.

### Stage 3 — GPU acceleration
9. **7.1–7.4, 7.6** — `turbomemory_gpu` crate, cuBLAS batch distance, CUDA HNSW build, device manager.
10. **7.5, 7.8** — Vulkan compute build path, CAGRA integration.

### Stage 4 — Operations & polish
11. **9.1–9.7** — Tracing, metrics, Docker, cross-platform builds.
12. **10.1–10.7** — Million-scale benchmarks, comparison harness, continuous tracking.

---

## Legacy Items (from previous TODO.md)

All previously tracked items are subsumed above. The following remain relevant but are now lower priority until Stage 1 is complete:

- ~~Cognitive graph durable persistence (graph shard per collection shard).~~ **Done (2026-06-19).** Graph JSON is now loaded on open and merged with new records, preserving learned edge weights, reinforcement timestamps, and abstraction nodes across restarts. Per-shard graph persistence still pending sharding (0.1).
- **Graph merge/forget policies** — partially done (2026-06-19): `reinforce` (retain on retrieval), `decay_edges` (forget stale reinforced edges), `build_abstractions` (generalize from co-occurrence), `deduplicate` (merge near-duplicates, transfer edges). Still pending: memory *evolution* (revise/contradict existing memories when new info arrives — the SAGE/A-Mem pattern), and importance-weighted edge strengthening on access (currently reinforcement is uniform per retrieval, not scaled by the retriever's confidence).
- Sparse vectors.
- Multi-agent scoping / distributed shards.
- Advanced full-text index tuning beyond per-segment Tantivy.

---

## Final Note

Scaling to **1M+ nodes × 4k dimensions** is a multi-month project, not a few quick fixes. The order matters:

1. **Sharding and lock-free segment list** remove the single-node ceiling.
2. **Plain Hot segments + background indexing** remove per-insert HNSW cost.
3. **Multi-threaded HNSW build + byte-threshold sealing** make large segments feasible.
4. **SIMD batched distance + scalar/PQ quantization** make high-dimension search fast.
5. **GPU/cuBLAS/CAGRA** provide the final throughput multiplier for batch workloads.

Do not start with GPU. A GPU-accelerated bad architecture is still a bad architecture.
