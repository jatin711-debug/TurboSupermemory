# Qdrant Architecture Deep Dive

> Codebase: `d:\personal-projects\TurboSuperMemory\qdrant` (version 1.18.2 from `Cargo.toml`).
> This document focuses on the design choices that make Qdrant's ingestion and search fast, and how they compare to TurboSuperMemory (TSM).

## 1. High-level crate layout

Qdrant is a workspace split into a thin binary/API crate and a set of library crates:

| Crate | Role | Key paths |
|-------|------|-----------|
| `src/` | REST (Actix) + gRPC (Tonic) entry points, auth, telemetry | `src/actix/api/`, `src/tonic/api/` |
| `lib/api` | REST schema + gRPC protobuf types | `lib/api/src/rest/`, `lib/api/src/grpc/` |
| `lib/storage` | Cluster-level dispatch, TableOfContents, consensus | `lib/storage/src/content_manager/` |
| `lib/collection` | Per-collection orchestration, update handler, optimizers | `lib/collection/src/` |
| `lib/shard` | Segment holder, local shard ops, optimizer planning, WAL wrapper | `lib/shard/src/` |
| `lib/segment` | The actual segment: vectors, payload, HNSW, payload indexes | `lib/segment/src/` |
| `lib/wal` | Append-only WAL segment implementation | `lib/wal/src/` |
| `lib/quantization` | Scalar / Product / Binary / Turbo quantization | `lib/quantization/src/` |
| `lib/common` | Shared utils (mmap, counters, budget, flags) | `lib/common/common/src/` |
| `lib/gridstore` | In-memory appendable payload index backend | `lib/gridstore/src/` |

Storage-related entry points:

- `lib/segment/src/lib.rs` exports `segment`, `index`, `vector_storage`, `payload_storage`, `id_tracker`, `entry`.
- `lib/collection/src/lib.rs` exports `collection`, `collection_manager`, `shards`, `update_handler`, `update_workers`.
- `lib/shard/src/lib.rs` exports `segment_holder`, `optimizers`, `update`, `wal`, `operations`.

---

## 2. Ingestion path: API → WAL → segment append

### 2.1 API entry

- **REST upsert:** `src/actix/api/update_api.rs:33` (`upsert_points`) → `do_upsert_points`.
- **gRPC upsert:** `src/tonic/api/points_api.rs:64` (`upsert`) → `src/tonic/api/update_common.rs:40`.
- Common helpers: `src/common/update.rs`:
  - `do_upsert_points` at **line 313**
  - `update(...)` at **line 1221**

### 2.2 Routing to the shard

`update()` calls `TableOfContent::update` in `lib/storage/src/content_manager/toc/point_ops.rs:489`, which resolves the shard selector and calls:

- `Collection::update_from_client` — `lib/collection/src/collection/point_ops.rs:144`
- `Collection::update_from_peer` — same file, **line 94**

`update_from_client` splits the operation per shard and dispatches each shard update in a `FuturesUnordered`.

### 2.3 Replica-set / local shard

Inside a shard replica set:

- `ShardReplicaSet::update_with_consistency` — `lib/collection/src/shards/replica_set/update.rs:145`
- `ShardReplicaSet::update` — **line 246**
- `ShardReplicaSet::update_impl` — **line 317**

For a local replica:

- `LocalShard::submit_update` — `lib/collection/src/shards/local_shard/shard_ops.rs:60`

### 2.4 WAL write

`submit_update` first durably logs the operation:

```rust
let (operation_id, _wal_lock) = match self.wal.lock_and_write(&mut operation).await { ... };
```

- `RecoverableWal::lock_and_write` — `lib/collection/src/wal_delta.rs:50`
- It serializes with `serde_cbor`, appends to the underlying `wal` crate, and returns the op number + a WAL lock guard.
- The lock guard is held while the operation is sent into the update worker channel, guaranteeing ordering.
- Then the operation is sent as an `UpdateSignal::Operation` into the single update worker channel (`shard_ops.rs:112`).

### 2.5 Update worker applies to segments

- `UpdateWorkers::update_worker_fn` — `lib/collection/src/update_workers/update_worker.rs:44`
- For each signal it spawns blocking work in `update_worker_internal` — **line 305**
- `update_worker_internal` calls `CollectionUpdater::update` — `lib/collection/src/collection_manager/collection_updater.rs:41`

`CollectionUpdater::update` serializes the operation across segments by acquiring:

1. `update_operation_lock.blocking_write()` (a `tokio::sync::RwLock`)
2. `segments.acquire_updates_lock()` (a `parking_lot::Mutex` in `LockedSegmentHolder`)

Then it dispatches to operation-type processors in `lib/shard/src/update.rs`:

- `process_point_operation` — **line 29**
- `upsert_points` — **line 181**
- `set_payload` / `delete_payload` / etc.

### 2.6 Segment-level upsert

- `Segment::upsert_point` — `lib/segment/src/segment/entry.rs:658`
- It either replaces an existing internal point or calls `insert_new_vectors` for a new point, then sets payload.

---

## 3. Segment model — the key to Qdrant's speed

### 3.1 What is a segment

A `Segment` is an independent, self-contained group of points (vectors + payload + indexes) with its own versioning and persistence.

- `Segment` struct — `lib/segment/src/segment/mod.rs:61`
- `VectorData` struct — same file, **line 99** (`vector_index`, `vector_storage`, `quantized_vectors`)
- Segment state is persisted in `segment.json` (`SEGMENT_STATE_FILE` at **line 37**)

A segment is classified by `segment_type` (`SegmentType::Plain` or `SegmentType::Indexed`) and by `appendable_flag`.

### 3.2 Appendable vs immutable

| Mode | Storage | Id tracker | Payload index | Vector index |
|------|---------|------------|---------------|--------------|
| **Appendable** | chunked/appendable mmap | mutable | mutable (Gridstore) | **plain / brute-force** |
| **Indexed / sealed** | single-file mmap | immutable compressed | mmap field indexes | **HNSW** |

**This is the most important ingestion-speed decision:** small, appendable segments use a **plain vector index** (exact brute-force), so inserts are just an append + O(N) exact scan at query time. There is **no per-insert HNSW graph mutation**. HNSW is built later, offline, by the optimizer.

### 3.3 Segment creation

- `build_segment(...)` — `lib/segment/src/segment_constructor/segment_constructor_base.rs:771`
- `create_segment(...)` — same file, **line 400**
- Creates payload storage, id tracker, vector storage(s), payload index, and vector indexes according to `SegmentConfig`.
- `SegmentHolder::create_appendable_segment` — `lib/shard/src/segment_holder/mod.rs:754`
- `LocalShard` ensures at least one appendable segment exists at load time — `lib/collection/src/shards/local_shard/mod.rs:494`

### 3.4 Segment size / sealing policy

Configured in `OptimizersConfig` (`lib/collection/src/optimizers_builder.rs:33`):

- `max_segment_size` (KB) — default computed per CPU: `DEFAULT_MAX_SEGMENT_PER_CPU_KB = 256_000` per indexing thread (`lib/shard/src/optimizers/config.rs:14`, helper at **line 185**).
- `indexing_threshold` (KB) — default `10_000` KB (`lib/shard/src/optimizers/config.rs:15`, helper at **line 175**).
- `memmap_threshold` (KB) — deprecated, maps to `on_disk` flags.

**Sealing** is done by rebuilding: the optimizer constructs a new segment in a temp directory, then atomically swaps it into the segment holder (see §10).

---

## 4. WAL / update log design

### 4.1 WAL crate

The low-level WAL lives in `lib/wal/src/`:

- `Wal` struct — `lib/wal/src/lib.rs:76`
- `append` — **line 292**
- `flush_open_segment` — **line 308**
- `prefix_truncate` — **line 408** (acknowledges old records)
- Segment capacity defaults to **32 MiB** (`WalOptions::default`, **line 37**)

Each WAL segment is an append-only mmap file with CRC32-C per entry:

- `Segment` struct — `lib/wal/src/segment.rs:99`
- `Segment::append` — **line 335**
- `Segment::flush` — **line 420**
- `Segment::flush_async` — **line 450**

### 4.2 Typed WAL wrapper

- `SerdeWal<R>` — `lib/shard/src/wal.rs:20`
  - serializes records with `serde_cbor`
  - `write` — **line 108**
  - `flush` — **line 260**
  - `ack` / `prefix_truncate` — **line 219**

- `RecoverableWal` — `lib/collection/src/wal_delta.rs:14`
  - Adds clock-tag handling for distributed consistency.
  - `lock_and_write` — **line 50**

### 4.3 Durability / commit frequency

- Every `wait=true` request explicitly flushes the WAL before applying (`update_worker_internal` at `update_worker.rs:317`).
- A background **flush worker** flushes the WAL and all segments periodically:
  - `flush_worker_fn` — `lib/collection/src/update_workers/flush_workers.rs:101`
  - interval is `flush_interval_sec` (default 60s) from `OptimizersConfig` (`optimizers_builder.rs:86`)
- After segment flush, the WAL is acknowledged up to the confirmed version:
  - `flush_worker_internal` — `flush_workers.rs:32`, `ack` at **line 95**

### 4.4 Recovery

On startup the local shard replays un-applied WAL entries:

- `LocalShard::load_from_wal` — `lib/collection/src/shards/local_shard/mod.rs:711`
- Reads from `first_index` up to the applied-seq upper bound and re-applies through `CollectionUpdater::update`.

### 4.5 Batching

Inside the update processors:

- Point upserts are chunked by `UPDATE_OP_CHUNK_SIZE = 32` — `lib/shard/src/update.rs:176`
- Point deletions are batched by `DELETION_BATCH_SIZE = 512` — same file, **line 341**

---

## 5. HNSW index integration

### 5.1 Configuration

- `HnswConfig` (collection-level) — `lib/segment/src/types.rs:661`
  - `m`, `ef_construct`, `full_scan_threshold`, `max_indexing_threads`, `on_disk`, `payload_m`
- `HnswGraphConfig` (persisted per index) — `lib/segment/src/index/hnsw_index/config.rs:12`
  - `m`, `m0 = m * 2`, `ef_construct`, `ef = ef_construct`, `full_scan_threshold`, `payload_m`, `payload_m0`

### 5.2 HNSW index struct

- `HNSWIndex` — `lib/segment/src/index/hnsw_index/hnsw.rs:39`
- `HNSWIndex::open` — **line 61**
- `HNSWIndex::build` — `lib/segment/src/index/hnsw_index/hnsw/build.rs:52`
- Search implementation — `lib/segment/src/index/hnsw_index/hnsw/vector_index_impl.rs:22`

### 5.3 Graph structure

- `GraphLayers` — `lib/segment/src/index/hnsw_index/graph_layers.rs`
- `GraphLayersBuilder` — `lib/segment/src/index/hnsw_index/graph_layers_builder.rs`
  - `new` — **line 326**
  - `link_new_point` — **line 414** (incremental insertion)
  - `get_random_layer` — **line 385** (geometric level distribution)

### 5.4 How incremental add works (during optimization / build)

1. First `SINGLE_THREADED_HNSW_BUILD_THRESHOLD` points (256 in release, 32 in debug) are inserted single-threaded to avoid disconnected components — `hnsw.rs:34`.
2. Remaining points are inserted in parallel via `rayon` (`hnsw/build.rs:353-355`).
3. Each insertion uses `link_new_point`, which:
   - picks the entry point
   - greedily searches down to the new point's level
   - links `ef_construct` nearest neighbors per level
   - optionally applies the heuristic (`HNSW_USE_HEURISTIC = true`)

### 5.5 Large segment builds

Large segments are built offline by the optimizer:

- `SegmentBuilder::update` copies points from source segments — `lib/segment/src/segment_constructor/segment_builder.rs:266`
- `SegmentBuilder::build` then materializes payload indexes and vector indexes — **line 506**
- HNSW is built via `build_vector_index` — `lib/segment/src/segment_constructor/segment_constructor_base.rs:289`

The build can reuse old HNSW graphs and supports GPU acceleration (behind the `gpu` feature).

---

## 6. Search path: HNSW, filtering, and aggregation

### 6.1 Search API entry points

| Layer | File | Key function / line |
|-------|------|---------------------|
| REST endpoints | `src/actix/api/search_api.rs` | `search_points` (l. 25), `batch_search_points` (l. 87) |
| gRPC endpoints | `src/tonic/api/points_api.rs` | `PointsService::search` (l. 382), `search_batch` (l. 404) |
| API → TOC dispatcher | `src/common/query.rs` | `do_core_search_points` (l. 21) → `do_core_search_batch_points` (l. 100) |
| Collection-level batch | `lib/collection/src/collection/search.rs` | `Collection::core_search_batch` (l. 51) |
| Shard replica set | `lib/collection/src/shards/replica_set/read_ops.rs` | `ShardReplicaSet::core_search` (l. 97) |
| Local shard | `lib/collection/src/shards/local_shard/search.rs` | `LocalShard::do_search` (l. 30) |
| Segment searcher | `lib/collection/src/collection_manager/segments_searcher.rs` | `SegmentsSearcher::search` (l. 211) |
| Per-segment dispatch | `lib/segment/src/segment/read_view/search.rs` | `SegmentReadView::search_batch` (l. 212) |

### 6.2 HNSW search implementation

`HNSWIndex::search_with_graph` resolves `ef` as (`hnsw/search.rs:34-36`):

```rust
let ef = params
    .and_then(|params| params.hnsw_ef)
    .unwrap_or(self.config.ef);
```

Then `GraphLayers::search` enforces `ef = max(ef, top)` (`graph_layers.rs:552`).

The candidate pool is implemented by `SearchContext` (`search_context.rs`):

- `nearest: FixedLengthPriorityQueue<ScoredPointOffset>` sized to `ef` (l. 18)
- `candidates: BinaryHeap<ScoredPointOffset>` (l. 19)
- `lower_bound()` returns the worst score currently in `nearest`; traversal stops when the best candidate is worse than this bound (l. 23–28)

Graph traversal (`GraphLayersBase::search_on_level`, `graph_layers.rs:109-149`):

1. Start from an entry point, greedy-descend to level 0 (`search_entry`, beam size 1).
2. On level 0, run beam search with `ef`:
   - Pop best candidate from `candidates`.
   - Iterate its links (`for_each_link`), score unvisited neighbors with `FilteredScorer::score_points`.
   - Add improved neighbors to `nearest` and `candidates`.
   - Stop when the candidate is worse than `search_context.lower_bound()`.
3. Return `nearest.into_iter_sorted().take(top)`.

### 6.3 Filtering integration

For a filtered query, `HNSWIndex::search` estimates cardinality (`vector_index_impl.rs:83-173`):

1. `payload_index.estimate_cardinality(filter)` → `query_cardinality`.
2. If `query_cardinality.max < full_scan_threshold` → use **plain** search.
3. If `query_cardinality.min > full_scan_threshold` → use **HNSW** graph search.
4. In the ambiguous middle, sample check (`sample_check_cardinality`) decides.

Plain path (`search_vectors_plain`, `hnsw/search.rs:293-325`) uses the payload index to enumerate filtered point IDs and scores them exhaustively.

Graph path passes a `filter_context` into `FilteredScorer`, so the HNSW traversal checks the filter lazily as it scores neighbors.

**ACORN adaptive filtered search** (`hnsw/search.rs:36-85`) optionally enables ACORN when `params.acorn.enable == true`, `m0 != 0`, and estimated filter selectivity ≤ `acorn_max_selectivity` (default 0.4, `types.rs:556`). ACORN traversal expands 2-hop neighbors when a direct neighbor fails the filter.

### 6.4 Multi-segment search

`SegmentsSearcher::search` (`lib/collection/src/collection_manager/segments_searcher.rs:211`):

1. Acquires a brief read lock on `LockedSegmentHolder`, collects segments via `non_appendable_then_appendable_segments()` (l. 233).
2. Spawns one blocking task per segment via `runtime_handle.spawn_blocking` (l. 254).
3. Each task calls `search_in_segment` (l. 615), which batches queries with identical params and calls `SegmentReadView::search_batch`.
4. Results are collected and processed by `process_search_result_step1` (l. 87):
   - Aggregates with `BatchResultAggregator`.
   - Uses probabilistic sampling (`sampling_limit`, l. 571) to avoid over-fetching from many segments.
   - Detects under-sampled segments and re-runs them without sampling if needed (l. 304-372).

Cross-shard merging in `Collection::merge_from_shards` (`lib/collection/src/collection/search.rs:273-325`) does k-way merge, deduplication by point ID, and version tracking.

### 6.5 Exact reranking / rescore

`postprocess_search_result` (`lib/segment/src/index/vector_index_search_common.rs:48-87`) runs after approximate/quantized search. When quantization is enabled and `rescore` is true (or default for the quantizer), the top-k approximate results are re-scored using the **original full-precision vectors**.

---

## 7. Payload indexing

### 7.1 Core payload index

- `StructPayloadIndex` — `lib/segment/src/index/struct_payload_index/mod.rs:42`
- It owns:
  - `payload: PayloadStorageEnum`
  - `id_tracker`
  - `field_indexes: IndexesMap`
- Build / apply API:
  - `build_index` — `lib/segment/src/index/struct_payload_index/payload_index.rs:21`
  - `overwrite_payload` — **line 140**
  - `set_payload` — **line 165**

### 7.2 Field index types

`FieldIndex` enum — `lib/segment/src/index/field_index/field_index_base/field_index.rs:28`:

- `KeywordIndex`, `IntMapIndex`, `UuidMapIndex` — map / inverted-style indexes
- `IntIndex`, `FloatIndex`, `DatetimeIndex`, `UuidIndex` — numeric BTree-style + histogram
- `GeoIndex`
- `FullTextIndex`
- `BoolIndex`
- `NullIndex`

### 7.3 Storage backends

The `IndexSelector` (`lib/segment/src/index/field_index/index_selector.rs:32`) chooses between:

- **Mmap** (`IndexSelectorMmap`) — non-appendable, on-disk or in-RAM-mmap.
- **Gridstore** (`IndexSelectorGridstore`) — appendable in-memory index used while a segment is appendable.

Concrete index implementations live in `lib/segment/src/index/field_index/`.

### 7.4 Payload storage

- `PayloadStorageType` enum — `lib/segment/src/types.rs:1434`
  - `Mmap`, `InRamMmap`
- `MmapPayloadStorage` — `lib/segment/src/payload_storage/mmap_payload_storage.rs`

---

## 8. Quantization

Quantization is applied during segment optimization/build.

- `QuantizationConfig` enum — `lib/segment/src/types.rs:894`
  - `Scalar`, `Product`, `Binary`, `Turbo`
- `ScalarQuantizationConfig` — **line 768**
- `ProductQuantizationConfig` — **line 800**
- `BinaryQuantizationConfig` — **line 844**

Quantized data is managed by `QuantizedVectors` (`lib/segment/src/vector_storage/quantized/quantized_vectors.rs`).

The `quantization` crate implements the algorithms:

- `encoded_vectors_u8.rs` — scalar int8
- `encoded_vectors_pq.rs` — product quantization
- `encoded_vectors_binary.rs` — binary quantization
- `turboquant/` — TurboQuant / 4-bit path

---

## 9. Concurrency model

### 9.1 Per-shard update serialization

Each local shard has:

- A **single update worker** (`update_worker_fn`) consuming a bounded `tokio::mpsc` channel.
- `update_operation_lock` — a `tokio::sync::RwLock<()>` held for write during every update.
- `LockedSegmentHolder::updates_mutex` — a `parking_lot::Mutex<()>` acquired before segment-holder writes.

**Only one update operation mutates segments at a time** per shard, but reads can proceed against `RwLockReadGuard`s.

### 9.2 Segment-holder locking

- `LockedSegmentHolder` — `lib/shard/src/segment_holder/locked.rs:19`
  - wraps `Arc<RwLock<SegmentHolder>>`
  - plus `updates_mutex`
- `SegmentHolder` — `lib/shard/src/segment_holder/mod.rs:45`
  - `appendable_segments: BTreeMap<SegmentId, LockedSegment>`
  - `non_appendable_segments: BTreeMap<SegmentId, LockedSegment>`

### 9.3 Per-segment locking

- `LockedSegment` enum — `lib/shard/src/locked_segment.rs:18`
  - `Original(Arc<RwLock<Segment>>)`
  - `Proxy(Arc<RwLock<ProxySegment>>)`
- Search/update code takes read or write guards on individual segments.

### 9.4 Optimizer concurrency

- Optimizations run in separate stoppable tasks (`spawn_stoppable`).
- A `ResourceBudget` limits CPU/IO permits across all optimizations per shard.
- Optimizers acquire permits via `optimizer_resource_budget.try_acquire(...)` — `lib/collection/src/update_workers/optimization_worker.rs:148`.

---

## 10. Memory vs mmap

### 10.1 Vector storage types

`VectorStorageType` — `lib/segment/src/types.rs:1585`:

| Variant | Use |
|---------|-----|
| `Memory` | Deprecated |
| `Mmap` | Single-file mmap, non-appendable |
| `InRamMmap` | Single-file mmap, pre-populated into RAM |
| `ChunkedMmap` | Appendable chunked mmap |
| `InRamChunkedMmap` | Appendable chunked mmap locked in RAM |
| `Empty` | Placeholder |

Default for new appendable segments is `InRamChunkedMmap`.

### 10.2 HNSW index storage

HNSW graph can be mmaped or loaded into RAM:

- `LoadOption::on_disk_mmap()` / `LoadOption::ram_from_mmap()` — used in `HNSWIndex::open` (`hnsw.rs:103-107`)
- `GraphLayers::load(...)` — `lib/segment/src/index/hnsw_index/graph_layers.rs`

### 10.3 Payload storage

Always mmap-backed by default (`PayloadStorageType::Mmap`), with optional `InRamMmap`.

---

## 11. Optimizer / merger background processes

### 11.1 Workers

Each local shard's `UpdateHandler` spawns three long-running tasks:

1. **Update worker** — drains update signals and applies ops (`lib/collection/src/update_workers/update_worker.rs:44`).
2. **Optimizer worker** — decides when/what to optimize and launches jobs (`lib/collection/src/update_workers/optimization_worker.rs:45`).
3. **Flush worker** — periodically flushes WAL + segments and acknowledges the WAL (`lib/collection/src/update_workers/flush_workers.rs:101`).

### 11.2 Optimizer types

Built in `lib/collection/src/optimizers_builder.rs:275`:

- `MergeOptimizer` — reduces segment count.
- `IndexingOptimizer` — builds HNSW / quantization / mmap for large segments.
- `VacuumOptimizer` — reclaims deleted vectors.
- `ConfigMismatchOptimizer` — rebuilds when config changes.

### 11.3 Optimization execution

The optimizer:

1. Wraps input segments in `ProxySegment`s so reads/updates can continue.
2. Builds a new segment in a temp directory.
3. Atomically swaps the new segment in and drops the proxies.

Key functions:

- `execute_optimization` — `lib/shard/src/optimize.rs:695`
- `build_new_segment` — **line 208**
- `finish_optimization` — **line 457**
- `SegmentBuilder::build` — `lib/segment/src/segment_constructor/segment_builder.rs:506`

The swap uses `SegmentHolder::swap_new` (`lib/shard/src/segment_holder/mod.rs:198`) under the `updates_mutex`, ensuring readers see a consistent transition.

---

## 12. Why Qdrant ingestion is so fast — summary

1. **Lazy HNSW indexing.** Small/appendable segments use a plain (brute-force) vector index. Inserts are just appends; no HNSW graph mutation happens on the hot path.
2. **Offline HNSW builds.** HNSW is built by background optimizers using temp segments and atomic swap-in, so insert latency is not coupled to graph construction cost.
3. **Cheap appendable segment ops.** Appendable segments use in-RAM chunked mmap and mutable in-memory payload indexes (Gridstore), avoiding random I/O and expensive index rebuilds.
4. **WAL-first durability with buffering.** Writes are appended to an mmap WAL; explicit flushes are per `wait=true` request or by the periodic flush worker, not per point.
5. **Update batching.** Upserts are chunked into 32-point batches inside the update processor, reducing lock/ FFI overhead.
6. **Single update worker per shard.** Serialization avoids complex locking and lock contention, at the cost of single-threaded write throughput per shard.
7. **Mmap-centric storage.** Vectors and payloads live in mmap-backed files, keeping the working set near OS cache and avoiding large user-space copies.

The combination means that for small-to-medium in-memory collections (like our benchmarks), Qdrant behaves like a fast append-only log plus periodic background indexing — not like an HNSW index that mutates on every insert.
