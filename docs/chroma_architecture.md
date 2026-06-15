# Chroma Architecture Deep Dive

> Codebase: `d:\personal-projects\TurboSuperMemory\chroma`.
> This document focuses on the design choices that shape Chroma's ingestion and search performance, and how they compare to TurboSuperMemory (TSM).

## 1. High-level layout

```
chromadb/api/          # Client-facing API
  client.py            # Client proxy (forwards to ServerAPI singleton)
  segment.py           # Legacy Python ServerAPI (SegmentAPI)
  rust.py              # New Rust-backed ServerAPI shim
  models/Collection.py # Collection.add / .query / .search

chromadb/db/           # Local persistence
  impl/sqlite.py       # SqliteDB, TxWrapper, connection pools
  mixins/embeddings_queue.py   # SQLite WAL producer/consumer
  mixins/sysdb.py      # SQLite sysdb

chromadb/segment/      # Segment abstractions & local manager
  __init__.py          # SegmentScope, SegmentManager, reader/writer traits
  impl/manager/local.py# Python LocalSegmentManager
  impl/metadata/sqlite.py# Legacy SQLite metadata segment
  impl/vector/local_hnsw.py          # Python in-memory HNSW
  impl/vector/local_persistent_hnsw.py # Python persistent HNSW
  impl/vector/brute_force_index.py   # Small/degraded fallback

chromadb/logservice/   # Distributed gRPC log service (hosted)
  logservice.py

rust/                  # Rust implementation (new local path)
  python_bindings/     # PyO3 module chromadb_rust_bindings
  frontend/            # ServiceBasedFrontend business logic
  log/                 # SqliteLog, LocalCompactionManager
  segment/             # LocalSegmentManager, HNSW writers, metadata writers
  blockstore/          # Arrow-backed blockfiles
  index/               # hnswlib wrapper, SPANN, usearch
```

---

## 2. Ingestion flow

### 2.1 Python API entry

- `Collection.add` (`chromadb/api/models/Collection.py:78`) validates the request and forwards to the underlying client.
- `Client._add` (`chromadb/api/client.py:416`) injects `tenant`/`database` and forwards to `self._server._add`.

### 2.2 Two server backends

| Path | Class | File |
|------|-------|------|
| Legacy Python | `SegmentAPI` | `chromadb/api/segment.py` |
| New Rust-backed | `RustBindingsAPI` | `chromadb/api/rust.py` |

#### Legacy Python path

`SegmentAPI._add` (`chromadb/api/segment.py:508`):

1. Fetches collection.
2. Converts columnar inputs to `OperationRecord`s via `_records` (`chromadb/api/segment.py:1096`).
3. Embeds `chroma:document` and `chroma:uri` into metadata keys.
4. Calls `self._producer.submit_embeddings(...)` — the local SQLite WAL queue.

#### Rust bindings path

`RustBindingsAPI._add` (`chromadb/api/rust.py:468`):

1. Emits telemetry.
2. Calls `self.bindings.add(...)` on the PyO3 `Bindings` object.
3. Inside Rust, `Bindings::add` (`rust/python_bindings/src/bindings.rs:456`) builds `AddCollectionRecordsRequest` and `runtime.block_on`s `frontend.add(req)`.

### 2.3 Frontend add (Rust path)

`ServiceBasedFrontend::add` (`rust/frontend/src/impls/service_based_frontend.rs:1427`):

- Validates embedding dimension.
- Converts inputs to `OperationRecord`s with `to_records(..., Operation::Add)`.
- Calls `retryable_push_logs` (`rust/frontend/src/impls/service_based_frontend.rs:1414`), retrying only on `PushLogsError::Backoff`.

### 2.4 Log write

Local SQLite: `SqliteLog::push_logs` (`rust/log/src/sqlite_log.rs:326`):

- Batches inserts sized by `MAX_VARIABLE_NUMBER / VARIABLE_PER_RECORD`.
- Inserts into `embeddings_queue` (`operation`, `topic`, `id`, `vector`, `encoding`, `metadata`), commits, then triggers backfill + purge via the `LocalCompactionManager` handle.

Legacy Python queue: `SqlEmbeddingsQueue.submit_embeddings` (`chromadb/db/mixins/embeddings_queue.py:189`) does the same table insert and notifies in-process consumers.

Distributed hosted: `LogService.submit_embeddings` (`chromadb/logservice/logservice.py:98`) converts records to protobuf and calls gRPC `PushLogs`.

---

## 3. WAL / log service

### 3.1 Abstractions

`Producer` / `Consumer` interfaces in `chromadb/ingest/__init__.py`:

- `Producer.submit_embeddings(collection_id, embeddings) -> Sequence[SeqId]`
- `Consumer.subscribe(..., consume_fn, start, end) -> UUID`

### 3.2 Implementations

| Mode | Producer/Consumer | Notes |
|------|-------------------|-------|
| Local embedded | `SqlEmbeddingsQueue` | Same `embeddings_queue` table; in-process pub/sub |
| Rust local | `SqliteLog` + `LocalCompactionManager` | Async SQLite log client, actor-driven compaction |
| Distributed | `LogService` gRPC stub | `PushLogs` / `PullLogs` |

### 3.3 Compaction / backfill

`LocalCompactionManager` (`rust/log/src/local_compaction_manager.rs`):

- `BackfillMessage` handler (`local_compaction_manager.rs:132`): reads log delta from `min(mt_max_seq_id, hnsw_max_seq_id)` to latest, materializes logs, applies to metadata writer and HNSW writer.
- `PurgeLogsMessage` handler (`local_compaction_manager.rs:265`): computes safe purge offset from segment max seq IDs and deletes older WAL entries.

`materialize_logs` (`rust/segment/src/lib.rs:746`) collapses `LogRecord` chunks into final operations per user ID, resolves existing offset IDs via the record-segment reader, and assigns new offset IDs atomically.

### 3.4 Key design point

Chroma separates the **write-ahead log** (SQLite `embeddings_queue`) from the **index mutation**. Writers append to the log cheaply; a background consumer (`LocalCompactionManager` or the Python segment subscription) applies batches of log records to the HNSW and metadata indexes. This amortizes index mutation cost.

---

## 4. Segment model

### 4.1 Scopes and types

From `chromadb/types.py` and `chromadb/segment/__init__.py`:

- `SegmentScope`: `VECTOR`, `METADATA`, `RECORD`
- `SegmentType`: `SQLITE`, `HNSW_LOCAL_MEMORY`, `HNSW_LOCAL_PERSISTED`, `BLOCKFILE_METADATA`, etc.

Every collection has:

- a `VECTOR` segment (HNSW),
- a `METADATA` segment (SQLite or blockfile),
- in distributed mode, a `RECORD` segment.

### 4.2 Python local segment manager

`LocalSegmentManager` (`chromadb/segment/impl/manager/local.py`):

- `prepare_segments_for_new_collection` (`local.py:140`) creates an HNSW vector segment and a SQLite metadata segment.
- `get_segment` (`local.py:203`) caches by `collection_id`; instantiates under `self._lock`.
- `hint_use_collection` (`local.py:227`) preloads vector segment and pins it in an LRU file-handle cache.

### 4.3 Rust local segment manager

`LocalSegmentManager` (`rust/segment/src/local_segment_manager.rs`):

- Uses a Foyer cache for `LocalHnswIndex` to bound open file descriptors.
- `get_hnsw_reader` / `get_hnsw_writer` (`local_segment_manager.rs:96/124`): cache hit returns cloned index; miss builds from segment and starts FDs.

### 4.4 Sealing / flushing

Segment writes are applied through writers (`SqliteMetadataWriter`, `LocalHnswSegmentWriter`, blockfile-based `MetadataSegmentWriter`). Blockfile writers produce `ArrowBlockfileFlusher` on `commit`; flusher persists blocks + root.

---

## 5. Rust / Python boundary

### 5.1 PyO3 module

`rust/python_bindings/src/lib.rs` exposes `chromadb_rust_bindings` with a `Bindings` class.

`Bindings` struct (`rust/python_bindings/src/bindings.rs:38`) holds:

- a `tokio::runtime::Runtime`,
- `System`,
- `Frontend`,
- `SqliteDb`,
- `LocalCompactionManager` handle.

### 5.2 Configuration wiring

`Bindings::py_new` (`bindings.rs:80`) builds a `FrontendConfig` with:

- `ExecutorConfig::Local(LocalExecutorConfig {})`,
- `LogConfig::Sqlite`,
- `SysDbConfig::Sqlite`,
- `LocalSegmentManagerConfig` with Foyer cache sized by `hnsw_cache_size`,
- `enable_schema = true`,
- default kNN index = HNSW.

### 5.3 GIL handling

- Mutating methods (`add`, `update`, `upsert`, `delete`) use `runtime.block_on(...)` while holding the GIL.
- Long-running reads (`get`, `query`) wrap the blocking call in `py.allow_threads(...)` (`bindings.rs:698`, `bindings.rs:746`) so the Python GIL is released.

### 5.4 Python shim

`RustBindingsAPI` (`chromadb/api/rust.py`) implements `ServerAPI` by delegating to `chromadb_rust_bindings.Bindings`, e.g. `_add` → `self.bindings.add(...)`.

---

## 6. HNSW vector indexing

### 6.1 Python in-memory HNSW — `chromadb/segment/impl/vector/local_hnsw.py`

- **Class:** `LocalHnswSegment(VectorReader)` (line 37)
- **State:** maintains `_id_to_label`, `_label_to_id`, `_id_to_seq_id` mappings (lines 52-57) and a single `hnswlib.Index` instance.
- **Index lifecycle:**
  - `_init_index()` (line 205) creates the index with `space`, `ef_construction`, `M`, then sets `ef` (search) and `num_threads`.
  - `_ensure_index()` (line 223) lazily initializes and resizes the index by `resize_factor`.
- **Writes:** `_write_records()` (line 291) batches incoming `LogRecord`s into a `Batch`, then `_apply_batch()` (line 244) calls `hnswlib.Index.add_items(...)` and `mark_deleted(...)`. All writes are serialized under a `WriteRWLock` (line 297).
- **Query:** `query_vectors()` (line 134) validates `k`, builds an `allowed_ids` filter set, and calls `hnswlib.Index.knn_query(..., filter=filter_function)` under a `ReadRWLock` (line 161).
- **Deletion:** logical deletes via `index.mark_deleted(label)` (line 260); no immediate rebuild.

### 6.2 Python persistent HNSW — `chromadb/segment/impl/vector/local_persistent_hnsw.py`

- **Class:** `PersistentLocalHnswSegment(LocalHnswSegment)` (line 95)
- **Persistence metadata:** `PersistentData` stored as `index_metadata.pickle` (lines 60-92).
- **Two-tier write path:**
  - New/updated records first land in an in-memory `BruteForceIndex` (line 101, `_brute_force_index`).
  - When `_batch_size` records accumulate (`hnsw:batch_size`, default 100), `_apply_batch()` flushes the batch into the persisted `hnswlib.Index` and clears the brute-force buffer (lines 362-365).
  - When `_sync_threshold` log records accumulate (`hnsw:sync_threshold`, default 1000), `_persist()` is triggered (line 296): calls `index.persist_dirty()` and rewrites the pickle metadata (lines 253-288).
- **Query merging:** `query_vectors()` (line 428) runs the brute-force index and the persisted HNSW in parallel, then merges results by distance, filtering deleted IDs from the HNSW result set and deduplicating against the brute-force index (lines 460-516).
- **File handles:** persistent index supports `open_persistent_index()` / `close_persistent_index()` to manage OS file descriptors (lines 546-559).

### 6.3 Rust local HNSW segment — `rust/segment/src/local_hnsw.rs`

- **Reader:** `LocalHnswSegmentReader` (line 29) and `LocalHnswSegmentWriter` (line 442) wrap a `LocalHnswIndex` (an `Arc<RwLock<Inner>>`).
- **Persistence:** same pickle-based `IdMap` and on-disk hnswlib files under `persist_root/<segment_id>/` (line 25).
- **Reads:** `query_embedding()` (line 300) contains a **degradation heuristic**: if `delete_percentage > 0.2` and `actual_len < 100`, it falls back to brute-force over valid IDs (lines 320-372). Otherwise it delegates to `HnswIndex::query()` (lines 378-381).
- **Writes:** `apply_log_chunk()` (line 670) collects operations into a per-label `HashMap`, resizes the index to the next power of two if needed (lines 797-804), and then applies adds/deletes in parallel with Rayon (`into_par_iter()`, line 809).
- **Sync threshold:** persists when `num_elements_since_last_persist >= sync_threshold` (lines 841-856).

### 6.4 Rust HNSW index wrapper — `rust/index/src/hnsw.rs`

- **Library:** thin Rust wrapper around the external `hnswlib` crate (chroma-core/hnswlib, v0.8.2).
- **Config:** `HnswIndexConfig` (line 22) stores `max_elements`, `m`, `ef_construction`, `ef_search`, `random_seed`, optional `persist_path`. `DEFAULT_MAX_ELEMENTS = 100` (line 13) to limit S3 fetch size.
- **Operations:** `init`, `add`, `delete`, `query`, `get`, `resize`, `save`, `load`, `load_from_hnsw_data` (lines 148-266).

### 6.5 HNSW parameters

| Parameter | Python default | Rust default | Meaning |
|---|---|---|---|
| `hnsw:space` | `"l2"` | from config | distance metric |
| `hnsw:M` / `max_neighbors` | 16 | 16 | graph connectivity |
| `hnsw:construction_ef` | 100 | 100 | beam width at build |
| `hnsw:search_ef` | 100 | 100 | beam width at query |
| `hnsw:num_threads` | CPU count | — | threads for `knn_query` |
| `hnsw:resize_factor` | 1.2 | — | growth factor |
| `hnsw:batch_size` | 100 | — | records before HNSW batch flush |
| `hnsw:sync_threshold` | 1000 | from config | log records before disk sync |

### 6.6 Incremental add vs. batch build

- **Python persistent segment:** incremental ingestion into the brute-force buffer, periodic batch promotion to hnswlib (`_apply_batch()` in `local_persistent_hnsw.py:294`).
- **Rust local segment:** per-log-chunk batch processed with Rayon parallelism (`local_hnsw.rs:809`).
- **Rust distributed segment:** each materialized log record triggers `HnswIndex::add()` immediately (`distributed_hnsw.rs:233`).
- **No full graph rebuild on every add:** hnswlib supports incremental insertion; deletes are logical (`mark_deleted`).

### 6.7 Persistence format

- **hnswlib on-disk files (Rust distributed):** `header.bin`, `data_level0.bin`, `length.bin`, `link_lists.bin` (`rust/index/src/hnsw_provider.rs:29-34`).
- **HnswIndexProvider** (`rust/index/src/hnsw_provider.rs:52`) fetches these four files from object storage in parallel, builds an `hnswlib::HnswData`, and loads the index on a blocking thread (`tokio::task::spawn_blocking`, line 211).
- **Flush:** serializes the in-memory index to `HnswData`, uploads the four files in parallel (`flush_from_memory`, line 427).
- **Fork-on-write:** distributed writers do not mutate the persisted index in place; they fork it to a new UUID (`fork`, line 168).

---

## 7. SPANN / quantized SPANN

### 7.1 Non-quantized SPANN

`rust/index/src/spann/types.rs` defines the full SPANN index: HNSW centroid index + posting lists stored in blockfiles + versions map + RNG selection, split/merge/balance. This is the production distributed SPANN path.

### 7.2 Quantized SPANN (usearch-based, gated by `feature = "usearch"`)

- **Writer core:** `rust/index/src/spann/quantized_spann.rs` (`QuantizedSpannIndexWriter<I: VectorIndex>`).
- **Centroid indexes:** two `USearchIndex` instances:
  - `raw_centroid` — full-precision centroid HNSW for writing-time navigation and center drift rebuilds.
  - `quantized_centroid` — RaBitQ-quantized centroid HNSW for fast search (`create`, lines 1036-1051).
- **Quantization scheme:** 4-bit RaBitQ with random rotation matrix and a global quantization center.
  - `rotate()` normalizes (cosine) and multiplies by `rotation` (lines 525-534).
  - `Code::<4>::quantize()` stores quantized residuals relative to the cluster centroid (`register`, lines 443-444).
- **Cluster maintenance:** dynamic split/merge/balance with `split_threshold`, `merge_threshold`, `write_nprobe`, `nreplica_count`, `write_rng_epsilon`, `write_rng_factor`, `reassign_neighbor_count`, `center_drift_threshold` (lines 189-206).
- **Persistence:** commits to three blockfiles:
  - `quantized_cluster` — `QuantizedCluster` values (center + codes + ids + versions).
  - `scalar_metadata` — lengths, `next_cluster_id`, versions.
  - `embedding_metadata` — global quantization center and rotation matrix.
  Plus two usearch indexes (`raw_centroid`, `quantized_centroid`) (`commit`, lines 820-994; `QuantizedSpannFlusher::flush`, lines 1460-1506).
- **Reader:** `rust/segment/src/quantized_spann.rs` (`QuantizedSpannSegmentReaderShard`, line 335). Loads rotation matrix + center, opens quantized centroid usearch index, cluster blockfile reader, and version reader. `navigate()` searches quantized centroids; `get_cluster()` reads a full posting list.
- **Integration:** `rust/segment/src/spann_provider.rs` wires the quantized reader/writer into the segment lifecycle (`read_quantized_usearch`, `write_quantized_usearch`, lines 125-155).

---

## 8. Brute-force fallback

### 8.1 Python buffer index — `chromadb/segment/impl/vector/brute_force_index.py`

- **Class:** `BruteForceIndex` (line 17). Fixed-capacity numpy array of vectors with id↔index maps.
- **Distance functions:** `l2`, `ip`, `cosine` from `chromadb.utils.distance_functions` (lines 33-39).
- **Query:** `np.apply_along_axis(self.distance_fn, 1, self.vectors, query)` for every query, then `np.argsort` (lines 125-131). Filters deleted IDs and `allowed_ids` (lines 136-150).
- **Used by:** `PersistentLocalHnswSegment` to serve not-yet-flushed records.

### 8.2 Rust local fallback

`rust/segment/src/local_hnsw.rs:320-372` triggers brute-force when the HNSW index is small and heavily tombstoned (>20% deleted, <100 live).

### 8.3 Small-collection fallback in Python

`LocalHnswSegment` simply returns empty results when the index is uninitialized (line 137); no separate brute-force path beyond the persistent segment.

---

## 9. Metadata indexing

### 9.1 Legacy Python metadata segment

`SqliteMetadataSegment` (`chromadb/segment/impl/metadata/sqlite.py`):

- Subscribes to the log `Consumer`.
- Writes via `_write_metadata` (`sqlite.py:494`): ADD / UPDATE / UPSERT / DELETE against `embeddings`, `embedding_metadata`, `embedding_fulltext_search`.
- Builds `where` filters (`_where_map_criterion`) and document filters (`_where_doc_criterion`).

### 9.2 Rust SQLite metadata writer/reader

`rust/segment/src/sqlite_metadata.rs`:

- `SqliteMetadataWriter::apply_logs` (`sqlite_metadata.rs:410`) applies operations in a transaction and updates schema when new metadata keys appear.
- `SqliteMetadataReader::get` / `count` query `Embeddings`, `EmbeddingMetadata`, `EmbeddingMetadataArray`, `EmbeddingFulltextSearch`.
- Where evaluation uses union subqueries for int/float equality and comparisons.

### 9.3 Blockfile-based metadata segment (distributed / advanced)

`MetadataSegmentWriter` / `MetadataSegmentWriterShard` (`rust/segment/src/blockfile_metadata.rs`):

- `from_segment` (`blockfile_metadata.rs:497`) forks/creates:
  - FTS index (`Trigram` or `TokenBitmap`),
  - scalar metadata bitmap indexes (string, bool, int, float),
  - sparse vector index (`WAND` or `MaxScore`).
- `apply_materialized_log_chunk` (`blockfile_metadata.rs:359`) applies partitioned materialized logs across shards concurrently.
- `set_metadata` (`blockfile_metadata.rs:949`) routes values to the appropriate bitmap writer.

### 9.4 Schema

The Rust local bindings enable schema-aware metadata handling (`enable_schema = true` in `Bindings::py_new`), which gates typed storage and query planning.

---

## 10. Blockstore design

### 10.1 Core idea

`rust/blockstore/src/arrow/` implements an Arrow-backed blockfile store:

- A blockfile is a collection of immutable **blocks**.
- Each block is an Arrow `RecordBatch` with schema `(prefix, key, value)`.
- A **sparse index** (root) maps key ranges to block UUIDs.

### 10.2 Block

`Block` (`rust/blockstore/src/arrow/block/types.rs:92`):

- Holds `RecordBatch` + `id`.
- Supports binary-search reads: `get`, `get_prefix`, `get_range`, `get_raw`.
- Serialized as Arrow IPC with 64-byte alignment.
- Caches via Foyer use custom `Code`/`Weighted` impls.

### 10.3 Sparse index

`SparseIndexWriter` / `SparseIndexReader` (`rust/blockstore/src/arrow/sparse_index.rs`):

- `BTreeMap<SparseIndexDelimiter, Uuid>` (writer) or `SparseIndexValue` (reader).
- Supports `get_target_block_id`, `get_all_target_block_ids`, `get_block_ids_for_prefixes`, `get_block_ids_range`.

`RootWriter` / `RootReader` (`rust/blockstore/src/arrow/root.rs`):

- Serializes sparse index as an Arrow record batch with metadata fields `version`, `id`, `max_block_size_bytes`.
- Supports fork semantics for versioning.

### 10.4 Writers

- `ArrowUnorderedBlockfileWriter` (`rust/blockstore/src/arrow/blockfile.rs`): general random-write blockfile.
  - `set` (`blockfile.rs:177`) uses `AsyncPartitionedMutex<Uuid>` per block to serialize deltas.
  - Splits blocks when size exceeds `max_block_size_bytes`.
  - `commit` drains deltas and returns `ArrowBlockfileFlusher`.
- `ArrowOrderedBlockfileWriter` (`rust/blockstore/src/arrow/ordered_blockfile_writer.rs`): append-ordered writes, used for record segments.

### 10.5 Flush and storage

- `ArrowBlockfileFlusher::flush` (`rust/blockstore/src/arrow/flusher.rs:44`) flushes blocks with bounded concurrency (`buffer_unordered(num_concurrent_block_flushes)`), then flushes the root.
- `BlockManager` / `RootManager` (`rust/blockstore/src/arrow/provider.rs`) manage block/root caches, storage fetches, and bounded concurrent loads.

### 10.6 Vector/value types in `rust/blockstore/src/arrow/block/value/`

| File | Value type | Purpose |
|---|---|---|
| `data_record_value.rs` | `DataRecord` | record segment: id + embedding + metadata + document |
| `float32array_value.rs` | `Vec<f32>` | raw embeddings, rotation matrix columns |
| `quantized_cluster_value.rs` | `QuantizedCluster` | SPANN posting list: center + 4-bit codes + ids + versions |
| `spann_posting_list_value.rs` | `SpannPostingList` | non-quantized SPANN posting lists |
| `f32_value.rs`, `u32_value.rs`, `str_value.rs`, `uint32array_value.rs`, `roaring_bitmap_value.rs` | scalar / metadata / full-text bitmap values |

---

## 11. Search path: query → segment manager → HNSW → result merging

### 11.1 Python local executor

`chromadb/execution/executor/local.py:107` (`LocalExecutor.knn()`):

1. **Prefilter:** if `user_ids`, `where`, or `where_document` exist, query the metadata segment first to get `prefiltered_ids` (lines 108-119).
2. **Vector query:** build `VectorQuery` with `vectors`, `k`, `allowed_ids=prefiltered_ids` (lines 127-134).
3. **Segment dispatch:** `self._vector_segment(plan.scan.collection).query_vectors(query)` (line 135).
4. **Hydration:** if documents/metadata/uris are requested, fetch records by merged result IDs (lines 153-187).

### 11.2 Segment manager

`chromadb/segment/impl/manager/local.py:50` (`LocalSegmentManager`):

- Maintains `_instances` dict and per-scope `SegmentCache` (lines 54-82).
- For persistent mode, vector segments are `HNSW_LOCAL_PERSISTED` (line 88).
- LRU segment cache keyed by collection id, sized by `chroma_memory_limit_bytes` and disk size (lines 72-82).
- File-handle LRU cache bounded by OS fd limit / handles per segment (lines 94-101).
- `get_segment()` returns cached instance or creates/starts a new one under a lock (lines 203-220).

### 11.3 HNSW query dispatch

- Python persistent segment merges brute-force + HNSW results (see §6.2).
- Rust distributed HNSW reader `DistributedHNSWSegmentReader::query()` (`rust/segment/src/distributed_hnsw.rs:402`) directly calls `HnswIndex::query()` with `allowed_ids` / `disallowed_ids`.

### 11.4 Result merging

- Python: distance-ordered merge of brute-force and HNSW streams, with dedup and deletion filtering (`local_persistent_hnsw.py:460-516`).
- Rust local: simple top-k max-heap over brute-force results; otherwise returns HNSW results directly (`local_hnsw.rs:300-392`).

---

## 12. Filtering: metadata filtering integration

### 12.1 Prefilter architecture

Filtering is **not pushed into HNSW** in the Python local path. The executor resolves the metadata filter first to a list of `allowed_ids`, then passes that list to the vector segment (`local.py:108-134`).

The vector segment uses the list as an `allowed_ids` filter:

- Python HNSW: builds a `Set[int]` of labels and passes `filter=filter_function` to `knn_query` (`local_hnsw.py:149-166`).
- Brute-force: filters during numpy post-processing (`brute_force_index.py:136-150`).

This is a **pre-filter** design: recall quality depends on HNSW being able to reach the allowed IDs from the graph; for small/degraded indexes, Rust falls back to brute-force.

### 12.2 Disallowed IDs

Rust distributed HNSW query signature includes both `allowed_ids` and `disallowed_ids` (`rust/index/src/hnsw.rs:190-200`; `distributed_hnsw.rs:402-413`), though the Python local path only exposes allowed IDs.

---

## 13. Caching and memory management

### 13.1 HNSW index cache (distributed Rust)

- `HnswIndexProvider` (`rust/index/src/hnsw_provider.rs:52`) caches `HnswIndexRef` per `CollectionUuid`.
- Cache key is the collection ID, limiting at most one index per collection.
- Weight is estimated as `len * sizeof(f32) * dimensionality` in MB (lines 110-126).
- `open()` uses double-checked locking to avoid duplicate loads (lines 296-368).
- `fork()` clones the index by serializing/deserializing on a blocking thread to avoid blocking Tokio (lines 168-238).

### 13.2 USearch index cache

- `USearchIndexProvider` (`rust/index/src/usearch.rs:330`) caches `USearchIndex` by collection and type (`Raw` vs `Quantized`).
- Weight is `memory_usage() / 1024 / 1024` (line 322).

### 13.3 Local segment manager cache (Rust)

- `LocalSegmentManager` (`rust/segment/src/local_segment_manager.rs:36`) caches `LocalHnswIndex` by `IndexUuid`.
- Uses a Foyer in-memory cache with eviction listener that closes file descriptors on eviction (lines 19-72).
- Default capacity 65536 weighted units (line 19).

### 13.4 Python segment cache

`LocalSegmentManager.segment_cache` (line 69):

- `SegmentScope.METADATA`: `BasicCache`.
- `SegmentScope.VECTOR`: `SegmentLRUCache` when `chroma_segment_cache_policy == "LRU"` and `chroma_memory_limit_bytes > 0`; otherwise `BasicCache`.
- Vector segment cache size is computed from on-disk directory size (`_get_segment_disk_size`, line 173).
- Eviction callback stops the segment instance and removes it from `_instances` (lines 107-112).

### 13.5 File-handle cache

Persistent Python HNSW keeps a separate `LRUCache[UUID, PersistentLocalHnswSegment]` (`_vector_instances_file_handle_cache`, line 55) bounded by the OS file-descriptor limit divided by the number of handles per index (lines 94-101). Eviction calls `close_persistent_index()`.

### 13.6 Block cache

- `BlockManager` wraps a `PersistentCache<Uuid, Block>` (`rust/blockstore/src/arrow/provider.rs:414-421`).
- Foyer-backed cache supports memory-only or hybrid disk tiers (`rust/cache/src/foyer.rs`).
- `insert_to_disk()` bypasses memory to avoid polluting the cache during prefetch (`provider.rs:516-527`).
- `Block::weight()` is `ceil(size / 1 MiB)` (`block/types.rs:709-713`), so the cache is weighted by actual Arrow buffer size.
- `max_concurrent_block_loads` bounds S3 fetch concurrency during range scans (lines 538-543).

### 13.7 MMAP / off-heap

- hnswlib itself memory-maps or loads its index files depending on persistence flags.
- Rust local segment provides explicit `open_fd()` / `close_fd()` on the hnswlib index (`rust/index/src/hnsw.rs:140-146`), and the eviction listener calls `close()` to release descriptors.
- Arrow blocks are loaded into memory as `Bytes` → `Buffer`; no explicit mmap in the blockstore layer, but the Foyer hybrid disk cache provides disk-tier caching.

---

## 14. Concurrency

### 14.1 Python side

- `SqliteDB` (`chromadb/db/impl/sqlite.py:63`):
  - Persistent mode uses `PerThreadPool`.
  - In-memory mode uses `LockPool` with a shared-cache URI.
- `TxWrapper` (`chromadb/db/impl/sqlite.py:28`) provides per-thread/lock connection-pool transactions.
- `LocalSegmentManager` (`chromadb/segment/impl/manager/local.py:203`) uses a `Lock` to ensure only one thread creates a segment instance.
- `SqlEmbeddingsQueue` uses in-process subscription callbacks.

### 14.2 Rust side

- `ArrowUnorderedBlockfileWriter`:
  - `AsyncPartitionedMutex<Uuid>` per block serializes block-delta mutations (`blockfile.rs:189`).
  - Block reads are bounded by `max_concurrent_block_loads`.
  - Block flushes are bounded by `num_concurrent_block_flushes`.
- `LocalSegmentManager` uses a Foyer cache with an eviction listener that closes HNSW indices.
- Rust async runtime is single multi-thread Tokio runtime owned by `Bindings`.

### 14.3 GIL / FFI

- `Bindings::get` / `Bindings::query` release the Python GIL via `py.allow_threads(...)`.
- Mutating calls keep the GIL during validation but block the runtime on async DB/log work.

---

## 15. Why Chroma's numbers look the way they do — summary

In our benchmarks Chroma's ingestion is **slower than Qdrant** but its search is **very fast for small-to-medium collections**. The design explains both:

1. **Log + compaction amortizes index cost, but adds layers.** Every insert goes through the SQLite queue and is later applied to the HNSW and metadata segments. For small in-memory workloads this is more overhead than Qdrant's direct appendable-segment append.
2. **Persistent HNSW uses a brute-force buffer + periodic batch flush.** Records land in a numpy brute-force buffer first (default batch size 100) and are only periodically promoted to hnswlib. This amortizes Python/C++ boundary crossings and index mutation cost, but the Python-layer bookkeeping still adds latency.
3. **hnswlib is fast, but the Python wrapper serializes writes.** `_write_records` takes a `WriteRWLock` and calls `hnswlib.Index.add_items` under it; no parallel HNSW construction on the local Python path.
4. **Search is fast because hnswlib's `knn_query` is efficient and the persistent segment can merge a small brute-force buffer with the HNSW index.** For N=5k D=128, search is ~1.6 ms vs TSM's ~3.1 ms.
5. **Recall degrades on larger/random data** because hnswlib's default `search_ef=100` with `k=5` and heavy logical deletes can miss true neighbors; our benchmark saw ~89.6% recall@5 at N=5k D=128.
6. **Heavy Python/Rust layering.** The legacy path pays for SQLite queue writes, metadata segment updates, and Python object conversions. The new Rust path is closer to Qdrant's model but still uses SQLite as the log and hnswlib for the index.

In short, Chroma trades some raw ingest throughput for a clean log/segment architecture, persistence, and multi-level caching aimed at hosted/distributed deployments.
