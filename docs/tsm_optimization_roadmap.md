# TSM Optimization Roadmap

> Approved plan: **docs and roadmap only** (`C:\Users\User\.kimi\plans\aquaman-vision-sentry.md`).
> No TSM code changes are authorized yet; this document is the input for the next implementation phase.

## 1. Where we are

The benchmark suite measured TSM against Qdrant and Chroma across `N=200/1k/5k` and `D=8/128/768`.

| N | D | TSM ingest (ms/item) | TSM search (ms/q) | TSM recall@5 | Qdrant ingest | Chroma ingest | Qdrant recall | Chroma recall |
|---|---|----------------------|-------------------|--------------|---------------|---------------|---------------|---------------|
| 200 | 8 | 0.628 | 0.137 | — | — | — | — | — |
| 1k | 128 | 2.975 | 0.803 | 100% | 0.068 | 5.297 | 100% | 100% |
| 5k | 128 | 15.161 | 3.119 | 88.0% | 0.068 | 5.468 | 100% | 89.6% |
| 5k | 768 | 26.96 | 6.33 | — | — | — | — | — |

A 10k×768 run timed out at 87% ingestion. Ingestion is super-linear, search is slower than Chroma, and recall drops at larger N.

---

## 2. Root-cause analysis

### 2.1 Ingestion super-linear slowdown

**What happens on every insert today**

`StorageEngine::insert_with_payload` (and every item inside `insert_batch_with_payload`) performs:

1. **Vector append** to the mmap `VectorStore`.
2. **WAL append** (`Wal::append`) for every record.
3. **Metadata cache update** (`MetadataStore::put`).
4. **Id index update** (`id_index`).
5. **Payload index update** (`payload_index.add`).
6. **Full-text index update** (`text_index.add`) — buffered in Tantivy but adds document-per-call.
7. **Cognitive graph update** (`graph.add_record`, `ccs.record_observation`).
8. **Hot segment HNSW insert** (`HotSegment::insert` via `usearch::Index::add`).
9. **Hot-segment seal** when `count == hot_capacity` (default 10,000), which triggers:
   - creation of a `SealedHotSegment` or `WarmSegment` from all Hot vectors,
   - a full HNSW rebuild (`from_vectors` adds every vector one-by-one, then saves/views),
   - payload-index building,
   - Tantivy `commit`,
   - metadata store flush,
   - WAL flush,
   - compaction to Cold if `WarmSegment` count exceeds policy.

All of this happens **synchronously** in the caller thread. There is no appendable plain segment and no background optimizer.

**Why it hurts**

| Symptom | Cause |
|---------|-------|
| Ingest rises from ~0.6 ms/item to ~15 ms/item at N=5k D=128 | The Hot segment grows incrementally with `usearch::Index::add`; at some point the working set no longer fits CPU caches and each graph mutation touches more memory. |
| D=768 is ~2× slower than D=128 at same N | Vector copies, distance computations, and HNSW neighbor scoring are all linear in D. |
| 10k×768 timed out at 87% | Likely the first Hot-segment seal at 10,000 records plus Tantivy commit + full payload rebuild + HNSW save is expensive enough that individual items pushed past the timeout budget. |

**Why Qdrant is flat**

Qdrant keeps small **appendable plain segments** (brute-force) for recent writes. Insert = append to chunked mmap + in-memory payload index + WAL append. HNSW is built **offline** by background optimizers once the segment crosses `indexing_threshold` or `max_segment_size`. Insert latency is therefore decoupled from graph construction cost.

**Why Chroma is slower than Qdrant but more stable**

Chroma writes to an SQLite `embeddings_queue` and applies logs in background batches. It also buffers recent writes in a **brute-force numpy buffer** (batch size 100) before promoting to hnswlib. So it pays SQLite + Python overhead, but it does not rebuild HNSW on every insert either.

### 2.2 Recall drop at N=5k

`SegmentHolder::search` computes the per-segment candidate pool as:

```rust
let pool_k = if filter.is_some() { top_k * 8 } else { top_k * 4 };
```

For `top_k = 5`, that is only **20** unfiltered candidates per segment.

At N=5k there are multiple segments (Hot + at least one Warm/Cold from compaction), so the global rerank receives 20–40 candidates total. usearch returns approximate neighbors from each segment, and a small pool does not give reranking enough raw material to recover the true top-5 across the whole dataset.

By contrast:

- Qdrant's HNSW search uses `ef = max(ef, top)` where `ef` defaults to `ef_construct` (100). It also merges per-segment results with probabilistic sampling and reranks with full vectors when quantization is enabled.
- Chroma's hnswlib uses `ef=100` by default for `k=5`.

TSM's default `search_list_size = 100` is used for HNSW construction (`ef_construction`) but is **not used as the search-time `ef`**. The runtime pool is hard-coded to `top_k * 4`.

### 2.3 Search latency

TSM search at N=5k D=128 is ~3.1 ms vs Chroma ~1.6 ms. Contributing factors:

- Small per-segment pool means more queries may need to scan fallback segments.
- `WarmSegment` / `ColdSegment` brute-force scans are not accelerated; at N=5k the Warm tier may still be scanned linearly.
- Tantivy full-text index `commit` is called inside `evaluate_filter` on every full-text query (current `TextIndex::commit` flushes the writer and reloads the reader).
- Multi-segment result merge uses heap merge but lacks sampling or prefetching.

### 2.4 Text index overhead

- `TextIndex::add` is called once per record. Tantivy documents are buffered in a 50 MB heap, but the writer is not committed/flushed periodically; instead `evaluate_filter` calls `commit` on every text query.
- For ingestion, this means the index grows uncommitted in memory until a text search happens, then pays a large commit cost.
- For search, repeated commits stall queries.

### 2.5 Graph / CCS overhead per insert

Every record also updates the cognitive graph (`graph.add_record`) and the compressed context store (`ccs.record_observation`). These are in-memory, but they run for every record and every concept tag. For payloads with many concepts this can dominate CPU.

### 2.6 Single-threaded ingestion

`insert_batch` releases the Python GIL, but the Rust work is single-threaded inside `StorageEngine`. There is no update queue, no background optimizer, and no CPU-budgeted indexing thread. `UpdateHandler` only triggers periodic consolidation/flush, not index building.

---

## 3. Comparative design matrix

| Area | TSM (current) | Qdrant | Chroma | Impact on TSM |
|------|---------------|--------|--------|---------------|
| **Hot write path** | HNSW `usearch::Index::add` on every insert | Plain/brute-force appendable segment; HNSW built offline | Brute-force numpy buffer + periodic hnswlib batch | High |
| **Segment sealing** | Synchronous seal at 10k items: rebuild HNSW + Tantivy commit + payload rebuild | Background optimizer builds new segment in temp dir, atomic swap-in | Background `LocalCompactionManager` applies log batches | High |
| **WAL durability** | Per-record `bincode` + CRC32-C + `sync_data` on caller path | mmap WAL, explicit flush on `wait=true` or periodic flush worker | SQLite `embeddings_queue` insert + background compaction/purge | Medium |
| **Metadata updates** | Per-record `MetadataStore::put`, later flushed | In-memory mutable until segment seal; flush with segment | Applied in log-compaction batches | Medium |
| **Payload index** | In-memory Roaring bitmap, rebuilt on seal | Mutable in-memory (Gridstore) or mmap field indexes | SQLite metadata segment or blockfile bitmap indexes | Medium |
| **Full-text index** | Tantivy document per record; commit on every query | Optional; built during optimization | FTS blockfile or SQLite FTS | Medium |
| **Search-time candidate pool** | `top_k * 4` (or `*8`) per segment | `ef = max(ef, top)` (default 100+) with sampling | `ef=100` default in hnswlib | High |
| **Multi-segment search** | Query all segments, merge + rerank | Parallel per-segment tasks + probabilistic sampling | Brute-force buffer + HNSW result merge | Medium |
| **Quantization / tiers** | Hot FP32, Warm 8-bit scalar, Cold 1-bit sign | Scalar/PQ/Binary/Turbo, optional rescore | RaBitQ 4-bit SPANN (gated), full-vector fallback | Medium |
| **Concurrency** | Single-threaded caller | Single update worker per shard + parallel optimizers | SQLite + Rust async + Rayon for batch applies | Medium |

---

## 4. Prioritized improvements

### P0 — Ingestion: separate appendable plain segment from indexed segment

**Goal:** Make the hot path `O(1)` per insert and decouple HNSW construction from insert latency.

**Approach (Qdrant-style):**

1. Introduce a small **appendable plain segment** (brute-force, in-RAM vectors) that receives all new writes.
2. While the collection is small (e.g. ≤ 4,096 vectors or below a configurable threshold), search uses this plain segment exclusively.
3. Once the plain segment crosses a size/record threshold, hand it off to a **background optimizer** that:
   - builds a `usearch` HNSW index from the vectors in bulk,
   - commits the Tantivy text index,
   - flushes metadata + payload indexes,
   - atomically swaps the new immutable segment in.

**Approach (Chroma-style, lower-effort variant):**

1. Keep the current Hot HNSW segment but insert into an in-memory **write buffer** first.
2. Flush the buffer to `usearch::Index::add` in batches (e.g. every 100 records or every N ms).
3. Still build the sealed HNSW offline when the Hot segment is full.

**Recommended:** adopt the Qdrant-style plain segment because it also fixes the recall path (small-N exact search) and naturally supports background builds. The Chroma-style buffer is a smaller incremental change if plain-segment work is too large for a first pass.

### P0 — Search: use `ef` = `search_list_size` at query time, not `top_k * 4`

**Goal:** Restore near-100% recall@5 at N=5k.

**Approach:**

1. Change `SegmentHolder::search` to compute `pool_k = max(top_k * 4, self.config.search_list_size)`.
2. Optionally expose `ef` in the Python `search_ann` API so callers can trade latency for recall.
3. For filtered search, keep the larger multiplier (`*8`) but still floor it at `search_list_size`.

**Expected effect:** recall should climb back to ≥ 98% at N=5k with modest latency increase (~0.5–1 ms).

### P0 — Text index: batch commits and decouple from search

**Goal:** Remove per-query Tantivy commit stalls and reduce memory pressure during ingestion.

**Approach:**

1. Run a background commit every N documents or every T seconds instead of committing on every full-text query.
2. In `evaluate_filter`, only commit if the writer has uncommitted documents.
3. For sealed segments, build a read-only Tantivy index at seal time and keep it open; do not share a mutable writer with the Hot segment.
4. (Longer term) split text indexing out of the insert hot path entirely and build it during segment optimization, like payload indexes in Qdrant.

### P1 — Background optimizer / update worker

**Goal:** Move HNSW seal + compaction off the caller thread.

**Approach:**

1. Add an async or blocking update worker thread per `StorageEngine`:
   - Producer: `insert_batch` pushes records into a bounded channel (or appends to WAL + shared buffer).
   - Consumer: applies WAL, updates metadata/payload/text indexes, inserts into the plain/HNSW segment, and triggers background seal/compaction.
2. Seal/compaction runs in a separate task with a resource budget (max one CPU-heavy optimizer at a time, like Qdrant's `ResourceBudget`).
3. `insert_batch` returns as soon as the WAL record is durable (or buffered, depending on durability setting), not after seal.

### P1 — Multi-segment search improvements

**Goal:** Reduce search latency and improve recall on multi-segment collections.

**Approach:**

1. Query segments in parallel using a thread pool (Rayon or dedicated workers) instead of sequentially.
2. Use per-segment `ef` candidates and a global k-way merge with deduplication.
3. For filtered search, push the allowed-offsets bitmap into each segment so the HNSW traversal skips filtered neighbors (Qdrant `FilteredScorer` pattern) instead of post-filtering.
4. Add probabilistic over-fetch (`sampling_limit`) when many segments exist, to avoid under-sampling small segments.

### P1 — WAL / metadata batching

**Goal:** Reduce per-record serialization and fsync overhead.

**Approach:**

1. Buffer `WalOp`s in memory and flush as a batch (length-prefixed records + single CRC or one CRC per record) instead of one syscall per record.
2. `MetadataStore::put_batch` already exists; make sure `insert_batch` uses it once per batch rather than record-by-record.
3. Add a configurable `sync_policy`:
   - `EveryWrite` (current default, safest),
   - `EveryBatch` (flush after each `insert_batch`),
   - `Periodic` (background flush every N ms, like Qdrant/Chroma).

### P2 — Tiered search acceleration

**Goal:** Make Warm/Cold tiers cheaper at query time.

**Approach:**

1. **Warm (8-bit scalar):** keep the current linear scan but add SIMD dot-product over quantized vectors (already 8-bit; easy to vectorize).
2. **Cold (1-bit sign):** use Popcnt / SIMD Hamming distance, then rerank top candidates with full vectors.
3. Add an optional exact top-k accumulator that stops scanning Warm/Cold early when their best possible scores are below the current global top-k threshold.
4. Consider pre-filtering Warm/Cold with payload bitmaps before scanning.

### P2 — Recall audit / auto-tune

**Goal:** Continuously verify that recall targets are met and adjust parameters automatically.

**Approach:**

1. Extend `audit_recall.py` to sample query vectors and ground-truth brute-force neighbors.
2. If recall falls below a threshold (e.g. 95%), raise `search_list_size` dynamically or warn the operator.
3. Add a configuration knob `target_recall` and let the engine pick `ef` per query using a small calibration set.

### P3 — Hosted/distributed features

These are out of scope for the current local-engine bottleneck but noted for completeness:

- Shard-level replication and distributed query routing (Qdrant collection/shard model).
- Object-storage backed segment persistence with lazy fetch (Chroma Rust blockstore + HnswIndexProvider).
- SPANN / RaBitQ quantized indexes for billion-scale collections (Chroma Rust path).

---

## 5. Suggested first implementation phase

If the user approves moving from docs to code, the recommended order is:

1. **Increase search candidate pool** — one-line change in `segment_holder.rs`, big recall win, low risk.
2. **Batch Tantivy commits** — move commit out of `evaluate_filter`; add background timer. Medium risk, fixes search stalls.
3. **Introduce a plain appendable segment** — larger architectural change, but fixes the super-linear ingest curve and naturally improves recall for small N.
4. **Background optimizer** — move Hot-seal + compaction off the caller thread; needed to keep insert latency flat as data grows.
5. **Parallel multi-segment search + filtered HNSW traversal** — latency and recall improvements on top of (3) and (4).

Each step should be benchmarked with `make benchmark`, `make audit`, and `cargo test --workspace --exclude turbomemory_python` before proceeding to the next.
