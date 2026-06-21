# Storage, Durability, and Segment Tiers Subsystem

This document provides a detailed technical overview of `turbomemory_storage` (located in [crates/turbomemory_storage](file:///d:/personal-projects/TurboSuperMemory/crates/turbomemory_storage)), which handles multi-agent namespace isolation, mmap vector storage, transactional durability, tiered indexing, and concurrent read/write locks.

---

## 1. Concurrency and Lock Architecture

`StorageEngine` is designed for high-concurrency environments. It uses a **lock-free-reader / exclusive-writer** design for searching, while using granular locking for mutations.

```mermaid
graph TD
    Reader["Reader Thread"] -- "Atomic Read" --> Snapshot["SegmentSnapshot Pointer"]
    Snapshot -- "Read Guardless" --> Segments["Active Segment Set"]
    Writer["Writer Thread"] -- "Acquire Write Lock" --> SegsLock["SegmentHolder RwLock Write"]
    SegsLock -- "Mutate & Clone" --> NewSnapshot["New SegmentSnapshot"]
    NewSnapshot -- "Atomic Pointer Swap" --> Snapshot
```

* **`arc_swap` Snapshots**: The active searchable segments are published atomically using `arc_swap::ArcSwap<SegmentSnapshot>`. Read paths (like searches) perform a lock-free pointer swap clone of the snapshot and scan the segments without holding any mutexes or read locks.
* **Granular locks**: Internal structures use `parking_lot::RwLock` and `parking_lot::Mutex` which are faster and more memory-efficient than standard library synchronization primitives.
  - `segments: Arc<RwLock<SegmentHolder>>`
  - `graph: Arc<RwLock<SpreadingActivation>>`
  - `id_index: Arc<RwLock<AHashMap<Arc<str>, PointOffset>>>`
  - `payload_index: Arc<RwLock<PayloadIndex>>`
  - `scope_index: Arc<RwLock<ScopeIndex>>`
  - `wal: Arc<Mutex<Wal>>`
  - `gpu: Arc<Mutex<Option<Arc<dyn GpuBackend>>>>` (lazy-initialized GPU backend)

### Lock Compatibility Matrix

| Operation / Resource | `segments` Lock | `graph` Lock | `id_index` Lock | `wal` Lock | `gpu` Lock |
|---|---|---|---|---|---|
| **Search (Read Path)** | None (Lock-free `arc_swap`) | Read Lock (Shared) | Read Lock (Shared) | None | None (GPU ops are lock-free after init) |
| **Insert / Update (Write Path)** | Write Lock (Exclusive) | Write Lock (Exclusive) | Write Lock (Exclusive) | Mutex (Exclusive Append) | None (GPU not used on write) |
| **Consolidation / Optimize** | Write Lock (Exclusive Swap) | Write Lock (Exclusive) | Read/Write Lock | Mutex (Exclusive Flush/Truncate) | None (GPU build uses owned data) |

---

## 2. Multi-Agent Scoping

To support multi-agent systems, memories can be partitioned using the `scope` field:
* **Global/Shared Scope** (`scope = None`): Visible to all agents.
* **Scoped Namespace** (`scope = Some(agent_id)`): Private memory visible only to the specified agent.

The [`ScopeIndex`](file:///d:/personal-projects/TurboSuperMemory/crates/turbomemory_storage/src/scope_index.rs) maintains in-memory roaring bitmaps:
* `by_scope: AHashMap<String, RoaringBitmap>`
* `global: RoaringBitmap`

During a search query with `Some(agent_id)`, the engine performs a fast bitmap union:
\[
\text{Allowed Offsets} = \text{global} \cup \text{by\_scope}[\text{agent\_id}]
\]
This bitmap filter is passed directly to the segment search traversal, ensuring strict isolation at zero extra query cost.

---

## 3. Vector Store (`vectors.bin`)

Full-precision vector embeddings are kept in a single mmap-backed file, [`VectorStore`](file:///d:/personal-projects/TurboSuperMemory/crates/turbomemory_storage/src/vector_store.rs). 

* **Binary Format**:
  - **Header** (32 bytes): Magic signature (`b"TMDV"`), Version (`u32`), Dimension (`u32`), record count (`u64`), and a CRC32-C checksum of the header.
  - **Contiguous Floats**: Raw `f32` vectors appended sequentially.
* **Mmap Growth Strategy**: To prevent frequent memory-mapping operations, the backing file grows geometrically (e.g., doubling size or allocating in large chunks).
* **Reranking Utility**: The `VectorStore` does not store metadata. It is solely responsible for storing raw float coordinates. Tiered segments search using low-precision quantizers, and then load full-precision coordinates from the mmap'd `VectorStore` to perform high-fidelity cosine reranking.

---

## 4. Durability Model (WAL & redb Snapshots)

TurboSuperMemory uses a tiered persistence model to balance write throughput with transactional correctness:

```mermaid
sequenceDiagram
    participant App as "Client Application"
    participant VS as "VectorStore (mmap)"
    participant WAL as "Write-Ahead Log (disk)"
    participant Cache as "Metadata Cache (RAM)"
    participant redb as "redb Snapshot (lazy)"

    App->>VS: Append float embedding
    App->>WAL: Append metadata (WalOp::Insert) & sync
    App->>Cache: Cache MetaRecord (in-memory)
    Note over App, redb: Transaction Complete
    Note over redb: On flush() or background consolidation
    Cache->>redb: Flush dirty MetaRecords to redb table
    WAL->>WAL: Truncate/reset WAL log
```

### 4.1 Write-Ahead Log (WAL)
Every metadata write (insert, update, delete) is appended to an append-only Write-Ahead Log (`wal_meta.bin`).
* **Frame format**: `[magic: 4 bytes "TMSW"] [version: u32] [payload length: u32] [payload bytes (bincode WalOp)] [crc32-c: u32]`.
* **Zero Embeddings in WAL**: Full embeddings are written directly to `VectorStore` and are *never* written to the WAL. The WAL only logs the `PointOffset`, `MetaRecord` (attributes, text, concepts), and monotonic `seq` number. This keeps the write throughput high.

### 4.2 Lazy Snapshotting via `redb`
`redb` acts as a lazy snapshot database (`memory.redb`):
* All records are stored in the `records` table, keyed by `PointOffset`, serialized using `bincode`.
* Engine configurations, current sequence counters, serialized cognitive graph state, and Compressed Cognitive State (CCS) are stored in the `meta` key-value table.
* The WAL is truncated/cleared only when `redb` is successfully flushed to disk.

### 4.3 Crash Recovery Protocol
On database open:
1. Load the last consistent snapshot from `memory.redb` (populating the graph, CCS, and metadata cache).
2. Open the WAL. Replay any records found in the WAL that have a sequence number higher than the snapshot.
3. Repopulate the in-memory indexes (ID index, text search index, scope index, payload filter index).
4. Persist the updated snapshot to `memory.redb` and truncate the WAL.

---

## 5. Segment Tiers & Lifecycle

TurboSuperMemory employs four distinct tiers to optimize vector search speed, memory usage, and build times.

```mermaid
stateDiagram-v2
    [*] --> Hot : "Insert (In-memory, FP32)"
    Hot --> SealedHot : "Hot capacity reached (HNSW Index build)"
    Hot --> Warm : "Hot capacity reached (If size is small)"
    SealedHot --> Cold : "Merge multiple segments"
    Warm --> Cold : "Total Warm capacity reached (Quantize sign/MSE)"
```

| Tier | Mutability | Quantization / Compression | Search Index Technology | Storage Type | Transition / Seal Trigger |
|---|---|---|---|---|---|
| **Hot** | Read-Write | FP32 (No compression) | In-memory `Vec<PointOffset>` + brute-force exact scan | Volatile RAM + mmap `vectors.bin` | Reaches capacity (e.g. `hot_capacity` = 10,000) |
| **SealedHot** | Read-Only | FP32 (No compression) | `usearch` HNSW index graph walk (falls back to exact scan on filter selectivity < 1%) | Persisted disk file (`segments/sealed_hot/`) | Promoted to Warm/Cold or merged during background consolidations |
| **Warm** | Read-Only | 8-bit Scalar or TurboQuant Product Quantizer (4x smaller) | SIMD-accelerated quantized Lookup Table (LUT) dot-product scan + top-k full FP32 reranking | Mmap array index file (`segments/warm/`) | Accumulated Warm records exceed `warm_capacity` |
| **Cold** | Read-Only | 1-bit Sign or TurboQuant MSE Quantizer (32x smaller) | XOR + Popcount byte-level LUT index scan + top-k full FP32 reranking | Mmap array index file (`segments/cold/`) | Long-term archival; evicted if importance decays below floor |

### 5.1 Hot Segment
* New insertions land here.
* Searches perform a fast brute-force dot product of the query against the memory slice.
* **GPU Acceleration**: When the `cuda` feature is enabled and a CUDA device is available, `SegmentSnapshot::search_gpu()` uses cuBLAS `sgemv` for batched exact scan over Hot segment offsets, falling back to CPU SIMD on any error.

### 5.2 SealedHot Segment
* Once the Hot capacity (e.g., 10,000 records) is reached, the Hot segment is sealed.
* A background thread builds an HNSW (Hierarchical Navigable Small World) index. Two implementations are available:
  - **`usearch` HNSW** (default CPU path): Header-only HNSW library. Index file written to `segments/sealed_hot/`.
  - **`CudaAnnIndex` HNSW** (GPU path, `cuda` feature): Custom CUDA HNSW implementation. For small N (≤4096), uses GPU brute-force all-pairs; for large N, uses random-projection bucketing + local search + probabilistic hierarchical layer construction. Transparently falls back to `usearch` on CUDA error.

### 5.3 Warm Segment
* Compresses embeddings to 8-bit integers using `ScalarQuantizer` or `TurboQuantProdQuantizer`.
* Computes similarity using LUTs and AVX2-accelerated math directly on quantized bytes, then reranks the top candidates with full floats.
* **GPU Rerank**: When `cuda` feature is enabled, quantized tier candidate reranking can use GPU batched exact scan via `exact_search_over_offsets_gpu()`.

### 5.4 Cold Segment
* Compresses embeddings to 1-bit representations using `SignQuantizer` or `TurboQuantMseQuantizer`.
* Computes similarity using bitwise XOR and popcount lookups (extremely compact).
* **GPU Rerank**: Same GPU rerank path as Warm tier when `cuda` feature is enabled.

---

## 6. GPU Acceleration in Storage Engine

The `StorageEngine` integrates GPU acceleration through a lazy-initialized, trait-based backend:

```mermaid
graph TD
    subgraph StorageEngine["StorageEngine"]
        gpu_field["gpu: Arc<Mutex<Option<Arc<dyn GpuBackend>>>>"]
        gpu_backend["gpu_backend() -> Option<Arc<dyn GpuBackend>>"]
        is_gpu["is_gpu_accelerated() -> bool"]
    end
    
    subgraph GpuBackend_Trait["GpuBackend Trait"]
        init["init() -> Result<Self>"]
        upload["upload_vectors(vectors) -> GpuBuffer"]
        batch_dot["batch_dot(query, vectors) -> Vec<f32>"]
        exact_topk["exact_topk(query, vectors, k) -> Vec<(idx, score)>"]
        build_hnsw["build_hnsw(vectors) -> GpuHnswIndex"]
    end
    
    subgraph CudaBackend_Impl["CudaBackend (cuda feature)"]
        cudarc["cudarc: CudaContext + CudaBlas"]
        sgemv["cuBLAS sgemv: vectors^T × query"]
        cuda_ann["CudaAnnIndex: custom HNSW"]
    end
    
    subgraph CpuFallback_Impl["CpuFallback"]
        unavailable["Always returns GpuUnavailable"]
    end
    
    gpu_field --> gpu_backend
    gpu_backend --> GpuBackend_Trait
    GpuBackend_Trait --> CudaBackend_Impl
    GpuBackend_Trait --> CpuFallback_Impl
```

### 6.1 GPU-Accelerated Search Paths

When `cuda` feature is enabled and a CUDA device is detected:

1. **Hot Segment Exact Scan**: `SegmentSnapshot::search_gpu()` uploads the query and candidate offset vectors to GPU, computes batched cosine similarity via cuBLAS `sgemv`, and returns top-k results.
2. **Quantized Tier Rerank**: After quantized LUT scan produces candidates, `gpu_rerank_candidates()` can optionally rerank using GPU exact distance compute.
3. **HNSW Build**: `SealedHotSegment::from_vectors()` attempts `GpuHnswIndex::build()` first; on any error, transparently falls back to CPU `UsearchIndex::build()`.

### 6.2 Transparent Fallback

Every GPU path implements silent CPU fallback:
- **CUDA unavailable** (no driver, no device): `CpuFallback` returns `GpuUnavailable` on `init()`.
- **Out of GPU memory**: `CudaBackend` catches allocation errors and propagates them as fallback triggers.
- **Kernel errors**: Any CUDA API error triggers fallback to the equivalent CPU path.
- **Runtime detection**: `is_gpu_accelerated()` checks both feature flag AND device availability at runtime.

---

## 7. Background Consolidation and Optimizer

The [`BackgroundOptimizer`](file:///d:/personal-projects/TurboSuperMemory/crates/turbomemory_storage/src/optimizer.rs) runs continuously as a worker thread:
* **Consolidation**: Merges fragmented small segments into larger ones to keep search parallelization balanced.
* **Tiering**: Promotes/demotes segments based on access frequency. Frequently accessed Cold records can be promoted back to Hot via `promote_hot` if configured.
* **Vacuuming**: Deletes marked records from indices and rewrites segment tables to reclaim storage space.
* **GPU HNSW Build**: When the `cuda` feature is enabled, the optimizer attempts GPU-accelerated HNSW construction for sealed segments. If the GPU path fails (OOM, CUDA error), it falls back to the standard CPU `usearch` build transparently. The build uses a `Weak<StorageEngine>` reference so the optimizer does not keep the engine alive if the engine is dropped.
