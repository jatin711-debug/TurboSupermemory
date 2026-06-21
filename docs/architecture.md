# TurboSuperMemory — Unified Architecture Documentation

Welcome to the architectural documentation for **TurboSuperMemory**, a high-performance, production-oriented AI Memory Engine written in Rust with PyO3 Python bindings. 

Unlike standard vector databases which focus solely on dense index retrieval (e.g. HNSW, IVF), TurboSuperMemory treats memory as a **first-class persistent cognitive layer** for AI agents. It models memory using a tiered vector storage system and an episodic-semantic graph with spreading activation.

---

## 1. System Crate Overview

The codebase is organized as a Cargo workspace split into five specialized crates.

```mermaid
graph TD
    api["turbomemory_api - gRPC/REST Service"] --> storage["turbomemory_storage - Segment & Persistence Engine"]
    python["turbomemory_python - PyO3 Bindings"] --> storage
    storage --> graph_crate["turbomemory_graph - Cognitive Reasoner & BM25"]
    storage --> core["turbomemory_core - SIMD Math & Quantization"]
    storage --> gpu["turbomemory_gpu - GPU Acceleration (CUDA)"]
    graph_crate --> core
    gpu --> core
```

* [**`turbomemory_core`**](file:///d:/personal-projects/TurboSuperMemory/docs/core_quantization.md): SIMD math kernels (AVX2/NEON), Fast Walsh-Hadamard Transform (FWHT) preconditioning, Lloyd-Max centroids, and quantization encoders (Scalar, Sign, TurboQuant).
* [**`turbomemory_storage`**](file:///d:/personal-projects/TurboSuperMemory/docs/storage_persistence.md): Mmap-backed dense vector storage, Write-Ahead Log (WAL) append-only durability, `redb` snapshot persistence, multi-agent scoping (`ScopeIndex`), metadata tables, and the segment consolidation engine.
* [**`turbomemory_graph`**](file:///d:/personal-projects/TurboSuperMemory/docs/cognitive_graph.md): The episodic-semantic memory graph, BM25 indexing, spreading activation scoring, Working Memory compression (CCS), synonym vocabulary evolution, and automatic importance recomputation.
* [**`turbomemory_python`**](file:///d:/personal-projects/TurboSuperMemory/docs/bindings_api.md): High-performance PyO3 bindings exposing the memory engine as a Python package, including zero-copy NumPy array mappings and GIL-free concurrency.
* [**`turbomemory_api`**](file:///d:/personal-projects/TurboSuperMemory/docs/bindings_api.md): Multi-protocol service providing REST (Axum) and gRPC (Tonic) frontends over a unified memory service.
* [**`turbomemory_gpu`**](file:///d:/personal-projects/TurboSuperMemory/docs/gpu_acceleration.md): Optional GPU acceleration layer with a trait-based backend system (`GpuBackend`), CUDA implementation via `cudarc` (cuBLAS batched distance + custom HNSW build), and transparent CPU fallback.

---

## 2. Core Architectural Decisions

### 2.1 Why Rust?
Memory engines for AI agents are highly CPU-bound (due to dense vector math and graph traversals) and memory-bound. Rust provides absolute control over memory allocation, SIMD vector instructions, and lock-free data structures, while providing complete safety from segmentation faults.

### 2.2 Why Segmented Tiers?
Standard vector databases rebuild large monolithic indices, which can create high write latencies. TurboSuperMemory splits storage into four distinct tiers:
1. **Hot**: Appendable in-memory buffers (fast writes, brute-force exact scan).
2. **SealedHot**: Indexed HNSW files built asynchronously in the background (CPU via `usearch` or GPU via `CudaAnnIndex` when `cuda` feature is enabled).
3. **Warm**: 8-bit quantized vectors (4x memory reduction) scanned via SIMD.
4. **Cold**: 1-bit sign-quantized or TurboQuant MSE vectors (32x memory reduction) scanned via fast bitwise XOR/popcount lookup tables.
This keeps write latency low while optimizing search speeds for old/cold memories. GPU acceleration is available for HNSW build and exact-scan reranking via the optional `turbomemory_gpu` crate.

### 2.3 The Split-Persistence Model (WAL + redb)
To achieve durability without duplicating large vector data:
* Vector float arrays are written directly to the mmap'd `vectors.bin` file.
* Metadata and record attributes are appended immediately to a lightweight Write-Ahead Log (`wal_meta.bin`).
* A snapshot is written lazily to `redb` (`memory.redb`) during background consolidation.
* On crash/reboot, the engine replays the lightweight WAL over the last `redb` snapshot.

### 2.4 GPU Acceleration Strategy
GPU acceleration is treated as an **optional performance multiplier**, not a requirement:
* **Trait-based design**: `GpuBackend` trait enables multiple GPU APIs (CUDA today, Vulkan/ROCm tomorrow).
* **Silent fallback**: Every GPU operation falls back to CPU on error — no crashes, no user-visible errors.
* **Build-focused wins**: GPU accelerates HNSW index construction (the clear GPU win), not single-query search (upload overhead dominates).
* **Opt-in compilation**: The `cuda` feature must be explicitly enabled; default builds are CPU-only.

```mermaid
graph TD
    subgraph GPU_Strategy["GPU Acceleration Strategy"]
        Trait["Trait-based: GpuBackend"]
        Fallback["Silent CPU fallback"]
        OptIn["Opt-in: cuda feature"]
        BuildFocus["Build-focused: HNSW construction"]
    end
    
    subgraph Integration["Storage Engine Integration"]
        Lazy["Lazy init: first call to gpu_backend()"]
        ArcSwap["ArcSwap: lock-free GPU state"]
        Transparent["Transparent to callers"]
    end
    
    Trait --> Integration
    Fallback --> Integration
    OptIn --> Integration
    BuildFocus --> Integration
```

---

## 3. System Dataflow

### 3.1 Write Path (Ingestion)
```mermaid
sequenceDiagram
    participant App as Client Application
    participant Core as StorageEngine
    participant VS as VectorStore (mmap)
    participant WAL as WAL (append-only)
    participant Cache as Metadata Cache (RAM)
    
    App->>Core: Ingest Record (ID, Text, Vector, Scope, Concepts)
    Core->>VS: Append raw f32 vector
    Core->>WAL: Append metadata entry (WalOp::Insert) & flush to disk
    Core->>Cache: Add record to metadata cache
    Core->>Core: Update memory indices (ID index, Scope index, Text index)
    Note over Core: Record is now durable and searchable in Hot segment
```

### 3.2 Read Path (Cognitive Retrieval)
```mermaid
flowchart TD
    Q["Query Input"] --> ANN["Parallel Segment Search: Hot/SealedHot/Warm/Cold"]
    ANN --> Rerank["Full f32 Vector Rerank via VectorStore"]
    Q --> BM25["BM25 Lexical Score Trigger"]
    Rerank --> GraphTrigger["Assemble Seeds: Cosine Similarity"]
    BM25 --> GraphTrigger
    GraphTrigger --> FOK{"Peak Energy >= fok_threshold?"}
    FOK -- "No" --> ReturnNone["Return None / Feeling-of-Knowing Gate Blocks"]
    FOK -- "Yes" --> Spreading["Spreading Activation Iterations"]
    Spreading -- "Traverses Refines/Contradicts/Temporal" --> FinalEnergy["Final Activation Scores"]
    FinalEnergy --> Fusion["Score Fusion: blended with Cosine Similarity"]
    Fusion --> Sort["Sort & Return Top K Results"]
```

### 3.3 Full System Component Diagram

```mermaid
graph TB
    subgraph Clients["Clients"]
        PyClient["Python Agent (PyO3)"]
        RESTClient["HTTP REST Client"]
        gRPCClient["gRPC Client"]
    end
    
    subgraph API_Layer["API Layer (turbomemory_api)"]
        Axum["Axum REST Server"]
        Tonic["Tonic gRPC Server"]
        Service["MemoryService (shared logic)"]
    end
    
    subgraph Bindings["Bindings (turbomemory_python)"]
        PyEngine["MemoryEngine Class"]
        PyNumpy["Zero-copy NumPy"]
        PyGIL["GIL Release"]
    end
    
    subgraph Storage["Storage Engine (turbomemory_storage)"]
        Engine["StorageEngine"]
        Segments["SegmentHolder (ArcSwap)"]
        VectorStore["VectorStore (mmap)"]
        WAL["WAL (append-only)"]
        redb["redb (lazy snapshot)"]
        Optimizer["BackgroundOptimizer"]
        ScopeIndex["ScopeIndex (RoaringBitmap)"]
        TextIndex["TextIndex (Tantivy)"]
        PayloadIndex["PayloadIndex (RoaringBitmap)"]
    end
    
    subgraph Tiers["Segment Tiers"]
        Hot["Hot (FP32, exact scan)"]
        SealedHot["SealedHot (HNSW: usearch or GPU)"]
        Warm["Warm (8-bit scalar/TurboQuant)"]
        Cold["Cold (1-bit sign/TurboQuant MSE)"]
    end
    
    subgraph GPU["GPU Layer (turbomemory_gpu, optional)"]
        GpuBackend["GpuBackend Trait"]
        CudaBackend["CudaBackend (cudarc + cuBLAS)"]
        CudaAnnIndex["CudaAnnIndex (custom HNSW)"]
        CpuFallback["CpuFallback"]
    end
    
    subgraph Graph["Cognitive Graph (turbomemory_graph)"]
        MemoryGraph["MemoryGraph"]
        Spreading["SpreadingActivation"]
        BM25["BM25 Index"]
        CCS["CompressedCognitiveState"]
        Compressor["CognitiveCompressor"]
        Vocab["ConceptVocabulary"]
    end
    
    subgraph Core["Math Core (turbomemory_core)"]
        SIMD["SIMD Kernels (AVX2/NEON)"]
        FWHT["FWHT Preconditioning"]
        Quantizers["Quantizers (Scalar/Sign/TurboQuant)"]
        Metrics["Distance Metrics"]
    end
    
    PyClient --> PyEngine
    RESTClient --> Axum
    gRPCClient --> Tonic
    Axum --> Service
    Tonic --> Service
    Service --> Engine
    PyEngine --> PyNumpy
    PyEngine --> PyGIL
    PyEngine --> Engine
    
    Engine --> Segments
    Engine --> VectorStore
    Engine --> WAL
    Engine --> redb
    Engine --> Optimizer
    Engine --> ScopeIndex
    Engine --> TextIndex
    Engine --> PayloadIndex
    Engine --> MemoryGraph
    Engine --> GpuBackend
    
    Segments --> Hot
    Segments --> SealedHot
    Segments --> Warm
    Segments --> Cold
    
    Hot --> SIMD
    SealedHot --> CudaAnnIndex
    Warm --> Quantizers
    Cold --> Quantizers
    
    GpuBackend --> CudaBackend
    GpuBackend --> CpuFallback
    CudaBackend --> Core
    
    MemoryGraph --> Spreading
    MemoryGraph --> BM25
    MemoryGraph --> Vocab
    Spreading --> CCS
    CCS --> Compressor
    
    Quantizers --> SIMD
    Quantizers --> FWHT
    Metrics --> SIMD
    FWHT --> Metrics
```

### 3.4 GPU-Accelerated Search Path (Opt-in via `cuda` feature)

When the `cuda` feature is enabled and a CUDA device is available, the search path can leverage GPU acceleration for exact-scan and HNSW build operations:

```mermaid
flowchart TD
    subgraph GPU_Backend["GPU Backend (turbomemory_gpu)"]
        Cuda["CudaBackend: cudarc + cuBLAS"]
        Fallback["CpuFallback: transparent fallback"]
    end
    
    subgraph Search_Ops["Search Operations"]
        Exact["Hot Segment Exact Scan"]
        Rerank["Quantized Tier Candidate Rerank"]
        HNSW["SealedHot HNSW Build"]
    end
    
    Cuda --"sgemv batched dot"--> Exact
    Cuda --"sgemv batched dot"--> Rerank
    Cuda --"CudaAnnIndex build"--> HNSW
    Fallback -."any CUDA error".- Exact
    Fallback -."any CUDA error".- Rerank
    Fallback -."any CUDA error".- HNSW
```

**GPU Path Design Principles:**
1. **Trait-based backend**: `GpuBackend` trait allows future Vulkan/ROCm/Metal implementations without touching storage code.
2. **Silent CPU fallback**: Every GPU operation falls back to CPU on error (CUDA unavailable, OOM, kernel error).
3. **Opt-in only**: GPU acceleration is only active when the `cuda` feature is enabled at compile time AND a CUDA device is detected at runtime.
4. **HNSW build focus**: GPU accelerates the HNSW graph build (the clear GPU win), not single-query search (upload overhead dominates for single queries).

---

## 4. Subsystem Documentation Links

For in-depth explanations of specific features, browse the detailed sub-documents:

1. [**Core Math, SIMD, and Quantization Subsystem**](file:///d:/personal-projects/TurboSuperMemory/docs/core_quantization.md)
   * Hard-level SIMD routines (AVX2/NEON), preconditioning approximate rotations, Lloyd-Max tables, and TurboQuant MSE/Prod quantizers.
2. [**Storage, Durability, and Segment Tiers Subsystem**](file:///d:/personal-projects/TurboSuperMemory/docs/storage_persistence.md)
   * The Write-Ahead Log (WAL) durability, `redb` lazy snapshot persistence, segmented search execution, background optimization, and thread-safe concurrency.
3. [**Cognitive Memory Graph Subsystem**](file:///d:/personal-projects/TurboSuperMemory/docs/cognitive_graph.md)
   * Episodic-semantic graph nodes/edges, Spreading Activation, Feeling-of-Knowing (FOK) gating, pluggable CCS working memory compaction, synonym vocabulary evolution, and automatic importance scoring.
4. [**Python Bindings and API Services Subsystem**](file:///d:/personal-projects/TurboSuperMemory/docs/bindings_api.md)
   * PyO3 binding structures, zero-copy NumPy array operations, thread GIL releases, Tonic gRPC, and Axum REST controllers.
5. [**GPU Acceleration Subsystem**](file:///d:/personal-projects/TurboSuperMemory/docs/gpu_acceleration.md)
   * Trait-based GPU backend design, CUDA implementation with cuBLAS batched distance compute, custom HNSW build algorithm, and transparent CPU fallback architecture.

---

## 5. Roadmap and Development Status

The development of TurboSuperMemory follows a structured progression outlined in [TODO.md](file:///d:/personal-projects/TurboSuperMemory/TODO.md):

* **Stage 1: Cognitive Deepening (Completed 2026-06-21)**
  * [x] **C1: Contradiction Detection** (Belief revision, weakening outdated records).
  * [x] **C2: Automatic Importance Scoring** (Salience-based moving averages, edge weight re-scaling).
  * [x] **C3: Online Concept Vocabulary Evolution** (Alias co-occurrence merging, hub suppression).
  * [x] **C4: Per-Agent Memory Scoping** (Bitmap-based multi-tenant namespace isolation).
  * [x] **C5: Real-Embedding Cognitive Scale Benchmark** (768-dim clustered GMM testing with 1000+ distractors).
  * [x] **C6: Pluggable LLM Working Memory Compressor** (Bridge PyO3 custom callbacks).
  * [x] **C7: Graph Introspection API** (Stats, concept degrees, refinements, contradictions).
  * [x] **C8: Automated Concept Extraction** (TF ranking, length bonuses, alias mapping).
* **Stage 1.5: GPU Acceleration (Completed 2026-06-21)**
  * [x] **G1: GPU Backend Trait** (`GpuBackend` with `CudaBackend` + `CpuFallback`).
  * [x] **G2: cuBLAS Batched Distance** (cuBLAS `sgemv` for exact scan and rerank).
  * [x] **G3: CUDA HNSW Build** (Custom `CudaAnnIndex` with brute-force + random projection).
  * [x] **G4: GPU Integration** (Storage engine integration, Python `gpu_accelerated` property).
* **Stage 2: Structural Scaling (In Progress / Next)**
  * [ ] **S1: Collection Sharding** (Distribute partitions).
  * [ ] **S2: Asynchronous index builds** (Improve SealedHot build performance).
  * [ ] **S3: Memory-mapped indices** (Scale storage bounds).
