# TurboSuperMemory: An Operating System for AI Memory

[![Rust Version](https://img.shields.io/badge/rust-1.96%2B-blue.svg)](https://www.rust-lang.org/)
[![Python Version](https://img.shields.io/badge/python-3.12-green.svg)](https://www.python.org/)

**TurboSuperMemory** is a high-performance, production-oriented AI Memory Engine written in Rust with PyO3 Python bindings. It treats memory as a first-class persistent intelligence layer for AI agents rather than a simple vector database.

---

## 🌌 System Architecture & Topology

![TurboSuperMemory Conceptual Architecture](assets/img_2.png)

---

## 🚀 Key Features

* **Hierarchical Storage Tiering** — Hot (RAM, FP32 HNSW), Warm (mmap, scalar-quantized), and Cold (mmap, 1-bit sign-quantized) tiers with automatic demotion/promotion and background consolidation.
* **ANN + Cognitive Retrieval** — HNSW approximate search combined with BM25 lexical triggers, episodic-semantic graph spreading activation, and Feeling-of-Knowing (FOK) gating.
* **Compression-First Design** — FWHT preconditioning, Lloyd-Max tables, scalar and 1-bit sign quantization with LUT-based search.
* **Durable & Deterministic** — Records, graph, and Compressed Cognitive State (CCS) are persisted via `redb`; reloads reproduce identical rankings.
* **Agent-Native API** — Python `MemoryEngine` plus gRPC and REST servers for ingest, ANN/cognitive search, CCS updates, and consolidation triggers.

---

## 🧬 Mathematical Foundations & Techniques

### 1. Preconditioning via Fast Walsh-Hadamard Transform (FWHT)
Rotates the vector space to spread coordinate variance uniformly, enabling efficient scalar quantization.

### 2. Lloyd-Max Quantization
Pre-computed optimal centroids for a standard normal distribution at 1–4 bits, scalable to higher bit widths.

### 3. HNSW Approximate Retrieval
Uses the battle-tested HNSW graph algorithm for sub-linear nearest-neighbor search, with an exact-scan fallback for small collections to guarantee deterministic results during the MVP phase.

### 4. Episodic-Semantic Graph & Spreading Activation
Memories and concepts form a property graph. Retrieval fuses dense semantic seeds with BM25 lexical triggers and propagates activation through the graph with lateral inhibition.

### 5. Feeling-of-Knowing (FOK) Gating
Retrieval requests whose peak activation falls below a configurable threshold are rejected, preventing irrelevant context from reaching the agent.

### 6. Compressed Cognitive State (CCS)
A bounded, schema-governed working-memory state updated online each turn. The MVP ships a deterministic CCS stub; an LLM-based compressor can be plugged in later.

---

## 🛠️ Developer Guide

### Prerequisites

* Latest stable Rust (1.96+ recommended)
* Python 3.12 with development libraries
  * Set `PYO3_PYTHON` if your Python is not at the default path:
    ```powershell
    $env:PYO3_PYTHON = "C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe"
    ```

### Workspace Compilation & Testing

```bash
make test
# or
cargo test --workspace
```

### Python Bindings

```bash
make build-python
```

This produces `target/release/turbomemory.dll`. The verification/benchmark scripts copy it to `turbomemory.pyd` for import.

### End-to-End Verification

```bash
make verify
# or manually:
cp target/release/turbomemory.dll turbomemory.pyd
python verify.py
```

### Retrieval Correctness Audit

```bash
make audit
```

### Performance Benchmark

```bash
make benchmark
python benchmark.py --tsm-only --num-items 1000
```

---

## 🐍 Python Quickstart

```python
import numpy as np
import turbomemory

engine = turbomemory.MemoryEngine(
    db_path="./test_db",
    dimension=8,
    max_edges=3,
    search_list_size=5,
    outlier_count=0
)

engine.insert(
    id="mem_1",
    text="The Rust programming language is known for its memory safety, concurrency, and speed.",
    embedding=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    importance_score=1.0,
    concepts=["rust", "concurrency", "safety"]
)

results = engine.search(
    query_text="Rust safety speed",
    query_embedding=np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    top_k=2
)
print(results)

ccs_json = engine.step_session(
    user_input="What is Python used for?",
    assistant_response="Python is widely used for AI and data science."
)
print(ccs_json)

engine.trigger_consolidation()
engine.flush()
```

---

## 📁 Repository Structure

```text
├── ResearchPapers/           # Core + freshly fetched arXiv papers
├── assets/
│   └── img_2.png
├── crates/
│   ├── turbomemory_core/     # Vector math, FWHT, Lloyd-Max, scalar/sign quantization, LUT search
│   ├── turbomemory_storage/  # Tiered StorageEngine, mmap segments, WAL, consolidation worker
│   ├── turbomemory_graph/    # BM25, spreading activation, FOK gate, CCS
│   ├── turbomemory_python/   # PyO3 MemoryEngine bindings
│   └── turbomemory_api/      # gRPC (tonic) + REST (axum) server and shared service layer
├── verify.py                 # E2E integration tests
├── audit_recall.py           # Recall + restart-correctness audit
├── benchmark.py              # Performance benchmarking harness
├── Makefile                  # Build/test/verify orchestration
└── Cargo.toml                # Workspace manifest
```

---

## 📚 Research Foundations

Local papers include:

* HNSW (Malkov & Yashunin 2016)
* DiskANN / FreshDiskANN
* ScaNN
* TurboQuant
* QJL
* SYNAPSE
* SAGE
* ACC / CCS
* Mem0, MAGMA, A-Mem, MemOS, HiMem, MemGPT

The architecture intentionally borrows production patterns from **Qdrant** (segmented mmap, WAL, update/optimize/flush workers, cardinality-aware filters), **mem0** (provider adapters, entity linking, additive extraction), and **Chroma** (WAL-as-source-of-truth, separate vector/metadata segments, typed metadata columns).

---

## 🛣️ Roadmap

1. **v0.1 (MVP)** — Working Python API, durable storage, HNSW + exact fallback, cognitive graph, CCS stub. ✅
2. **v0.2** — Tiered storage (Hot/Warm/Cold), scalar/sign quantization, mmap-backed segments, background consolidation worker. ✅
3. **v0.3** — Full WAL-driven update/optimize/flush pipeline, promotion from Warm/Cold, graph merge/forget policies.
4. **v0.4** — Multi-agent scoping, distributed shard interfaces.
5. **v0.5** — GPU/Vulkan backend abstraction, advanced filtering, benchmarks on million-scale datasets.
