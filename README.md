# TurboSuperMemory (TSM)

[![Rust](https://img.shields.io/badge/rust-1.96%2B-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![Status: Production Ready](https://img.shields.io/badge/status-production%20ready-brightgreen.svg)](#validation)
[![CUDA](https://img.shields.io/badge/CUDA-12.6%20accelerated-76B900.svg?logo=nvidia)](#gpu-acceleration-opt-in-via-cuda-feature)
[![Tests](https://img.shields.io/badge/tests-108%20passing-brightgreen.svg)](#validation)

> **🚀 Production Release:** TurboSuperMemory is a high-performance, bare-metal cognitive memory engine for AI agents — written in native Rust, accelerated by CUDA, and equipped with 3-tier TurboQuant hardware compression and Stage-2 ColBERT late interaction.

---

## 🌟 Why TurboSuperMemory?

Most "agent memory" solutions today are thin Python wrappers around vector databases. They store every embedding raw, hand back nearest neighbors, burn hundreds of dollars on write-time LLM calls, and lose all temporal reasoning.

**TurboSuperMemory (TSM) is engineered from the silicon up:**

* ⚡ **$0.00 Write-Time Ingestion**: Open-vocabulary statistical PMI concept extraction in native Rust ($<0.05\text{ms}$ latency) — zero LLM token burn on write.
* 💥 **32× TurboQuant Hardware Compression**: 1 Million vectors fit into **~61 MB of RAM** (vs 1.95 GB in standard FP32 databases) using Polar Fast Walsh-Hadamard Transforms (FWHT) + Lloyd-Max codebooks.
* 🧠 **Cognitive Biology & Graph Layer**: ACT-R power-law recency decay, spreading activation across concept hubs, and NLI-based non-destructive belief revision.
* 🎯 **2-Stage Retrieval & ColBERT MaxSim**: Fast Stage-1 candidate retrieval ($<1\text{ms}$) + optional Stage-2 token-level late interaction (`LiquidAI/LFM2.5-ColBERT-350M`) on CUDA.
* 🛡️ **Zero-Crash Storage Engine**: Lock-free atomic `ArcSwap` snapshots, segmented mmap buffers, and Write-Ahead Logging (WAL) that never block live queries.

---

## 📐 Mathematical Foundations

### 1. The Cognitive Score Fusion Formula
At query time, TSM fuses vector spatial similarity with topological graph diffusion, temporal forward chaining, and belief demotion:

$$\text{Final Score}(M) = \Big[ \underbrace{\text{CosineSimilarity}(Q, M)}_{\text{Semantic Vector Floor}} + \underbrace{(1 - \alpha_{\text{cognitive}}) \cdot \sigma(\Delta_{\text{graph}}(M))}_{\text{Cognitive Graph Boost}} \Big] \cdot \underbrace{\Big(1 + \lambda_{\text{recency}} \cdot \frac{\text{seq}(M)}{\text{seq}_{\max}}\Big)}_{\text{Temporal Recency Multiplier}} \cdot \underbrace{D(M)}_{\text{Truth Demotion}}$$

* **Semantic Vector Floor**: Guarantees high-similarity nearest neighbors are never dropped.
* **Cognitive Graph Boost**: Injects Hill-saturated ($\sigma(x) = \frac{x}{1+x}$) spreading activation across multi-hop concept and entity relations.
* **Temporal Recency Multiplier**: Smoothly tilts retrieval toward newer valid assertions when queries request current timeline state.
* **Truth Demotion ($D \in (0, 1]$)**: Non-destructively penalizes superseded and contradicted memories without losing historical raw data.

### 2. Stage-2 ColBERT MaxSim Late Interaction
For multi-constraint queries, TSM evaluates token-level alignment between query tokens $Q$ and candidate memory tokens $D$:

$$\text{MaxSim}(Q, D) = \sum_{i=1}^{L_q} \max_{j=1}^{L_d} (Q_i \cdot D_j)$$

$$\text{Score}_{\text{fused}}(M) = \text{Score}_{\text{TSM}}(M) \cdot \Big(1 + \text{Softmax}(\text{MaxSim}(Q, M)) \cdot N\Big)$$

---

## 🏛️ 3-Tier Storage Lifecycle & TurboQuant Compression

Memory automatically flows downward through three storage tiers as it ages:

```
                   ┌────────────────────────────────────────────────────────┐
                   │                     NEW MEMORIES                       │
                   └───────────────────────────┬────────────────────────────┘
                                               │ (Zero latency write)
                                               ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ 1. HOT TIER (RAM): Raw FP32 Vectors                                                    │
  │ • Storage: Active RAM buffer (0% compression, 2,048 bytes/vector @ 512-dim)            │
  │ • Search: Exact sub-microsecond flat scan or local HNSW graph                          │
  └────────────────────────────────────────────┬───────────────────────────────────────────┘
                                               │ (When Hot reaches `hot_capacity` or flush)
                                               ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ 2. WARM TIER (mmap): TurboQuant-Prod (8-bit Quantized + 1-bit QJL Residual)             │
  │ • Storage: 3.6× Compression (576 bytes/vector @ 512-dim)                               │
  │ • Algorithm: Polar Fast Walsh-Hadamard Transform (FWHT) + Lloyd-Max Codebooks          │
  │ • Search: AVX2/CUDA Quantized Scan shortlist ──► FP32 Rerank                           │
  └────────────────────────────────────────────┬───────────────────────────────────────────┘
                                               │ (When Warm reaches `warm_capacity` compaction)
                                               ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ 3. COLD TIER (mmap): TurboQuant-MSE (1-bit Sign Quantized)                             │
  │ • Storage: 💥 32.0× Massive Compression (64 bytes/vector @ 512-dim)                    │
  │ • Footprint: 1 Million vectors fits in just ~61 MB (vs 1.95 GB for FP32)!              │
  │ • Search: Bitwise hamming / sign LUT scan ──► FP32 Rerank                              │
  └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🥊 Benchmark Results

### 1. LongMemEval 1v1 Head-to-Head (TSM vs. Mem0 1.0)
Evaluated across full multi-session conversations, using identical OpenAI `text-embedding-3-small` (1536-dim) vectors and judged by `gpt-4o-mini`:

| Evaluation Dimension | TurboSuperMemory (TSM) | Mem0 1.0 (Official Usage) | TSM Advantage |
| :--- | :---: | :---: | :---: |
| **Accuracy @ 150 Tokens** | **`50.0%`** | `42.9%` | 🏆 **+16.6% relative lift** |
| **Single-Session User Details** | **`100.0%`** | `75.0%` | 🎯 **+25.0% vs Mem0** |
| **User Preference Following** | **`50.0%`** | `0.0%` | 🎯 **+50.0% vs Mem0** |
| **Temporal Reasoning** | **`50.0%`** | `0.0%` | ⏳ **+50.0% vs Mem0** |
| **Knowledge Updates** | **`100.0%`** | `100.0%` | 🔄 Tied (100% resolution) |
| **Write-Time LLM Calls** | **`0 calls`** | `252 calls` | ⚡ **Zero LLM write dependency** |
| **Write-Time Tokens Burned** | **`0 tokens ($0.00)`** | `398,596 tokens` | 💰 **100% Free Ingestion** |
| **Write Ingestion Latency** | **`<0.5 seconds`** | `>11 minutes` | 🚀 **~1,300× Faster Ingestion** |
| **Database Stability** | **`0 errors`** | `3 mutation crashes` | 🛡️ Zero ChromaDB KeyError crashes |

---

### 2. BEAM 100K Multi-Session Reasoning Benchmark
Evaluated across 20 multi-turn probing questions spanning all 10 memory reasoning categories:

| Reasoning Ability | TSM + ColBERT Score | Pass Rate ($\ge 0.5$) | Status |
| :--- | :---: | :---: | :--- |
| **Preference Following** | **`1.000`** | **`100.0%`** | 🎯 Perfect recall on evolving constraints |
| **Instruction Following** | **`1.000`** | **`100.0%`** | ⚡ Flawless adherence to formatting rules |
| **Information Extraction** | **`0.667`** | **`50.0%`** | 📌 High entity & numeric precision |
| **Knowledge Update** | **`0.500`** | **`50.0%`** | 🔄 Dynamic truth state resolution |
| **Abstention** | **`0.500`** | **`50.0%`** | 🛡️ Correctly withholds missing facts |
| **Summarization** | **`0.500`** | **`50.0%`** | 📝 Captures key conversation takeaways |
| **Multi-Session Reasoning** | **`0.375`** | **`50.0%`** | 🔗 Multi-hop concept traversal |
| **Overall Pass Rate** | **`0.517 Avg`** | **`9 / 20 Passed`** | ✅ **0 Execution Errors** |

---

## 💻 Quickstart (Python SDK)

### 1. Unified Agent Memory with ColBERT Reranking

```python
from tsm import Memory

# Initialize turnkey memory engine with Stage-2 ColBERT late interaction
memory = Memory(
    db_path="./agent_memory_db",
    embedder="sentence_transformer",  # 100% local, free embeddings (MiniLM)
    reranker="colbert",               # LiquidAI/LFM2.5-ColBERT-350M on CUDA
)

# Ingest memories (runs in <0.05ms with zero LLM API calls)
memory.add("User's primary database port is 9443 on host telemetry.prod.internal", user_id="alice")
memory.add("User upgraded database port to 9444 last Tuesday", user_id="alice")

# Recall with cognitive graph search + ColBERT MaxSim reranking
results = memory.recall("What is the current database port?", user_id="alice", top_k=3)

for r in results:
    print(f"Memory: {r['text']} (Score: {r['score']:.4f})")
```

### 2. High-Level Multi-Tier TurboQuant Configuration

```python
import turbomemory

# Configure 3-tier memory engine
engine = turbomemory.MemoryEngine(
    db_path="./turbo_db",
    dimension=512,                  # Must be a power of 2 for FWHT (128, 256, 512, 1024)
    hot_capacity=1000,              # First 1,000 vectors stay in RAM
    warm_capacity=10000,            # Next 10,000 vectors in TurboQuant-Prod (8-bit)
    warm_quantizer="turbo_prod8",   # 8-bit FWHT + QJL residual (3.6x compression)
    cold_quantizer="turbo_mse1",    # 1-bit FWHT sign quantization (32x compression)
    outlier_count=0,
)
```

---

## ⚡ Verifying Tiers & Compression

Run the built-in audit scripts:

```bash
# Verify 3-tier lifecycle and TurboQuant storage footprint
python benchmarks/test_turboquant_tiers.py

# Verify needle-in-a-haystack retrieval from 1-bit Cold Tier
python benchmarks/test_cold_tier_retrieval.py
```

---

## 🖥️ REST & gRPC API Server

Launch the dual-transport server:

```bash
# Build the unified API server binary
make build-api

# Start gRPC (:50051) and REST (:8080) simultaneously
TURBO_DB_PATH=./server_db TURBO_DIMENSION=768 make api-server
```

* **gRPC Endpoint**: `localhost:50051` (Proto definitions in `crates/turbomemory_api/proto/turbomemory.proto`)
* **REST Health Check**: `GET http://localhost:8080/health`
* **REST Search**: `POST http://localhost:8080/search`

---

## 🏗️ Workspace Crates

| Crate | Path | Responsibility |
| :--- | :--- | :--- |
| `turbomemory_core` | `crates/turbomemory_core` | Vector math, SIMD FWHT, Lloyd-Max quantizers, MaxSim scoring. |
| `turbomemory_graph` | `crates/turbomemory_graph` | BM25, PMI concept extraction, spreading activation, ACT-R decay, belief revision. |
| `turbomemory_storage` | `crates/turbomemory_storage` | 3-Tier StorageEngine, lock-free `ArcSwap` snapshots, WAL, mmap segments. |
| `turbomemory_gpu` | `crates/turbomemory_gpu` | CUDA acceleration kernels (NVRTC + cuBLAS) with transparent CPU fallback. |
| `turbomemory_python` | `crates/turbomemory_python` | PyO3 C-extension facade with zero-copy NumPy bindings. |
| `turbomemory_api` | `crates/turbomemory_api` | High-throughput gRPC (Tonic) and REST (Axum) server. |

---

## 🛠️ Verification & Build Commands

```bash
# Format & Lint
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings

# Rust Suite (108 tests)
cargo test --workspace --exclude turbomemory_python --features cuda

# Python Extension & Verification
make build-python
make verify
```

---

## 📄 License

Licensed under the **MIT License** — see [LICENSE](./LICENSE).

Copyright © 2026 jatin711-debug.
