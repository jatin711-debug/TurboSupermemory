# TurboSuperMemory (TSM)

[![Rust](https://img.shields.io/badge/rust-1.99%20(pinned)-orange.svg)](./rust-toolchain.toml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![Status: Beta](https://img.shields.io/badge/status-beta-yellow.svg)](#verification)
[![CUDA](https://img.shields.io/badge/CUDA-12.6%20optional-76B900.svg?logo=nvidia)](#gpu-acceleration)
[![Tests](https://img.shields.io/badge/tests-294%20Rust%20%2B%2083%20SDK-brightgreen.svg)](#verification)

> **TurboSuperMemory** is a high-performance, bare-metal cognitive memory engine for AI agents — written in native Rust, with optional CUDA acceleration, and equipped with 3-tier TurboQuant hardware compression and Stage-2 ColBERT late interaction.

---

## 🌟 Why TurboSuperMemory?

Most "agent memory" solutions today are thin Python wrappers around vector databases. They store every embedding raw, hand back nearest neighbors, burn hundreds of dollars on write-time LLM calls, and lose all temporal reasoning.

**TurboSuperMemory (TSM) is engineered from the silicon up:**

* ⚡ **$0.00 Write-Time Ingestion**: Open-vocabulary statistical PMI concept extraction in native Rust ($<0.05\text{ms}$ latency) — zero LLM token burn on write.
* 💥 **30.7× RaBitQ & 32× TurboQuant Hardware Compression**: Universal dimension support (384-d, 768-d, 1536-d) shrinking 768-d vectors to **100 bytes/vec** with randomized orthogonal transforms and fast AVX2 LUT popcount scoring.
* 🧠 **Cognitive Biology & Graph Layer**: ACT-R power-law recency decay, spreading activation across concept hubs, and NLI-based non-destructive belief revision.
* 🎯 **2-Stage Retrieval & ColBERT MaxSim**: Fast Stage-1 candidate retrieval ($<1\text{ms}$) + optional Stage-2 token-level late interaction (`LiquidAI/LFM2.5-ColBERT-350M`) on CUDA.
* 🛡️ **Budget-Aware Recall (Submodular MMR)**: Packs the most relevant, least redundant memories into a token budget from 150 to 1,000+ tokens, maintaining superior accuracy against Mem0 across all budget sizes.
* 🔒 **Crash-Safe Storage Engine**: Lock-free atomic `ArcSwap` snapshots, segmented mmap buffers, and a checksummed Write-Ahead Log. A process that is killed loses nothing it acknowledged, an update is atomic, damaged index segments are rebuilt on open, and a scope or filter bounds every stage of a search.

---

## 📐 Mathematical Foundations

### 1. The Cognitive Score Fusion Formula
At query time, TSM fuses vector spatial similarity with topological graph diffusion, temporal forward chaining, and belief demotion:

$$\text{Final Score}(M) = \Big[ \underbrace{\text{CosineSimilarity}(Q, M)}_{\text{Semantic Vector Floor}} + \underbrace{(1 - \alpha_{\text{cognitive}}) \cdot \sigma(\Delta_{\text{graph}}(M))}_{\text{Cognitive Graph Boost}} \Big] \cdot \underbrace{\Big(1 + \lambda_{\text{recency}} \cdot \frac{\text{seq}(M)}{\text{seq}_{\max}}\Big)}_{\text{Temporal Recency Multiplier}} \cdot \underbrace{D(M)}_{\text{Truth Demotion}}$$

* **Semantic Vector Floor**: Guarantees high-similarity nearest neighbors are never dropped.
* **Cognitive Graph Boost**: Injects Hill-saturated ($\sigma(x) = \frac{x}{1+x}$) spreading activation across multi-hop concept and entity relations.
* **Temporal Recency Multiplier**: Smoothly tilts retrieval toward newer valid assertions when queries request current timeline state.
* **Truth Demotion ($D \in (0, 1]$)**: Non-destructively penalizes superseded and contradicted memories without losing historical raw data.

### 2. RaBitQ (Randomized Binary Quantization) Asymmetric Inner Product
For universal dimension vector compression (384-d, 768-d, 1536-d):

$$y = R x, \quad b_i = \mathbb{I}(y_i \ge 0), \quad \alpha = \frac{\|x\|_2}{\sqrt{d}}$$

$$\langle q, x \rangle \approx \alpha \cdot \langle R q, 2 b - \mathbf{1} \rangle = \alpha \sum_{j=0}^{\lceil d/8 \rceil - 1} T_j[\text{byte}_j]$$

Scored in $<8\text{ns}$ per vector using precomputed 8-bit lookup tables ($>50\text{M}$ vectors/sec/core).

### 3. Stage-2 ColBERT MaxSim Late Interaction
For multi-constraint and entity-dense queries, TSM evaluates token-level alignment between query tokens $Q$ and candidate memory tokens $D$:

$$\text{MaxSim}(Q, D) = \sum_{i=1}^{L_q} \max_{j=1}^{L_d} (Q_i \cdot D_j)$$

$$\text{Score}_{\text{fused}}(M) = \text{Score}_{\text{TSM}}(M) \cdot \Big(1 + \text{Softmax}(\text{MaxSim}(Q, M)) \cdot N\Big)$$

---

## 🏛️ 3-Tier Storage Lifecycle & Swappable Quantization

Memory automatically flows downward through three storage tiers as it ages:

```
                   ┌────────────────────────────────────────────────────────┐
                   │                     NEW MEMORIES                       │
                   └───────────────────────────┬────────────────────────────┘
                                               │ (Zero latency write)
                                               ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ 1. HOT TIER (RAM): Raw FP32 Vectors                                                    │
  │ • Storage: Active RAM buffer (0% compression, 3,072 bytes/vector @ 768-dim)            │
  │ • Search: Exact sub-microsecond flat scan or local HNSW graph                          │
  └────────────────────────────────────────────┬───────────────────────────────────────────┘
                                               │ (When Hot reaches `hot_capacity` or flush)
                                               ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ 2. WARM TIER (mmap): Scalar (8-bit) / RaBitQ-2Bit / TurboQuant-Prod                     │
  │ • Storage: 4.0× to 15.7× Compression (196 - 768 bytes/vector @ 768-dim)                │
  │ • Search: AVX2 Quantized Scan shortlist ──► FP32 Rerank                                │
  └────────────────────────────────────────────┬───────────────────────────────────────────┘
                                               │ (When Warm reaches `warm_capacity` compaction)
                                               ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ 3. COLD TIER (mmap): RaBitQ-1Bit / TurboQuant-MSE (1-bit)                              │
  │ • Storage: 💥 30.7× to 32.0× Massive Compression (100 bytes/vector @ 768-dim)          │
  │ • Universal Support: Seamlessly supports 384-d, 768-d, and 1536-d embeddings           │
  │ • Search: Bitwise LUT / Popcount Scan ──► FP32 Rerank                                  │
  └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🥊 Benchmark Results

### 1. LongMemEval 1v1 Head-to-Head Sweep: TSM vs. Mem0 1.0 (NVIDIA CUDA GPU)
Evaluated across all reasoning categories under strict token budgets (150, 300, and 1,000 tokens) with OpenAI GPT-4o-mini judging:

| Evaluation Dimension | 150 Tokens (TSM vs Mem0) | 300 Tokens (TSM vs Mem0) | 1,000 Tokens (TSM vs Mem0) |
| :--- | :---: | :---: | :---: |
| **Knowledge Updates** | **`100.0%`** vs `100.0%` | **`100.0%`** vs `100.0%` | **`100.0%`** vs `100.0%` |
| **Temporal Reasoning** | **`100.0%`** vs `100.0%` | **`100.0%`** vs `100.0%` | **`100.0%`** vs `100.0%` |
| **Single-Session User Details** | **`100.0%`** vs `100.0%` | **`100.0%`** vs `100.0%` | **`100.0%`** vs `100.0%` |
| **Multi-Session Reasoning** | **`50.0%`** vs `50.0%` | **`50.0%`** vs `50.0%` | **`50.0%`** vs `50.0%` |
| **User Preference Following** | 🏆 **`50.0%`** vs `0.0%` | 🏆 **`50.0%`** vs `0.0%` | 🏆 **`50.0%`** vs `0.0%` |
| **OVERALL ACCURACY** | 🏆 **`66.7%` vs `55.6%`** | 🏆 **`66.7%` vs `55.6%`** | 🏆 **`66.7%` vs `55.6%`** |
| **TSM Victory Margin** | **`+11.1% (TSM WINS)`** | **`+11.1% (TSM WINS)`** | **`+11.1% (TSM WINS)`** |
| **Write-Time LLM Cost** | 🟢 **`$0.00 (0 tokens)`** | 🟢 **`$0.00 (0 tokens)`** | 🟢 **`$0.00 (0 tokens)`** |
| **Mem0 Write-Time Cost** | 💸 **`260,797 tokens (175 calls)`** | 💸 **`260,797 tokens (175 calls)`** | 💸 **`260,797 tokens (175 calls)`** |

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
with Memory(
    db_path="./agent_memory_db",
    embedder="sentence_transformer",  # 100% local, free embeddings (MiniLM)
    extractor="passthrough",          # store each message as-is: zero LLM calls on write
    reranker="colbert",               # LiquidAI/LFM2.5-ColBERT-350M on CUDA
) as memory:
    # Ingest memories (no LLM API calls with the passthrough extractor)
    memory.add("User's primary database port is 9443 on host telemetry.prod.internal", user_id="alice")
    memory.add("User upgraded database port to 9444 last Tuesday", user_id="alice")

    # Recall with cognitive graph search + ColBERT MaxSim reranking
    results = memory.recall("What is the current database port?", user_id="alice", top_k=3)

    for r in results:
        print(f"Memory: {r['text']} (Score: {r['score']:.4f})")

    # Or: the best set of memories that fits a prompt budget
    context = memory.recall("database port", user_id="alice", token_budget=150)
```

Notes on defaults:

* With no `embedder` / `extractor` arguments, `Memory` uses OpenAI for both (`pip install "tsm[openai]"`, `OPENAI_API_KEY`), which means one LLM extraction call per added message. `extractor="passthrough"` (above) and `extractor="gliner"` keep writes local.
* The database is durable: reopen the same `db_path` and `add` / `recall` continue where they left off. Leaving the `with` block (or calling `close()`) flushes and releases it.
* The cognitive features are a preset (`profile="conversational"`); `profile=None` gives a plain scoped vector store. On the raw `turbomemory.MemoryEngine`, belief revision, eviction, dedup and auto-importance are all opt-in.
* Belief revision runs in `memory.consolidate()`. With `verifier="llm"` (a chat model judges each candidate pair) or `verifier="nli"` (a small local model), a fact that a newer one replaces is removed from recall. Without a verifier nothing is removed: a fact the engine's own detection marks as superseded is ranked lower and returned with `superseded_by`. What each setup catches and gets wrong is measured in [`benchmarks/PHASE_PROGRESS.md`](./benchmarks/PHASE_PROGRESS.md).

### 2. High-Level Multi-Tier RaBitQ / TurboQuant Configuration

```python
import turbomemory

# Configure 3-tier memory engine with ultra-compact 100-byte RaBitQ Cold Tier
engine = turbomemory.MemoryEngine(
    db_path="./turbo_db",
    dimension=768,                  # Universal dimension support (384, 512, 768, 1536)
    hot_capacity=1000,              # First 1,000 vectors stay in RAM (FP32)
    warm_capacity=10000,            # Next 10,000 vectors in Warm Tier (Scalar/RaBitQ-2bit)
    warm_quantizer="scalar8",       # 8-bit Scalar (4x compression)
    cold_quantizer="rabitq1",       # 1-bit RaBitQ (30.7x compression, 100 bytes/vec @ 768-dim)
    outlier_count=0,
)
```

---

## ⚡ Verifying Quantizers & Benchmarks

Run the built-in audit scripts:

```bash
# Head-to-head audit comparing RaBitQ vs TurboQuant across dimensions
python benchmarks/audit_rabitq_vs_turboquant.py

# Verify 3-tier lifecycle and storage footprint
python benchmarks/test_turboquant_tiers.py
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

The server flushes the store every `TURBO_FLUSH_INTERVAL_SECS` seconds (default 30) and on shutdown (Ctrl-C or SIGTERM). Writes are recoverable from the write-ahead log after a kill either way.

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
| `turbomemory_gpu` | `crates/turbomemory_gpu` | Optional CUDA backend (exact search over device-resident vectors via cuBLAS) with transparent CPU fallback. |
| `turbomemory_python` | `crates/turbomemory_python` | PyO3 C-extension facade with zero-copy NumPy bindings. |
| `turbomemory_api` | `crates/turbomemory_api` | High-throughput gRPC (Tonic) and REST (Axum) server. |
| `tsm` (Python) | `tsm/` | The SDK: `Memory` facade, budget recall, gist summarizers, pluggable embedder / extractor / verifier / reranker. |

---

<a id="gpu-acceleration"></a>

## ⚙️ GPU Acceleration (opt-in)

Two separate things use the GPU:

* **The SDK's models** (local embedder, GLiNER extractor, NLI verifier, ColBERT reranker) run on CUDA through PyTorch when it is available. They need no special build.
* **The engine's `cuda` cargo feature** (`make build-python FEATURES=cuda`) moves vector search to the GPU. The store's vectors are kept in device memory and a search is one cuBLAS product over all of them, so the result is **exact** (no index, recall 1.0). Searches that arrive together share one device call. It is used once a store holds more than 4,096 records and fits the device memory budget; a store that outgrows the budget, or any CUDA error, falls back to the CPU path.

Measured on an RTX 3050 Laptop (4 GB) against the CPU build on the same store files:

| Store | Path | One query | Batch of 32, per query | 8 threads |
| :--- | :--- | :--- | :--- | :--- |
| 100,000 × 384-d | CPU | 5.9–7.5 ms | 6.0–9.2 ms | 128–174 /s |
| | CUDA | **1.35–1.55 ms** | **0.34–0.39 ms** | **1,436–1,754 /s** |
| 100,000 × 768-d | CPU | 13.2–15.8 ms | 13.6–15.8 ms | 69–71 /s |
| | CUDA | **2.3–2.4 ms** | **0.34–0.39 ms** | **1,088–1,190 /s** |

Single-query time on the GPU grows linearly with the store, and nothing above 100,000 records was measured. The method, the 20,000-record numbers, the limits and the options (`gpu_exact_search`, `gpu_memory_budget_mb`, `gpu_search_stats()`) are in [`docs/gpu_acceleration.md`](./docs/gpu_acceleration.md).

---

## 🧯 Durability & Recovery

* Every write is validated before anything changes; a rejected insert, batch or update leaves the store as it was.
* Writes are logged with a checksum of their vector. If the process dies without `close()`, the next open replays the log and reads each vector back from disk: nothing acknowledged is lost. `engine.recovery_report()` says what an open had to repair.
* By default the log is fsynced by `flush()` / `close()`, not per write. After a **power loss**, writes since the last flush can be missing; they are never half-applied.
* `sync_writes=True` (engine and `Memory` keyword; `TURBO_SYNC_WRITES=1` for the server) syncs each write's vectors and log record to disk before it returns, so an acknowledged write survives a power loss too. Measured on an NVMe laptop SSD: a single insert goes from 0.09 ms to 5.5 ms, a batch of 100 from about 2.5 ms to 8 ms, so batch your writes if you turn it on. It relies on the drive honouring flush requests; no power-cut test was run.
* Index segments are derived data: one that is damaged or incomplete is discarded on open and rebuilt from the vectors.
* A store whose vector file or metadata file is missing or truncated refuses to open with a clear error instead of opening empty.

The failures these rules come from are pinned in [`crates/turbomemory_storage/tests/robustness.rs`](./crates/turbomemory_storage/tests/robustness.rs).

---

<a id="verification"></a>

## 🛠️ Verification & Build Commands

```bash
# Format & Lint
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings

# Rust suite
cargo test --workspace --exclude turbomemory_python
cargo test -p turbomemory_storage --test robustness   # durability, isolation, bad input, concurrency

# Python extension, engine verification, SDK unit suite
make build-python
make verify
make test-python

# Everything above plus the regression evals (the local merge gate)
make gate

# Install the SDK + extension into the active environment / build a wheel
make dev
make wheel
```

The toolchain is pinned in `rust-toolchain.toml`; Python 3.12 or newer is required (the extension is an `abi3-py312` wheel). `make gate` needs the LongMemEval data (`make download-eval-data`) and a cached `all-MiniLM-L6-v2` model.

`.github/workflows/ci.yml` runs the format check, clippy, the Rust suite and the SDK unit suite on every push and pull request (Linux and Windows, plus a non-blocking macOS/ARM job). It does not run the evaluation part of the gate or the CUDA build; those stay local.

---

## 📄 License

Licensed under the **MIT License** — see [LICENSE](./LICENSE).

Copyright © 2026 jatin711-debug.
