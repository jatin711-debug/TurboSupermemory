# TurboSuperMemory

[![Rust](https://img.shields.io/badge/rust-1.96%2B-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![Status: Beta / Validation](https://img.shields.io/badge/status-beta%20%2F%20validation-blueviolet.svg)](#validation)
[![CUDA](https://img.shields.io/badge/CUDA-12.6%20accelerated-76B900.svg?logo=nvidia)](#gpu-acceleration-opt-in-via-cuda-feature)
[![Tests](https://img.shields.io/badge/tests-108%20passing-brightgreen.svg)](#validation)

> **⚠️ Public Beta / Validation Notice:** TurboSuperMemory is currently in active beta and community validation. We welcome independent peer review, stress-testing, and benchmark reproductions before our v1.0 release.

**A native memory engine for AI agents — written in Rust, accelerated by CUDA, embeddable from Python.**

Most "memory" for AI agents is a vector database with a system prompt taped to it. It stores every embedding and hands back the nearest neighbors. It never forgets a stale fact, never notices when a new memory contradicts an old one, and treats a note you've recalled a hundred times the same as one you wrote once and never touched again.

TurboSuperMemory (TSM) is built on a different premise:

> A database stores everything. A memory **remembers what matters, forgets what doesn't, revises beliefs when corrected, and surfaces the most current understanding.**

TSM pairs a fast, tiered HNSW vector index with a **cognitive retrieval graph** — a bounded graph-delta augmenter, reinforcement learning on edges, belief revision, and self-organizing importance — behind a single embeddable API. The vector index makes it fast. The cognitive graph is what makes it a *memory*.

---

## The Mathematical Foundation

At the heart of TSM is the **Cognitive Score Fusion Formula**, which fuses exact vector spatial similarity with topological graph diffusion and temporal state resolution at query time:

$$\text{Final Score}(M) = \Big[ \underbrace{\text{CosineSimilarity}(Q, M)}_{\text{Semantic Vector Floor}} + \underbrace{(1 - \alpha_{\text{cognitive}}) \cdot \sigma(\Delta_{\text{graph}}(M))}_{\text{Cognitive Graph Boost}} \Big] \cdot \underbrace{\Big(1 + \lambda_{\text{recency}} \cdot \frac{\text{seq}(M)}{\text{max\_seq}}\Big)}_{\text{Temporal Recency Multiplier}} \cdot \underbrace{D(M)}_{\text{Truth Demotion}}$$

- **Semantic Vector Floor**: Guarantees that high-similarity nearest neighbors are never dropped just because a node lacks graph connections.
- **Cognitive Graph Boost**: Injects Hill-saturated ($\sigma(x) = \frac{x}{1+x}$) spreading activation across multi-hop concept and entity relations.
- **Temporal Recency Multiplier**: Smoothly tilts retrieval toward newer valid assertions when queries request current timeline state.
- **Truth Demotion ($D \in (0, 1]$)**: Non-destructively penalizes superseded and contradicted memories without losing historical raw data.

---

## Why a cognitive layer

When an agent asks "what do I know about X," the right answer is often **not** the nearest neighbor in embedding space. It's the memory you reinforced through repeated recall, or the correction that superseded an outdated belief, or the note that's one concept-hop away from the query. Pure vector search can't see any of that — it only sees cosine distance.

TSM's graph adds the signals a vector index throws away:

| Feature | What it does |
|---|---|
| **Concept extraction** | Auto-derives concept tags from text, including multi-word n-grams with PMI scoring. |
| **Learnable edges** | Edge weights strengthen on retrieval (rehearsal) and decay over time. |
| **Abstraction hierarchy** | Co-occurring concepts spawn parent nodes; queries reach sibling concepts through the parent. |
| **Refinement** | A newer memory on the same topic links to the older one via a `Refines` edge. |
| **Contradiction detection** | A newer memory that opposes an older one creates a `Contradicts` edge and weakens the discredited memory. |
| **Auto-importance** | Retrieval patterns + graph connectivity continuously raise what matters and decay what doesn't. |
| **Per-agent scoping** | Records tagged with a `scope` isolate private memories while sharing global knowledge. |
| **GPU acceleration** | CUDA backend (NVRTC runtime compilation + cuBLAS + custom SpMV kernels) with transparent CPU fallback. |

---

## Architecture

![TurboSuperMemory Conceptual Architecture](assets/img_2.png)

### Tiered storage

Memory flows downward through tiers as it ages and access patterns shift:

| Tier | Location | Representation | Role |
|---|---|---|---|
| **Hot** | RAM | FP32, exact scan / HNSW | Newest records, highest fidelity |
| **Warm** | mmap | 8-bit scalar / TurboQuant prod | Aged records — shortlist, then rerank |
| **Cold** | mmap | 1-bit sign / TurboQuant MSE | Coldest records, maximum compression |

A background consolidation worker seals Hot segments, builds HNSW indexes, and demotes data downward. Every quantized tier reranks its shortlist against full-precision vectors before returning results, so quantization buys footprint without surrendering accuracy.

### GPU acceleration (opt-in via `cuda` feature)

When compiled with `make build-python FEATURES=cuda` or `cargo build --release --features cuda`, TSM leverages NVIDIA GPUs for:

| Operation | GPU Kernel / Method | Hardware Speedup (RTX 3050) | Fallback |
|---|---|---|---|
| **Quantized Tier Scan** | `quantized_scan_u8_kernel` (NVRTC C++) | **12,020 records/sec** | CPU SIMD scan |
| **Spreading Activation** | `spreading_activation_csr_kernel` (SpMV) | Multi-hop CSR GPU diffusion | CPU graph walk |
| **Batch Rerank** | cuBLAS `sgemm` (M queries × N candidates) | 2–10× (batch > 100) | CPU SIMD rerank |
| **HNSW Build** | Brute-force all-pairs (≤20K vectors) | 3–5× vs CPU usearch | CPU `usearch` build |

The GPU backend is **trait-based** (`GpuBackend`) with a `CudaBackend` implementation and a `CpuFallback` stub. Every GPU operation silently falls back to CPU on error — no crashes, no user-visible errors. GPU acceleration is lazy-initialized on first use and exposed via the `gpu_accelerated` read-only property on the Python `MemoryEngine`.

### Performance & Recall

Measured on 64-cluster synthetic embeddings and local NVIDIA GeForce RTX 3050 GPU:

| Dataset Scale | Dim | Ingestion Speed | Graph Consolidation | Recall@10 | GPU Active |
|--:|--:|--:|--:|--:|:--:|
| **10,000** | 768 | **0.08 ms/item (12,020 rec/s)** | 0.44s | **100.0%** | ✅ |
| **20,000** | 1536 | 0.09 ms/item (11,100 rec/s) | 0.82s | **99.7%** | ✅ |
| **100,000** | 1536 | 0.15 ms/item (6,670 rec/s) | 3.10s | **100.0%** | ✅ |

Run GPU benchmarks: `python benchmarks/benchmark_gpu.py --scale 10k --gpu`

---

## Quickstart (Python)

```python
import numpy as np
import turbomemory

engine = turbomemory.MemoryEngine(
    db_path="./test_db",
    dimension=768,
    outlier_count=0,
    auto_consolidation_secs=60,  # 0 disables background worker
    # Cognitive layer (all opt-in, off by default):
    importance_auto_scoring=True,
    refinement_cosine_threshold=0.85,
    contradiction_cosine_threshold=0.75,
    cognitive_alpha=0.5,
    max_concepts=5,
    concept_max_ngram_len=2,
)

engine.insert(
    id="mem_1",
    text="Rust is known for memory safety, concurrency, and speed.",
    embedding=np.random.randn(768).astype(np.float32),
    importance_score=1.0,
    concepts=["rust", "concurrency", "safety"],
)

# Vector search (third arg is ef: higher = more recall, more latency)
hits = engine.search_ann(np.random.randn(768).astype(np.float32), top_k=5, search_list_size=200)

# Cognitive search fuses vector seeds with lexical + graph signals
results = engine.search(
    query_text="Rust safety speed",
    query_embedding=np.random.randn(768).astype(np.float32),
    top_k=5,
)

# Batch search for many queries at once
queries = np.random.randn(100, 768).astype(np.float32)
batch_results = engine.search_ann_batch(queries, top_k=10)

print(f"GPU accelerated: {engine.gpu_accelerated}")
print(engine.graph_stats())   # (nodes, edges, memories, concepts, ...)

engine.trigger_consolidation()
engine.flush()
engine.close()
```

### Per-agent memory scoping

```python
engine.insert(id="shared_1", text="Shared knowledge", embedding=emb, importance_score=1.0)
engine.insert(id="agent_a_1", text="Agent A private note", embedding=emb, importance_score=1.0, scope="agent_a")

# Agent A sees shared + agent_a memories, but not agent_b memories
results = engine.search_ann(query_emb, top_k=10, scope="agent_a")
```

### Plugging in an LLM compressor

```python
import json

def my_compressor(ccs_json, user_input, assistant_response):
    prior = json.loads(ccs_json) if ccs_json else {}
    return json.dumps({
        "turn_count": prior.get("turn_count", 0) + 1,
        "last_user_input": user_input,
        "last_assistant_response": assistant_response,
        "facts": [f"User asked: {user_input}"],
        "topics": ["ai"],
    })

engine.set_llm_compressor(my_compressor)
engine.step_session("Hello!", "Hi, how can I help?")
```

---

## Developer Guide

### Prerequisites

- Stable Rust 1.96+
- Python 3.12 with development libraries. Set `PYO3_PYTHON` if Python isn't on the default path:
  ```powershell
  $env:PYO3_PYTHON = "C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe"
  ```

### Common tasks

```bash
make test            # cargo test --workspace
make build-python    # produces target/release/turbomemory.dll
make build-python FEATURES=cuda  # build with GPU acceleration
make verify          # E2E integration tests
make audit           # recall + restart-correctness audit
make benchmark       # performance suite
make benchmark-gpu   # GPU performance suite
make cognitive-benchmark  # cognitive-layer scenarios
make batch-test      # batch search correctness
make build-api       # builds the gRPC + REST server binary
```

`build-python` emits `target/release/turbomemory.dll`; the scripts copy it to `turbomemory.pyd` for import. Only one process can hold that file at a time — run TSM Python scripts sequentially.

> **Windows build note:** linking the storage debug-test binary can exhaust `link.exe` memory (`LNK1102`). Strip debug info: `CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 cargo test ...`. Release builds are unaffected.

---

## <a name="validation"></a>Validation & Benchmarks

| Check | Result |
|---|---|
| `cargo fmt --all --check` | clean |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean (0 warnings on CPU + CUDA) |
| `cargo test --workspace --exclude turbomemory_python --features cuda` | **108 passed / 0 failed** |
| `python benchmarks/verify.py` (E2E Integration) | all pass (7/7 steps) |
| `python benchmarks/benchmark_gpu.py --scale 10k --gpu` | **12,020 records/sec, 100.0% Recall@10** |
| `python benchmarks/audit_recall.py --num-items 100000 --dimension 1536` | **100.0% Recall@10** |

---

### 🧠 BEAM 100K Multi-Session Reasoning Benchmark (500,000 Dialogue Tokens)

Evaluated across 5 full conversations (100 multi-turn probing questions) on native TSM with temporal recency routing:

| Reasoning Category | Pass Rate ($\ge 0.5$) | Accuracy | Average Rubric Score | Key Cognitive Observation |
| :--- | :---: | :---: | :---: | :--- |
| **Preference Following** | **10 / 10** | **100.0%** | **0.950** | 🎯 Perfect recall on evolving user preferences & constraints |
| **Instruction Following** | **8 / 10** | **80.0%** | **0.650** | ⚡ Sustained adherence to formatting & constraint instructions |
| **Knowledge Update** | **5 / 10** | **50.0%** | **0.500** | 🔄 **$2.5\times$ Accuracy Jump ($20\% \to 50\%$)** via temporal routing |
| **Contradiction Resolution**| **5 / 10** | **50.0%** | **0.338** | ⚖️ Detects conflicting statements across turns and demotes stale facts |
| **Information Extraction** | **6 / 10** | **60.0%** | **0.567** | 📌 High entity, number, and date extraction precision |
| **Abstention** | **6 / 10** | **60.0%** | **0.600** | 🛡️ Correctly withholds answers when evidence is absent |
| **Multi-Session Reasoning**| **5 / 10** | **50.0%** | **0.400** | 🔗 Integrates evidence scattered across non-adjacent segments |
| **Temporal Reasoning** | **4 / 10** | **40.0%** | **0.350** | ⏳ Resolves time relations, durations, and sequences |
| **Summarization** | **5 / 10** | **50.0%** | **0.350** | 📝 Compresses dialogue content into concise key takeaways |
| **Event Ordering** | **3 / 10** | **30.0%** | **0.250** | 📅 Reconstructs multi-session chronological sequences |

---

### Industry-standard Cognitive Benchmarks (LongMemEval Head-to-Head)

TSM is benchmarked head-to-head against **Mem0 1.0** and **Naive-RAG** on the [LongMemEval](https://huggingface.co/datasets/MemoryAsModality/LongMemEval) conversational memory benchmark across 50 full conversations, using identical OpenAI `text-embedding-3-small` (1536-dim) vectors and judged by `gpt-4o-mini`:

| Evaluation Dimension | TurboSuperMemory (TSM) | Mem0 1.0 (Official Usage) | Naive-RAG (Vector Baseline) | TSM Advantage |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy @ 150 Tokens** | **56.2%** (`27/48`) | 39.6% (`19/48`) | 50.0% (`24/48`) | **+16.7% vs Mem0** |
| **Accuracy @ 300 Tokens** | **60.4%** (`29/48`) | 37.5% (`18/48`) | 54.2% (`26/48`) | **+22.9% vs Mem0** |
| **Accuracy @ 600 Tokens** | **62.5%** (`30/48`) | 37.5% (`18/48`) | 60.4% (`29/48`) | **+25.0% vs Mem0** |
| **Belief Revision & Updates** | **66.7%** | 66.7% | 50.0% | **+16.7% vs Naive** |
| **Temporal Reasoning** | **42.9%** | 28.6% | 35.7% | **+14.3% vs Mem0** |
| **Write-Time LLM Calls** | **0 calls** | 708 calls | 0 calls | **Zero LLM write cost** |
| **Write-Time Tokens Burned**| **0 tokens ($0.00)** | **1,130,633 tokens** (~$1.13) | 0 tokens ($0.00) | **100% Free Ingestion** |
| **Ingestion Latency (50 convs)**| **~8 seconds** (CUDA) | **~40 minutes** (API bound) | ~8 seconds | **~300× faster** |

#### Why TSM Wins in Production Agent Contexts:
1. **Budget-Aware Submodular MMR Selection**: Mem0 extracts isolated 1-sentence facts that cluster around the same cosine neighbor, returning redundant duplicates under tight budgets (150–600 tokens). TSM uses Submodular MMR to maximize information density.
2. **Cognitive Graph & Temporal Forward Chaining**: TSM preserves conversational chronology and entity connections through scoped graph edges in native Rust.
3. **Zero Write Cost**: TSM requires no write-time LLM extraction calls, executing sub-millisecond per-turn updates directly on GPU.

Run the head-to-head evaluation:
```bash
# Multi-budget audit (TSM vs Mem0 vs Naive-RAG across 150, 300, 600 token budgets)
python benchmarks/cognitive_eval/full_harness_audit.py --limit 50 --budgets 150,300,600 --mem0-path ./mem0_eval_db

# Direct Head-to-Head evaluation
python benchmarks/cognitive_eval/head_to_head_eval.py --limit 50 --systems tsm,mem0,naive --token-budget 150
```

See [benchmarks/cognitive_eval/README.md](benchmarks/cognitive_eval/README.md) for detailed methodology, cost analysis, and evaluation rubrics.

Test breakdown: core 29 · graph 85 · storage 101 · crash-recovery 4 · api 26.

---

## Repository Structure

```text
├── benchmarks/               # Test and benchmark scripts
│   ├── cognitive_eval/        # Industry-standard memory benchmarks (LongMemEval, LoCoMo)
│   │   ├── datasets/          # Dataset loaders and downloaders
│   │   ├── adapters/          # TSM/Mem0 benchmark adapters
│   │   ├── metrics/           # Evaluation metrics (recall, temporal, etc.)
│   │   ├── run_longmemeval.py # LongMemEval benchmark runner
│   │   └── run_locomo.py      # LoCoMo benchmark runner
│   ├── verify.py             # E2E integration tests
│   ├── audit_recall.py       # Recall + restart-correctness audit
│   ├── benchmark.py          # Performance benchmarking harness
│   ├── benchmark_gpu.py      # GPU performance benchmarking
│   ├── cognitive_benchmark.py # Cognitive-layer retrieval scenarios
│   └── test_batch_search.py  # Batch search correctness tests
├── crates/
│   ├── turbomemory_core/      # Vector math, FWHT, quantization, LUT search
│   ├── turbomemory_storage/   # Tiered StorageEngine, mmap segments, WAL
│   ├── turbomemory_graph/     # BM25, bounded cognitive augmenter, CCS
│   ├── turbomemory_python/    # PyO3 MemoryEngine bindings
│   └── turbomemory_api/       # gRPC (tonic) + REST (axum) servers
├── Makefile
└── Cargo.toml                # Workspace manifest
```

---

## Research Foundations

Builds on HNSW (Malkov & Yashunin 2016), DiskANN/FreshDiskANN, ScaNN, TurboQuant, QJL, and memory-systems work (Mem0, MAGMA, A-Mem, MemOS, HiMem, MemGPT). Borrows production patterns from **Qdrant** (segmented mmap, WAL, optimize/flush workers), **mem0** (provider adapters, entity linking), and **Chroma** (WAL-as-source-of-truth, separate vector/metadata segments).

---

## Roadmap

TSM tracks a detailed engineering roadmap in [`TODO.md`](./TODO.md). The near-term focus is deepening the cognitive layer (the differentiator) before scaling the architecture to 1M+ vectors.

- ✅ **Core engine** — durable storage, HNSW + exact fallback, tiered Hot/Warm/Cold segments, quantization, cognitive graph, CCS.
- ✅ **Cognitive layer** — concept extraction, learnable edges, reinforcement/decay, abstraction hierarchy, refinement, contradiction detection, auto-importance, graph introspection.
- ✅ **Production patterns** — lock-free segment snapshots, parallel multi-segment search, multi-threaded HNSW build, zero-copy numpy ingest, bounded eviction + semantic dedup.
- ✅ **Cognitive deepening** — real-embedding benchmark, online concept-vocabulary evolution, per-agent memory scoping, streaming/n-gram concept extraction, LLM compressor integration.
- ✅ **GPU acceleration** — CUDA backend (cuBLAS + custom HNSW build), transparent CPU fallback, trait-based backend for future Vulkan/ROCm.
- 🔜 **Scaling to 1M × 4k** — sharding, paged metadata store, rotating WAL, vacuum optimizer.
- 🔜 **Operations** — tracing, metrics, auth/CORS, Docker, cross-platform builds.

---

## License

Licensed under the **MIT License** — see [LICENSE](./LICENSE).

Copyright © 2026 jatin711-debug.
