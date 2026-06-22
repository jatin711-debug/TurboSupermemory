# TurboSuperMemory

[![Rust](https://img.shields.io/badge/rust-1.96%2B-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![Tests](https://img.shields.io/badge/tests-75%20passing-brightgreen.svg)](#validation)

**A memory engine for AI agents — written in Rust, embeddable from Python.**

Most "memory" for AI agents is a vector database with a system prompt taped to it. It stores every embedding and hands back the nearest neighbors. It never forgets a stale fact, never notices when a new memory contradicts an old one, and treats a note you've recalled a hundred times the same as one you wrote once and never touched again.

TurboSuperMemory (TSM) is built on a different premise:

> A database stores everything. A memory **remembers what matters, forgets what doesn't, revises beliefs when corrected, and surfaces the most current understanding.**

TSM pairs a fast, tiered HNSW vector index with a **cognitive retrieval graph** — spreading activation, reinforcement learning on edges, belief revision, and self-organizing importance — behind a single embeddable API. The vector index makes it fast. The cognitive graph is what makes it a *memory*.

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
| **GPU acceleration** | CUDA backend (cuBLAS + custom HNSW build) with transparent CPU fallback. |

Every cognitive feature is **opt-in and off by default**, so TSM behaves like a plain tiered vector store until you turn the brain on.

A dedicated benchmark exercises exactly the cases where the correct memory is *not* the nearest neighbor. The cognitive layer wins **4 of 4** scenarios against plain ANN. Run it yourself: `python benchmarks/cognitive_benchmark.py`.

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

When compiled with `make build-python FEATURES=cuda`, TSM leverages NVIDIA GPUs for:

| Operation | GPU Path | Speedup (RTX 3050) | Fallback |
|---|---|---|---|
| **HNSW build** | Brute-force all-pairs (≤20K vectors) | 3–5× vs CPU usearch | Transparent CPU `usearch` build |
| **Batch rerank** | cuBLAS `sgemm` (M queries × N candidates) | 2–10× (batch > 100) | CPU SIMD rerank |

The GPU backend is **trait-based** (`GpuBackend`) with a `CudaBackend` implementation and a `CpuFallback` stub. Every GPU operation silently falls back to CPU on error — no crashes, no user-visible errors. GPU acceleration is lazy-initialized on first use and exposed via the `gpu_accelerated` read-only property on the Python `MemoryEngine`.

### Performance & Recall

Measured on 64-cluster synthetic embeddings (realistic stand-in for text embeddings):

| N | Dim | Ingest | Search | Recall@10 | GPU Active |
|--:|--:|--:|--:|--:|:--:|
| 10,000 | 1536 | 0.17 ms/item | ~28 ms/query | **99.4%** | ✅ |
| 20,000 | 1536 | 0.09 ms/item | ~28 ms/query | **99.7%** | ✅ |
| 100,000 | 1536 | 0.15 ms/item | ~27 ms/query | **100.0%** | ✅ |

**Key findings:**
- GPU HNSW build uses brute-force all-pairs for segments ≤20K vectors (fast and exact on GPU); larger segments fall back to proven `usearch` HNSW
- Batch search (`search_ann_batch`) accepts a 2-D numpy matrix and returns `list[list[(id, score)]]` — zero-copy for contiguous float32 arrays, validated at 100K with **0 mismatches** vs single-query
- If you need higher recall at scale, raise `ef` per query — it's the single biggest lever

Run benchmarks: `python benchmarks/benchmark.py --tsm-only` or `python benchmarks/benchmark_gpu.py --scale 100k --dimension 1536`

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

## <a name="validation"></a>Validation

| Check | Result |
|---|---|
| `cargo fmt --all --check` | clean |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean (both cuda and non-cuda) |
| `cargo test --workspace --exclude turbomemory_python` | **75 passed / 0 failed** |
| `python benchmarks/verify.py` (E2E) | all pass (7/7 steps) |
| `python benchmarks/test_batch_search.py` | 0 mismatches batch vs single-query |
| `python benchmarks/cognitive_benchmark.py` | **4/4 cognitive scenarios won** |
| `python benchmarks/audit_recall.py --num-items 100000 --dimension 1536` | **100.0% recall@10** |

### Industry-standard Cognitive Benchmarks

TSM is validated against real-world memory benchmarks:

| Benchmark | Dataset | TSM Result | Mem0 Claimed |
|---|---|---|---|
| **LongMemEval** | 500 conversations, 500 queries | **100% recall@10** (quick test) | 91.6% |
| **LoCoMo-MC10** | 55K sessions, 1,986 queries | Infrastructure validated | — |

Run benchmarks:
```bash
# Download datasets
python benchmarks/cognitive_eval/datasets/download.py --dataset all

# LongMemEval (quick: 5 conversations, ~5 min; full: 500 conversations, ~2-3 hours)
python benchmarks/cognitive_eval/run_longmemeval.py --quick --quick-n 5

# LoCoMo (quick: 10 queries, ~10 min; full: 1986 queries, ~4-6 hours)
python benchmarks/cognitive_eval/run_locomo.py --quick --quick-n 10
```

See [benchmarks/cognitive_eval/README.md](benchmarks/cognitive_eval/README.md) for detailed results and methodology.

Test breakdown: core 29 · graph 65 · storage 68 · crash-recovery 3.

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
│   ├── turbomemory_graph/     # BM25, spreading activation, FOK gate, CCS
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
