# TurboSuperMemory

[![Rust](https://img.shields.io/badge/rust-1.96%2B-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![Tests](https://img.shields.io/badge/tests-165%20passing-brightgreen.svg)](#validation)

**A memory engine for AI agents — written in Rust, embeddable from Python.**

Most "memory" for AI agents is a vector database with a system prompt taped to it. It stores every embedding you give it and hands back the nearest neighbors. It never forgets a stale fact, never notices when a new memory contradicts an old one, and treats a note you've recalled a hundred times the same as one you wrote once and never touched again.

TurboSuperMemory (TSM) is built on a different premise:

> A database stores everything. A memory **remembers what matters, forgets what doesn't, revises beliefs when corrected, and surfaces the most current understanding.**

TSM pairs a fast, tiered HNSW vector index with a **cognitive retrieval graph** — spreading activation, reinforcement learning on edges, belief revision, and self-organizing importance — behind a single embeddable API. The vector index makes it fast. The cognitive graph is what makes it a *memory*.

---

## Why a cognitive layer

When an agent asks "what do I know about X," the right answer is often **not** the nearest neighbor in embedding space. It's the memory you reinforced through repeated recall, or the correction that superseded an outdated belief, or the note that's one concept-hop away from the query. Pure vector search can't see any of that — it only sees cosine distance.

TSM's graph adds the signals a vector index throws away:

| Feature | What it does | Why it matters |
|---|---|---|
| **Concept extraction** | Auto-derives concept tags from record text, including multi-word n-grams ("memory safety", "borrow checker") with PMI collocation scoring. | The graph works turnkey — callers don't have to hand-tag concepts. |
| **Learnable edges** | Edge weights derive from importance, strengthen on retrieval (rehearsal), and decay over time. | Retrieval itself is the learning signal. Frequently recalled memories get easier to recall. |
| **Abstraction hierarchy** | Co-occurring concepts (`rust` + `safety`) spawn a parent node. | A query hitting one concept can reach memories of a sibling concept. |
| **Refinement** *(belief update)* | A newer memory on the same topic (high text overlap) links to the older one via a `Refines` edge. | The current version of a fact surfaces; the old one is preserved, not deleted. |
| **Contradiction detection** *(belief revision)* | A newer memory that *opposes* an older one (same topic, low text overlap) creates a `Contradicts` edge and weakens the discredited memory. | The correction surfaces above the false claim — without erasing history. |
| **Automatic importance scoring** | Retrieval patterns + graph connectivity continuously raise what matters and decay what doesn't. | Self-organizing memory. No manual importance tagging. |
| **Graph introspection** | `graph_stats()`, `get_concepts()`, `get_memory_concepts()`, `get_refinements()`, `get_contradictions()`. | "What does the agent actually know?" — debuggable, not a black box. |
| **Online vocabulary evolution** | Merges synonymous concept nodes ("coding" → "programming") and suppresses over-general hubs ("system"). | The concept graph stays coherent as it accumulates thousands of surface forms. |
| **Per-agent memory scoping** | Records can be tagged with an agent `scope`; scoped searches return that agent's memories plus global/shared memories. | Multiple agents, assistants, or applications can share one engine while isolating private memories. |
| **Pluggable CCS compressor** | A bounded working-memory state with a deterministic compressor; an LLM compressor swaps in at runtime. | Keeps a coherent rolling summary without unbounded context growth. |

Every cognitive feature is **opt-in and off by default**, so TSM behaves like a plain tiered vector store until you turn the brain on.

A dedicated benchmark exercises exactly the cases where the correct memory is *not* the nearest neighbor — abstraction traversal, refinement surfacing, reinforcement boosting, and contradiction surfacing. The cognitive layer wins **4 of 4** of these scenarios against plain ANN. Run it yourself: `python cognitive_benchmark.py`.

---

## Architecture

![TurboSuperMemory Conceptual Architecture](assets/img_2.png)

### Tiered storage

Memory flows downward through tiers as it ages and access patterns shift:

| Tier | Location | Representation | Role |
|---|---|---|---|
| **Hot** | RAM | FP32, exact scan / HNSW | Newest records, highest fidelity |
| **Warm** | mmap | 8-bit scalar quant | Aged records — shortlist, then rerank |
| **Cold** | mmap | 8-bit scalar quant | Coldest records, compact long-term store |

A background consolidation worker seals Hot segments, builds an HNSW index once a segment crosses `hnsw_threshold`, and demotes data downward under a resource budget. Every quantized tier reranks its shortlist against full-precision vectors before returning results, so quantization buys footprint without surrendering accuracy.

Because search is dispatched **per segment** and each sealed segment carries a bounded HNSW graph, query latency tracks per-segment work rather than total collection size — adding more records grows the number of segments searched in parallel, not the cost of any single traversal.

### How it works

**Preconditioning (FWHT).** A Fast Walsh–Hadamard Transform rotates the vector space to spread coordinate variance uniformly, which makes scalar quantization far more accurate.

**Quantization.** Lloyd–Max optimal centroids (1–4 bits), scalar quantization, and TurboQuant compress aged vectors. Search runs over compact codes via lookup tables, then reranks survivors with full-precision vectors.

**HNSW retrieval.** Sub-linear approximate nearest-neighbor search over the Hot/Sealed tiers, built multi-threaded (256-point single-threaded seed, then parallel insert). An exact-scan fallback for small collections (≤ 4,096 records) guarantees deterministic results.

**Cognitive graph.** Memories and concepts form a property graph. Retrieval fuses dense semantic seeds with BM25 lexical triggers and propagates activation through the graph with lateral inhibition. Score fusion blends cosine similarity with graph activation, so reinforcement, refinement, and abstraction can re-rank memories above pure vector distance.

**Memory evolution.** On consolidation, the engine detects refinements (same topic, updated content → `Refines`) and contradictions (same topic, opposing content → `Contradicts` + weakened old memory), and rescores importance from retrieval patterns. The newer belief surfaces; history is never deleted.

**FOK gating.** Retrievals whose peak activation falls below a configurable threshold are rejected, keeping irrelevant context out of the agent's window.

**Compressed Cognitive State (CCS).** A bounded, schema-governed working-memory state updated each turn. Ships with a deterministic compressor; an LLM-based compressor can be plugged in via a trait.

---

## Quickstart (Python)

```python
import numpy as np
import turbomemory

engine = turbomemory.MemoryEngine(
    db_path="./test_db",
    dimension=768,
    max_edges=16,
    search_list_size=100,
    outlier_count=0,
    auto_consolidation_secs=60,  # 0 disables the background worker
    # Cognitive-layer knobs (all opt-in, off by default):
    importance_auto_scoring=True,        # self-organizing memory importance
    refinement_cosine_threshold=0.85,    # link newer memories that refine older ones
    contradiction_cosine_threshold=0.75, # detect + weaken contradicted beliefs
    cognitive_alpha=0.5,                 # blend cosine with graph activation
    max_concepts=5,                      # concepts per memory
    concept_max_ngram_len=2,             # extract bigrams like "memory safety"
)

engine.insert(
    id="mem_1",
    text="Rust is known for memory safety, concurrency, and speed.",
    embedding=np.random.randn(768).astype(np.float32),
    importance_score=1.0,
    concepts=["rust", "concurrency", "safety"],
)

# Approximate vector search; the third arg is ef (higher = more recall, more latency).
hits = engine.search_ann(np.random.randn(768).astype(np.float32), top_k=5, search_list_size=200)

# Cognitive search fuses vector seeds with lexical + graph signals.
results = engine.search(
    query_text="Rust safety speed",
    query_embedding=np.random.randn(768).astype(np.float32),
    top_k=5,
)

# Inspect what the engine has learned.
print(engine.graph_stats())   # (nodes, edges, memories, concepts, ...)
print(engine.get_concepts())  # [(concept, degree), ...] sorted by degree
print(engine.get_memory_concepts("mem_1"))

engine.trigger_consolidation()  # seals tiers + runs cognitive maintenance
engine.flush()
engine.close()
```

For high-throughput ingest, use `insert_batch(ids, texts, embeddings, scores, concepts)` to amortize the Python↔Rust boundary. Contiguous `float32` numpy arrays are borrowed zero-copy.

### Per-agent memory scoping

Records can be tagged with a `scope`. Scoped searches return records in that
scope plus global (un-scoped) records, so multiple agents can share one engine
while keeping private memories isolated.

```python
engine.insert(id="shared_1", text="Shared knowledge", embedding=emb, importance_score=1.0)
engine.insert(id="agent_a_1", text="Agent A private note", embedding=emb, importance_score=1.0, scope="agent_a")

# Agent A sees shared + agent_a memories, but not agent_b memories.
results = engine.search_ann(query_emb, top_k=10, scope="agent_a")
```

### Plugging in an LLM compressor

The working-memory compressor can be replaced at runtime with any Python
callable that accepts `(current_ccs_json, user_input, assistant_response)` and
returns a `CompressedCognitiveState` JSON string:

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
make verify          # end-to-end integration checks
make audit           # recall + restart-correctness audit
make benchmark       # performance suite
make build-api       # builds the gRPC + REST server binary
```

`build-python` emits `target/release/turbomemory.dll`; the verification and benchmark scripts copy it to `turbomemory.pyd` for import. Only one process can hold that file at a time — run TSM Python scripts sequentially.

> **Windows build note:** linking the storage debug-test binary can exhaust `link.exe` memory (`LNK1102`). Strip debug info for the test run as a workaround: `CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 cargo test ...`. Release builds are unaffected.

---

## <a name="validation"></a>Validation

The full verification matrix passes on the current `main`:

| Check | Result |
|---|---|
| `cargo fmt --all --check` | clean |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean |
| `cargo test --workspace --exclude turbomemory_python` | **165 passed / 0 failed** |
| `python verify.py` (E2E) | all pass |
| `python cognitive_benchmark.py` | **4/4 cognitive scenarios won** |

Test breakdown: core 29 · graph 65 · storage 68 · crash-recovery 3.

---

## Repository Structure

```text
├── ResearchPapers/           # Reference arXiv papers
├── assets/img_2.png          # Architecture diagram
├── crates/
│   ├── turbomemory_core/      # Vector math, FWHT, Lloyd-Max, quantization, LUT search
│   ├── turbomemory_storage/   # Tiered StorageEngine, mmap segments, WAL, consolidation
│   ├── turbomemory_graph/     # BM25, spreading activation, FOK gate, CCS, memory evolution
│   ├── turbomemory_python/    # PyO3 MemoryEngine bindings
│   └── turbomemory_api/       # gRPC (tonic) + REST (axum) servers
├── verify.py                 # E2E integration tests
├── audit_recall.py           # Recall + restart-correctness audit
├── benchmark.py              # Performance benchmarking harness
├── cognitive_benchmark.py    # Cognitive-layer retrieval scenarios
├── Makefile
└── Cargo.toml                # Workspace manifest
```

---

## Research Foundations

Builds on HNSW (Malkov & Yashunin 2016), DiskANN/FreshDiskANN, ScaNN, TurboQuant, QJL, and memory-systems work (Mem0, MAGMA, A-Mem, MemOS, HiMem, MemGPT). It borrows production patterns from **Qdrant** (segmented mmap, WAL, optimize/flush workers), **mem0** (provider adapters, entity linking), and **Chroma** (WAL-as-source-of-truth, separate vector/metadata segments).

---

## Roadmap

TSM tracks a detailed engineering roadmap in [`TODO.md`](./TODO.md). The near-term focus is deepening the cognitive layer (the differentiator) before scaling the architecture to 1M+ vectors.

- ✅ **Core engine** — durable storage, HNSW + exact fallback, tiered Hot/Warm/Cold segments, quantization, cognitive graph, CCS.
- ✅ **Cognitive layer** — concept extraction, learnable edges, reinforcement/decay, abstraction hierarchy, refinement, contradiction detection, automatic importance scoring, graph introspection API.
- ✅ **Production patterns** — lock-free segment snapshots, parallel multi-segment search, multi-threaded HNSW build, zero-copy numpy ingest, bounded eviction + semantic dedup.
- ✅ **Cognitive deepening** — real-embedding benchmark, online concept-vocabulary evolution, per-agent memory scoping, streaming/n-gram concept extraction, LLM compressor integration.
- 🔜 **Scaling to 1M × 4k** — sharding, paged metadata store, rotating WAL, vacuum optimizer.
- 🔜 **Operations** — tracing, metrics, auth/CORS, Docker, cross-platform builds.

---

## License

Licensed under the **MIT License** — see [LICENSE](./LICENSE).

Copyright © 2026 jatin711-debug.
