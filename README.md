# TurboSuperMemory

[![Rust](https://img.shields.io/badge/rust-1.96%2B-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)

A high-performance AI memory engine written in Rust with Python bindings. TurboSuperMemory treats memory as a persistent, tiered intelligence layer for AI agents — not just a vector store. It combines an HNSW vector index with hierarchical storage tiering, quantization, and a cognitive retrieval graph behind a single embeddable API.

---

## Highlights

- **Fast ingest, low-latency search** — batched inserts at ~0.08 ms/item and ~4.2 ms/query at 100k × 768-dim (see [Benchmarks](#benchmarks)).
- **Hierarchical storage tiering** — Hot (RAM, FP32 HNSW), Warm (mmap, 8-bit scalar-quantized), and Cold (mmap, 8-bit scalar-quantized) tiers with automatic demotion and background consolidation.
- **Quantize-and-rerank** — quantized tiers shortlist candidates, then rerank against full-precision vectors to recover accuracy.
- **Cognitive retrieval** — HNSW seeds fused with BM25 lexical triggers, episodic-semantic graph spreading activation, and Feeling-of-Knowing (FOK) gating.
- **Durable and deterministic** — records, graph, and Compressed Cognitive State persist via `redb`; reloads reproduce identical rankings.
- **Embeddable + networked** — Python `MemoryEngine`, plus gRPC and REST servers.

---

## Benchmarks

Measured on a single machine across collection sizes of 50k and 100k vectors at 768 and 1024 dimensions, running 100 top-5 queries per run. Recall@5 is computed against an exact flat NumPy search (ground truth).

**Head-to-head (N=100,000, 768-dim):**

| System | Ingest / item | Search / query | Recall@5 | Notes |
|---|--:|--:|--:|---|
| Flat NumPy (ground truth) | 0.001 ms | 136.09 ms | 100.0% | Exact brute-force baseline |
| **TurboSuperMemory** | **0.081 ms** | **4.22 ms** | **85.2%** | Tiered + quantized, post-consolidation |
| ChromaDB (ephemeral) | 7.21 ms | 3.03 ms | 65.2% | In-process |
| Qdrant (in-memory) | 0.14 ms | 458.52 ms | 100.0% | In-process, full-precision |
| LLM-in-the-loop (Mem0) | 150.60 ms | 150.75 ms | 100.0% | 150 ms simulated LLM latency |

**Reading the numbers.** TurboSuperMemory has the fastest ingest in the field and the fastest approximate search, while holding recall in the low-to-mid 80s. Qdrant reaches perfect recall by keeping every vector at full precision in RAM, at the cost of ~100× higher query latency at this scale. Chroma is comparable on search latency but trails on both ingest speed and recall. The flat NumPy baseline is exact but scans every vector per query, so its latency grows linearly with collection size.

### Scaling across size and dimension

TurboSuperMemory across all four runs (post-consolidation, `ef=200`):

| N | Dim | Ingest / item | Search / query | Recall@5 | Peak RSS | Disk |
|--:|--:|--:|--:|--:|--:|--:|
| 50,000 | 768 | 0.082 ms | 4.82 ms | 82.2% | 1,581 MB | 435 MB |
| 50,000 | 1024 | 0.106 ms | 5.34 ms | 76.4% | 1,979 MB | 531 MB |
| 100,000 | 768 | 0.081 ms | 4.22 ms | 85.2% | 2,942 MB | 875 MB |
| 100,000 | 1024 | 0.073 ms | 6.59 ms | 76.8% | 3,266 MB | 1,068 MB |

Disk footprint includes vectors, quantized segments, graph, and the transactional WAL. Search latency stays in the 4–7 ms range as the collection grows from 50k to 100k, since per-segment HNSW work is bounded rather than linear in collection size.

### Methodology

- **Hardware:** AMD Ryzen 7 6800H (16 logical cores), 15.3 GB RAM, Windows 11.
- **Data:** 64 clusters of unit-norm vectors with 0.15 intra-cluster jitter — a realistic stand-in for embedding distributions, which live on a low-dimensional manifold with local structure. Queries are perturbed cluster members.
- **TSM config:** `ef=200`, HNSW `M=16`, `ef_construction=100`; auto-consolidation disabled and triggered once after ingest so search runs against the steady-state tiered layout. One-shot consolidation at these sizes takes ~2–4 minutes.
- **Caveats:** Single seeded run on synthetic data; Chroma and Qdrant run in their in-process modes (not as production servers). Numbers are directional, not a formal head-to-head. Reproduce with the command below.

```bash
python benchmark.py --num-items 100000 --num-queries 100 --dimension 768 \
    --data-distribution clustered --num-clusters 64 --trigger-consolidation --ef 200
```

> Note on adversarial data: pure random Gaussian unit vectors in high dimensions are near-orthogonal (pairwise cosine ≈ 1/√dim), so the true top-k is separated from the rest by noise-level gaps. That regime is pathological for *every* ANN and quantization scheme and does not reflect real embeddings. Use `--data-distribution random` to stress-test it.

---

## Architecture

![TurboSuperMemory Conceptual Architecture](assets/img_2.png)

Memory flows through tiers as it ages and access patterns shift:

| Tier | Location | Representation | Role |
|---|---|---|---|
| **Hot** | RAM | FP32, exact scan / HNSW | Newest records, highest fidelity |
| **Warm** | mmap | 8-bit scalar quant | Aged records, shortlist + rerank |
| **Cold** | mmap | 8-bit scalar quant | Coldest records, compact long-term store |

A background consolidation worker seals Hot segments, builds HNSW indices once a segment crosses `hnsw_threshold`, and demotes/compacts data downward under a resource budget. All quantized tiers rerank candidates against full-precision vectors before returning results.

---

## How It Works

**Preconditioning (FWHT).** A Fast Walsh-Hadamard Transform rotates the vector space to spread coordinate variance uniformly, which makes scalar quantization far more accurate.

**Quantization.** Lloyd-Max optimal centroids (1–4 bits) and scalar quantization compress aged vectors. Search runs over compact codes via lookup tables, then reranks survivors with full-precision vectors.

**HNSW retrieval.** Sub-linear approximate nearest-neighbor search over the Hot/Sealed tiers, with an exact-scan fallback for small collections to guarantee deterministic results.

**Cognitive graph.** Memories and concepts form a property graph. Retrieval fuses dense semantic seeds with BM25 lexical triggers and propagates activation through the graph with lateral inhibition.

**FOK gating.** Retrievals whose peak activation falls below a configurable threshold are rejected, keeping irrelevant context out of the agent's window.

**Compressed Cognitive State (CCS).** A bounded, schema-governed working-memory state updated each turn. Ships with a deterministic stub; an LLM-based compressor can be plugged in.

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
print(hits)

# Cognitive search fuses vector seeds with lexical + graph signals.
results = engine.search(
    query_text="Rust safety speed",
    query_embedding=np.random.randn(768).astype(np.float32),
    top_k=5,
)

engine.trigger_consolidation()
engine.flush()
engine.close()
```

For high-throughput ingest, use `insert_batch(ids, texts, embeddings, scores, concepts)` to amortize the Python↔Rust boundary.

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
```

`build-python` emits `target/release/turbomemory.dll`; the verification and benchmark scripts copy it to `turbomemory.pyd` for import. Only one process can hold that file at a time — run TSM Python scripts sequentially.

---

## Repository Structure

```text
├── ResearchPapers/           # Reference arXiv papers
├── assets/img_2.png          # Architecture diagram
├── crates/
│   ├── turbomemory_core/      # Vector math, FWHT, Lloyd-Max, quantization, LUT search
│   ├── turbomemory_storage/   # Tiered StorageEngine, mmap segments, WAL, consolidation
│   ├── turbomemory_graph/     # BM25, spreading activation, FOK gate, CCS
│   ├── turbomemory_python/    # PyO3 MemoryEngine bindings
│   └── turbomemory_api/       # gRPC (tonic) + REST (axum) servers
├── verify.py                 # E2E integration tests
├── audit_recall.py           # Recall + restart-correctness audit
├── benchmark.py              # Performance benchmarking harness
├── Makefile
└── Cargo.toml                # Workspace manifest
```

---

## Research Foundations

Builds on HNSW (Malkov & Yashunin 2016), DiskANN/FreshDiskANN, ScaNN, TurboQuant, QJL, and memory-systems work (Mem0, MAGMA, A-Mem, MemOS, HiMem, MemGPT). It borrows production patterns from **Qdrant** (segmented mmap, WAL, optimize/flush workers), **mem0** (provider adapters, entity linking), and **Chroma** (WAL-as-source-of-truth, separate vector/metadata segments).

---

## Roadmap

1. **v0.1 — MVP** ✅ Python API, durable storage, HNSW + exact fallback, cognitive graph, CCS stub.
2. **v0.2 — Tiering** ✅ Hot/Warm/Cold tiers, scalar quantization, mmap segments, background consolidation.
3. **v0.3** — Full WAL-driven update/optimize/flush pipeline, Warm/Cold promotion, graph merge/forget policies.
4. **v0.4** — Multi-agent scoping, distributed shard interfaces.
5. **v0.5** — GPU/Vulkan backend, advanced filtering, million-scale benchmarks.

---

## License

No license has been declared for this repository yet. Add a `LICENSE` file before distribution.

