# Agent Guide — TurboSuperMemory

## Build & Test

* **Rust:** `cargo test --workspace --exclude turbomemory_python`
* **Python bindings:** `make build-python` (requires Python 3.12 dev libs; set `PYO3_PYTHON` if needed)
* **E2E verification:** `make verify`
* **Recall audit:** `make audit`
* **Benchmark:** `make benchmark`

## Workspace Layout

```text
crates/
├── turbomemory_core/      # Vector math, FWHT, Lloyd-Max, quantization primitives
├── turbomemory_storage/   # MemoryStore, HNSW index, redb persistence
├── turbomemory_graph/     # BM25, episodic-semantic graph, spreading activation, CCS
└── turbomemory_python/    # PyO3 bindings exposing MemoryEngine
```

## Coding Conventions

* **Error handling:** use `thiserror` enums per crate; convert to `PyErr` in the Python binding.
* **Persistence:** `redb` is the durable source of truth for records, graph JSON, and CCS.
* **Determinism:** graph nodes/edges use `BTreeMap`/`Vec` sorted by key so reloads reproduce the same retrieval ranking.
* **Concurrency:** Python binding wraps `MemoryStore` in `Arc<Mutex<_>>`; Rust API uses `&mut self` for mutating operations.
* **Quantization:** keep compression as a pluggable metric/tier trait; the MVP uses FP32 Hot storage with scalar quantization stubs.
* **Keep it simple:** prefer exact correctness for small N; add approximations only when they are benchmarked and gated.
