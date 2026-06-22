# Agent Guide — TurboSuperMemory

TurboSuperMemory is a high-performance, production-oriented AI Memory Engine written in Rust with PyO3 Python bindings. It treats memory as a first-class persistent intelligence layer for AI agents: dense vector retrieval (HNSW + exact fallback), BM25 lexical triggers, an episodic-semantic graph with spreading activation, tiered storage (Hot/Warm/Cold), and a Compressed Cognitive State (CCS) working-memory stub.

This guide is written for AI coding agents that need to build, test, and modify the project. Assume the reader knows nothing about the repository.

---

## 1. Technology Stack

| Layer | Technology |
|-------|------------|
| Language | Rust (edition 2021, tested on 1.96+) |
| Python bindings | PyO3 0.25 + numpy 0.25, `abi3-py312` |
| Build system | Cargo workspace + GNU Make |
| Vector index | `usearch` 2.25 (HNSW) for sealed Hot segments; plain brute-force for mutable Hot segment |
| Vector storage | mmap-backed `VectorStore` (`vectors.bin`) with CRC-validated header |
| Metadata durability | `redb` 4.1 for lazy snapshots; append-only WAL (`wal/wal_meta.bin`) is the runtime source of truth |
| Quantization | FWHT preconditioning, Lloyd-Max tables, scalar quantizer, 1-bit sign quantizer, and TurboQuant MSE/prod quantizers (configurable per tier) |
| Full-text search | Tantivy 0.22 (`text_index/`) |
| Payload filtering | In-memory Roaring bitmap index (`payload_index.rs`) |
| Concurrency | `parking_lot` RwLock/Mutex inside `StorageEngine`; Python binding holds `Arc<StorageEngine>` directly |
| Parallelism | Rayon for cross-segment search |
| API server | `tokio` + `tonic` (gRPC) + `axum` (REST) |

---

## 2. Workspace Layout

```text
crates/
├── turbomemory_core/      # Vector math, SIMD kernels, FWHT, Lloyd-Max, scalar/sign/TurboQuant quantization, LUT search
├── turbomemory_storage/   # MemoryStore/StorageEngine, tiered segments, HNSW, WAL, redb persistence, payload/text indexes
├── turbomemory_graph/     # BM25, episodic-semantic graph, spreading activation, FOK gate, CCS stub
├── turbomemory_python/    # PyO3 bindings exposing MemoryEngine
└── turbomemory_api/       # gRPC (tonic) + REST (axum) server and shared service layer
```

Key root files:

- `Cargo.toml` — workspace manifest and shared dependencies.
- `Makefile` — build/test/verify orchestration.
- `verify.py` — E2E integration tests for the Python binding.
- `audit_recall.py` — recall + restart-correctness audit.
- `benchmark.py` — performance benchmark harness (optionally compares Chroma/Qdrant/flat NumPy).
- `TODO.md` — engineering backlog with status and execution phases.

---

## 3. Build & Test Commands

All commands should be run from the project root (`D:/personal-projects/TurboSuperMemory`).

### Prerequisites

- Latest stable Rust (1.96+ recommended).
- Python 3.12 with development libraries.
- Set `PYO3_PYTHON` if Python is not at the default path. The Makefile defaults to `C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe`.

### Rust

```bash
# Build the whole workspace (excludes Python extension by default)
cargo build --workspace

# Run all Rust tests (excluding the PyO3 crate, which needs Python libs)
cargo test --workspace --exclude turbomemory_python

# Linting (must pass before merging)
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
```

### Python Bindings

```bash
make build-python
```

Produces `target/release/turbomemory.dll`; the verify/audit/benchmark scripts copy it to `turbomemory.pyd`.

### End-to-End Verification

```bash
make verify       # build-python + copy DLL + python benchmarks/verify.py
make audit        # build-python + copy DLL + python benchmarks/audit_recall.py
make benchmark    # build-python + copy DLL + python benchmarks/benchmark.py --tsm-only
make benchmark-gpu # build-python + copy DLL + python benchmarks/benchmark_gpu.py
make cognitive-benchmark # build-python + copy DLL + python benchmarks/cognitive_benchmark.py
make batch-test   # build-python + copy DLL + python benchmarks/test_batch_search.py
```

### API Server

```bash
make build-api    # builds target/release/turbomemory-server.exe
make api-server   # runs with env defaults: TURBO_DB_PATH=./turbo_db, TURBO_DIMENSION=768,
                  # TURBO_GRPC_ADDR=0.0.0.0:50051, TURBO_REST_ADDR=0.0.0.0:8080
```

### Cleanup

```bash
make clean        # cargo clean + remove turbomemory.pyd
```

---

## 4. Architecture Overview

### 4.1 Durability Model

1. Full embeddings are written to the mmap-backed `VectorStore` first.
2. A metadata-only WAL entry is appended; the WAL is the source of truth for record metadata and ordering.
3. `redb` (`memory.redb`) is a lazy snapshot; it is flushed only on explicit `flush()` / background consolidation.
4. On open, un-flushed WAL entries are replayed into the metadata cache, the snapshot is persisted, and the id index, graph, payload/text indexes, and tiered segments are rebuilt from the snapshot.

### 4.2 Storage Engine

`StorageEngine` in `crates/turbomemory_storage/src/engine.rs` is the central type. It is internally synchronized with `parking_lot` locks and is wrapped in `Arc<_>` for sharing:

- `vectors: Arc<VectorStore>` — mmap-backed dense f32 storage keyed by `PointOffset`.
- `segments: Arc<RwLock<SegmentHolder>>` — Hot, SealedHot, Warm, and Cold segments.
- `graph: Arc<RwLock<SpreadingActivation>>` — cognitive graph + BM25 + spreading activation.
- `ccs: Arc<Mutex<Option<CompressedCognitiveState>>>` — compressed cognitive state.
- `id_index: Arc<RwLock<AHashMap<Arc<str>, PointOffset>>>` — O(1) id lookup.
- `payload_index: Arc<RwLock<PayloadIndex>>` — Roaring-bitmap payload filter index.
- `text_index: Arc<TextIndex>` — Tantivy full-text index over memory text.
- `wal: Arc<Mutex<Wal>>` — append-only metadata WAL.
- `optimizer: Arc<BackgroundOptimizer>` — background seal/build/flush worker.

`StorageEngine` is `Clone` by cloning the `Arc`s; it does **not** require an external mutex.

### 4.3 Tiered Segments

| Tier | Mutability | Backing | Search |
|------|-----------|---------|--------|
| Hot | Appendable | Plain offset list + shared `VectorStore` | Exact scan (SIMD batched) |
| SealedHot | Immutable | `usearch` HNSW index file + manifest | HNSW; selective filters fall back to exact scan |
| Warm | Immutable | Quantized mmap data + manifest (scalar or TurboQuant prod) | Quantized LUT scan + full-f32 rerank |
| Cold | Immutable | Quantized mmap data + manifest (sign or TurboQuant MSE) | Binary/quantized LUT scan + full-f32 rerank |

Lifecycle: records land in Hot; when `hot_capacity` is reached the Hot segment is sealed. Large sealed segments become SealedHot (HNSW); smaller ones become Warm. When total Warm records exceed `warm_capacity`, all Warm segments are merged into a Cold segment. Frequently accessed records can be promoted back to Hot via `promote_hot`.

### 4.4 Retrieval Pipeline

1. `search_ann` / `search_ann_candidates` searches tiered segments (parallel across segments) and reranks with full f32 embeddings.
2. For collections ≤ 4,096 records the engine uses an exact flat scan for determinism.
3. `search` fuses ANN seeds with BM25 lexical triggers and propagates activation through the memory graph.
4. The Feeling-of-Knowing (FOK) gate returns `None` if peak activation is below the configured threshold.

### 4.5 API Server

`crates/turbomemory_api/src/main.rs` starts a single binary that serves:

- gRPC on `TURBO_GRPC_ADDR` (default `0.0.0.0:50051`) — see `proto/turbomemory.proto`.
- REST on `TURBO_REST_ADDR` (default `0.0.0.0:8080`) — see `rest.rs` for routes.

Shared service logic lives in `service.rs`, including filter conversion for both frontends.

---

## 5. Code Organization Conventions

### Crate Responsibilities

- `turbomemory_core`: pure math/quantization; no I/O, no concurrency, no persistence.
- `turbomemory_graph`: in-memory graph, BM25, spreading activation, CCS; serde JSON for state.
- `turbomemory_storage`: all persistence, indexing, concurrency, and the public `StorageEngine` API.
- `turbomemory_python`: thin PyO3 wrapper translating Python types/exceptions to the storage API.
- `turbomemory_api`: gRPC/REST frontends and shared service layer.

### Module Naming

- `lib.rs` is the crate root.
- Error enums are named `<Crate>Error` and live in `lib.rs`.
- `Result<T>` aliases are crate-local.
- Tests live in `#[cfg(test)] mod tests` inside source files or under `tests/` for integration tests.

### Key Types

- `PointOffset = u64` — stable dense offset used inside vector segments and the WAL.
- `Record` — full record including embedding (`Arc<[f32]>`).
- `MetaRecord` — record without embedding; kept in metadata cache and WAL.

---

## 6. Coding Style & Conventions

### Error Handling

- Use `thiserror` enums per crate (`TurboError`, `StorageError`, `ApiError`).
- Convert storage errors to Python exceptions in `turbomemory_python::storage_err`:
  - `DuplicateId` / `DimensionMismatch` / `InvalidArgument` → `ValueError`
  - `NotFound` → `KeyError`
  - everything else → `RuntimeError`
- API errors map to `tonic::Status` and `axum::http::StatusCode` in `service.rs`.

### Persistence

- `redb` is the durable source of truth for metadata snapshots, sequence counters, graph JSON, and CCS JSON.
- The WAL is the runtime source of truth and is replayed on open.
- Full embeddings live only in `VectorStore`; metadata records never duplicate them.

### Determinism

- Graph nodes and edges use `BTreeMap`/`Vec` sorted by key so reloads reproduce the same retrieval ranking.
- WAL entries carry monotonic `seq` and stable `PointOffset`.
- `SegmentHolder::search` deduplicates by offset, keeping the highest score.

### Concurrency

- `StorageEngine` uses internal locking; callers (Python, API server) hold `Arc<StorageEngine>` directly.
- Use `py.allow_threads(...)` in PyO3 methods for every heavy Rust call.
- Segment search runs in parallel with Rayon when there are multiple segments; it falls back to sequential for a single segment.
- The background optimizer uses a `Weak<StorageEngine>` so it does not keep the engine alive.

### Quantization

- Keep compression as a pluggable metric/tier trait (`Quantizer`).
- The MVP uses FP32 Hot storage with scalar-quantized Warm and sign-quantized Cold tiers by default.
- TurboQuant MSE and inner-product-optimal (`prod`) quantizers are available via `QuantizerKind` and can be assigned to Warm/Cold tiers.
- Quantized scoring uses LUT-based kernels in `metrics_quantized.rs` for scalar/sign quantizers and rotated/projected query buffers for TurboQuant.

### Simplicity

- Prefer exact correctness for small N (≤ 4,096 records uses exact scan).
- Add approximations only when they are benchmarked and gated by thresholds.

### Formatting & Linting

- Run `cargo fmt --all` before finishing changes.
- Run `cargo clippy --workspace --all-targets -- -D warnings`; warnings are treated as errors in CI-like local checks.

---

## 7. Testing Strategy

### Rust Unit Tests

Every crate uses `#[cfg(test)] mod tests` in source files. Run with:

```bash
cargo test --workspace --exclude turbomemory_python
```

Notable test areas:

- `turbomemory_core`: SIMD distance kernels against reference implementations, quantizer round-trips (scalar/sign/TurboQuant), FWHT invertibility, TurboQuant distortion vs paper bounds.
- `turbomemory_storage`: insert/search, tier sealing/reload, WAL replay, crash recovery, payload/text filtering, promotion/demotion, batch idempotency.
- `turbomemory_graph`: graph construction, BM25 scoring, spreading activation, FOK gating.

### Integration Tests

- `crates/turbomemory_storage/tests/crash_recovery.rs` — process-restart scenarios: WAL replay, tier reload, truncated WAL tolerance.

### Python Verification

- `verify.py` — structured E2E tests: ingest, spreading-activation search, FOK gate, CCS step, consolidation.
- `audit_recall.py` — measures recall@k against flat NumPy ground truth and checks restart correctness.
- `benchmark.py` — performance comparison; `--tsm-only` skips optional Chroma/Qdrant baselines.

### Benchmarks

- `crates/turbomemory_storage/benches/vector_search.rs` — Criterion benchmark for exact/HNSW/Warm/Cold paths.

### Manual API Server Smoke Test

```bash
make build-api
make api-server
# In another terminal:
curl -X POST http://localhost:8080/health
```

---

## 8. Deployment Processes

There is no automated CI/CD in this repository (no `.github/workflows`). Deployment is currently manual:

1. Build the Python extension with `make build-python`.
2. Build the server with `make build-api`.
3. Run the full verification matrix before releasing:
   - `cargo test --workspace --exclude turbomemory_python`
   - `cargo clippy --workspace --all-targets -- -D warnings`
   - `make verify`
   - `make audit`
   - `make benchmark`
4. Run the API server via `make api-server` or directly:
   ```bash
   TURBO_DB_PATH=./turbo_db TURBO_DIMENSION=768 \
     TURBO_GRPC_ADDR=0.0.0.0:50051 TURBO_REST_ADDR=0.0.0.0:8080 \
     ./target/release/turbomemory-server.exe
   ```

Database state is stored in the directory passed to `MemoryEngine`/server:

```text
<db_path>/
├── memory.redb          # redb metadata snapshot
├── vectors.bin          # mmap-backed f32 vector store
├── wal/
│   └── wal_meta.bin     # append-only metadata WAL
├── text_index/          # Tantivy index
└── segments/
    ├── sealed_hot/      # persisted usearch HNSW segments
    ├── warm/            # quantized segments (scalar or TurboQuant prod)
    └── cold/            # quantized segments (sign or TurboQuant MSE)
```

---

## 9. Security Considerations

- **Secrets/credentials**: never commit `.env`, API keys, or credentials. There are no secret-handling files in the project currently.
- **Python extension loading**: `verify.py`, `audit_recall.py`, and `benchmark.py` copy the compiled DLL into `turbomemory.pyd` in the project root. Ensure the source DLL is trusted and built from source.
- **API server**: the gRPC/REST server currently has no authentication, TLS, CORS, or request-size limits. Do not expose it to untrusted networks.
- **Payload parsing**: JSON payloads are validated as syntactically correct JSON on insert, but the engine does not enforce a schema beyond top-level field indexing.
- **File permissions**: the engine creates database directories and mmap files with default OS permissions. On multi-user systems, restrict access to the database directory.
- **Crash safety**: the WAL uses CRC32-C framed records and tolerates trailing truncation. Always call `flush()` or use the Python context manager/`close()` before shutdown to avoid losing un-snapshotted metadata.
- **Resource limits**: `VectorStore` grows the backing file geometrically. Monitor disk space; out-of-disk errors propagate as `StorageError::Io`.

---

## 10. Common Gotchas for Agents

- **Dimension must be a power of two for FWHT preconditioning**, but the storage engine does not require FWHT for normal operation; it normalizes embeddings on insert.
- **Do not wrap `StorageEngine` in an external `Mutex`** in new bindings; use `Arc<StorageEngine>` and rely on internal locks.
- **HNSW is only built for sealed segments**; the mutable Hot segment is always plain brute-force.
- **`flush()` must be called explicitly** (or via `close()`/context manager) to durably persist metadata and clear the WAL.
- **The Python binding releases the GIL** on every heavy call; do not reintroduce GIL-held Rust work.
- **`search` can return `None`** when the FOK gate rejects the query; handle it in callers.

---

## 11. Where to Start When Modifying

| Task | Start in |
|------|----------|
| Add a distance metric or SIMD kernel | `crates/turbomemory_core/src/metrics.rs` |
| Add a quantizer | `crates/turbomemory_core/src/quantization.rs` + `metrics_quantized.rs` |
| Change storage/concurrency behavior | `crates/turbomemory_storage/src/engine.rs` |
| Change tier policy or segment lifecycle | `crates/turbomemory_storage/src/segment_holder.rs` + `optimizer.rs` |
| Change HNSW build/search | `crates/turbomemory_storage/src/segments/sealed_hot.rs` |
| Change graph/retrieval semantics | `crates/turbomemory_graph/src/activation.rs` + `graph.rs` |
| Change Python API | `crates/turbomemory_python/src/lib.rs` |
| Change gRPC/REST surface | `crates/turbomemory_api/proto/turbomemory.proto` + `service.rs` + `rest.rs` + `grpc.rs` |
| Change build/test orchestration | `Makefile` + `Cargo.toml` |
