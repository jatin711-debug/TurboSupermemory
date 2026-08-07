# TurboSuperMemory Agent Notes

## Toolchain And Shell

- This is a Rust 2021 workspace with six crates. There is no pinned `rust-toolchain`; the README targets stable Rust 1.96+ and the PyO3 extension uses `abi3-py312`.
- Python extension and evaluation work must use Python 3.12. Set both `PYO3_PYTHON` and `PYTHON` when overriding the interpreter; the Windows Makefile default is the machine-specific `C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe`.
- Make recipes use POSIX `export`, `cp`, and `rm` even on Windows. Run them under GNU Make with a POSIX shell (for example Git Bash/MSYS), not `nmake` or a PowerShell-only shell.
- Building `turbomemory_api` runs `tonic-build` over `crates/turbomemory_api/proto/turbomemory.proto`; `protoc` must be installed and on `PATH`.
- CUDA is opt-in: `make build-python FEATURES=cuda`. The feature propagates from `turbomemory_python` through storage to `turbomemory_gpu`; the normal build always uses the CPU fallback.

## Workspace Boundaries

- `turbomemory_core`: vector math, SIMD/FWHT, quantizers, and quantized scoring. Keep I/O and persistence out.
- `turbomemory_graph`: BM25, concept extraction, graph/reinforcement/belief semantics, spreading activation, FOK, and CCS/compressors.
- `turbomemory_gpu`: the `GpuBackend` abstraction, CPU fallback, and optional CUDA implementation.
- `turbomemory_storage`: the central `StorageEngine`, persistence, indexes, tier lifecycle, filtering, and background workers. Start at `engine.rs`; tier decisions live in `config.rs`, `segment_holder.rs`, and `optimizer.rs`.
- `turbomemory_python`: the `MemoryEngine` PyO3 facade. Keep it thin and preserve `py.allow_threads(...)` around heavy engine calls and zero-copy contiguous `float32` NumPy paths.
- `turbomemory_api`: shared behavior belongs in `service.rs`; transport conversion belongs in `grpc.rs`/`rest.rs`. Edit the proto source, never generated files under `target/`.

## Focused Commands

Run commands from the workspace root.

```bash
# Fast feedback
cargo test -p turbomemory_storage search_ann_batch_matches_single_query
cargo test -p turbomemory_storage --test crash_recovery reopen_replays_unflushed_wal
cargo test -p turbomemory_graph

# Rust verification used by the regression gate
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --exclude turbomemory_python

# Python extension and E2E
make build-python
make verify

# API
make build-api
make api-server
```

- `make test` includes the PyO3 crate and therefore needs a working Python development/link environment; use the explicit workspace command above for the normal Rust suite.
- `make verify` rebuilds the release extension and places it at repo-root `turbomemory.pyd`/`.so`. Python benchmark scripts share that artifact; run them sequentially because a loaded `.pyd` cannot be replaced on Windows.
- On Windows, storage test linking can fail with `LNK1102`; set `CARGO_PROFILE_DEV_DEBUG=0` and `CARGO_PROFILE_TEST_DEBUG=0` for the test command. The regression gate sets these automatically.
- `benchmarks/audit_recall.py` discovers the built library per-platform (`.dll`/`.so`/`.dylib`) and inserts the repo root into `sys.path`, so `make audit` is portable. Makefile recipes quote `"$(PYTHON)"`; do not unquote — POSIX shells otherwise mangle Windows paths with backslashes.

## Required Verification

- For local iteration, run the narrow crate/test first, then format check, clippy, and the Rust suite.
- For storage-engine or cognitive-layer changes, finish with `make gate`; use `make gate GATE_ARGS=--quick` only for a faster, noisier pass. The gate rebuilds the extension and runs formatting, clippy, Rust tests, synthetic belief checks, a role-filtered LongMemEval smoke test, and an ANN recall floor.
- `make gate` must run under Python 3.12 and expects the checked-in LongMemEval data plus a locally cached embedding model to be usable offline (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`).
- Full cognitive evaluations and performance benchmarks are expensive and are not substitutes for the regression gate. Their runners and prerequisites are documented in `benchmarks/cognitive_eval/README.md` and `setup.sh`.
- There is no CI workflow in this repository; local verification is the merge gate.

## Engine Invariants

- `StorageEngine::open` returns `Arc<StorageEngine>` and the engine owns its synchronization. Do not add an external mutex around it; clones share internal `Arc`s.
- Durability order is vector mmap write, metadata-only WAL append, then lazy `redb` snapshot. `flush()` drains pending seals/access counters, syncs vectors/text/metadata/segments/graph/CCS, and only then clears the WAL. Preserve this order and cover persistence changes with restart/crash tests.
- Full embeddings live in `vectors.bin`; `MetaRecord`/WAL/redb metadata must not duplicate them. Derived id, scope, payload, text, graph, and segment indexes are rebuilt or reloaded on open. Graph snapshots are stored binary (`meta_bin` table, `TMGR` magic) with legacy JSON snapshots still readable on open; the next flush rewrites them as binary.
- Search uses an exact scan at 4,096 records or fewer. Above that, immutable segment snapshots search Hot/SealedHot/Warm/Cold candidates and rerank against full-f32 vectors.
- Hot sealing only swaps the mutable segment into a pending queue. The optimizer later chooses HNSW SealedHot or quantized Warm based on thresholds/resource budget; Warm compacts to Cold after `warm_capacity`.
- Default Warm and Cold quantizers are both scalar 8-bit. TurboQuant variants require a power-of-two dimension, so they are invalid with the default dimension 768 and must return `InvalidArgument`, not panic.
- Belief revision, abstraction, auto-importance, vocabulary evolution, eviction, and deduplication are opt-in. Do not assume the README's cognitive examples are default behavior.
- Cognitive `search` may return no result when the FOK gate rejects a query; ANN search does not have that optional-result contract.
- GPU failures deliberately fall back to CPU. CUDA currently accelerates bounded HNSW construction and batched full-f32 reranking; GPU-native HNSW search and quantized-tier CUDA scan are not implemented.

## Change Routing

- Distance/SIMD/FWHT/quantization: `crates/turbomemory_core/src/{metrics,quantization,turbo_quant,metrics_quantized}.rs`.
- Retrieval, durability, filtering, or concurrency: `crates/turbomemory_storage/src/engine.rs` plus the relevant index/store module.
- Tier lifecycle/HNSW: `crates/turbomemory_storage/src/{segment_holder,optimizer}.rs` and `src/segments/`.
- Cognitive behavior: `crates/turbomemory_graph/src/{activation,graph,extract,ccs}.rs`; belief orchestration also exists in storage consolidation.
- Public Python behavior: `crates/turbomemory_python/src/lib.rs`; preserve storage-to-Python exception mapping (`ValueError` for invalid/duplicate/dimension, `KeyError` for missing ids, `RuntimeError` otherwise).
- Public server behavior: `crates/turbomemory_api/proto/turbomemory.proto` plus `src/{service,grpc,rest}.rs`. The server starts gRPC and REST together with graceful shutdown (Ctrl-C or one server's failure stops both); defaults come from `TURBO_DB_PATH`, `TURBO_DIMENSION`, `TURBO_GRPC_ADDR`, and `TURBO_REST_ADDR` in `main.rs`. Setting `TURBO_API_KEY` enables bearer-token auth on both transports; unset means open access (a wildcard bind logs a warning). REST errors are JSON (`{"error":{code,message}}`), the JSON filter DSL caps nesting at `MAX_FILTER_DEPTH = 32`, and batch inserts validate parallel-array lengths.
