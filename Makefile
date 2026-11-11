# TurboSuperMemory — build orchestration
#
# Requires:
#   - Rust toolchain pinned in rust-toolchain.toml (rustup installs it automatically)
#   - Python 3.12 with development libraries (adjust PYO3_PYTHON if needed)
#   - Optional: CUDA toolkit for GPU acceleration (set FEATURES=cuda)

# Auto-detect OS for Python path
ifeq ($(OS),Windows_NT)
    PYO3_PYTHON ?= C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe
    PYTHON ?= $(PYO3_PYTHON)
    EXE_EXT := .exe
    DLL_EXT := .dll
    PYD_EXT := .pyd
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        PYO3_PYTHON ?= $(shell which python3 2>/dev/null || which python 2>/dev/null || echo /usr/bin/python3)
        PYTHON ?= $(PYO3_PYTHON)
        EXE_EXT :=
        DLL_EXT := .so
        PYD_EXT := .so
    else ifeq ($(UNAME_S),Darwin)
        PYO3_PYTHON ?= $(shell which python3 2>/dev/null || which python 2>/dev/null || echo /usr/bin/python3)
        PYTHON ?= $(PYO3_PYTHON)
        EXE_EXT :=
        DLL_EXT := .dylib
        PYD_EXT := .so
    endif
endif

FEATURES    ?= 

# Build flags: add --features cuda if FEATURES=cuda
CARGO_FEATURES := $(if $(FEATURES),--features $(FEATURES),)

.PHONY: build build-python build-api test test-python verify audit benchmark clippy fmt clean api-server gate dev wheel

build:
	export PYO3_PYTHON="$(PYO3_PYTHON)" && cargo build --workspace $(CARGO_FEATURES)

build-python:
	export PYO3_PYTHON="$(PYO3_PYTHON)" && cargo build --release --package turbomemory_python $(CARGO_FEATURES)

dev:
	uv pip install -e .

wheel:
	maturin build --release $(if $(FEATURES),--features $(FEATURES),)

build-api:
	export PYO3_PYTHON="$(PYO3_PYTHON)" && cargo build --release --package turbomemory_api --bin turbomemory-server $(CARGO_FEATURES)

test:
	export PYO3_PYTHON="$(PYO3_PYTHON)" && cargo test --workspace $(CARGO_FEATURES)

# SDK unit suite: fake embedder/extractor against the real engine (no API
# keys, no model downloads). Also part of `make gate`.
test-python: build-python
	cp target/release/libturbomemory$(DLL_EXT) turbomemory$(PYD_EXT) 2>/dev/null || cp target/release/turbomemory$(DLL_EXT) turbomemory$(PYD_EXT) 2>/dev/null || true
	"$(PYTHON)" -m unittest discover -s tsm/tests -t .

verify: build-python
	cp target/release/libturbomemory$(DLL_EXT) turbomemory$(PYD_EXT) 2>/dev/null || cp target/release/turbomemory$(DLL_EXT) turbomemory$(PYD_EXT) 2>/dev/null || true
	"$(PYTHON)" benchmarks/verify.py

audit: build-python
	cp target/release/libturbomemory$(DLL_EXT) turbomemory$(PYD_EXT) 2>/dev/null || cp target/release/turbomemory$(DLL_EXT) turbomemory$(PYD_EXT) 2>/dev/null || true
	"$(PYTHON)" benchmarks/audit_recall.py

benchmark: build-python
	cp target/release/libturbomemory$(DLL_EXT) turbomemory$(PYD_EXT) 2>/dev/null || cp target/release/turbomemory$(DLL_EXT) turbomemory$(PYD_EXT) 2>/dev/null || true
	"$(PYTHON)" benchmarks/benchmark.py --tsm-only

clippy:
	export PYO3_PYTHON="$(PYO3_PYTHON)" && cargo clippy --workspace --all-targets $(CARGO_FEATURES) -- -D warnings

fmt:
	cargo fmt --all

# Regression gate: fmt + clippy + Rust tests + tsm SDK tests + synthetic belief
# + recall floor. Run before every commit that touches the engine or cognitive
# layer. Rebuilds the extension first so the Python checks load the current
# code. Needs no datasets, models or keys.
gate: build-python
	cp target/release/libturbomemory$(DLL_EXT) turbomemory$(PYD_EXT) 2>/dev/null || cp target/release/turbomemory$(DLL_EXT) turbomemory$(PYD_EXT) 2>/dev/null || true
	export PYO3_PYTHON="$(PYO3_PYTHON)" && "$(PYTHON)" benchmarks/regression_gate.py $(GATE_ARGS)

clean:
	cargo clean
	rm -f turbomemory$(PYD_EXT)

api-server: build-api
	export TURBO_DB_PATH=./turbo_db && \
	export TURBO_DIMENSION=768 && \
	export TURBO_GRPC_ADDR=0.0.0.0:50051 && \
	export TURBO_REST_ADDR=0.0.0.0:8080 && \
	./target/release/turbomemory-server$(EXE_EXT)
