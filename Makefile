# TurboSuperMemory — build orchestration
#
# Requires:
#   - Rust 1.75+ (tested on 1.96)
#   - Python 3.12 with development libraries (adjust PYO3_PYTHON if needed)

PYO3_PYTHON ?= C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe
PYTHON      ?= $(PYO3_PYTHON)

.PHONY: build build-python build-api test verify audit benchmark clippy fmt clean api-server

build:
	cargo build --workspace

build-python:
	export PYO3_PYTHON="$(PYO3_PYTHON)" && cargo build --release --package turbomemory_python

build-api:
	export PYO3_PYTHON="$(PYO3_PYTHON)" && cargo build --release --package turbomemory_api --bin turbomemory-server

test:
	export PYO3_PYTHON="$(PYO3_PYTHON)" && cargo test --workspace

verify: build-python
	cp target/release/turbomemory.dll turbomemory.pyd
	$(PYTHON) verify.py

audit: build-python
	cp target/release/turbomemory.dll turbomemory.pyd
	$(PYTHON) audit_recall.py

benchmark: build-python
	cp target/release/turbomemory.dll turbomemory.pyd
	$(PYTHON) benchmark.py --tsm-only

clippy:
	export PYO3_PYTHON="$(PYO3_PYTHON)" && cargo clippy --workspace --all-targets -- -D warnings

fmt:
	cargo fmt --all

clean:
	cargo clean
	rm -f turbomemory.pyd

api-server: build-api
	export TURBO_DB_PATH=./turbo_db && \
	export TURBO_DIMENSION=768 && \
	export TURBO_GRPC_ADDR=0.0.0.0:50051 && \
	export TURBO_REST_ADDR=0.0.0.0:8080 && \
	./target/release/turbomemory-server.exe
