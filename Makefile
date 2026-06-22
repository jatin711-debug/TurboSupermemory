# TurboSuperMemory — build orchestration
#
# Requires:
#   - Rust 1.75+ (tested on 1.96)
#   - Python 3.12 with development libraries (adjust PYO3_PYTHON if needed)
#   - Optional: CUDA toolkit for GPU acceleration (set FEATURES=cuda)

PYO3_PYTHON ?= C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe
PYTHON      ?= $(PYO3_PYTHON)
FEATURES    ?= 

# Build flags: add --features cuda if FEATURES=cuda
CARGO_FEATURES := $(if $(FEATURES),--features $(FEATURES),)

.PHONY: build build-python build-api test verify audit benchmark benchmark-gpu cognitive-benchmark batch-test clippy fmt clean api-server download-eval-data longmemeval locomoco compare report

build:
	export PYO3_PYTHON="$(PYO3_PYTHON)" && cargo build --workspace $(CARGO_FEATURES)

build-python:
	export PYO3_PYTHON="$(PYO3_PYTHON)" && cargo build --release --package turbomemory_python $(CARGO_FEATURES)

build-api:
	export PYO3_PYTHON="$(PYO3_PYTHON)" && cargo build --release --package turbomemory_api --bin turbomemory-server $(CARGO_FEATURES)

test:
	export PYO3_PYTHON="$(PYO3_PYTHON)" && cargo test --workspace $(CARGO_FEATURES)

verify: build-python
	cp target/release/turbomemory.dll turbomemory.pyd
	$(PYTHON) benchmarks/verify.py

audit: build-python
	cp target/release/turbomemory.dll turbomemory.pyd
	$(PYTHON) benchmarks/audit_recall.py

benchmark: build-python
	cp target/release/turbomemory.dll turbomemory.pyd
	$(PYTHON) benchmarks/benchmark.py --tsm-only

benchmark-gpu: build-python
	cp target/release/turbomemory.dll turbomemory.pyd
	$(PYTHON) benchmarks/benchmark_gpu.py

cognitive-benchmark: build-python
	cp target/release/turbomemory.dll turbomemory.pyd
	$(PYTHON) benchmarks/cognitive_benchmark.py

batch-test: build-python
	cp target/release/turbomemory.dll turbomemory.pyd
	$(PYTHON) benchmarks/test_batch_search.py

clippy:
	export PYO3_PYTHON="$(PYO3_PYTHON)" && cargo clippy --workspace --all-targets $(CARGO_FEATURES) -- -D warnings

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

# Cognitive evaluation benchmarks (LongMemEval, LoCoMo)
download-eval-data:
	$(PYTHON) benchmarks/cognitive_eval/datasets/download.py

longmemeval: build-python
	cp target/release/turbomemory.dll turbomemory.pyd
	$(PYTHON) benchmarks/cognitive_eval/run_longmemeval.py --quick

locomoco: build-python
	cp target/release/turbomemory.dll turbomemory.pyd
	$(PYTHON) benchmarks/cognitive_eval/run_locomo.py --quick

compare: build-python
	cp target/release/turbomemory.dll turbomemory.pyd
	$(PYTHON) benchmarks/cognitive_eval/run_comparison.py --quick --output benchmarks/cognitive_eval/results/

report:
	$(PYTHON) benchmarks/cognitive_eval/report.py --input benchmarks/cognitive_eval/results/ --output benchmarks/cognitive_eval/results/report.md
