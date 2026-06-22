#!/bin/bash
# Run LoCoMo benchmark

set -e

echo "=========================================="
echo "  LoCoMo Benchmark"
echo "  (temporal reasoning test)"
echo "=========================================="

source .venv/bin/activate

# Build if needed
if [ ! -f "turbomemory.so" ]; then
    echo "Building TSM..."
    if [ "$USE_CUDA" = "true" ] || [ "$FEATURES" = "cuda" ]; then
        make FEATURES=cuda build-python
    else
        make build-python
    fi
fi

# Run benchmark
python benchmarks/cognitive_eval/run_locomo.py \
    --dataset data/locomo \
    --output benchmarks/cognitive_eval/results/ \
    --quick

echo ""
echo "Benchmark complete! Results saved to:"
echo "  benchmarks/cognitive_eval/results/"
