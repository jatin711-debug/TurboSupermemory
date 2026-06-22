#!/bin/bash
# Run full LongMemEval benchmark (500 conversations)

set -e

echo "=========================================="
echo "  LongMemEval Full Benchmark"
echo "  (500 conversations, ~30-60 min)"
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
python benchmarks/cognitive_eval/run_longmemeval.py \
    --dataset data/longmemeval \
    --output benchmarks/cognitive_eval/results/ \
    --use-cognitive

echo ""
echo "Benchmark complete! Results saved to:"
echo "  benchmarks/cognitive_eval/results/"
