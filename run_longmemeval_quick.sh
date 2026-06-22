#!/bin/bash
# Run LongMemEval quick benchmark (50 conversations)

set -e

echo "=========================================="
echo "  LongMemEval Quick Benchmark"
echo "  (50 conversations, ~5-10 min)"
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
    --limit 50

echo ""
echo "Benchmark complete! Results saved to:"
echo "  benchmarks/cognitive_eval/results/"
