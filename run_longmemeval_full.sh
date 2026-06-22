#!/bin/bash
# Run full LongMemEval benchmark (500 conversations)
# NOTE: Cognitive features are DISABLED - using ANN only for fast benchmarking
#       Use --cognitive flag for production-quality cognitive search

set -e

echo "=========================================="
echo "  LongMemEval Full Benchmark"
echo "  (500 conversations, ~30-60 min)"
echo "  Cognitive: DISABLED (ANN only)"
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
    --data-dir benchmarks/cognitive_eval/data \
    --output benchmarks/cognitive_eval/results/

echo ""
echo "Benchmark complete! Results saved to:"
echo "  benchmarks/cognitive_eval/results/"
echo ""
echo "NOTE: This benchmark uses ANN search (fast). For cognitive search comparison,"
echo "      use: python benchmarks/cognitive_eval/run_longmemeval.py --compare-cognitive"