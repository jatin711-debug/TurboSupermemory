#!/bin/bash
# Run LongMemEval quick benchmark (50 conversations)
# NOTE: Cognitive features are DISABLED - using ANN only for fast benchmarking
#       Use --cognitive flag for production-quality cognitive search

set -e

echo "=========================================="
echo "  LongMemEval Quick Benchmark"
echo "  (50 conversations, ~5-10 min)"
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
    --data-dir data/longmemeval \
    --output benchmarks/cognitive_eval/results/ \
    --quick \
    --quick-n 50

echo ""
echo "Benchmark complete! Results saved to:"
echo "  benchmarks/cognitive_eval/results/"
echo ""
echo "NOTE: This benchmark uses ANN search (fast). For cognitive search comparison,"
echo "      use: python benchmarks/cognitive_eval/run_longmemeval.py --compare-cognitive"