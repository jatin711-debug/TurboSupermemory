#!/bin/bash
# Run recall audit (restart correctness test)

set -e

echo "=========================================="
echo "  Recall Audit"
echo "  (restart correctness + recall@K)"
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

# Run audit
python benchmarks/audit_recall.py

echo ""
echo "Audit complete!"
