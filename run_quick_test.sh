#!/bin/bash
# Quick test script for TurboSuperMemory
# Tests basic functionality with 5 items
# NOTE: Cognitive features are DISABLED for benchmarks (ANN only)
#       Use --cognitive flag for production-quality search (slower)

set -e

echo "=========================================="
echo "  TurboSuperMemory Quick Test (ANN only)"
echo "  Cognitive: DISABLED for speed"
echo "=========================================="

source .venv/bin/activate

python -c "
import turbomemory
import numpy as np

print('Creating engine...')
engine = turbomemory.MemoryEngine(
    db_path='./quick_test_db',
    dimension=384
)

print('Adding 5 test items...')
for i in range(5):
    embedding = np.random.randn(384).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    engine.insert(
        id=f'test_{i}',
        text=f'This is test item number {i}',
        embedding=embedding.tolist(),
        importance_score=1.0,
        concepts=[],
        payload='{}'
    )

print('Searching with ANN (fast)...')
query = np.random.randn(384).astype(np.float32)
query = query / np.linalg.norm(query)
results = engine.search_ann(
    query.tolist(),
    3
)

print(f'Found {len(results)} results')
for r in results:
    print(f'  - {r[0]}: score={r[1]:.4f}')

engine.close()
print('Quick test PASSED!')
print('')
print('NOTE: This test uses ANN search (fast). For cognitive search (slower but smarter),')
print('      use use_cognitive=True in adapter or --cognitive flag in benchmarks.')
"

echo ""
echo "Quick test completed successfully!"
