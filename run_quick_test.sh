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
    engine.add(
        id=f'test_{i}',
        embedding=embedding.tolist(),
        text=f'This is test item number {i}',
        payload={'index': i}
    )

print('Searching with ANN (fast)...')
query = np.random.randn(384).astype(np.float32)
query = query / np.linalg.norm(query)
results = engine.search(query=query.tolist(), top_k=3)

print(f'Found {len(results)} results')
for r in results:
    print(f'  - {r[\"id\"]}: score={r[\"score\"]:.4f}')

engine.close()
print('Quick test PASSED!')
print('')
print('NOTE: This test uses ANN search (fast). For cognitive search (slower but smarter),')
print('      use use_cognitive=True in adapter or --cognitive flag in benchmarks.')
"

echo ""
echo "Quick test completed successfully!"
