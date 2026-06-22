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
import tempfile
import shutil

# Use a fresh temp directory to avoid CRC mismatch from previous runs
test_db = tempfile.mkdtemp(prefix='tsm_quick_test_')
print(f'Creating engine at {test_db}...')
engine = turbomemory.MemoryEngine(test_db, 384)

print('Adding 5 test items...')
for i in range(5):
    embedding = np.random.randn(384).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    engine.insert(
        f'test_{i}',
        f'This is test item number {i}',
        embedding.tolist(),
        1.0,
        [],
        '{}'
    )

print('Searching with ANN (fast)...')
query = np.random.randn(384).astype(np.float32)
query = query / np.linalg.norm(query)
results = engine.search_ann(query.tolist(), 3)

print(f'Found {len(results)} results')
for r in results:
    print(f'  - {r[0]}: score={r[1]:.4f}')

engine.close()
shutil.rmtree(test_db, ignore_errors=True)
print('Quick test PASSED!')
print('')
print('NOTE: This test uses ANN search (fast). For cognitive search (slower but smarter),')
print('      use use_cognitive=True in adapter or --cognitive flag in benchmarks.')
"

echo ""
echo "Quick test completed successfully!"
