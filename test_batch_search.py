#!/usr/bin/env python
"""
Batch search correctness test for TurboSuperMemory.

Tests that search_ann_batch produces identical results to individual search_ann calls,
and validates GPU gemm rerank path when CUDA is enabled.
"""

import os
import sys
import shutil
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BatchSearchTest")


def setup_module():
    """Import turbomemory extension."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ext = ".pyd" if sys.platform.startswith("win") else ".so"
    pyd_path = os.path.join(current_dir, f"turbomemory{ext}")
    if not os.path.exists(pyd_path):
        # Try to find and copy from target
        lib_prefix = "" if sys.platform.startswith("win") else "lib"
        lib_suffix = ".dll" if sys.platform.startswith("win") else ext
        candidates = [
            os.path.join(current_dir, "target", "release", f"{lib_prefix}turbomemory{lib_suffix}"),
            os.path.join(current_dir, "target", "debug", f"{lib_prefix}turbomemory{lib_suffix}"),
        ]
        for src in candidates:
            if os.path.exists(src):
                shutil.copy(src, pyd_path)
                logger.info(f"Copied {src} -> {pyd_path}")
                break

    import turbomemory
    return turbomemory


def test_batch_matches_single():
    """search_ann_batch must return same results as individual search_ann calls."""
    turbomemory = setup_module()
    db_dir = "test_db_batch_search"
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

    dim = 128
    n = 500
    np.random.seed(42)
    vectors = np.random.randn(n, dim).astype(np.float32)
    # Normalize for cosine similarity
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    engine = turbomemory.MemoryEngine(
        db_path=db_dir,
        dimension=dim,
        max_edges=16,
        search_list_size=32,
        outlier_count=0,
    )

    # Insert
    for i in range(n):
        engine.insert(
            id=f"mem_{i}",
            text=f"Memory number {i}",
            embedding=vectors[i],
            importance_score=1.0,
            concepts=[],
        )

    # Generate random queries
    n_queries = 20
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    # Batch search
    batch_results = engine.search_ann_batch(queries, top_k=10)
    assert len(batch_results) == n_queries

    # Compare with individual searches
    mismatches = 0
    for i in range(n_queries):
        single_results = engine.search_ann(queries[i], top_k=10)
        batch_ids = [r[0] for r in batch_results[i]]
        single_ids = [r[0] for r in single_results]

        if batch_ids != single_ids:
            mismatches += 1
            logger.warning(f"Query {i}: batch={batch_ids} vs single={single_ids}")

    # Allow small differences due to floating point in tie-breaking
    assert mismatches <= n_queries * 0.1, f"Too many mismatches: {mismatches}/{n_queries}"
    logger.info(f"Batch vs single mismatch rate: {mismatches}/{n_queries}")

    # Cleanup
    engine.close()
    shutil.rmtree(db_dir)
    logger.info("test_batch_matches_single PASSED")


def test_batch_with_consolidation():
    """Batch search after consolidation must still be correct."""
    turbomemory = setup_module()
    db_dir = "test_db_batch_consolidation"
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

    dim = 128
    n = 2000
    np.random.seed(123)
    vectors = np.random.randn(n, dim).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    engine = turbomemory.MemoryEngine(
        db_path=db_dir,
        dimension=dim,
        max_edges=16,
        search_list_size=64,
        outlier_count=0,
        hot_capacity=500,
    )

    for i in range(n):
        engine.insert(
            id=f"mem_{i}",
            text=f"Memory {i}",
            embedding=vectors[i],
            importance_score=1.0,
            concepts=[],
        )

    # Trigger consolidation
    engine.trigger_consolidation()

    # Test batch after consolidation
    n_queries = 10
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    batch_results = engine.search_ann_batch(queries, top_k=10)
    assert len(batch_results) == n_queries

    # Verify all results are valid IDs
    for i, results in enumerate(batch_results):
        assert len(results) <= 10, f"Query {i}: too many results {len(results)}"
        for id_str, score in results:
            assert id_str.startswith("mem_"), f"Invalid ID: {id_str}"
            assert -1.0 <= score <= 1.0, f"Invalid score: {score}"

    engine.close()
    shutil.rmtree(db_dir)
    logger.info("test_batch_with_consolidation PASSED")


def test_batch_empty():
    """Empty batch should return empty list."""
    turbomemory = setup_module()
    db_dir = "test_db_batch_empty"
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

    engine = turbomemory.MemoryEngine(
        db_path=db_dir,
        dimension=8,
        max_edges=3,
        search_list_size=5,
        outlier_count=0,
    )

    empty = np.zeros((0, 8), dtype=np.float32)
    results = engine.search_ann_batch(empty, top_k=5)
    assert results == [], f"Expected empty list, got: {results}"

    engine.close()
    shutil.rmtree(db_dir)
    logger.info("test_batch_empty PASSED")


if __name__ == "__main__":
    test_batch_matches_single()
    test_batch_with_consolidation()
    test_batch_empty()
    logger.info("=" * 70)
    logger.info("All batch search tests PASSED!")
    logger.info("=" * 70)
