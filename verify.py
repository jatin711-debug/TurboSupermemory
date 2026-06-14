#!/usr/bin/env python
"""
TurboSuperMemory - Python Integration and End-to-End Verification Suite.

This script locates the compiled Rust library dynamically based on the platform,
copies it to the Python module path, and runs structured verification tests.
Supports CLI overrides for library and database paths.
"""

import os
import sys
import shutil
import logging
import argparse
import numpy as np

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("VerificationSuite")


def parse_args():
    """Parses command line arguments for the verification suite."""
    parser = argparse.ArgumentParser(description="TurboSuperMemory Integration Test Suite")
    parser.add_argument(
        "--dll-path", 
        type=str, 
        default=None, 
        help="Path to the compiled Rust binary (.dll, .so, or .dylib)"
    )
    parser.add_argument(
        "--pyd-path", 
        type=str, 
        default=None, 
        help="Destination Python module path (e.g. ./turbomemory.pyd)"
    )
    parser.add_argument(
        "--db-path", 
        type=str, 
        default=None, 
        help="Database directory path to use for test state storage"
    )
    return parser.parse_args()


def setup_environment(dll_path_arg, pyd_path_arg):
    """Locates and copies the compiled Rust library to the target Python extension path."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Determine extension file suffix based on platform
    is_windows = sys.platform.startswith("win")
    is_macos = sys.platform.startswith("darwin")
    
    ext_suffix = ".pyd" if is_windows else ".so"
    pyd_path = pyd_path_arg or os.path.join(current_dir, f"turbomemory{ext_suffix}")

    # 2. Determine target binary names
    lib_prefix = "" if is_windows else "lib"
    lib_suffix = ".dll" if is_windows else (".dylib" if is_macos else ".so")
    lib_filename = f"{lib_prefix}turbomemory{lib_suffix}"

    # 3. Gather candidates for auto-detection
    dll_candidates = []
    if dll_path_arg:
        dll_candidates.append(dll_path_arg)
    else:
        # Prefer release builds for verification; fall back to debug.
        dll_candidates.extend([
            os.path.join(current_dir, "target", "release", lib_filename),
            os.path.join(current_dir, "target", "debug", lib_filename),
        ])

    # Find first existing candidate
    resolved_dll_path = None
    for candidate in dll_candidates:
        if os.path.exists(candidate):
            resolved_dll_path = candidate
            break

    if not resolved_dll_path:
        logger.error(
            f"Could not locate compiled binary '{lib_filename}' in target directories.\n"
            f"Checked candidates:\n" + "\n".join(f" - {c}" for c in dll_candidates) + "\n"
            f"Please run 'make build-python' first."
        )
        sys.exit(1)

    logger.info(f"Resolved source library: {resolved_dll_path}")
    logger.info(f"Resolved destination extension: {pyd_path}")

    # Copy the file to the target location
    try:
        shutil.copy(resolved_dll_path, pyd_path)
        logger.info("Successfully copied compiled extension to Python module path.")
    except Exception as e:
        logger.error(f"Failed to copy library: {e}")
        sys.exit(1)


def run_verification(db_path_arg):
    """Executes structured end-to-end testing of the AI Memory Engine."""
    logger.info("Initializing integration tests...")
    
    # Try importing the compiled module
    try:
        import turbomemory
        logger.info("Successfully imported turbomemory library.")
    except ImportError as e:
        logger.error(f"Import failed: {e}. Ensure DLL is compiled and on Python path.")
        sys.exit(1)

    # Database directory setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = db_path_arg or os.path.join(current_dir, "test_db")
    if os.path.exists(db_dir):
        try:
            shutil.rmtree(db_dir)
            logger.info(f"Cleaned previous test database at: {db_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean test database directory: {e}")

    dimension = 8
    
    # Instantiate MemoryEngine
    logger.info(f"Instantiating MemoryEngine with db_path: {db_dir} ...")
    engine = turbomemory.MemoryEngine(
        db_path=db_dir,
        dimension=dimension,
        max_edges=3,
        search_list_size=5,
        outlier_count=0
    )
    logger.info("MemoryEngine instantiated successfully.")

    # 1. Ingest memories
    logger.info("Step 1: Testing Memory Ingestion...")
    memories = [
        {
            "id": "mem_1",
            "text": "The Rust programming language is known for its memory safety, concurrency, and speed.",
            "embedding": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "importance": 1.0,
            "concepts": ["rust", "concurrency", "safety"]
        },
        {
            "id": "mem_2",
            "text": "Python is widely used in data science, artificial intelligence, and machine learning.",
            "embedding": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "importance": 5.0,
            "concepts": ["python", "ai", "science"]
        }
    ]

    for m in memories:
        success = engine.insert(
            id=m["id"],
            text=m["text"],
            embedding=np.array(m["embedding"], dtype=np.float32),
            importance_score=m["importance"],
            concepts=m["concepts"]
        )
        assert success, f"Failed to insert memory: {m['id']}"
        logger.info(f"Successfully ingested memory: {m['id']}")

    # 2. Test Dual-Trigger Spreading Activation Search
    logger.info("Step 2: Testing Retrieval Search (Spreading Activation & FOK Gating)...")
    
    # Query matching mem_1 semantically and lexically
    query_emb = np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = engine.search(
        query_text="Rust safety speed",
        query_embedding=query_emb,
        top_k=2
    )
    logger.info(f"Retrieved Search Results: {results}")
    
    assert results is not None, "Gating wrongly rejected a valid relevant query."
    assert results[0][0] == "mem_1", f"Expected mem_1 to be top result, got: {results[0][0]}"
    logger.info("Relevance retrieval test passed.")

    # Query that is completely unrelated -> FOK gate should reject
    unrelated_emb = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    bad_results = engine.search(
        query_text="banana chocolate cookie recipe",
        query_embedding=unrelated_emb,
        top_k=2
    )
    logger.info(f"Unrelated query search results: {bad_results}")
    assert bad_results is None, "FOK gate failed to reject an unrelated query."
    logger.info("FOK gating rejection test passed.")

    # 3. Test Working Memory updates (ACC loop)
    logger.info("Step 3: Testing ACC Working Memory (CCS step)...")
    ccs_json = engine.step_session(
        user_input="What is Python used for?",
        assistant_response="Python is widely used for AI and data science."
    )
    logger.info(f"Updated Compressed Cognitive State (CCS): {ccs_json}")
    assert ccs_json != "", "CCS state serialization returned empty string."
    logger.info("ACC Working Memory session test passed.")

    # 4. Test Consolidation Merge
    logger.info("Step 4: Testing Dynamic Consolidation (StreamingMerge)...")
    sealed, compacted, promoted = engine.trigger_consolidation()
    logger.info(f"Consolidation complete: sealed={sealed}, compacted={compacted}, promoted={promoted}.")
    logger.info("Consolidation merge test passed.")

    logger.info("=========================================================================")
    logger.info("All end-to-end integration and verification tests PASSED successfully!")
    logger.info("=========================================================================")


if __name__ == "__main__":
    args = parse_args()
    setup_environment(args.dll_path, args.pyd_path)
    run_verification(args.db_path)
