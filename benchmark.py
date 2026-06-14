#!/usr/bin/env python3.12
"""
TurboSuperMemory - Performance and Recall Benchmarking Suite.

Runs empirical performance comparisons between:
1. TurboSuperMemory (Our Rust-powered 3-tier engine)
2. Flat NumPy Search (Standard vector search baseline)
3. Local ChromaDB (Ephemeral Client)
4. Local Qdrant (In-Memory client)
5. LLM-in-the-Loop Memory (Simulating Mem0/MemGPT API call overheads)
"""

import os
import sys
import time
import shutil
import argparse
import numpy as np

# Optional memory measurement
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Attempt to locate and import the compiled Python extension module
def setup_environment():
    """Locates and copies the compiled Rust library to the target Python extension path."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    is_windows = sys.platform.startswith("win")
    is_macos = sys.platform.startswith("darwin")
    
    ext_suffix = ".pyd" if is_windows else ".so"
    pyd_path = os.path.join(current_dir, f"turbomemory{ext_suffix}")

    lib_prefix = "" if is_windows else "lib"
    lib_suffix = ".dll" if is_windows else (".dylib" if is_macos else ".so")
    lib_filename = f"{lib_prefix}turbomemory{lib_suffix}"

    # Prefer release builds for benchmarking. Only fall back to debug if a
    # release artifact is not present.
    dll_candidates = [
        os.path.join(current_dir, "target", "release", lib_filename),
        os.path.join(current_dir, "target", "debug", lib_filename),
    ]

    resolved_dll_path = None
    for candidate in dll_candidates:
        if os.path.exists(candidate):
            resolved_dll_path = candidate
            break

    if not resolved_dll_path:
        print(f"Error: Could not locate compiled binary '{lib_filename}' in target directories.")
        print(f"Checked locations:\n" + "\n".join(f" - {c}" for c in dll_candidates))
        print("Please build the library first using 'make build-python' or cargo build.")
        sys.exit(1)

    try:
        shutil.copy(resolved_dll_path, pyd_path)
    except Exception as e:
        print(f"Error: Failed to copy library from {resolved_dll_path} to {pyd_path}: {e}")
        sys.exit(1)


setup_environment()

try:
    import turbomemory
except ImportError as e:
    print(f"Error: Failed to import 'turbomemory' module: {e}")
    sys.exit(1)

# Check for optional ChromaDB library
HAS_CHROMA = False
try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    pass

# Check for optional Qdrant Client library
HAS_QDRANT = False
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    HAS_QDRANT = True
except ImportError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="TurboSuperMemory Empirical Benchmarking Suite")
    parser.add_argument("--num-items", type=int, default=200, help="Number of database items to ingest")
    parser.add_argument("--num-queries", type=int, default=50, help="Number of search queries to execute")
    parser.add_argument("--dimension", type=int, default=8, help="Embedding dimension")
    parser.add_argument("--llm-delay", type=float, default=0.15, help="Simulated LLM latency per write/read (seconds)")
    parser.add_argument("--tsm-only", action="store_true", help="Only benchmark TurboSuperMemory (skip baselines)")
    return parser.parse_args()


def get_dir_size(path):
    """Recursively calculates directory size in bytes."""
    total_size = 0
    if os.path.exists(path):
        if os.path.isdir(path):
            for dirpath, _, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if not os.path.islink(fp):
                        total_size += os.path.getsize(fp)
        else:
            total_size = os.path.getsize(path)
    return total_size


def peak_memory_mb():
    """Returns the current process peak RSS in MB, or None if unavailable."""
    if not HAS_PSUTIL:
        return None
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024.0 * 1024.0)


class FlatNumPyBaseline:
    """A pure-Python/NumPy flat search baseline representing standard uncompressed vector search."""
    def __init__(self, dimension):
        self.dimension = dimension
        self.ids = []
        self.texts = []
        self.embeddings = []

    def insert(self, id, text, embedding):
        self.ids.append(id)
        self.texts.append(text)
        self.embeddings.append(embedding)
        return True

    def search(self, query_embedding, top_k):
        if not self.embeddings:
            return []
        embs = np.array(self.embeddings)
        scores = np.dot(embs, query_embedding)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.ids[i], float(scores[i])) for i in top_indices]


class LLMInTheLoopBaseline:
    """Simulates Mem0/MemGPT-style memory engines that execute LLM extraction/compaction loops."""
    def __init__(self, dimension, llm_delay):
        self.dimension = dimension
        self.llm_delay = llm_delay
        self.db = FlatNumPyBaseline(dimension)

    def insert(self, id, text, embedding):
        time.sleep(self.llm_delay)
        return self.db.insert(id, text, embedding)

    def search(self, query_embedding, top_k):
        time.sleep(self.llm_delay)
        return self.db.search(query_embedding, top_k)


if HAS_CHROMA:
    class ChromaDBBaseline:
        """Baseline wrapper around local ChromaDB Ephemeral client."""
        def __init__(self, dimension):
            self.client = chromadb.EphemeralClient()
            self.collection = self.client.create_collection("benchmark")

        def insert(self, id_str, text, embedding):
            self.collection.add(
                ids=[id_str],
                embeddings=[embedding.tolist()],
                documents=[text]
            )
            return True

        def search(self, query_embedding, top_k):
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            ids = results['ids'][0]
            distances = results['distances'][0] if results['distances'] else [0.0]*len(ids)
            return list(zip(ids, distances))


if HAS_QDRANT:
    class QdrantBaseline:
        """Baseline wrapper around local Qdrant in-memory client."""
        def __init__(self, dimension):
            self.client = QdrantClient(":memory:")
            self.client.create_collection(
                collection_name="benchmark",
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
            )
            self.current_id = 0

        def insert(self, id_str, text, embedding):
            self.client.upsert(
                collection_name="benchmark",
                points=[
                    PointStruct(
                        id=self.current_id,
                        vector=embedding.tolist(),
                        payload={"id": id_str, "text": text}
                    )
                ]
            )
            self.current_id += 1
            return True

        def search(self, query_embedding, top_k):
            search_result = self.client.search(
                collection_name="benchmark",
                query_vector=embedding.tolist(),
                limit=top_k
            )
            return [(hit.payload["id"], hit.score) for hit in search_result]


def calculate_recall(ground_truth, evaluated_results):
    matched_queries = 0
    total_expected = 0
    for i, gt_res in enumerate(ground_truth):
        eval_res = evaluated_results[i]
        gt_ids = {item[0] for item in gt_res}
        eval_ids = {item[0] for item in eval_res} if eval_res else set()
        matches = len(gt_ids.intersection(eval_ids))
        matched_queries += matches
        total_expected += len(gt_ids)
    return (matched_queries / total_expected * 100.0) if total_expected > 0 else 100.0


def run_benchmark():
    args = parse_args()
    print("=========================================================================")
    print("      TURBOSUPERMEMORY EMPIRICAL BENCHMARKING SUITE                      ")
    print("=========================================================================")
    print(f"Configuration:")
    print(f"  - Database Items : {args.num_items}")
    print(f"  - Query Count    : {args.num_queries}")
    print(f"  - Vector Dim     : {args.dimension}")
    print(f"  - LLM Latency    : {args.llm_delay * 1000:.1f} ms")
    print(f"  - Qdrant Status  : {'AVAILABLE (In-Memory)' if HAS_QDRANT else 'NOT INSTALLED'}")
    if args.tsm_only:
        print("  - Mode           : TurboSuperMemory only")
    print("=========================================================================\n")

    # Generate synthetic dataset
    np.random.seed(42)
    raw_embeddings = np.random.randn(args.num_items, args.dimension).astype(np.float32)
    norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
    embeddings = raw_embeddings / np.where(norms == 0, 1.0, norms)
    
    raw_queries = np.random.randn(args.num_queries, args.dimension).astype(np.float32)
    q_norms = np.linalg.norm(raw_queries, axis=1, keepdims=True)
    queries = raw_queries / np.where(q_norms == 0, 1.0, q_norms)

    # Initialize databases
    db_dir = "./benchmark_test_db"
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

    # HNSW parameters aligned with Malkov & Yashunin (2016): M=16 gives a
    # denser, more navigable graph; ef_construction=100 provides enough
    # backtracking during build so that search can use a modest ef_search.
    tsm_engine = turbomemory.MemoryEngine(
        db_path=db_dir,
        dimension=args.dimension,
        max_edges=16,
        search_list_size=100,
        outlier_count=0
    )
    
    flat_baseline = FlatNumPyBaseline(args.dimension)
    llm_baseline = LLMInTheLoopBaseline(args.dimension, args.llm_delay)
    
    chroma_baseline = ChromaDBBaseline(args.dimension) if HAS_CHROMA else None
    qdrant_baseline = QdrantBaseline(args.dimension) if HAS_QDRANT else None

    # Ingestion
    print("[1/3] Benchmarking memory ingestion...")
    
    # 1. NumPy
    if not args.tsm_only:
        t0 = time.perf_counter()
        for i in range(args.num_items):
            flat_baseline.insert(f"mem_{i}", f"Sample text {i}", embeddings[i])
        flat_ingest = (time.perf_counter() - t0) * 1000.0 / args.num_items

    # 2. LLM loop
    if not args.tsm_only:
        t0 = time.perf_counter()
        for i in range(min(args.num_items, 15)):
            llm_baseline.insert(f"mem_{i}", f"Sample text {i}", embeddings[i])
        llm_ingest = (time.perf_counter() - t0) * 1000.0 / min(args.num_items, 15)

    # 3. TurboSuperMemory — batched insert amortizes Python ↔ Rust boundary
    # cost and lets the storage layer batch durability work.
    batch_size = 512
    tsm_peak_rss = 0.0
    t0 = time.perf_counter()
    progress_interval = max(1, args.num_items // (batch_size * 20))
    for batch_idx, start in enumerate(range(0, args.num_items, batch_size)):
        end = min(start + batch_size, args.num_items)
        ids = [f"mem_{i}" for i in range(start, end)]
        texts = [f"Sample memory text content {i}" for i in range(start, end)]
        batch_embeddings = embeddings[start:end]
        scores = [1.0] * (end - start)
        concepts = [[f"concept_{i%5}"] for i in range(start, end)]
        tsm_engine.insert_batch(ids, texts, batch_embeddings, scores, concepts)
        if HAS_PSUTIL:
            tsm_peak_rss = max(tsm_peak_rss, peak_memory_mb())
        if (batch_idx + 1) % progress_interval == 0:
            elapsed = time.perf_counter() - t0
            pct = 100.0 * end / args.num_items
            ms_per_item = elapsed * 1000.0 / end if end > 0 else 0.0
            print(
                f"  ingest progress: {pct:5.1f}% ({end}/{args.num_items}) "
                f"| {ms_per_item:.3f} ms/item | peak {tsm_peak_rss:.1f} MB",
                flush=True,
            )
    tsm_ingest = (time.perf_counter() - t0) * 1000.0 / args.num_items

    # 4. ChromaDB
    chroma_ingest = 0.0
    if not args.tsm_only and chroma_baseline:
        t0 = time.perf_counter()
        for i in range(args.num_items):
            chroma_baseline.insert(f"mem_{i}", f"Sample text {i}", embeddings[i])
        chroma_ingest = (time.perf_counter() - t0) * 1000.0 / args.num_items

    # 5. Qdrant
    qdrant_ingest = 0.0
    if not args.tsm_only and qdrant_baseline:
        t0 = time.perf_counter()
        for i in range(args.num_items):
            qdrant_baseline.insert(f"mem_{i}", f"Sample text {i}", embeddings[i])
        qdrant_ingest = (time.perf_counter() - t0) * 1000.0 / args.num_items

    # Retrieval
    print("[2/3] Benchmarking query retrieval...")
    top_k = 5
    
    # 1. NumPy
    ground_truth = []
    if not args.tsm_only:
        t0 = time.perf_counter()
        for q in queries:
            ground_truth.append(flat_baseline.search(q, top_k))
        flat_search = (time.perf_counter() - t0) * 1000.0 / args.num_queries

    # 2. LLM Loop
    if not args.tsm_only:
        t0 = time.perf_counter()
        for q in queries[:10]:
            llm_baseline.search(q, top_k)
        llm_search = (time.perf_counter() - t0) * 1000.0 / 10

    # 3. TurboSuperMemory
    tsm_results = []
    t0 = time.perf_counter()
    for q in queries:
        tsm_results.append(tsm_engine.search_ann(q, top_k))
        if HAS_PSUTIL:
            tsm_peak_rss = max(tsm_peak_rss, peak_memory_mb())
    tsm_search = (time.perf_counter() - t0) * 1000.0 / args.num_queries

    # 4. ChromaDB
    chroma_results = []
    chroma_search = 0.0
    if not args.tsm_only and chroma_baseline:
        t0 = time.perf_counter()
        for q in queries:
            chroma_results.append(chroma_baseline.search(q, top_k))
        chroma_search = (time.perf_counter() - t0) * 1000.0 / args.num_queries

    # 5. Qdrant
    qdrant_results = []
    qdrant_search = 0.0
    if not args.tsm_only and qdrant_baseline:
        t0 = time.perf_counter()
        for q in queries:
            qdrant_results.append(qdrant_baseline.search(q, top_k))
        qdrant_search = (time.perf_counter() - t0) * 1000.0 / args.num_queries

    # Calculations
    print("[3/3] Calculating metrics...")
    tsm_recall = calculate_recall(ground_truth, tsm_results) if not args.tsm_only else 0.0
    chroma_recall = calculate_recall(ground_truth, chroma_results) if (not args.tsm_only and chroma_baseline) else 0.0
    qdrant_recall = calculate_recall(ground_truth, qdrant_results) if (not args.tsm_only and qdrant_baseline) else 0.0

    tsm_size_kb = get_dir_size(db_dir) / 1024.0
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

    # Output Table
    print("\n=========================================================================")
    print("      BENCHMARK RESULTS                                                  ")
    print("=========================================================================")
    print(f"{'System Under Test':<25} | {'Ingest / Item':<15} | {'Search / Query':<15} | {'Recall@5':<10} | {'Peak RSS':<12}")
    print("-" * 88)
    def rss_str(val):
        return f"{val:>9.1f} MB" if val is not None else "N/A"
    if not args.tsm_only:
        print(f"{'Flat NumPy (GT)':<25} | {flat_ingest:>10.3f} ms | {flat_search:>10.3f} ms | {'100.0%':<10} | {'N/A':<12}")
        print(f"{'LLM-in-the-Loop (Mem0)':<25} | {llm_ingest:>10.3f} ms | {llm_search:>10.3f} ms | {'100.0%':<10} | {'N/A':<12}")
        
        if HAS_CHROMA:
            print(f"{'ChromaDB (Ephemeral)':<25} | {chroma_ingest:>10.3f} ms | {chroma_search:>10.3f} ms | {chroma_recall:>8.1f}% | {'N/A':<12}")
        else:
            print(f"{'ChromaDB':<25} | {'NOT INSTALLED':>13} | {'NOT INSTALLED':>13} | {'N/A':<10} | {'N/A':<12}")
            
        if HAS_QDRANT:
            print(f"{'Qdrant (In-Memory)':<25} | {qdrant_ingest:>10.3f} ms | {qdrant_search:>10.3f} ms | {qdrant_recall:>8.1f}% | {'N/A':<12}")
        else:
            print(f"{'Qdrant':<25} | {'NOT INSTALLED':>13} | {'NOT INSTALLED':>13} | {'N/A':<10} | {'N/A':<12}")

    print(f"{'TurboSuperMemory (Ours)':<25} | {tsm_ingest:>10.3f} ms | {tsm_search:>10.3f} ms | {tsm_recall:>8.1f}% | {rss_str(tsm_peak_rss):<12}")
    print("=========================================================================")
    print(f"Local Disk/RAM footprints (N={args.num_items}):")
    print(f"  - TurboSuperMemory Sled DB Size : {tsm_size_kb:.2f} KB (includes indices & transactional WAL)")
    if tsm_peak_rss > 0.0:
        print(f"  - TurboSuperMemory Peak RSS     : {tsm_peak_rss:.1f} MB")
    print("=========================================================================\n")


if __name__ == "__main__":
    run_benchmark()
