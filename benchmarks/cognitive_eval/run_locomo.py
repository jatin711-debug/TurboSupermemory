#!/usr/bin/env python3
"""LoCoMo benchmark runner for TurboSuperMemory.

Evaluates TSM's temporal reasoning on the LoCoMo benchmark,
which tests whether a memory system retrieves the temporally
correct fact (current vs past state).

Usage:
    python run_locomo.py --quick  # Quick mode: 10 queries
    python run_locomo.py --adapter tsm --embedding-model BAAI/bge-large-en-v1.5
    python run_locomo.py --data-dir benchmarks/cognitive_eval/data
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognitive_eval.adapters.tsm_adapter import TSMAdapter
from cognitive_eval.datasets.locomo import load_locomo
from cognitive_eval.metrics.recall import (
    average_precision,
    hit_rate_at_k,
    mrr,
    recall_at_k,
)
from cognitive_eval.metrics.temporal import (
    recency_bias_score,
    temporal_accuracy,
    temporal_confusion_matrix,
)

logger = logging.getLogger("cognitive_eval.run_locomo")


def run_benchmark(
    adapter,
    dataset,
    top_k: int = 10,
    quick: bool = False,
    quick_n: int = 10,
) -> dict:
    """Run LoCoMo benchmark on a given adapter.
    
    Args:
        adapter: Memory adapter (TSMAdapter or Mem0Adapter)
        dataset: LoCoMoDataset with sessions and queries
        top_k: Number of results to retrieve per query
        quick: If True, only evaluate first quick_n queries
        quick_n: Number of queries for quick mode
        
    Returns:
        Dict with benchmark results and metrics
    """
    all_metrics = []
    all_temporal_metrics = []
    latencies = []
    
    sessions = dataset.sessions
    queries = dataset.queries
    
    if quick:
        queries = queries[:quick_n]
        logger.info("QUICK MODE: Evaluating %d/%d queries", quick_n, len(dataset.queries))
    
    # For quick mode, only ingest sessions relevant to the queries
    if quick:
        relevant_session_ids = set(q.relevant_session for q in queries)
        # Also sample some other sessions for diversity
        sampled_sessions = [s for s in sessions if s.session_id in relevant_session_ids]
        if len(sampled_sessions) < 100:
            # Add more random sessions
            import random
            random.seed(42)
            other_sessions = [s for s in sessions if s.session_id not in relevant_session_ids]
            sampled_sessions.extend(random.sample(other_sessions, min(100 - len(sampled_sessions), len(other_sessions))))
        sessions = sampled_sessions
        logger.info("Quick mode: sampled %d sessions for evaluation", len(sessions))
    
    logger.info("Processing %d sessions with %d queries", len(sessions), len(queries))
    
    # Ingest all sessions first
    logger.info("Ingesting %d sessions...", len(sessions))
    for i, session in enumerate(sessions):
        if i % 1000 == 0:
            logger.info("  Ingested %d/%d sessions", i, len(sessions))
        
        # Create messages from session facts
        messages = []
        for fact in session.facts:
            messages.append({
                "role": "user",
                "content": fact,
                "timestamp": session.timestamp,
            })
        
        # Ingest session
        adapter.add(messages, user_id=session.session_id)
    
    # Trigger consolidation after ingestion (for TSM)
    if hasattr(adapter, 'trigger_consolidation'):
        adapter.trigger_consolidation()
    
    # Evaluate each query
    logger.info("Evaluating %d queries...", len(queries))
    for i, query in enumerate(queries):
        if i % 100 == 0:
            logger.info("  Query %d/%d", i, len(queries))
        
        start = time.perf_counter()
        results = adapter.search(query.query_text, user_id=query.relevant_session, top_k=top_k)
        end = time.perf_counter()
        
        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)
        
        # Extract text from results for evaluation
        retrieved_texts = [r.get("text", r.get("content", "")) for r in results]
        
        # Check if answer appears in retrieved results
        answer = query.answer_text.lower()
        hit_at_k = []
        for j, text in enumerate(retrieved_texts):
            text_lower = text.lower()
            if answer in text_lower or text_lower in answer or any(
                word in text_lower for word in answer.split() if len(word) > 3
            ):
                hit_at_k.append(j)
        
        has_hit = len(hit_at_k) > 0
        first_hit = hit_at_k[0] + 1 if hit_at_k else float('inf')
        
        metrics = {
            "query_id": query.query_id,
            "query": query.query_text,
            "query_type": query.query_type,
            "answer": query.answer_text,
            "recall_at_1": 1.0 if hit_at_k and hit_at_k[0] < 1 else 0.0,
            "recall_at_3": 1.0 if hit_at_k and hit_at_k[0] < 3 else 0.0,
            "recall_at_10": 1.0 if has_hit else 0.0,
            "mrr": 1.0 / first_hit if has_hit else 0.0,
            "hit_rate_at_3": 1.0 if has_hit and hit_at_k[0] < 3 else 0.0,
            "latency_ms": latency_ms,
            "num_results": len(results),
        }
        all_metrics.append(metrics)
        
        # Temporal-specific metrics
        # For LoCoMo-MC10, we classify queries as "recent" (current) or "distant" (past)
        temporal_type = "current" if query.query_type == "recent" else "past" if query.query_type == "distant" else "unknown"
        temporal = temporal_accuracy(
            retrieved_texts,
            query.answer_text,  # expected current
            query.answer_text,  # expected past (same for MC10)
            temporal_type,
        )
        temporal["latency_ms"] = latency_ms
        all_temporal_metrics.append(temporal)
    
    # Aggregate results
    if not all_metrics:
        logger.warning("No metrics computed!")
        return {}
    
    # Standard metrics
    result = {
        "num_sessions": len(sessions),
        "num_queries": len(all_metrics),
        "recall_at_1": float(np.mean([m["recall_at_1"] for m in all_metrics])),
        "recall_at_3": float(np.mean([m["recall_at_3"] for m in all_metrics])),
        "recall_at_10": float(np.mean([m["recall_at_10"] for m in all_metrics])),
        "mrr": float(np.mean([m["mrr"] for m in all_metrics])),
        "hit_rate_at_3": float(np.mean([m["hit_rate_at_3"] for m in all_metrics])),
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
        },
        "raw_metrics": all_metrics,
    }
    
    # Temporal metrics
    temporal_correct = sum(1 for t in all_temporal_metrics if t.get("correct", False))
    result["temporal_accuracy"] = temporal_correct / len(all_temporal_metrics) if all_temporal_metrics else 0.0
    result["temporal_breakdown"] = {
        "correct": temporal_correct,
        "total": len(all_temporal_metrics),
        "by_type": {},
    }
    
    # Group by query type
    by_type = {}
    for t in all_temporal_metrics:
        qt = t.get("query_type", "unknown")
        if qt not in by_type:
            by_type[qt] = {"correct": 0, "total": 0}
        by_type[qt]["total"] += 1
        if t.get("correct", False):
            by_type[qt]["correct"] += 1
    
    for qt, stats in by_type.items():
        result["temporal_breakdown"]["by_type"][qt] = {
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
            "correct": stats["correct"],
            "total": stats["total"],
        }
    
    return result


def print_results(results: dict, adapter_name: str = "TSM"):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 70)
    print(f"LoCoMo Benchmark Results — {adapter_name}")
    print("=" * 70)
    
    print(f"\nDataset: {results['num_sessions']} sessions, {results['num_queries']} queries")
    
    print("\nRetrieval Metrics:")
    print(f"  recall@1:    {results['recall_at_1']:.4f}")
    print(f"  recall@3:    {results['recall_at_3']:.4f}")
    print(f"  recall@10:   {results['recall_at_10']:.4f}")
    print(f"  MRR:         {results['mrr']:.4f}")
    print(f"  Hit Rate@3:  {results['hit_rate_at_3']:.4f}")
    
    print("\nTemporal Accuracy:")
    print(f"  Overall:     {results['temporal_accuracy']:.4f}")
    if "by_type" in results.get("temporal_breakdown", {}):
        for qt, stats in results["temporal_breakdown"]["by_type"].items():
            print(f"  {qt}:        {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
    
    print("\nLatency (ms):")
    latency = results['latency_ms']
    print(f"  Mean:  {latency['mean']:>8.2f}")
    print(f"  P50:   {latency['p50']:>8.2f}")
    print(f"  P95:   {latency['p95']:>8.2f}")
    print(f"  P99:   {latency['p99']:>8.2f}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run LoCoMo benchmark on TSM")
    parser.add_argument("--data-dir", type=str, help="Path to data directory")
    parser.add_argument("--adapter", type=str, default="tsm", choices=["tsm", "mem0"])
    parser.add_argument("--embedding-model", type=str, default=None, 
                        help="Embedding model (default: auto-select based on VRAM)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding")
    parser.add_argument("--lightweight", action="store_true", 
                        help="Use lightweight model (all-MiniLM-L6-v2, 384 dim) for fast testing")
    parser.add_argument("--extractor", type=str, default="mock", choices=["mock", "ollama"])
    parser.add_argument("--top-k", type=int, default=10, help="Number of results per query")
    parser.add_argument("--quick", action="store_true", help="Quick mode: only first 10 queries")
    parser.add_argument("--quick-n", type=int, default=10, help="Number of queries for quick mode")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    # Load dataset
    logger.info("Loading LoCoMo dataset...")
    try:
        dataset = load_locomo(args.data_dir)
    except FileNotFoundError as e:
        logger.error("Dataset not found: %s", e)
        logger.info("Run: python benchmarks/cognitive_eval/datasets/download.py")
        sys.exit(1)
    
    logger.info("Dataset loaded: %d sessions, %d queries", len(dataset.sessions), len(dataset.queries))
    
    # Initialize adapter
    db_path = tempfile.mkdtemp(prefix="tsm_locomo_")
    
    try:
        if args.adapter == "tsm":
            if args.lightweight:
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
            else:
                model_name = args.embedding_model
            
            logger.info("Initializing TSM adapter with model=%s, extractor=%s (cognitive=OFF for benchmarks)",
                        model_name, args.extractor)
            adapter = TSMAdapter(
                db_path=db_path,
                embedding_model=model_name,
                extractor=args.extractor,
                batch_size=args.batch_size,
                cognitive_features=False,  # Disabled for benchmarks - ANN only
            )
        else:
            from cognitive_eval.adapters.mem0_adapter import Mem0Adapter
            adapter = Mem0Adapter()
        
        # Run benchmark
        logger.info("Running benchmark...")
        results = run_benchmark(
            adapter,
            dataset,
            top_k=args.top_k,
            quick=args.quick,
            quick_n=args.quick_n,
        )
        
        # Print results
        print_results(results, adapter_name=args.adapter.upper())
        
        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Don't save raw_metrics in JSON (too large)
            save_results = {k: v for k, v in results.items() if k != "raw_metrics"}
            
            with open(output_path, "w") as f:
                json.dump(save_results, f, indent=2)
            logger.info("Results saved to %s", output_path)
        
    finally:
        adapter.close()
        import shutil
        shutil.rmtree(db_path, ignore_errors=True)


if __name__ == "__main__":
    main()
