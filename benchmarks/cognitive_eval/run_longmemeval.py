#!/usr/bin/env python3
"""LongMemEval benchmark runner for TurboSuperMemory.

Evaluates TSM's memory quality on the LongMemEval benchmark,
which tests long-context memory retrieval in conversational AI.

Usage:
    python run_longmemeval.py --quick  # Quick mode: 10 conversations
    python run_longmemeval.py --adapter tsm --embedding-model BAAI/bge-large-en-v1.5
    python run_longmemeval.py --data-dir benchmarks/cognitive_eval/data
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
from cognitive_eval.benchmark_datasets.longmemeval import load_longmemeval
from cognitive_eval.metrics.recall import (
    average_precision,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
)

logger = logging.getLogger("cognitive_eval.run_longmemeval")


def run_benchmark(
    adapter,
    conversations,
    top_k: int = 10,
    quick: bool = False,
    quick_n: int = 10,
    trigger_consolidation: bool = True,  # NEW: Enable consolidation by default
) -> dict:
    """Run LongMemEval benchmark on a given adapter.
    
    Args:
        adapter: Memory adapter (TSMAdapter or Mem0Adapter)
        conversations: List of Conversation dataclasses
        top_k: Number of results to retrieve per query
        quick: If True, only evaluate first quick_n conversations
        quick_n: Number of conversations for quick mode
        trigger_consolidation: Whether to trigger consolidation after ingestion (builds graph edges)
        
    Returns:
        Dict with benchmark results and metrics
    """
    all_metrics = []
    latencies = []
    
    # Per-step timing tracking
    step_times = {
        "fact_extraction": [],
        "embedding": [],
        "tsm_insert": [],
        "consolidation": [],
        "search": [],
        "total_per_conversation": [],
    }
    
    if quick:
        conversations = conversations[:quick_n]
        logger.info("QUICK MODE: Evaluating %d/%d conversations", quick_n, len(conversations))
    
    total_queries = sum(len(c.queries) for c in conversations)
    logger.info("Processing %d conversations with %d total queries", len(conversations), total_queries)
    
    for i, conversation in enumerate(conversations):
        conv_start = time.perf_counter()
        logger.info("Conversation %d/%d: %s (%d messages, %d queries)",
                    i + 1, len(conversations), conversation.conv_id,
                    len(conversation.messages), len(conversation.queries))
        
        # Ingest all messages with timing breakdown
        ingest_start = time.perf_counter()
        add_result = adapter.add(conversation.messages, user_id=conversation.conv_id)
        ingest_time = (time.perf_counter() - ingest_start) * 1000
        
        # Trigger consolidation after ingestion (for TSM) - builds graph edges
        consolidation_time = 0
        if trigger_consolidation and hasattr(adapter, 'trigger_consolidation'):
            cons_start = time.perf_counter()
            adapter.trigger_consolidation()
            consolidation_time = (time.perf_counter() - cons_start) * 1000
        
        conv_total = (time.perf_counter() - conv_start) * 1000
        step_times["total_per_conversation"].append(conv_total)
        
        # Log detailed breakdown from adapter.add()
        if isinstance(add_result, dict):
            logger.info("  → Facts: %d | Extract: %.1fms | Embed: %.1fms (%.1fms/fact) | Insert: %.1fms | Consolidation: %.1fms | Total: %.1fms",
                        add_result.get('num_facts', 0),
                        add_result.get('extract_ms', 0),
                        add_result.get('embed_ms', 0),
                        add_result.get('embed_ms', 0) / max(add_result.get('num_facts', 1), 1),
                        add_result.get('insert_ms', 0),
                        consolidation_time,
                        conv_total)
        else:
            logger.info("  → Ingest: %.1fms (%.1fms/msg), Consolidation: %.1fms, Total: %.1fms",
                        ingest_time, ingest_time / max(len(conversation.messages), 1), 
                        consolidation_time, conv_total)
        
        # Evaluate each query
        for query in conversation.queries:
            search_start = time.perf_counter()
            # Use cognitive search by default for TSM (it's the whole point of the cognitive layer)
            use_cognitive = hasattr(adapter, 'engine')
            if use_cognitive:
                results = adapter.search(query.query_text, user_id=conversation.conv_id, top_k=top_k, use_cognitive=True)
            else:
                results = adapter.search(query.query_text, user_id=conversation.conv_id, top_k=top_k, use_cognitive=False)
            search_time = (time.perf_counter() - search_start) * 1000
            
            latencies.append(search_time)
            step_times["search"].append(search_time)
            
            # Extract text from results for evaluation
            retrieved_texts = [r.get("text", r.get("content", "")) for r in results]
            
            # Check if answer appears in retrieved results (semantic match)
            answer = query.answer_text.lower()
            hit_at_k = []
            for j, text in enumerate(retrieved_texts):
                # Skip empty results (no text content available)
                if not text or not text.strip():
                    continue
                
                # Check if answer is contained in or similar to retrieved text
                text_lower = text.lower()
                if answer in text_lower or text_lower in answer or any(
                    word in text_lower for word in answer.split() if len(word) > 3
                ):
                    hit_at_k.append(j)
            
            # Compute metrics
            has_hit = len(hit_at_k) > 0
            first_hit = hit_at_k[0] + 1 if hit_at_k else float('inf')
            
            metrics = {
                "conversation_id": conversation.conv_id,
                "query": query.query_text,
                "answer": query.answer_text,
                "recall_at_1": 1.0 if hit_at_k and hit_at_k[0] < 1 else 0.0,
                "recall_at_3": 1.0 if hit_at_k and hit_at_k[0] < 3 else 0.0,
                "recall_at_10": 1.0 if has_hit else 0.0,
                "mrr": 1.0 / first_hit if has_hit else 0.0,
                "hit_rate_at_3": 1.0 if has_hit and hit_at_k[0] < 3 else 0.0,
                "latency_ms": search_time,
                "num_results": len(results),
            }
            all_metrics.append(metrics)
    
    # Aggregate results
    if not all_metrics:
        logger.warning("No metrics computed!")
        return {}
    
    # Print timing breakdown
    print("\n" + "=" * 70)
    print("PER-STEP TIMING BREAKDOWN")
    print("=" * 70)
    print(f"\nConversation ingestion (total):")
    print(f"  Mean per conversation: {np.mean(step_times['total_per_conversation']):.1f}ms")
    print(f"  Total for {len(conversations)} conversations: {np.sum(step_times['total_per_conversation']):.1f}ms")
    
    print(f"\nSearch latency:")
    print(f"  Mean: {np.mean(step_times['search']):.2f}ms")
    print(f"  P50:  {np.percentile(step_times['search'], 50):.2f}ms")
    print(f"  P95:  {np.percentile(step_times['search'], 95):.2f}ms")
    print(f"  P99:  {np.percentile(step_times['search'], 99):.2f}ms")
    
    print("\n" + "=" * 70)
    
    result = {
        "num_conversations": len(conversations),
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
        "step_times": {
            "ingestion_ms": float(np.sum(step_times["total_per_conversation"])),
            "search_mean_ms": float(np.mean(step_times["search"])),
        },
        "raw_metrics": all_metrics,
    }
    
    return result


def print_results(results: dict, adapter_name: str = "TSM"):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 70)
    print(f"LongMemEval Benchmark Results — {adapter_name}")
    print("=" * 70)
    
    print(f"\nDataset: {results['num_conversations']} conversations, {results['num_queries']} queries")
    
    print("\nRetrieval Metrics:")
    print(f"  recall@1:    {results['recall_at_1']:.4f}")
    print(f"  recall@3:    {results['recall_at_3']:.4f}")
    print(f"  recall@10:   {results['recall_at_10']:.4f}")
    print(f"  MRR:         {results['mrr']:.4f}")
    print(f"  Hit Rate@3:  {results['hit_rate_at_3']:.4f}")
    
    print("\nLatency (ms):")
    latency = results['latency_ms']
    print(f"  Mean:  {latency['mean']:>8.2f}")
    print(f"  P50:   {latency['p50']:>8.2f}")
    print(f"  P95:   {latency['p95']:>8.2f}")
    print(f"  P99:   {latency['p99']:>8.2f}")
    
    print("\n" + "=" * 70)
    
    # Comparison with Mem0 claim
    print("\nComparison with Mem0 (reported 91.6% on LongMemEval):")
    if results['recall_at_10'] >= 0.85:
        print(f"  ✅ TSM recall@10 = {results['recall_at_10']:.1%} (within 10% of Mem0 claim)")
    elif results['recall_at_10'] >= 0.75:
        print(f"  ⚠️  TSM recall@10 = {results['recall_at_10']:.1%} (15% below Mem0 claim)")
    else:
        print(f"  ❌ TSM recall@10 = {results['recall_at_10']:.1%} (significantly below Mem0 claim)")
    
    print("=" * 70)


def compare_search_modes(adapter, conversations, top_k: int = 10) -> dict:
    """Compare ANN vs cognitive search on the same queries.
    
    Returns dict with timing and recall comparison.
    """
    ann_times = []
    cog_times = []
    ann_hits = []
    cog_hits = []
    ann_results = []
    cog_results = []
    
    for conv in conversations:
        adapter.add(conv.messages, user_id=conv.conv_id)
        
        for query in conv.queries:
            answer = query.answer_text.lower()
            
            # ANN search
            start = time.perf_counter()
            ann_res = adapter.search(query.query_text, user_id=conv.conv_id, top_k=top_k, use_cognitive=False)
            ann_time = (time.perf_counter() - start) * 1000
            ann_times.append(ann_time)
            ann_results.append(ann_res)
            
            # Check if ANN found the answer
            ann_hit = False
            for r in ann_res:
                text = r.get('text', '').lower()
                if text and (answer in text or text in answer or any(word in text for word in answer.split() if len(word) > 3)):
                    ann_hit = True
                    break
            ann_hits.append(ann_hit)
            
            # Cognitive search
            start = time.perf_counter()
            cog_res = adapter.search(query.query_text, user_id=conv.conv_id, top_k=top_k, use_cognitive=True)
            cog_time = (time.perf_counter() - start) * 1000
            cog_times.append(cog_time)
            cog_results.append(cog_res)
            
            # Check if cognitive found the answer
            cog_hit = False
            for r in (cog_res or []):
                text = r.get('text', '').lower()
                if text and (answer in text or text in answer or any(word in text for word in answer.split() if len(word) > 3)):
                    cog_hit = True
                    break
            cog_hits.append(cog_hit)
    
    # Compare overlap
    overlaps = []
    for ann, cog in zip(ann_results, cog_results):
        ann_ids = {r['id'] for r in ann}
        cog_ids = {r['id'] for r in cog} if cog else set()
        if ann_ids:
            overlap = len(ann_ids & cog_ids) / len(ann_ids)
            overlaps.append(overlap)
    
    ann_recall = sum(ann_hits) / len(ann_hits) if ann_hits else 0.0
    cog_recall = sum(cog_hits) / len(cog_hits) if cog_hits else 0.0
    
    return {
        'ann_mean_ms': float(np.mean(ann_times)),
        'ann_p50_ms': float(np.percentile(ann_times, 50)),
        'ann_recall_at_k': ann_recall,
        'cog_mean_ms': float(np.mean(cog_times)),
        'cog_p50_ms': float(np.percentile(cog_times, 50)),
        'cog_recall_at_k': cog_recall,
        'speedup': float(np.mean(cog_times) / np.mean(ann_times)),
        'avg_overlap': float(np.mean(overlaps)) if overlaps else 0.0,
        'num_queries': len(ann_times),
        'ann_better_queries': sum(1 for a, c in zip(ann_hits, cog_hits) if a and not c),
        'cog_better_queries': sum(1 for a, c in zip(ann_hits, cog_hits) if c and not a),
        'both_correct': sum(1 for a, c in zip(ann_hits, cog_hits) if a and c),
        'both_wrong': sum(1 for a, c in zip(ann_hits, cog_hits) if not a and not c),
    }


def main():
    parser = argparse.ArgumentParser(description="Run LongMemEval benchmark on TSM")
    parser.add_argument("--data-dir", type=str, help="Path to data directory")
    parser.add_argument("--adapter", type=str, default="tsm", choices=["tsm", "mem0"])
    parser.add_argument("--embedding-model", type=str, default=None, 
                        help="Embedding model (default: auto-select based on VRAM)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding")
    parser.add_argument("--lightweight", action="store_true", 
                        help="Use lightweight model (all-MiniLM-L6-v2, 384 dim) for fast testing")
    parser.add_argument("--extractor", type=str, default="mock", choices=["mock", "ollama"],
                        help="Fact extractor to use")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results per query")
    parser.add_argument("--quick", action="store_true", help="Quick mode: only first 10 conversations")
    parser.add_argument("--quick-n", type=int, default=10, help="Number of conversations for quick mode")
    parser.add_argument("--compare-cognitive", action="store_true", 
                        help="Compare ANN vs cognitive search on same queries")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    # Load dataset
    logger.info("Loading LongMemEval dataset...")
    try:
        conversations = load_longmemeval(args.data_dir)
    except FileNotFoundError as e:
        logger.error("Dataset not found: %s", e)
        logger.info("Run: python benchmarks/cognitive_eval/benchmark_datasets/download.py")
        sys.exit(1)
    
    total_messages = sum(len(c.messages) for c in conversations)
    total_queries = sum(len(c.queries) for c in conversations)
    logger.info("Dataset loaded: %d conversations, %d messages, %d queries",
                len(conversations), total_messages, total_queries)
    
    # Initialize adapter
    db_path = tempfile.mkdtemp(prefix="tsm_longmemeval_")
    
    try:
        if args.adapter == "tsm":
            if args.lightweight:
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
            else:
                model_name = args.embedding_model
            
            logger.info("Initializing TSM adapter with model=%s, extractor=%s (cognitive enabled for benchmarks)",
                        model_name, args.extractor)
            adapter = TSMAdapter(
                db_path=db_path,
                embedding_model=model_name,
                extractor=args.extractor,
                batch_size=args.batch_size,
                cognitive_features=True,  # Always enable cognitive for benchmark
            )
        else:
            from cognitive_eval.adapters.mem0_adapter import Mem0Adapter
            adapter = Mem0Adapter()
        
        # Run benchmark
        logger.info("Running benchmark...")
        results = run_benchmark(
            adapter,
            conversations,
            top_k=args.top_k,
            quick=args.quick,
            quick_n=args.quick_n,
            trigger_consolidation=True,  # Always trigger consolidation for TSM
        )
        
        # Print results
        print_results(results, adapter_name=args.adapter.upper())
        
        # Compare ANN vs cognitive if requested
        if args.compare_cognitive and args.adapter == "tsm":
            print("\n" + "=" * 70)
            print("Comparing ANN vs Cognitive Search")
            print("=" * 70)
            
            # Run comparison on a subset
            test_convs = conversations[:args.quick_n] if args.quick else conversations[:5]
            comparison = compare_search_modes(adapter, test_convs, top_k=args.top_k)
            
            print(f"\nQueries tested: {comparison['num_queries']}")
            print(f"\nANN Search:")
            print(f"  Mean: {comparison['ann_mean_ms']:.1f} ms")
            print(f"  P50:  {comparison['ann_p50_ms']:.1f} ms")
            print(f"  Recall@{args.top_k}: {comparison['ann_recall_at_k']:.1%}")
            print(f"\nCognitive Search:")
            print(f"  Mean: {comparison['cog_mean_ms']:.1f} ms")
            print(f"  P50:  {comparison['cog_p50_ms']:.1f} ms")
            print(f"  Recall@{args.top_k}: {comparison['cog_recall_at_k']:.1%}")
            print(f"\nSpeedup: ANN is {comparison['speedup']:.1f}x faster")
            print(f"Result overlap: {comparison['avg_overlap']:.1%}")
            print(f"\nQuery breakdown:")
            print(f"  Both correct:   {comparison['both_correct']}")
            print(f"  Both wrong:     {comparison['both_wrong']}")
            print(f"  ANN only:       {comparison['ann_better_queries']}")
            print(f"  Cognitive only: {comparison['cog_better_queries']}")
            print(f"\nRecommendation:")
            if comparison['cog_recall_at_k'] > comparison['ann_recall_at_k']:
                improvement = (comparison['cog_recall_at_k'] - comparison['ann_recall_at_k']) * 100
                print(f"  🧠 Cognitive improves recall by {improvement:.1f} percentage points")
            elif comparison['ann_recall_at_k'] > comparison['cog_recall_at_k']:
                print(f"  ⚡ ANN has better recall. Cognitive may need tuning.")
            else:
                print(f"  ⚖️  Same recall. Use ANN for speed.")
            print("=" * 70)
        
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
