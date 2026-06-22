#!/usr/bin/env python3
"""Head-to-head comparison: TurboSuperMemory vs Mem0.

Runs both LongMemEval and LoCoMo benchmarks on both systems
and generates a comparison report.

Usage:
    python run_comparison.py --output results/
    python run_comparison.py --quick  # Fast subset for testing
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognitive_eval.adapters.tsm_adapter import TSMAdapter
from cognitive_eval.benchmark_datasets.longmemeval import LongMemEvalDataset
from cognitive_eval.benchmark_datasets.locomo import LoCoMoDataset
from cognitive_eval.run_longmemeval import run_benchmark as run_longmemeval
from cognitive_eval.run_locomo import run_benchmark as run_locomo

logger = logging.getLogger("cognitive_eval.run_comparison")


def run_comparison(
    embedding_model: str = "BAAI/bge-large-en-v1.5",
    extractor: str = "mock",
    quick: bool = False,
    quick_n: int = 10,
) -> dict:
    """Run both benchmarks on both systems and compare.
    
    Args:
        embedding_model: SentenceTransformer model name
        extractor: Fact extractor to use ("mock" or "ollama")
        quick: If True, use subset of data for fast testing
        quick_n: Number of conversations for quick mode
        
    Returns:
        Dict with comparison results
    """
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "embedding_model": embedding_model,
        "extractor": extractor,
        "quick_mode": quick,
        "tsm": {},
        "mem0": {},
    }
    
    # Load datasets
    logger.info("Loading datasets...")
    longmemeval = LongMemEvalDataset()
    locomoco = LoCoMoDataset()
    
    try:
        longmemeval.load()
        results["datasets"] = {
            "longmemeval": longmemeval.get_statistics(),
        }
    except FileNotFoundError:
        logger.error("LongMemEval dataset not found. Run: python -m cognitive_eval.benchmark_datasets.download")
        longmemeval = None
    
    try:
        locomoco.load()
        results["datasets"]["locomo"] = locomoco.get_statistics()
    except FileNotFoundError:
        logger.error("LoCoMo dataset not found. Run: python -m cognitive_eval.benchmark_datasets.download")
        locomoco = None
    
    if not longmemeval and not locomoco:
        logger.error("No datasets available!")
        return results
    
    # Run TSM benchmarks
    logger.info("=" * 70)
    logger.info("Running TSM benchmarks...")
    logger.info("=" * 70)
    
    tsm_db = tempfile.mkdtemp(prefix="tsm_compare_")
    
    try:
        tsm_adapter = TSMAdapter(
            db_path=tsm_db,
            embedding_model=embedding_model,
            extractor=extractor,
        )
        
        if longmemeval:
            logger.info("Running LongMemEval on TSM...")
            tsm_longmemeval = run_longmemeval(
                tsm_adapter,
                longmemeval,
                top_k=10,
                quick=quick,
                quick_n=quick_n,
            )
            results["tsm"]["longmemeval"] = {
                k: v for k, v in tsm_longmemeval.items()
                if k not in ("raw_metrics", "raw_temporal_metrics")
            }
        
        if locomoco:
            logger.info("Running LoCoMo on TSM...")
            tsm_locomo = run_locomo(
                tsm_adapter,
                locomoco,
                top_k=10,
                quick=quick,
                quick_n=quick_n,
            )
            results["tsm"]["locomo"] = {
                k: v for k, v in tsm_locomo.items()
                if k not in ("raw_metrics", "raw_temporal_metrics")
            }
        
        tsm_adapter.close()
        
    finally:
        import shutil
        shutil.rmtree(tsm_db, ignore_errors=True)
    
    # Run Mem0 benchmarks (if available)
    logger.info("=" * 70)
    logger.info("Running Mem0 benchmarks...")
    logger.info("=" * 70)
    
    try:
        from cognitive_eval.adapters.mem0_adapter import Mem0Adapter
        mem0_adapter = Mem0Adapter()
        
        if longmemeval:
            logger.info("Running LongMemEval on Mem0...")
            mem0_longmemeval = run_longmemeval(
                mem0_adapter,
                longmemeval,
                top_k=10,
                quick=quick,
                quick_n=quick_n,
            )
            results["mem0"]["longmemeval"] = {
                k: v for k, v in mem0_longmemeval.items()
                if k not in ("raw_metrics", "raw_temporal_metrics")
            }
        
        if locomoco:
            logger.info("Running LoCoMo on Mem0...")
            mem0_locomo = run_locomo(
                mem0_adapter,
                locomoco,
                top_k=10,
                quick=quick,
                quick_n=quick_n,
            )
            results["mem0"]["locomo"] = {
                k: v for k, v in mem0_locomo.items()
                if k not in ("raw_metrics", "raw_temporal_metrics")
            }
        
        mem0_adapter.close()
        
    except Exception as e:
        logger.warning("Mem0 benchmark failed (expected if mem0ai not installed): %s", e)
        results["mem0"]["error"] = str(e)
    
    return results


def print_comparison(results: dict):
    """Print comparison table."""
    print("\n" + "=" * 70)
    print("TurboSuperMemory vs Mem0 — Benchmark Comparison")
    print("=" * 70)
    
    print(f"\nTimestamp: {results['timestamp']}")
    print(f"Embedding Model: {results['embedding_model']}")
    print(f"Extractor: {results['extractor']}")
    print(f"Quick Mode: {results['quick_mode']}")
    
    # LongMemEval comparison
    if "longmemeval" in results.get("tsm", {}):
        print("\n" + "-" * 70)
        print("LongMemEval Results")
        print("-" * 70)
        
        tsm_lme = results["tsm"]["longmemeval"]
        print(f"\nTurboSuperMemory:")
        print(f"  recall@10:   {tsm_lme.get('recall_at_10', 0):.4f}")
        print(f"  MRR:         {tsm_lme.get('mrr', 0):.4f}")
        print(f"  NDCG@10:     {tsm_lme.get('ndcg_at_10', 0):.4f}")
        print(f"  Latency P50: {tsm_lme.get('latency_ms', {}).get('p50', 0):.2f}ms")
        
        if "longmemeval" in results.get("mem0", {}):
            mem0_lme = results["mem0"]["longmemeval"]
            print(f"\nMem0:")
            print(f"  recall@10:   {mem0_lme.get('recall_at_10', 0):.4f}")
            print(f"  MRR:         {mem0_lme.get('mrr', 0):.4f}")
            print(f"  NDCG@10:     {mem0_lme.get('ndcg_at_10', 0):.4f}")
            print(f"  Latency P50: {mem0_lme.get('latency_ms', {}).get('p50', 0):.2f}ms")
            
            # Comparison
            tsm_recall = tsm_lme.get('recall_at_10', 0)
            mem0_recall = mem0_lme.get('recall_at_10', 0)
            diff = tsm_recall - mem0_recall
            
            print(f"\nDifference (TSM - Mem0):")
            print(f"  recall@10: {diff:+.4f} ({diff/mem0_recall*100:+.1f}%)")
    
    # LoCoMo comparison
    if "locomo" in results.get("tsm", {}):
        print("\n" + "-" * 70)
        print("LoCoMo Results")
        print("-" * 70)
        
        tsm_lm = results["tsm"]["locomo"]
        print(f"\nTurboSuperMemory:")
        print(f"  recall@10:        {tsm_lm.get('recall_at_10', 0):.4f}")
        temporal = tsm_lm.get("temporal", {})
        print(f"  Temporal Accuracy: {temporal.get('overall_accuracy', 0):.4f}")
        print(f"  Temporal Error:    {temporal.get('overall_temporal_error_rate', 0):.4f}")
        
        if "locomo" in results.get("mem0", {}):
            mem0_lm = results["mem0"]["locomo"]
            print(f"\nMem0:")
            print(f"  recall@10:        {mem0_lm.get('recall_at_10', 0):.4f}")
            temporal = mem0_lm.get("temporal", {})
            print(f"  Temporal Accuracy: {temporal.get('overall_accuracy', 0):.4f}")
            print(f"  Temporal Error:    {temporal.get('overall_temporal_error_rate', 0):.4f}")
    
    print("\n" + "=" * 70)
    
    # Mem0 claim comparison
    print("\nComparison with Mem0 Claims:")
    print("  Mem0 claims 91.6% on LongMemEval")
    if "longmemeval" in results.get("tsm", {}):
        tsm_recall = results["tsm"]["longmemeval"].get("recall_at_10", 0)
        if tsm_recall >= 0.85:
            print(f"  ✅ TSM recall@10 = {tsm_recall:.1%} (within 10% of Mem0 claim)")
        elif tsm_recall >= 0.75:
            print(f"  ⚠️  TSM recall@10 = {tsm_recall:.1%} (15% below Mem0 claim)")
        else:
            print(f"  ❌ TSM recall@10 = {tsm_recall:.1%} (significantly below Mem0 claim)")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Compare TSM vs Mem0 on cognitive benchmarks")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--extractor", type=str, default="mock", choices=["mock", "ollama"])
    parser.add_argument("--quick", action="store_true", help="Quick mode for testing")
    parser.add_argument("--quick-n", type=int, default=10, help="Number of conversations for quick mode")
    parser.add_argument("--output", type=str, help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    logger.info("Starting TSM vs Mem0 comparison...")
    
    results = run_comparison(
        embedding_model=args.embedding_model,
        extractor=args.extractor,
        quick=args.quick,
        quick_n=args.quick_n,
    )
    
    print_comparison(results)
    
    # Save results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"comparison_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", output_file)
    
    logger.info("Comparison complete!")


if __name__ == "__main__":
    main()
