#!/usr/bin/env python3
"""Generate benchmark reports from results.

Converts benchmark result JSON files into markdown tables
suitable for README.md inclusion.

Usage:
    python report.py --input results/ --output report.md
    python report.py --input results/comparison_20240115_120000.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("cognitive_eval.report")


def load_results(input_path: Path) -> dict:
    """Load results from a JSON file or directory."""
    if input_path.is_file():
        with open(input_path, "r") as f:
            return json.load(f)
    
    # Load all JSON files from directory
    results = {}
    for json_file in input_path.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)
            results[json_file.stem] = data
    
    return results


def format_latency(latency: dict) -> str:
    """Format latency dict as string."""
    return f"{latency.get('p50', 0):.1f}ms (P50), {latency.get('p95', 0):.1f}ms (P95)"


def generate_longmemeval_table(results: dict) -> str:
    """Generate LongMemEval results markdown table."""
    lines = [
        "## LongMemEval Results",
        "",
        "| System | recall@10 | MRR | NDCG@10 | Latency |",
        "|--------|-----------|-----|---------|---------|",
    ]
    
    for system_name, system_results in results.items():
        if "longmemeval" not in system_results:
            continue
        
        lme = system_results["longmemeval"]
        recall = lme.get("recall_at_10", 0)
        mrr = lme.get("mrr", 0)
        ndcg = lme.get("ndcg_at_10", 0)
        latency = format_latency(lme.get("latency_ms", {}))
        
        lines.append(f"| {system_name} | {recall:.3f} | {mrr:.3f} | {ndcg:.3f} | {latency} |")
    
    lines.append("")
    return "\n".join(lines)


def generate_locomo_table(results: dict) -> str:
    """Generate LoCoMo results markdown table."""
    lines = [
        "## LoCoMo Results",
        "",
        "| System | recall@10 | Temporal Accuracy | Temporal Error Rate | Latency |",
        "|--------|-----------|-------------------|---------------------|---------|",
    ]
    
    for system_name, system_results in results.items():
        if "locomo" not in system_results:
            continue
        
        lm = system_results["locomo"]
        recall = lm.get("recall_at_10", 0)
        temporal = lm.get("temporal", {})
        accuracy = temporal.get("overall_accuracy", 0)
        error_rate = temporal.get("overall_temporal_error_rate", 0)
        latency = format_latency(lm.get("latency_ms", {}))
        
        lines.append(f"| {system_name} | {recall:.3f} | {accuracy:.3f} | {error_rate:.3f} | {latency} |")
    
    lines.append("")
    return "\n".join(lines)


def generate_comparison_report(results: dict) -> str:
    """Generate full comparison markdown report."""
    lines = [
        "# TurboSuperMemory Cognitive Benchmark Results",
        "",
        f"**Date:** {results.get('timestamp', 'N/A')}",
        f"**Embedding Model:** {results.get('embedding_model', 'N/A')}",
        f"**Extractor:** {results.get('extractor', 'N/A')}",
        f"**Quick Mode:** {results.get('quick_mode', False)}",
        "",
        "---",
        "",
    ]
    
    # Dataset statistics
    if "datasets" in results:
        lines.extend([
            "## Dataset Statistics",
            "",
        ])
        for dataset_name, stats in results["datasets"].items():
            lines.append(f"### {dataset_name}")
            lines.append("")
            for key, value in stats.items():
                if isinstance(value, dict):
                    lines.append(f"- {key}:")
                    for k, v in value.items():
                        lines.append(f"  - {k}: {v}")
                else:
                    lines.append(f"- {key}: {value}")
            lines.append("")
    
    # LongMemEval
    if "tsm" in results and "longmemeval" in results["tsm"]:
        lines.append(generate_longmemeval_table(results))
    
    # LoCoMo
    if "tsm" in results and "locomo" in results["tsm"]:
        lines.append(generate_locomo_table(results))
    
    # Mem0 comparison
    lines.extend([
        "## Comparison with Mem0",
        "",
        "Mem0 claims:",
        "- LongMemEval: 91.6% recall",
        "- LoCoMo: Not publicly reported",
        "",
    ])
    
    if "tsm" in results and "longmemeval" in results["tsm"]:
        tsm_recall = results["tsm"]["longmemeval"].get("recall_at_10", 0)
        lines.append(f"**TSM LongMemEval recall@10: {tsm_recall:.1%}**")
        lines.append("")
        
        if tsm_recall >= 0.85:
            lines.append("✅ Within 10% of Mem0 claim")
        elif tsm_recall >= 0.75:
            lines.append("⚠️ 15% below Mem0 claim")
        else:
            lines.append("❌ Significantly below Mem0 claim")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("*Generated by cognitive_eval/report.py*")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark reports")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file or directory")
    parser.add_argument("--output", type=str, help="Output markdown file")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input path not found: %s", input_path)
        sys.exit(1)
    
    logger.info("Loading results from %s", input_path)
    results = load_results(input_path)
    
    logger.info("Generating report...")
    report = generate_comparison_report(results)
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        logger.info("Report saved to %s", output_path)
    else:
        print(report)


if __name__ == "__main__":
    main()
