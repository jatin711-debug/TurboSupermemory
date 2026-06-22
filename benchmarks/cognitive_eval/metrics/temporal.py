"""Temporal reasoning metrics for LoCoMo benchmark.

LoCoMo tests whether a memory system can retrieve the temporally
correct fact (e.g., "current city" vs "past city"). These metrics
measure temporal accuracy beyond standard recall.
"""

from typing import Dict, List, Optional


def temporal_accuracy(
    retrieved: List[str],
    expected_current: str,
    expected_past: str,
    query_type: str = "current_state",
) -> Dict:
    """Compute temporal accuracy for a single query.
    
    Args:
        retrieved: Ordered list of retrieved memory IDs
        expected_current: The expected answer for "current state" queries
        expected_past: The expected answer for "past state" queries
        query_type: Type of query ("current_state", "past_state", "temporal_reasoning")
        
    Returns:
        Dict with temporal accuracy metrics
    """
    retrieved_top3 = set(retrieved[:3])
    
    current_found = expected_current in retrieved_top3
    past_found = expected_past in retrieved_top3
    
    # Temporal error: retrieving past when asking for current (or vice versa)
    if query_type == "current_state":
        temporal_error = past_found and not current_found
        correct = current_found
    elif query_type == "past_state":
        temporal_error = current_found and not past_found
        correct = past_found
    else:  # temporal_reasoning
        # For temporal reasoning, either could be correct depending on context
        correct = current_found or past_found
        temporal_error = False
    
    return {
        "correct": correct,
        "current_found": current_found,
        "past_found": past_found,
        "temporal_error": temporal_error,
        "query_type": query_type,
    }


def temporal_confusion_matrix(
    results: List[Dict],
) -> Dict:
    """Aggregate temporal confusion matrix across all queries.
    
    Args:
        results: List of temporal_accuracy() result dicts
        
    Returns:
        Dict with aggregated statistics
    """
    total = len(results)
    if total == 0:
        return {}
    
    by_type = {}
    for r in results:
        qtype = r["query_type"]
        if qtype not in by_type:
            by_type[qtype] = []
        by_type[qtype].append(r)
    
    summary = {
        "total_queries": total,
        "overall_accuracy": sum(1 for r in results if r["correct"]) / total,
        "overall_temporal_error_rate": sum(1 for r in results if r["temporal_error"]) / total,
        "by_type": {},
    }
    
    for qtype, type_results in by_type.items():
        type_total = len(type_results)
        summary["by_type"][qtype] = {
            "count": type_total,
            "accuracy": sum(1 for r in type_results if r["correct"]) / type_total,
            "temporal_error_rate": sum(1 for r in type_results if r["temporal_error"]) / type_total,
            "current_found_rate": sum(1 for r in type_results if r["current_found"]) / type_total,
            "past_found_rate": sum(1 for r in type_results if r["past_found"]) / type_total,
        }
    
    return summary


def recency_bias_score(
    retrieved: List[str],
    expected_current: str,
    expected_past: str,
) -> float:
    """Measure recency bias: does the system always prefer recent memories?
    
    A score of 1.0 means the system always retrieves the current fact.
    A score of 0.0 means it always retrieves the past fact.
    A score of 0.5 means it's balanced (or retrieves neither).
    
    Args:
        retrieved: Ordered list of retrieved memory IDs
        expected_current: Expected current fact ID
        expected_past: Expected past fact ID
        
    Returns:
        Recency bias score in [0.0, 1.0]
    """
    current_rank = None
    past_rank = None
    
    for i, item in enumerate(retrieved, start=1):
        if item == expected_current and current_rank is None:
            current_rank = i
        if item == expected_past and past_rank is None:
            past_rank = i
    
    if current_rank is None and past_rank is None:
        return 0.5  # Neither found
    
    if current_rank is not None and past_rank is None:
        return 1.0  # Only current found
    
    if current_rank is None and past_rank is not None:
        return 0.0  # Only past found
    
    # Both found - prefer the one ranked higher (lower rank number)
    if current_rank < past_rank:
        return 1.0
    elif past_rank < current_rank:
        return 0.0
    else:
        return 0.5  # Same rank (unlikely)


if __name__ == "__main__":
    # Test temporal metrics
    retrieved = ["mem_current", "mem_past", "mem_other"]
    
    print("Test 1: Current state query")
    result = temporal_accuracy(retrieved, "mem_current", "mem_past", "current_state")
    print(f"  Result: {result}")
    
    print("\nTest 2: Past state query")
    result = temporal_accuracy(retrieved, "mem_current", "mem_past", "past_state")
    print(f"  Result: {result}")
    
    print("\nTest 3: Temporal confusion matrix")
    results = [
        temporal_accuracy(["a", "b"], "a", "b", "current_state"),
        temporal_accuracy(["b", "a"], "a", "b", "current_state"),
        temporal_accuracy(["a", "b"], "a", "b", "past_state"),
    ]
    summary = temporal_confusion_matrix(results)
    print(f"  Summary: {summary}")
    
    print("\nTest 4: Recency bias")
    bias = recency_bias_score(["current", "past"], "current", "past")
    print(f"  Bias (current first): {bias}")
    
    bias = recency_bias_score(["past", "current"], "current", "past")
    print(f"  Bias (past first): {bias}")
