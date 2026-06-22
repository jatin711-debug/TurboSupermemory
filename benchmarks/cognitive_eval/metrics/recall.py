"""Standard information retrieval metrics for benchmark evaluation.

All metrics operate on lists of string IDs for retrieved and expected items.
This allows comparison across different memory systems (TSM, Mem0, etc.)
that may use different ID formats.
"""

import math
from typing import List, Set, Union


def _to_set(items: Union[List[str], Set[str]]) -> Set[str]:
    """Convert items to a set for set operations."""
    if isinstance(items, set):
        return items
    return set(items)


def recall_at_k(retrieved: List[str], expected: Union[List[str], Set[str]], k: int) -> float:
    """Compute recall@K: fraction of expected items found in top-K retrieved.
    
    Recall@K = |relevant ∩ retrieved| / |relevant|
    
    Args:
        retrieved: Ordered list of retrieved item IDs (best first)
        expected: Set or list of relevant/expected item IDs
        k: Number of top retrieved items to consider
        
    Returns:
        Recall@K score in [0.0, 1.0]
    """
    if not expected:
        return 1.0  # Nothing to find, so we found everything
    
    expected_set = _to_set(expected)
    retrieved_k = set(retrieved[:k])
    
    hits = len(expected_set & retrieved_k)
    return hits / len(expected_set)


def precision_at_k(retrieved: List[str], expected: Union[List[str], Set[str]], k: int) -> float:
    """Compute precision@K: fraction of retrieved items that are relevant.
    
    Precision@K = |relevant ∩ retrieved| / |retrieved|
    
    Args:
        retrieved: Ordered list of retrieved item IDs (best first)
        expected: Set or list of relevant/expected item IDs
        k: Number of top retrieved items to consider
        
    Returns:
        Precision@K score in [0.0, 1.0]
    """
    if k <= 0 or not retrieved:
        return 0.0
    
    expected_set = _to_set(expected)
    retrieved_k = retrieved[:k]
    
    if not retrieved_k:
        return 0.0
    
    hits = len(expected_set & set(retrieved_k))
    return hits / len(retrieved_k)


def f1_at_k(retrieved: List[str], expected: Union[List[str], Set[str]], k: int) -> float:
    """Compute F1@K: harmonic mean of precision and recall.
    
    F1@K = 2 * (precision * recall) / (precision + recall)
    
    Args:
        retrieved: Ordered list of retrieved item IDs (best first)
        expected: Set or list of relevant/expected item IDs
        k: Number of top retrieved items to consider
        
    Returns:
        F1@K score in [0.0, 1.0]
    """
    prec = precision_at_k(retrieved, expected, k)
    rec = recall_at_k(retrieved, expected, k)
    
    if prec + rec == 0:
        return 0.0
    
    return 2 * (prec * rec) / (prec + rec)


def mrr(retrieved: List[str], expected: Union[List[str], Set[str]]) -> float:
    """Compute Mean Reciprocal Rank (MRR).
    
    MRR = (1/|Q|) * Σ(1/rank_i) where rank_i is the rank of the first
    relevant item for query i. If no relevant item is found, contribution is 0.
    
    For multiple expected items, uses the best (lowest) rank among them.
    
    Args:
        retrieved: Ordered list of retrieved item IDs (best first)
        expected: Set or list of relevant/expected item IDs
        
    Returns:
        MRR score in [0.0, 1.0]
    """
    expected_set = _to_set(expected)
    
    if not expected_set:
        return 1.0  # Nothing to find
    
    if not retrieved:
        return 0.0
    
    # Find the best rank among all expected items
    best_rank = None
    for exp in expected_set:
        try:
            rank = retrieved.index(exp) + 1  # 1-based rank
            if best_rank is None or rank < best_rank:
                best_rank = rank
        except ValueError:
            pass  # Not found
    
    if best_rank is None:
        return 0.0  # None of the expected items were found
    
    return 1.0 / best_rank


def dcg_at_k(retrieved: List[str], expected: Union[List[str], Set[str]], k: int) -> float:
    """Compute Discounted Cumulative Gain (DCG) at K.
    
    DCG@K = Σ(relevance_i / log2(i + 1)) for i = 1 to K
    
    Binary relevance: 1 if item is in expected, 0 otherwise.
    
    Args:
        retrieved: Ordered list of retrieved item IDs (best first)
        expected: Set or list of relevant/expected item IDs
        k: Number of top retrieved items to consider
        
    Returns:
        DCG@K score (unbounded, higher is better)
    """
    expected_set = _to_set(expected)
    
    dcg = 0.0
    for i, item in enumerate(retrieved[:k], start=1):
        if item in expected_set:
            # Binary relevance = 1, discounted by position
            dcg += 1.0 / math.log2(i + 1)
    
    return dcg


def ndcg_at_k(retrieved: List[str], expected: Union[List[str], Set[str]], k: int) -> float:
    """Compute Normalized DCG (NDCG) at K.
    
    NDCG@K = DCG@K / IDCG@K
    
    Where IDCG is the ideal DCG (all relevant items at top positions).
    
    Args:
        retrieved: Ordered list of retrieved item IDs (best first)
        expected: Set or list of relevant/expected item IDs
        k: Number of top retrieved items to consider
        
    Returns:
        NDCG@K score in [0.0, 1.0]
    """
    expected_set = _to_set(expected)
    
    if not expected_set:
        return 1.0  # Nothing to find, perfect score
    
    dcg = dcg_at_k(retrieved, expected_set, k)
    
    # Compute ideal DCG: all relevant items at top positions
    ideal_relevance = [1.0] * min(len(expected_set), k)
    idcg = 0.0
    for i, rel in enumerate(ideal_relevance, start=1):
        idcg += rel / math.log2(i + 1)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def hit_rate_at_k(retrieved: List[str], expected: Union[List[str], Set[str]], k: int) -> float:
    """Compute hit rate@K: 1 if any expected item is in top-K, 0 otherwise.
    
    This is a binary metric useful for "did we find at least one relevant item?"
    
    Args:
        retrieved: Ordered list of retrieved item IDs (best first)
        expected: Set or list of relevant/expected item IDs
        k: Number of top retrieved items to consider
        
    Returns:
        Hit rate (0.0 or 1.0)
    """
    expected_set = _to_set(expected)
    
    if not expected_set:
        return 1.0
    
    retrieved_k = set(retrieved[:k])
    return 1.0 if (expected_set & retrieved_k) else 0.0


def average_precision(retrieved: List[str], expected: Union[List[str], Set[str]], k: int = None) -> float:
    """Compute Average Precision (AP).
    
    AP = (1/R) * Σ(P@k * rel(k)) for k = 1 to K
    
    Where R is the number of relevant items, P@k is precision at k,
    and rel(k) is 1 if item k is relevant, 0 otherwise.
    
    Args:
        retrieved: Ordered list of retrieved item IDs (best first)
        expected: Set or list of relevant/expected item IDs
        k: Maximum rank to consider (None = all retrieved)
        
    Returns:
        Average Precision in [0.0, 1.0]
    """
    expected_set = _to_set(expected)
    
    if not expected_set:
        return 1.0
    
    if k is None:
        k = len(retrieved)
    
    relevant_count = 0
    precision_sum = 0.0
    
    for i, item in enumerate(retrieved[:k], start=1):
        if item in expected_set:
            relevant_count += 1
            precision_sum += relevant_count / i
    
    if relevant_count == 0:
        return 0.0
    
    return precision_sum / relevant_count


if __name__ == "__main__":
    # Test metrics with example data
    retrieved = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
    expected = {"doc_2", "doc_4", "doc_6"}  # doc_6 is not retrieved
    
    print("Retrieved:", retrieved)
    print("Expected:", expected)
    print()
    
    for k in [1, 3, 5, 10]:
        print(f"@K={k}:")
        print(f"  Recall:    {recall_at_k(retrieved, expected, k):.4f}")
        print(f"  Precision: {precision_at_k(retrieved, expected, k):.4f}")
        print(f"  F1:        {f1_at_k(retrieved, expected, k):.4f}")
        print(f"  NDCG:      {ndcg_at_k(retrieved, expected, k):.4f}")
        print(f"  Hit Rate:  {hit_rate_at_k(retrieved, expected, k):.4f}")
        print()
    
    print(f"MRR:      {mrr(retrieved, expected):.4f}")
    print(f"AP@5:     {average_precision(retrieved, expected, 5):.4f}")
