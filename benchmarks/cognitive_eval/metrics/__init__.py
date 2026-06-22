"""Benchmark metrics for cognitive evaluation.

Provides standard information retrieval metrics:
- recall@K: What fraction of relevant items were retrieved?
- MRR: Mean Reciprocal Rank (how high up was the first relevant item?)
- NDCG: Normalized Discounted Cumulative Gain (accounts for ranking quality)
"""

__all__ = ["recall_at_k", "mrr", "ndcg_at_k", "precision_at_k", "f1_at_k"]
