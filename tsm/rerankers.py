"""MultiVector (ColBERT / LFM2.5) late-interaction rerankers for TSM.

Provides token-level MaxSim precision reranking over candidate shortlists
retrieved from TSM's tiered vector and cognitive graph layer.
"""

from typing import List, Optional, Sequence
import numpy as np
import logging

logger = logging.getLogger("tsm.rerankers")


class ColBertReranker:
    """Multi-Vector Late-Interaction (ColBERT / LFM2.5) Stage-2 Precision Reranker.

    Computes token-level MaxSim late interaction:
        Score(Q, D) = sum_{q in Q} max_{d in D} (E_q . E_d)
    """

    def __init__(
        self,
        model_name: str = "LiquidAI/LFM2.5-ColBERT-350M",
        device: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        import torch
        from sentence_transformers import MultiVectorEncoder

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name

        logger.info(f"Loading ColBERT MultiVectorEncoder '{model_name}' on {device}...")
        self.model = MultiVectorEncoder(model_name, trust_remote_code=trust_remote_code, device=device)

    def rerank(self, query: str, texts: Sequence[str]) -> np.ndarray:
        """Compute MaxSim late-interaction scores for candidate texts against query.

        Returns a 1-D float32 numpy array of MaxSim scores.
        """
        if not texts:
            return np.array([], dtype=np.float32)

        q_embs = self.model.encode_query([query])
        doc_embs = self.model.encode_document(list(texts))
        scores = self.model.similarity(q_embs, doc_embs)[0]

        if hasattr(scores, "cpu"):
            scores = scores.cpu().numpy()
        return np.asarray(scores, dtype=np.float32)
