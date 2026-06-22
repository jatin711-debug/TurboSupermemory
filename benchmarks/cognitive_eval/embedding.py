"""Embedding providers for cognitive benchmarks.

Supports both sentence-transformers and direct transformers fallback.
Includes batch encoding for efficient local hardware usage.
"""
import logging
import os
from typing import List, Union

import numpy as np
import torch

logger = logging.getLogger("cognitive_eval.embedding")


class SimpleEmbeddingProvider:
    """Embedding provider using transformers directly with batch support.
    
    Avoids sentence-transformers torchcodec/FFmpeg dependency issues on Windows.
    Supports batch encoding for efficient processing.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", device: str = None, batch_size: int = 32):
        """Initialize embedding provider.
        
        Args:
            model_name: HuggingFace model name
            device: 'cuda', 'cpu', or None for auto
            batch_size: Batch size for encoding (higher = faster but more memory)
        """
        from transformers import AutoModel, AutoTokenizer
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        logger.info("Loading model %s on %s (batch_size=%d)", model_name, self.device, batch_size)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Get dimension from model config
        self._dim = self.model.config.hidden_size
        logger.info("Model loaded: dim=%d", self._dim)
    
    @property
    def get_sentence_embedding_dimension(self) -> int:
        return self._dim
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling with attention mask."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """Encode text(s) to embeddings.
        
        Args:
            texts: Single text or list of texts
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            numpy array of embeddings (1D for single text, 2D for list)
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                encoded = self.tokenizer(
                    batch, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors='pt'
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                model_output = self.model(**encoded)
                embeddings = self._mean_pooling(model_output, encoded['attention_mask'])
                
                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        result = np.vstack(all_embeddings)
        return result[0] if single else result
    
    def encode_batch(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Batch encode with optional progress bar.
        
        Args:
            texts: List of texts to encode
            show_progress: Whether to show tqdm progress bar
            
        Returns:
            2D numpy array of embeddings
        """
        if show_progress:
            try:
                from tqdm import tqdm
                all_embeddings = []
                with torch.no_grad():
                    for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding"):
                        batch = texts[i:i + self.batch_size]
                        encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
                        encoded = {k: v.to(self.device) for k, v in encoded.items()}
                        model_output = self.model(**encoded)
                        embeddings = self._mean_pooling(model_output, encoded['attention_mask'])
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                        all_embeddings.append(embeddings.cpu().numpy())
                return np.vstack(all_embeddings)
            except ImportError:
                pass
        
        return self.encode(texts)


def create_embedding_provider(model_name: str = None, batch_size: int = 32) -> SimpleEmbeddingProvider:
    """Create embedding provider with smart defaults for local hardware.
    
    Args:
        model_name: Model name or None for auto-selection based on VRAM
        batch_size: Batch size for encoding
        
    Returns:
        SimpleEmbeddingProvider instance
    """
    if model_name is None:
        # Auto-select model based on available VRAM
        if torch.cuda.is_available():
            vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            logger.info("Detected VRAM: %.0f MB", vram_mb)
            
            if vram_mb < 6000:  # Less than 6GB VRAM
                model_name = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dim, ~80MB
                batch_size = min(batch_size, 64)  # Can use larger batch with small model
                logger.info("Low VRAM detected. Using lightweight model: %s", model_name)
            else:
                model_name = "BAAI/bge-large-en-v1.5"  # 1024 dim, ~1.3GB
                batch_size = min(batch_size, 32)
        else:
            # CPU only - use lightweight model
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            batch_size = min(batch_size, 32)
            logger.info("No GPU detected. Using lightweight model: %s", model_name)
    
    return SimpleEmbeddingProvider(model_name=model_name, batch_size=batch_size)


if __name__ == "__main__":
    # Test the provider
    provider = create_embedding_provider()
    print(f"Model: {provider.model_name}")
    print(f"Dim: {provider.get_sentence_embedding_dimension}")
    
    # Test single encoding
    emb = provider.encode("This is a test")
    print(f"Single embedding shape: {emb.shape}")
    
    # Test batch encoding
    texts = [f"This is test sentence {i}" for i in range(100)]
    embs = provider.encode_batch(texts, show_progress=True)
    print(f"Batch embedding shape: {embs.shape}")
