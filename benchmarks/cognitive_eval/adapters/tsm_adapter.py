"""TurboSuperMemory adapter for benchmark compatibility.

Makes TSM behave like Mem0 for benchmark evaluation:
- Mem0-style `add(messages, user_id)` API
- Mem0-style `search(query, user_id, top_k)` API
- Automatic fact extraction via LLM
- Temporal metadata tracking
"""

import json
import logging
import os
import shutil
import sys
import tempfile
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger("cognitive_eval.adapters.tsm")


def _setup_turbomemory():
    """Locate and load the compiled turbomemory extension."""
    # Find project root (2 levels up from this file: adapters/ -> cognitive_eval/ -> benchmarks/ -> project_root)
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    project_root = os.path.dirname(script_dir)
    ext = ".pyd" if sys.platform.startswith("win") else ".so"
    pyd = os.path.join(project_root, f"turbomemory{ext}")
    
    # Try to find and copy the DLL if needed
    if sys.platform.startswith("win") and not os.path.exists(pyd):
        dll = os.path.join(project_root, "target", "release", "turbomemory.dll")
        if os.path.exists(dll):
            shutil.copy2(dll, pyd)
    
    if not os.path.exists(pyd):
        raise RuntimeError(
            f"turbomemory extension not found at {pyd}. "
            "Run 'make build-python' first."
        )
    
    sys.path.insert(0, project_root)
    import turbomemory
    return turbomemory


class TSMAdapter:
    """Adapter that makes TSM behave like Mem0 for benchmark compatibility.
    
    Usage:
        adapter = TSMAdapter(
            db_path="./eval_db",
            embedding_model="BAAI/bge-large-en-v1.5",
            extractor="ollama",  # or "mock"
        )
        
        # Add conversation (Mem0-style)
        adapter.add(messages, user_id="user_123")
        
        # Search (Mem0-style)
        results = adapter.search("Where does the user live?", user_id="user_123", top_k=3)
    """
    
    def __init__(
        self,
        db_path: str,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        extractor: str = "mock",
        extractor_model: str = "llama3.2:3b",
        cognitive_features: bool = True,
        dimension: Optional[int] = None,
        **kwargs,
    ):
        """Initialize the TSM adapter.
        
        Args:
            db_path: Path to TSM database directory
            embedding_model: SentenceTransformer model name for embeddings
            extractor: Fact extractor to use ("ollama", "mock")
            extractor_model: Ollama model name (if using Ollama)
            cognitive_features: Enable cognitive layer features
            dimension: Embedding dimension (auto-detected if None)
        """
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        
        # Load embedding model (with fallback for sentence-transformers issues)
        # We try sentence-transformers first, but on Windows it often fails
        # due to torchcodec/FFmpeg dependency issues
        self.model = None
        self.dim = dimension
        
        # Try sentence-transformers first (only if not on Windows or if explicitly requested)
        if sys.platform != "win32":
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(embedding_model)
                self.dim = dimension or self.model.get_sentence_embedding_dimension()
                logger.info("Loaded embedding model: %s (dim=%d)", embedding_model, self.dim)
            except Exception as e:
                logger.warning("sentence-transformers failed: %s", e)
        
        # Fallback to transformers directly
        if self.model is None:
            logger.info("Using transformers fallback for embeddings")
            from ..embedding import create_embedding_provider
            # Pass batch_size if provided in kwargs
            batch_size = kwargs.get('batch_size', 32)
            self.model = create_embedding_provider(embedding_model, batch_size=batch_size)
            # Handle both property (SimpleEmbeddingProvider) and method (SentenceTransformer)
            dim_attr = self.model.get_sentence_embedding_dimension
            self.dim = dimension or (dim_attr() if callable(dim_attr) else dim_attr)
            logger.info("Loaded embedding model via fallback: %s (dim=%d)", embedding_model, self.dim)
        
        # Load turbomemory
        self.tsm = _setup_turbomemory()
        
        # Initialize extractor
        if extractor == "ollama":
            from ..extraction.ollama import OllamaExtractor
            self.extractor = OllamaExtractor(model=extractor_model)
        else:
            from ..extraction.mock import MockExtractor
            self.extractor = MockExtractor()
            logger.info("Using mock extractor (no LLM)")
        
        # Initialize TSM engine with cognitive features
        config = {
            "db_path": db_path,
            "dimension": self.dim,
            "max_concepts": 10,
            "auto_consolidation_secs": 0,  # Manual consolidation for determinism
        }
        
        if cognitive_features:
            config.update({
                "refinement_cosine_threshold": 0.5,
                "contradiction_cosine_threshold": 0.5,
                "contradiction_text_threshold": 0.3,
                "contradiction_weaken_factor": 0.5,
                "cognitive_alpha": 0.3,
                "spreading_iterations": 6,
                "spreading_decay": 0.7,
                "spreading_beta": 0.0,
                "importance_auto_scoring": True,
                "concept_evolution_enabled": True,
                "abstraction_co_occurrence_threshold": 3,
            })
        
        self.engine = self.tsm.MemoryEngine(**config)
        logger.info("TSM engine initialized with cognitive_features=%s", cognitive_features)
        
        # Counter for unique IDs
        self._insert_counter = 0
    
    def add(self, messages: List[Dict], user_id: Optional[str] = None, batch: bool = True) -> None:
        """Add conversation messages to memory (Mem0-compatible API).
        
        Args:
            messages: List of message dicts or Message objects with keys:
                - role: "user" or "assistant"
                - content: Message text
                - timestamp: ISO timestamp string
            user_id: Optional user/conversation ID for scoping
            batch: Whether to use batch embedding (faster, more memory)
        """
        # Normalize messages to dicts
        msg_dicts = []
        for msg in messages:
            if hasattr(msg, 'content'):
                msg_dicts.append({
                    'content': msg.content,
                    'role': getattr(msg, 'role', 'user'),
                    'timestamp': getattr(msg, 'timestamp', ''),
                })
            else:
                msg_dicts.append(msg)
        
        # Extract all facts first
        context = []
        all_facts = []
        fact_metadata = []
        
        for msg in msg_dicts:
            content = msg.get("content", "")
            if not content or not content.strip():
                continue
            
            facts = self.extractor.extract_facts(content, context)
            context.append(content)
            
            for fact in facts:
                all_facts.append(fact)
                fact_metadata.append({
                    'role': msg.get("role", "user"),
                    'timestamp': msg.get("timestamp", ""),
                    'content': content,
                })
        
        # Batch embed all facts
        if all_facts:
            if batch and len(all_facts) > 1:
                embeddings = self.model.encode(all_facts)
            else:
                embeddings = np.vstack([self.model.encode(f) for f in all_facts])
            
            # Insert all facts
            for i, (fact, meta) in enumerate(zip(all_facts, fact_metadata)):
                self._insert_counter += 1
                memory_id = f"{user_id}_{self._insert_counter}" if user_id else f"mem_{self._insert_counter}"
                
                self.engine.insert(
                    id=memory_id,
                    text=fact,
                    embedding=embeddings[i].astype(np.float32),
                    importance_score=1.0,
                    concepts=[],
                    payload=json.dumps({
                        "timestamp": meta['timestamp'],
                        "role": meta['role'],
                        "user_id": user_id,
                        "original_message": meta['content'],
                    }),
                    scope=user_id,
                )
    
    def search(self, query: str, user_id: Optional[str] = None, top_k: int = 3, use_cognitive: bool = False) -> List[Dict]:
        """Search memories (Mem0-compatible API).
        
        Args:
            query: Search query text
            user_id: Optional user/conversation ID for scoping
            top_k: Number of results to return
            use_cognitive: Whether to use cognitive graph (slower but smarter) or direct ANN (faster)
                
                - False (default): Direct ANN search. Fast (~25ms), pure vector similarity.
                  Use this for benchmarking retrieval speed and comparing with other systems.
                
                - True: Full cognitive search. Slower (~350ms), but includes:
                  - Spreading activation through memory graph
                  - Temporal reasoning (prefers recent facts)
                  - Contradiction handling (weakens outdated facts)
                  - Concept linking (related facts boost each other)
                  - FOK gate (returns None if confidence too low)
                  
                  Use this for production AI agents that need cognitive reasoning.
        """
        query_embedding = self.model.encode(query)
        
        if use_cognitive:
            # Full cognitive search (spreading activation, FOK gate, etc.)
            results = self.engine.search(
                query_text=query,
                query_embedding=query_embedding.astype(np.float32),
                top_k=top_k,
                scope=user_id,
            )
        else:
            # Direct ANN search (fast, no cognitive overhead)
            results = self.engine.search_ann(
                query_embedding.astype(np.float32),
                top_k,
            )
        
        if not results:
            return []
        
        return [
            {
                "id": r[0],
                "score": float(r[1]),
            }
            for r in results
        ]
    
    def search_ann(self, query: str, user_id: Optional[str] = None, top_k: int = 3) -> List[Dict]:
        """Pure ANN search without cognitive layer (for comparison).
        
        Args:
            query: Search query text
            user_id: Optional user/conversation ID for scoping
            top_k: Number of results to return
            
        Returns:
            List of result dicts with keys: id, score
        """
        query_embedding = self.model.encode(query)
        
        results = self.engine.search_ann(
            query_embedding=query_embedding.astype(np.float32),
            top_k=top_k,
            scope=user_id,
        )
        
        return [
            {
                "id": r[0],
                "score": float(r[1]),
            }
            for r in results
        ]
    
    def trigger_consolidation(self) -> None:
        """Trigger manual consolidation (for deterministic benchmarks)."""
        self.engine.trigger_consolidation()
    
    def close(self) -> None:
        """Close the TSM engine and clean up."""
        self.engine.close()
    
    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            "gpu_accelerated": self.engine.gpu_accelerated,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Quick test
    print("Testing TSMAdapter...")
    
    adapter = TSMAdapter(
        db_path=tempfile.mkdtemp(prefix="tsm_test_"),
        embedding_model="all-MiniLM-L6-v2",  # Small model for testing
        extractor="mock",
    )
    
    messages = [
        {"role": "user", "content": "I just moved to San Francisco.", "timestamp": "2024-01-15T10:00:00Z"},
        {"role": "assistant", "content": "Great! How do you like it?", "timestamp": "2024-01-15T10:01:00Z"},
        {"role": "user", "content": "I love the weather here.", "timestamp": "2024-01-15T10:02:00Z"},
    ]
    
    print("\nAdding messages...")
    adapter.add(messages, user_id="user_123")
    adapter.trigger_consolidation()
    
    print("\nSearching...")
    results = adapter.search("Where does the user live?", user_id="user_123", top_k=3)
    
    print(f"\nResults ({len(results)}):")
    for r in results:
        print(f"  {r['id']}: score={r['score']:.4f}")
    
    adapter.close()
    shutil.rmtree(adapter.db_path, ignore_errors=True)
    print("\nTest complete!")
