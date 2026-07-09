"""Mem0 adapter for benchmark comparison.

Provides the same interface as TSMAdapter but uses Mem0 underneath.
This allows head-to-head comparison on the same benchmarks.

Note: Mem0 must be installed: pip install mem0ai
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger("cognitive_eval.adapters.mem0")


class Mem0Adapter:
    """Adapter for Mem0 memory system.
    
    Usage:
        adapter = Mem0Adapter()
        adapter.add(messages, user_id="user_123")
        results = adapter.search("Where does the user live?", user_id="user_123", top_k=3)
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
    ):
        """Initialize the Mem0 adapter.
        
        Args:
            config: Optional Mem0 configuration dict
        """
        try:
            from mem0 import Memory
            from mem0.configs.base import MemoryConfig
            
            # Use local embedder to avoid API keys
            mem0_config = MemoryConfig(
                embedder={
                    "provider": "huggingface",
                    "config": {
                        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    }
                },
                vector_store={
                    "provider": "chroma",
                    "config": {
                        "collection_name": "mem0_benchmark",
                        "path": "/tmp/mem0_db",
                    }
                }
            )
            
            self.memory = Memory(config=mem0_config)
            logger.info("Mem0 adapter initialized")
        except ImportError:
            raise RuntimeError(
                "mem0ai not installed. "
                "Install with: pip install mem0ai"
            )
    
    def add(self, messages: List[Dict], user_id: Optional[str] = None) -> None:
        """Add conversation messages to memory.
        
        Args:
            messages: List of message dicts with keys:
                - role: "user" or "assistant"
                - content: Message text
            user_id: Optional user ID for scoping
        """
        # Mem0 expects a list of message dicts
        mem0_messages = [
            {
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            }
            for msg in messages
        ]
        
        self.memory.add(mem0_messages, user_id=user_id)
    
    def search(self, query: str, user_id: Optional[str] = None, top_k: int = 3) -> List[Dict]:
        """Search memories.
        
        Args:
            query: Search query text
            user_id: Optional user ID for scoping
            top_k: Number of results to return
            
        Returns:
            List of result dicts with keys: id, score, text
        """
        results = self.memory.search(
            query=query,
            user_id=user_id,
            top_k=top_k,
        )
        
        return [
            {
                "id": r.get("id", ""),
                "score": r.get("score", 0.0),
                "text": r.get("memory", ""),
            }
            for r in results
        ]
    
    def search_ann(self, query: str, user_id: Optional[str] = None, top_k: int = 3) -> List[Dict]:
        """Pure ANN search (Mem0 doesn't expose this separately)."""
        # Mem0 always uses its full retrieval pipeline
        # For comparison, we just call search() but note the limitation
        logger.warning("Mem0 does not expose pure ANN search. Using full retrieval.")
        return self.search(query, user_id, top_k)
    
    def trigger_consolidation(self) -> None:
        """No-op for Mem0 (consolidation is automatic)."""
        pass
    
    def close(self) -> None:
        """No-op for Mem0."""
        pass
    
    def get_stats(self) -> Dict:
        """Get adapter statistics."""
        return {}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Mem0Adapter...")
    
    try:
        adapter = Mem0Adapter()
        
        messages = [
            {"role": "user", "content": "I just moved to San Francisco."},
            {"role": "assistant", "content": "Great! How do you like it?"},
        ]
        
        print("\nAdding messages...")
        adapter.add(messages, user_id="user_123")
        
        print("\nSearching...")
        results = adapter.search("Where does the user live?", user_id="user_123", top_k=3)
        
        print(f"\nResults ({len(results)}):")
        for r in results:
            print(f"  {r['id']}: score={r['score']:.4f}, text={r['text'][:50]}...")
        
        adapter.close()
        print("\nTest complete!")
        
    except Exception as e:
        print(f"Mem0 test failed (expected if mem0ai not installed): {e}")
