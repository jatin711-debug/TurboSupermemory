"""Mock fact extractor for testing the benchmark harness without Ollama.

This extractor provides deterministic, fast fact extraction for testing
infrastructure. It does NOT use an LLM and produces simplified results.

Usage:
    extractor = MockExtractor()
    facts = extractor.extract_facts("I moved to San Francisco.")
    # Returns: ["I moved to San Francisco."]
"""

import logging
from typing import List, Optional

logger = logging.getLogger("cognitive_eval.extraction.mock")


class MockExtractor:
    """Mock fact extractor for testing.
    
    Splits messages on sentence boundaries and returns each sentence
    as a separate fact. No LLM calls, deterministic, fast.
    """
    
    def __init__(self, split_sentences: bool = True):
        self.split_sentences = split_sentences
        logger.info("MockExtractor initialized (split_sentences=%s)", split_sentences)
    
    def extract_facts(self, message: str, context: Optional[List[str]] = None) -> List[str]:
        """Extract facts by splitting on sentence boundaries.
        
        Args:
            message: The message text
            context: Ignored (mock doesn't use context)
            
        Returns:
            List of fact strings
        """
        if not message or not message.strip():
            return []
        
        if not self.split_sentences:
            return [message.strip()]
        
        # Simple sentence splitting
        import re
        sentences = re.split(r'(?<=[.!?])\s+', message.strip())
        facts = [s.strip() for s in sentences if s.strip()]
        
        return facts if facts else [message.strip()]
    
    def extract_facts_batch(
        self,
        messages: List[str],
        contexts: Optional[List[List[str]]] = None,
    ) -> List[List[str]]:
        """Extract facts from multiple messages."""
        return [self.extract_facts(msg) for msg in messages]
    
    def health_check(self) -> bool:
        """Always returns True (mock doesn't need external services)."""
        return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    extractor = MockExtractor()
    
    test_messages = [
        "I just moved to San Francisco. The weather is great!",
        "I love coffee. I drink it every morning. It's delicious.",
    ]
    
    print("Testing MockExtractor...")
    for msg in test_messages:
        print(f"\nMessage: {msg}")
        facts = extractor.extract_facts(msg)
        print("Facts:")
        for fact in facts:
            print(f"  - {fact}")
