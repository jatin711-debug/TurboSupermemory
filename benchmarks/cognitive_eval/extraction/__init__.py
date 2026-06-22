"""LLM-based fact extraction for benchmark compatibility.

Mem0 uses single-pass LLM extraction to derive atomic facts from
conversation messages. This module provides compatible extractors using
local models (Ollama) or mock implementations for testing.
"""

__all__ = ["OllamaExtractor", "MockExtractor", "FactExtractor"]
