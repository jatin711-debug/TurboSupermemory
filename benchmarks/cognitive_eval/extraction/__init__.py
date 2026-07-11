"""LLM-based fact extraction for benchmark compatibility.

Real extraction derives atomic facts from conversation messages with an LLM.
`create_extractor` selects a backend, preferring a real LLM:

  - "auto"   : Ollama if its server is reachable, else OpenAI (needs
               OPENAI_API_KEY), else a clear error — never a silent mock.
  - "ollama" : local Ollama.
  - "openai" : OpenAI API (needs OPENAI_API_KEY).
  - "mock"   : deterministic sentence-splitter — offline, no LLM. Kept ONLY for
               the fast offline regression gate; not for quality measurement.
"""

import logging
import os

logger = logging.getLogger("cognitive_eval.extraction")

__all__ = ["OllamaExtractor", "MockExtractor", "OpenAIExtractor", "create_extractor"]


def create_extractor(name: str = "auto", ollama_model: str = "qwen2.5:3b",
                     openai_model: str = "gpt-4o-mini", shared=None):
    """Build a fact extractor. `shared` (a prebuilt extractor) short-circuits so
    one instance — and its cross-arm cache — can be reused across adapters."""
    if shared is not None:
        return shared

    if name == "mock":
        from .mock import MockExtractor
        return MockExtractor()

    if name == "ollama":
        from .ollama import OllamaExtractor
        return OllamaExtractor(model=ollama_model)

    if name == "openai":
        from .openai_extractor import OpenAIExtractor
        return OpenAIExtractor(model=openai_model)

    if name == "auto":
        # Prefer a reachable local Ollama server (free, private).
        try:
            from .ollama import OllamaExtractor
            ext = OllamaExtractor(model=ollama_model)
            if ext.health_check():
                logger.info("auto extractor -> Ollama (%s)", ollama_model)
                return ext
            logger.info("Ollama installed but server unreachable; trying OpenAI.")
        except Exception as e:  # noqa: BLE001 — ollama missing/unavailable is expected
            logger.info("Ollama unavailable (%s); trying OpenAI.", e)
        if os.environ.get("OPENAI_API_KEY"):
            from .openai_extractor import OpenAIExtractor
            logger.info("auto extractor -> OpenAI (%s)", openai_model)
            return OpenAIExtractor(model=openai_model)
        raise RuntimeError(
            "auto extractor: no reachable Ollama server and no OPENAI_API_KEY. "
            "Start Ollama (`ollama serve`) or set OPENAI_API_KEY. (Use --extractor mock "
            "only for the offline regression gate.)"
        )

    raise ValueError(f"unknown extractor '{name}'")
