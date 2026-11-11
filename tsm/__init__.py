"""tsm — the TurboSuperMemory Python SDK.

A one-flag preset over the compiled turbomemory engine: scoped, self-correcting
conversational memory with verified belief revision and budget-aware recall.
"""

from .embedders import OpenAIEmbedder, SentenceTransformerEmbedder
from .extractors import GlinerExtractor, OpenAIExtractor
from .gist import ExtractiveGistSummarizer, OpenAIGistSummarizer
from .identity import EmbedderMismatchError
from .interfaces import AsymmetricEmbedder, Embedder, Extractor, Verifier
from .memory import CONVERSATIONAL_PROFILE, Memory

# Keep in step with `version` in pyproject.toml (checked by the unit tests).
__version__ = "0.2.0"

# Convenience alias matching the requested public surface: the preset mapping.
MemoryConfig = CONVERSATIONAL_PROFILE

__all__ = [
    "Memory",
    "MemoryConfig",
    "Embedder",
    "AsymmetricEmbedder",
    "EmbedderMismatchError",
    "Extractor",
    "Verifier",
    "OpenAIEmbedder",
    "SentenceTransformerEmbedder",
    "OpenAIExtractor",
    "GlinerExtractor",
    "OpenAIGistSummarizer",
    "ExtractiveGistSummarizer",
    "__version__",
]
