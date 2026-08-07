"""tsm — the TurboSuperMemory Python SDK.

A one-flag preset over the compiled turbomemory engine: scoped, self-correcting
conversational memory with verified belief revision and budget-aware recall.
"""

from .interfaces import Embedder, Extractor, Verifier
from .memory import CONVERSATIONAL_PROFILE, Memory

__version__ = "0.1.0"

# Convenience alias matching the requested public surface: the preset mapping.
MemoryConfig = CONVERSATIONAL_PROFILE

__all__ = [
    "Memory",
    "MemoryConfig",
    "Embedder",
    "Extractor",
    "Verifier",
    "__version__",
]
