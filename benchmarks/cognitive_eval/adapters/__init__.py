"""System adapters for benchmark comparison.

Provides unified interfaces for different memory systems (TSM, Mem0)
so benchmarks can run against any system with the same code.
"""

__all__ = ["TSMAdapter", "Mem0Adapter"]
