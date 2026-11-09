"""System adapters for benchmark comparison.

``TSMAdapter`` is the evaluation adapter the head-to-head runners measure
(its own settings and additions; ``tsm.Memory`` is measured by
``shipped_stack_eval.py``). Mem0 is driven directly by
``head_to_head_eval.make_mem0``.
"""

__all__ = ["TSMAdapter"]
