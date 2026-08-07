"""Locate and import the compiled ``turbomemory`` PyO3 extension.

The extension (``turbomemory.pyd`` / ``turbomemory.so``) is built by
``make build-python`` and placed at the repository root. When ``tsm`` is used
from inside that repository (or installed alongside the artifact), this loader
finds it without any manual ``sys.path`` plumbing. If ``turbomemory`` is
already importable (e.g. properly installed into site-packages), it is used
as-is.
"""

import os
import sys

_MAX_ANCESTORS = 4


def load_turbomemory():
    """Import and return the ``turbomemory`` extension module."""
    try:
        import turbomemory

        return turbomemory
    except ImportError:
        pass

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [here]
    parent = here
    for _ in range(_MAX_ANCESTORS):
        parent = os.path.dirname(parent)
        if parent not in candidates:
            candidates.append(parent)

    for directory in candidates:
        for ext in (".pyd", ".so"):
            artifact = os.path.join(directory, f"turbomemory{ext}")
            if os.path.exists(artifact):
                if directory not in sys.path:
                    sys.path.insert(0, directory)
                import turbomemory

                return turbomemory

    raise ImportError(
        "The compiled 'turbomemory' extension was not found. Build it with "
        "'make build-python' (it is placed at the repository root), or install "
        "it so that 'import turbomemory' works."
    )
