"""TurboSuperMemory Cognitive Evaluation Benchmark.

Industry-standard benchmark harness for validating memory quality
against LongMemEval and LoCoMo datasets.
"""

import os as _os
import sys as _sys

# The shipped SDK (`tsm`) lives at the repo root. The harness imports its
# budgeting, gist, concept and ranking logic from there, so the evaluations
# measure the code users actually run instead of a private copy of it.
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

__version__ = "0.1.0"
