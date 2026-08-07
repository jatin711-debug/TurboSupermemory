"""Safe OpenAI key loading for the eval harness.

The key is provided by the USER via a gitignored file (or the environment) and
read by this code at runtime — it is never passed through commands, printed, or
committed. `ensure_openai_key()` populates OPENAI_API_KEY from the first source
found, so the OpenAI SDK (which reads that env var) works without the key ever
crossing the tool/command boundary.

To supply the key, create ONE of these files containing just the key (sk-...):
    <repo-root>/openai_key.txt          (gitignored)
    <repo-root>/.openai_key             (gitignored)
    ~/.openai_key
or set OPENAI_API_KEY in an environment the eval process actually inherits.
"""

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANDIDATES = [
    _REPO_ROOT / "openai_key.txt",
    _REPO_ROOT / ".openai_key",
    Path.home() / ".openai_key",
]
_MINIMAX_CANDIDATES = [
    _REPO_ROOT / "minimax_key.txt",
    _REPO_ROOT / ".minimax_key",
    Path.home() / ".minimax_key",
]


def ensure_openai_key() -> bool:
    """Ensure OPENAI_API_KEY is in the environment. Returns True if a key is
    available (already-set or loaded from a key file), False otherwise. The raw
    value is never logged."""
    if os.environ.get("OPENAI_API_KEY"):
        return True
    for p in _CANDIDATES:
        try:
            if p.exists():
                key = p.read_text(encoding="utf-8").strip()
                # Tolerate `OPENAI_API_KEY=sk-...` or a bare `sk-...` line.
                if key.upper().startswith("OPENAI_API_KEY"):
                    key = key.split("=", 1)[-1].strip().strip('"').strip("'")
                if key:
                    os.environ["OPENAI_API_KEY"] = key
                    return True
        except OSError:
            continue
    return False


def key_file_hint() -> str:
    return (f"Create a file with your key (sk-...), e.g. {_CANDIDATES[0]} , or set "
            f"OPENAI_API_KEY in the environment. The file is gitignored and never printed.")


def ensure_minimax_key() -> bool:
    """Load MINIMAX_API_KEY from the environment or a gitignored key file."""
    if os.environ.get("MINIMAX_API_KEY"):
        return True
    for path in _MINIMAX_CANDIDATES:
        try:
            if not path.exists():
                continue
            key = path.read_text(encoding="utf-8").strip()
            if key.upper().startswith("MINIMAX_API_KEY"):
                key = key.split("=", 1)[-1].strip().strip('"').strip("'")
            if key:
                os.environ["MINIMAX_API_KEY"] = key
                return True
        except OSError:
            continue
    return False


def minimax_key_file_hint() -> str:
    return (
        f"Create a file containing the key, e.g. {_MINIMAX_CANDIDATES[0]}, or set "
        "MINIMAX_API_KEY. The file is gitignored and never printed."
    )
