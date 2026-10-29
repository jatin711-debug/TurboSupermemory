"""Messages as they were said, kept next to the facts taken from them.

Extraction is lossy. On LongMemEval the extractor stores nothing from about a
third of user messages, and on the 355 held-out questions every one whose
evidence missed the context had an evidence turn with no fact taken from it
(``benchmarks/cognitive_eval/evidence_rank_report.py``, 2026-10-28). Systems
that keep the raw log next to what they derive from it lose less (Scroll,
arXiv 2608.21690; Eywa, 2605.30771), and from a few hundred tokens of context
up, raw text read back beats consolidated text (Retain or Consolidate,
2607.17545; Reproducing LightMem, 2607.29104).

``Memory(keep_messages=True)`` therefore also stores every message, cut into
passages of at most ``PASSAGE_TOKENS`` (a long assistant reply would
otherwise be one vector that matches nothing in particular and fill a small
budget on its own). A passage is a record of its own under a role of its own
(``message_role``): the engine's belief revision only compares user facts, so
two passages are never taken for an update of each other, while the role
prior still ranks a passage like what its speaker said. Nothing here calls a
model or an engine.
"""

import re
from typing import List, Optional

from .budget import estimate_tokens

# About two sentences of an assistant reply; a typical user message (57
# tokens on average) stays one passage.
PASSAGE_TOKENS = 100

# What a passage's role is called: the speaker's role behind this prefix, the
# user's own as the bare prefix (as a gist of the user's facts is "summary").
MESSAGE_ROLE = "message"

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def message_role(role: Optional[str]) -> str:
    """The stored role of a passage of a message said by ``role``."""
    role = (role or "user").strip() or "user"
    return MESSAGE_ROLE if role == "user" else f"{MESSAGE_ROLE}-{role}"


def is_message(role: Optional[str]) -> bool:
    """True for the stored role of a passage (``message_role``)."""
    return bool(role) and (role == MESSAGE_ROLE or role.startswith(MESSAGE_ROLE + "-"))


def spoken_by(role: Optional[str]) -> str:
    """Who said a passage stored under ``role``: ``"user"``, ``"assistant"``, ..."""
    if role == MESSAGE_ROLE:
        return "user"
    return role[len(MESSAGE_ROLE) + 1:] if is_message(role) else (role or "")


def passages(text: str, max_tokens: int = PASSAGE_TOKENS) -> List[str]:
    """``text`` cut into consecutive passages of at most ``max_tokens``.

    Cuts fall between sentences; a sentence longer than the cap is cut
    between words. Nothing is dropped and nothing repeated: the passages,
    joined with spaces, read as the message (whitespace aside).
    """
    text = (text or "").strip()
    if not text:
        return []
    if estimate_tokens(text) <= max_tokens:
        return [text]
    units: List[str] = []
    for sentence in (s.strip() for s in _SENTENCE.split(text)):
        if not sentence:
            continue
        if estimate_tokens(sentence) <= max_tokens:
            units.append(sentence)
            continue
        words, piece = sentence.split(), []
        for word in words:
            if piece and estimate_tokens(" ".join(piece + [word])) > max_tokens:
                units.append(" ".join(piece))
                piece = []
            piece.append(word)
        if piece:
            units.append(" ".join(piece))
    out: List[str] = []
    current: List[str] = []
    for unit in units:
        if current and estimate_tokens(" ".join(current + [unit])) > max_tokens:
            out.append(" ".join(current))
            current = []
        current.append(unit)
    if current:
        out.append(" ".join(current))
    return out
