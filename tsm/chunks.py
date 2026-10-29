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
from .when import said_on

# About two sentences of an assistant reply; a typical user message (57
# tokens on average) stays one passage.
PASSAGE_TOKENS = 100

# What a passage's role is called: the speaker's role behind this prefix, the
# user's own as the bare prefix (as a gist of the user's facts is "summary").
MESSAGE_ROLE = "message"

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
# A list number ("27.", "3)", "(4)") belongs to the item after it. The
# sentence split leaves it at the end of the item before ("... etc.) 28.") or
# on its own, and a cut there separated an item from its number: asked for
# the 27th entry of a list, a reader found "Sound effects" with no number.
_LIST_NUMBER = r"\(?\d{1,3}[.)]"
_TRAILING_NUMBER = re.compile(r"\s(" + _LIST_NUMBER + r")$")
_ONLY_NUMBER = re.compile(r"^" + _LIST_NUMBER + r"$")
LIST_RUN = 3


def _number(mark: Optional[str]) -> Optional[int]:
    digits = re.sub(r"\D", "", mark or "")
    return int(digits) if digits else None


def _sentences(text: str) -> List[str]:
    """``text`` split between sentences, each list number kept with the item
    it numbers. A number that ends a sentence is taken for a list number
    only in a run (the sentence before or after ends with the number before
    or after it), so "... about topic 12." stays as it was said. The order
    of the words does not change."""
    raw = [s.strip() for s in _SENTENCE.split(text)]
    raw = [s for s in raw if s]
    ends = []
    for sentence in raw:
        trailing = _TRAILING_NUMBER.search(sentence)
        ends.append(_number(trailing.group(1)) if trailing else None)
    # Runs of sentences ending n, n+1, ...: a list when the run starts at 1
    # or is at least LIST_RUN long.
    in_list = [False] * len(raw)
    k = 0
    while k < len(raw):
        j = k
        while (ends[j] is not None and j + 1 < len(raw) and ends[j + 1] is not None
               and ends[j + 1] == ends[j] + 1):
            j += 1
        if j > k and (ends[k] == 1 or j - k + 1 >= LIST_RUN):
            in_list[k:j + 1] = [True] * (j - k + 1)
        k = j + 1
    out: List[str] = []
    carry = ""
    for k, sentence in enumerate(raw):
        if carry:
            sentence, carry = f"{carry} {sentence}", ""
        if _ONLY_NUMBER.match(sentence):
            carry = sentence
            continue
        trailing = _TRAILING_NUMBER.search(sentence)
        if trailing and in_list[k]:
            carry = trailing.group(1)
            sentence = sentence[:trailing.start()].rstrip()
        out.append(sentence)
    if carry:
        out.append(carry)
    return out


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


def occasion(role: Optional[str], timestamp) -> Optional[int]:
    """The day a passage of a message was said (a date ordinal), for the
    packer: two passages said on different days are two occurrences and are
    never dropped as duplicates of each other. ``None`` for anything that is
    not a passage or has no date."""
    if not is_message(role):
        return None
    day = said_on(timestamp)
    return day.toordinal() if day else None


def passages(text: str, max_tokens: int = PASSAGE_TOKENS) -> List[str]:
    """``text`` cut into consecutive passages of at most ``max_tokens``.

    Cuts fall between sentences, never between a list number and its item;
    a sentence longer than the cap is cut between words. Nothing is dropped
    and nothing repeated: the passages, joined with spaces, read as the
    message (whitespace aside).
    """
    text = (text or "").strip()
    if not text:
        return []
    if estimate_tokens(text) <= max_tokens:
        return [text]
    units: List[str] = []
    for sentence in _sentences(text):
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
