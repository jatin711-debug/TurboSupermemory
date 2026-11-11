"""What "forgotten" means for text that was derived from a forgotten memory.

``Memory.forget`` deletes memories outright. What it cannot delete outright
is text that was made from them and holds other things too: a gist that
folded a forgotten fact in with twenty others, a cached extraction, a lesson
that quotes it. Deleting those whole would lose what they hold besides;
keeping them would serve the forgotten content again. So a piece of text is
judged sentence by sentence:

``carries(text, forgotten)`` is true when ``text`` holds a forgotten
memory's content:

- it contains a forgotten text verbatim (case and spacing aside; a
  forgotten text under four words only when it is the whole text);
- or one of its distinctive tokens: a number of three or more digits that is
  not a year (a code, an amount, a phone number), or a token with ``@``
  (an address);
- or most (``TERM_SHARE``) of its salient words, when it has at least
  ``MIN_TERMS`` of them ("my sister Ana lives in Porto" is carried by "Ana's
  home is Porto, where her sister ..." and not by "I visited Porto").

``redact(text, forgotten)`` keeps the sentences that carry nothing.

Rule-based and English-leaning like the rest of the SDK's text rules; it
errs toward removing (a cache entry removed is only paid for again).
"""

import re
from typing import Iterable, List, Sequence, Set, Tuple

from .monitor import _stem, salient_terms

TERM_SHARE = 0.6
MIN_TERMS = 2

_TOKEN = re.compile(r"[\w@.+-]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_TAG = re.compile(r"^\s*\[[^\]]*\]\s*")
_YEAR = re.compile(r"^(?:19|20)\d\d$")


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def distinctive_tokens(text: str) -> Set[str]:
    """Numbers of three or more digits that are not years, and tokens with
    ``@``, lower-cased and stripped of surrounding punctuation."""
    out: Set[str] = set()
    for raw in _TOKEN.findall(text or ""):
        token = raw.strip(".,;:!?()[]{}\"'").lower()
        digits = re.sub(r"\D", "", token)
        if "@" in token and len(token) > 3:
            out.add(token)
        elif len(digits) >= 3 and not _YEAR.match(token):
            out.add(token)
    return out


class Forgotten:
    """The forgotten texts, prepared once for many ``carries`` checks."""

    def __init__(self, texts: Iterable[str]):
        self.texts: List[str] = []
        self.tokens: Set[str] = set()
        self.term_sets: List[List[str]] = []
        for text in texts:
            text = _TAG.sub("", text or "").strip()
            if not text:
                continue
            self.texts.append(_norm(text))
            self.tokens |= distinctive_tokens(text)
            terms = [_stem(t) for t in salient_terms(text)]
            if len(terms) >= MIN_TERMS:
                self.term_sets.append(terms)

    def __bool__(self) -> bool:
        return bool(self.texts)

    def carries(self, text: str) -> bool:
        if not text or not self:
            return False
        norm = _norm(text)
        # Verbatim: anywhere for a text of four words or more; a shorter one
        # ("yes.", "ok, thanks") only as the whole text.
        if any(t in norm if len(t.split()) >= 4 else t == norm for t in self.texts):
            return True
        if self.tokens & distinctive_tokens(text):
            return True
        words = {_stem(w) for w in re.findall(r"[a-z0-9]+", norm)}
        return any(sum(t in words for t in terms) >= TERM_SHARE * len(terms)
                   for terms in self.term_sets)


def carries(text: str, forgotten: Sequence[str]) -> bool:
    """True when ``text`` holds the content of one of ``forgotten``."""
    return Forgotten(forgotten).carries(text)


def redact(text: str, forgotten) -> Tuple[str, List[str]]:
    """``(kept, removed)``: the sentences of ``text`` that carry none of
    ``forgotten`` (a ``Forgotten`` or a list of texts), joined, and those
    that do."""
    if not isinstance(forgotten, Forgotten):
        forgotten = Forgotten(forgotten)
    kept, removed = [], []
    for sentence in _SENTENCE.split(text or ""):
        sentence = sentence.strip()
        if not sentence:
            continue
        (removed if forgotten.carries(sentence) else kept).append(sentence)
    return " ".join(kept), removed
