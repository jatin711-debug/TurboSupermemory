"""Facts as keys, the turn they came from as what is read.

A fact is a good thing to search on: short, one statement, worded the way a
question about it is worded. It is a worse thing to read. "The workshop was
free" does not say which workshop, and a third of what users say yields no
fact at all. LongMemEval's own finding (Wu et al., 2024, fact-augmented key
expansion) is to index the facts and hand the reader the turn they were
taken from: +9.4% recall and +5.4% accuracy there, where replacing the turns
with the facts lowered accuracy.

``turn_units`` does that for a scored pool of facts. Facts taken from one
message become one unit: the message as it was said, cut to the sentences
that matter when it is long, scored by its best fact. Memories without a
source message (gists, records written before the message was kept) stay
units of their own. Nothing here calls a model or an engine; ``Memory.recall``
uses it with ``sources="turn"`` and the evaluation harness imports it.
"""

import re
from typing import Dict, Iterable, List, Optional, Sequence, Set

from .budget import estimate_tokens

# A turn longer than this is cut to the sentences that bear on its facts and
# the question. User turns average about 57 tokens; assistant turns run to
# several hundred, and one of them would otherwise take a small budget.
SOURCE_TOKENS = 120

_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_STOP = frozenset(
    "the a an and or but of to in on at for with from by as is are was were be been being "
    "it its this that these those i me my we our you your he she they them his her their "
    "user assistant speaker what which who when where how did do does have has had not no "
    "so if then than there here just also about into over can could would should will".split())


def _words(text: str) -> Set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if len(w) > 2 and w not in _STOP}


def excerpt(message: str, keys: Iterable[str], max_tokens: int = SOURCE_TOKENS) -> str:
    """``message`` when it fits ``max_tokens``; otherwise its sentences that
    share the most words with ``keys`` (the facts taken from it, the
    question), in the order they were said, until the cap. Sentences that
    were not next to each other are joined with " ... "."""
    message = (message or "").strip()
    if estimate_tokens(message) <= max_tokens:
        return message
    sentences = [s.strip() for s in _SENTENCE.split(message) if s and s.strip()]
    if not sentences:
        return message[: max_tokens * 4].strip()
    wanted: Set[str] = set()
    for key in keys:
        wanted |= _words(key)
    scored = sorted(range(len(sentences)),
                    key=lambda i: (-len(_words(sentences[i]) & wanted), i))
    kept, used = [], 0
    for i in scored:
        tokens = estimate_tokens(sentences[i])
        if used + tokens > max_tokens:
            if not kept:  # one sentence longer than the cap: cut it
                kept.append(i)
                sentences[i] = sentences[i][: max_tokens * 4].rsplit(" ", 1)[0]
            continue
        kept.append(i)
        used += tokens
    kept.sort()
    parts = []
    for n, i in enumerate(kept):
        if n and i != kept[n - 1] + 1:
            parts.append("...")
        parts.append(sentences[i])
    return " ".join(parts)


def turn_units(pool: Sequence[Dict], query: str = "",
               max_tokens: int = SOURCE_TOKENS) -> List[Dict]:
    """Group a scored pool of facts by the message each was taken from.

    Each pool item needs ``"id"``, ``"text"`` and ``"score"``; one with a
    ``"source"`` (the message it was taken from) and a ``"turn_index"`` is
    grouped with the others from that turn. A unit is a copy of the turn's
    best-scoring fact with ``"text"`` set to the message (``excerpt``),
    ``"facts"`` the texts of the turn's facts in the pool, best first, and
    ``"ids"`` their ids. It is ``"superseded_by"`` something only when every
    fact of the turn in the pool is: a turn that also says something current
    is not stale. Items without a source pass through unchanged. Returned
    best first.
    """
    groups: Dict[tuple, List[Dict]] = {}
    units: List[Dict] = []
    for p in pool:
        source = p.get("source")
        turn = p.get("turn_index")
        if not source or turn is None:
            units.append(dict(p))
            continue
        groups.setdefault((p.get("scope"), turn, source), []).append(p)
    for members in groups.values():
        members = sorted(members, key=lambda p: -float(p["score"]))
        best = dict(members[0])
        source = best.pop("source")
        best["facts"] = [m["text"] or "" for m in members]
        best["ids"] = [m["id"] for m in members]
        best["text"] = excerpt(source, best["facts"] + [query], max_tokens)
        stale = [m.get("superseded_by") for m in members]
        if not all(stale):
            best.pop("superseded_by", None)
            best.pop("chain", None)
        units.append(best)
    units.sort(key=lambda u: -float(u["score"]))
    return units


def speaker(role: Optional[str]) -> str:
    """How a unit names who said it: ``"user: "``, ``"assistant: "``."""
    role = (role or "").strip()
    return f"{role}: " if role else ""
