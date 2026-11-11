"""The read-time monitor: is what ``recall()`` found enough to answer from?

``Memory.checked_recall`` runs this after an ordinary budgeted recall. The
monitor never gates retrieval (it always looks first, the Feeling-of-Knowing
gate that decided before looking stays removed) and never writes: what it
decides is a second, targeted look whose results are swapped into the same
token budget, never added on top of it. Bigger contexts read worse
(2,000 -> 8,000 tokens cost 9 of 96 judged answers on LongMemEval-S), so the
budget does not grow.

The decision rests first on signals that come from outside the reader's own
judgement and cost nothing:

- the question points at a time ("ten days ago") and nothing said then is in
  the context (``window_empty``);
- a counting question ("how many ...") whose candidates hold strong matches
  from occasions the context does not cover (``left_out_occasions``): every
  mention has to be in front of the reader before it can count them;
- salient words of the question that no memory in the context contains
  (``missing_terms``);
- a superseded memory in the context whose replacement is not
  (``stale_unresolved``).

Only when one of those fires (or ``always_judge``) is a ``SufficiencyJudge``
asked, if one is installed, whether the context answers the question and, if
not, what is missing. A judge that says "sufficient" cancels the second look.
Without a judge the signals decide alone.

Everything the monitor decides is returned as a ``MonitorReport``, whose
``to_dict()`` is a replayable record (the signals, the judge's reply, the
action, the ids swapped in and out). Keep those records: a hand-built
controller that loses its evaluation still leaves labelled data for a learned
one.

Status (2026-11-11): a prototype. It has unit tests and no judged run. The
evaluation it needs before it can be trusted is in
``reports/Cognitive architecture for TurboSuperMemory.md`` (Phase 1): five
reads per context, against a majority-of-three reader on the same contexts
(the same number of calls), and it is deleted if it does not beat that.

Nothing here calls the engine; the judge calls a model only when asked.
"""

import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .ranking import is_counting_query
from .when import said_on

# Words that say nothing about what a question is about: function words,
# question words, and the words of time expressions (``tsm.when`` handles
# those). What is left are the words the evidence has to contain.
_STOP = frozenset("""
a about above after again against all also am an and any are as at be because been
before being below between both but by can could did do does doing done down during each
few for from further had has have having he her here hers herself him himself his how i
if in into is it its itself just me more most my myself no nor not now of off on once
only or other our ours ourselves out over own same she should so some such than that the
their theirs them themselves then there these they this those through to too under until
up very was we were what when where which while who whom why will with would you your
yours yourself yourselves tell remind recall remember mentioned mention said say says
told know think thought get got go going went many much total number times time ago
last past ever previous previously earlier recently recent first second next again
day days week weeks weekend month months year years today yesterday tomorrow
monday tuesday wednesday thursday friday saturday sunday morning evening night
january february march april may june july august september october november december
""".split())

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

# A score this far below the weakest memory in the context still counts as a
# strong match for ``left_out_occasions``.
OCCASION_SLACK = 0.03
# How many of the chosen scores the margin compares (best against this one).
MARGIN_DEPTH = 5


def salient_terms(question: str) -> List[str]:
    """The words of ``question`` the evidence would have to contain: no
    function or question words, no time words, at least three letters, in
    order, without repeats."""
    seen: List[str] = []
    for word in _WORD.findall((question or "").lower()):
        word = word.split("'")[0]
        if len(word) < 3 or word in _STOP or word.isdigit() or word in seen:
            continue
        seen.append(word)
    return seen


def _stem(word: str) -> str:
    """A crude stem, enough to match "weddings" with "wedding" and
    "attended" with "attend"."""
    for suffix in ("ings", "ing", "ies", "ied", "es", "ed", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _occasion_key(item: Dict):
    """What tells two mentions apart as two occasions: the packer's occasion
    (the day of a message passage), else the day it was said, else its turn."""
    if item.get("occasion") is not None:
        return ("day", item["occasion"])
    day = said_on(item.get("timestamp"))
    if day is not None:
        return ("day", day.toordinal())
    if item.get("turn_index") is not None:
        return ("turn", item["turn_index"])
    return ("id", item.get("id"))


@dataclass
class Signals:
    """What the monitor measured on one recall. All free: no model call."""
    windows: int = 0
    window_hits: int = 0
    counting: bool = False
    occasions_in_context: int = 0
    occasions_left_out: int = 0
    left_out_ids: List[str] = field(default_factory=list)
    missing_terms: List[str] = field(default_factory=list)
    stale_unresolved: int = 0
    margin: Optional[float] = None


def grounded_signals(question: str, chosen: Sequence[Dict], pool: Sequence[Dict],
                     windows: int = 0) -> Signals:
    """Measure ``chosen`` (what ``recall()`` packed) against the question and
    ``pool`` (every candidate it chose from, with scores). ``windows`` is how
    many time windows the question pointed at (``tsm.when.question_windows``).
    """
    signals = Signals(windows=int(windows), counting=is_counting_query(question))
    chosen_ids = {item.get("id") for item in chosen}
    signals.window_hits = sum(1 for item in chosen if item.get("in_window"))

    occasions = {_occasion_key(item) for item in chosen}
    signals.occasions_in_context = len(occasions)
    if signals.counting and chosen:
        floor = min(float(item.get("score", 0.0)) for item in chosen) - OCCASION_SLACK
        best_per_occasion: Dict = {}
        for item in pool:
            if item.get("id") in chosen_ids or float(item.get("score", 0.0)) < floor:
                continue
            key = _occasion_key(item)
            if key in occasions:
                continue
            current = best_per_occasion.get(key)
            if current is None or item["score"] > current["score"]:
                best_per_occasion[key] = item
        left_out = sorted(best_per_occasion.values(), key=lambda p: -float(p["score"]))
        signals.occasions_left_out = len(left_out)
        signals.left_out_ids = [p["id"] for p in left_out]

    words = set()
    for item in chosen:
        for word in _WORD.findall((item.get("text") or "").lower()):
            words.add(_stem(word.split("'")[0]))
    signals.missing_terms = [t for t in salient_terms(question) if _stem(t) not in words]

    signals.stale_unresolved = sum(
        1 for item in chosen
        if item.get("superseded_by") and item["superseded_by"] not in chosen_ids)

    scores = sorted((float(item.get("score", 0.0)) for item in chosen), reverse=True)
    if len(scores) >= 2 and scores[0] > 0:
        other = scores[min(MARGIN_DEPTH, len(scores)) - 1]
        signals.margin = (scores[0] - other) / scores[0]
    return signals


def reasons_for_second_look(signals: Signals) -> List[str]:
    """The signals that say the context may not hold the answer, by name."""
    reasons = []
    if signals.windows and not signals.window_hits:
        reasons.append("window_empty")
    if signals.counting and signals.occasions_left_out:
        reasons.append("left_out_occasions")
    if signals.missing_terms:
        reasons.append("missing_terms")
    if signals.stale_unresolved:
        reasons.append("stale_unresolved")
    return reasons


def second_look_query(question: str, signals: Signals,
                      missing: Optional[str] = None) -> Optional[str]:
    """What to search for the second time: what a judge said is missing,
    else the question's words no memory in the context has, else nothing
    (the left-out occasions are already candidates and need no search)."""
    if missing and missing.strip():
        return missing.strip()
    if signals.missing_terms:
        return " ".join(signals.missing_terms)
    return None


SUFFICIENCY_SYSTEM = (
    "You check whether notes from a user's earlier conversations are enough to "
    "answer a question about them. Each note starts with the date it was said.\n"
    "Reply with exactly two lines:\n"
    "SUFFICIENT: YES or NO\n"
    "MISSING: if NO, a short search query (at most ten words) for what the notes "
    "lack; if YES, write NONE.\n"
    "Say NO when the question asks to count or list things and some may be "
    "missing, when it asks about a time the notes do not cover, or when it "
    "names something no note mentions. Do not answer the question."
)

_SUFFICIENT = re.compile(r"SUFFICIENT\s*:\s*(YES|NO)", re.I)
_MISSING = re.compile(r"MISSING\s*:\s*(.+)", re.I)


def sufficiency_request(question: str, contexts: Sequence[str],
                        today: Optional[str] = None) -> str:
    """The user message for ``SUFFICIENCY_SYSTEM``."""
    listing = "\n".join(f"{n}. {text}" for n, text in enumerate(contexts, 1))
    when = f"Current date: {today}\n" if today else ""
    return f"Notes:\n{listing or '(none)'}\n\n{when}Question: {question}"


def parse_sufficiency(reply: Optional[str]) -> Tuple[Optional[bool], str]:
    """``(sufficient, missing)`` from a reply to ``SUFFICIENCY_SYSTEM``;
    ``sufficient`` is ``None`` when the reply does not say."""
    text = reply or ""
    verdict = _SUFFICIENT.search(text)
    sufficient = None if verdict is None else verdict.group(1).upper() == "YES"
    missing = _MISSING.search(text)
    what = missing.group(1).strip() if missing else ""
    if what.upper().startswith("NONE"):
        what = ""
    return sufficient, what


class LLMSufficiencyJudge:
    """Asks a chat model whether a context answers a question
    (``SUFFICIENCY_SYSTEM``). One short request per question that the
    grounded signals flag; replies are cached when ``cache_dir`` is given.

    A failed request raises (``tsm._retry``): ``checked_recall`` then keeps the
    first recall and records the failure in its report.
    """

    def __init__(self, model: str = "gpt-4o-mini", client=None,
                 base_url: Optional[str] = None, api_key: Optional[str] = None,
                 cache_dir: Optional[str] = None, **chat_kwargs):
        from ._chat import CachedChat

        self._chat = CachedChat(SUFFICIENCY_SYSTEM, model=model, client=client,
                                base_url=base_url, api_key=api_key, cache_dir=cache_dir,
                                cache_name="sufficiency", max_tokens=60,
                                what="sufficiency check", **chat_kwargs)
        self.model = model

    @property
    def calls(self) -> int:
        return self._chat.calls

    def forget_texts(self, texts) -> int:
        """Remove cached replies that carry ``texts`` (``Memory.forget``)."""
        return self._chat.forget_texts(texts)

    def judge(self, question: str, contexts: Sequence[str],
              today: Optional[str] = None) -> Dict:
        """``{"sufficient": bool or None, "missing": str, "reply": str}``."""
        reply = self._chat.ask(sufficiency_request(question, contexts, today))
        sufficient, missing = parse_sufficiency(reply)
        return {"sufficient": sufficient, "missing": missing, "reply": reply}

    def flush_cache(self) -> None:
        self._chat.flush_cache()


@dataclass
class MonitorReport:
    """What ``checked_recall`` measured, asked and did, as a record."""
    question: str
    signals: Signals
    reasons: List[str] = field(default_factory=list)
    judge: Optional[Dict] = None
    judge_error: Optional[str] = None
    action: str = "none"            # "none" | "second_look"
    second_query: Optional[str] = None
    added_ids: List[str] = field(default_factory=list)
    dropped_ids: List[str] = field(default_factory=list)

    @property
    def confident(self) -> bool:
        """No signal fired, or a judge said the context is sufficient."""
        if self.judge and self.judge.get("sufficient") is True:
            return True
        return not self.reasons

    def to_dict(self) -> Dict:
        out = asdict(self)
        out["confident"] = self.confident
        return out


def swap_in(first: List[Dict], extra: Sequence[Dict], token_budget: int,
            share: float, tokens) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Fit ``extra`` (best first) into ``share`` of ``token_budget`` and drop
    the last-packed items of ``first`` until both fit the whole budget.
    ``tokens(item)`` is an item's cost. Returns ``(results, added,
    dropped)``; the first recall's order is kept, the added items follow.
    """
    seen = {item.get("id") for item in first}
    cap = max(1, int(token_budget * share))
    added: List[Dict] = []
    used = 0
    for item in extra:
        if item.get("id") in seen:
            continue
        cost = tokens(item)
        if used + cost > cap:
            continue
        added.append(item)
        seen.add(item.get("id"))
        used += cost
    if not added:
        return list(first), [], []
    kept = list(first)
    dropped: List[Dict] = []
    while kept and sum(tokens(i) for i in kept) + used > token_budget:
        dropped.append(kept.pop())
    return kept + added, added, dropped
