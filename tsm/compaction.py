"""Keeping one user's memory under a token budget.

When a user's memories no longer fit the budget, the newest facts are kept as
they are (what the user said first, then what the assistant said) and
everything older is folded into a few short gists: forget detail, keep gist.
This is the policy the bounded-storage evaluations measured
(``benchmarks/cognitive_eval/bounded_head_to_head.py``), and it is rolling: on
a later pass the earlier gists are folded again together with the facts that
have aged out since, so the store stays within its budget however long the
history grows.

``plan_compaction`` decides what stays, what is folded, and what the gists
say. It touches no store; ``tsm.Memory`` applies the plan
(``Memory(max_user_tokens=...)``).
"""

from dataclasses import dataclass, field
from typing import Callable, Collection, Dict, List, Optional, Sequence, Tuple

from .budget import (
    fit_complete_facts_to_budget,
    pack_role_priority_recent_indexes,
    partition_by_token_weight,
    total_tokens,
)

# source_role of a stored gist: of what the user said, and of everything else.
GIST_ROLE = "summary"
GIST_OTHER_ROLE = "summary-assistant"

Summarize = Callable[[Sequence[str], int], str]


@dataclass(frozen=True)
class CompactionPlan:
    """What a compaction pass does to one user's store.

    ``keep`` and ``fold`` are indexes into the memories that were passed in:
    the ones that stay untouched and the ones the gists replace. ``gists``
    are the new gist texts and ``gist_roles`` the role to store each under.
    ``gist_spans`` gives, per gist, the first and last date of what it was
    written from (``None`` when none of it was dated).
    """

    keep: List[int]
    fold: List[int]
    gists: List[str]
    gist_roles: List[str]
    gist_spans: List[Optional[Tuple[str, str]]] = field(default_factory=list)
    # How many merges each gist is the result of: 0 for a gist written from
    # facts, one more than its inputs for a merge (``policy="merge_oldest"``).
    gist_levels: List[int] = field(default_factory=list)


Span = Optional[Tuple[str, str]]


def _join(spans: Sequence[Span]) -> Span:
    """The span covering ``spans`` (ISO dates compare as strings)."""
    dated = [s for s in spans if s]
    if not dated:
        return None
    return (min(s[0] for s in dated), max(s[1] for s in dated))


POLICIES = ("refold", "merge_oldest")


def is_gist(role: Optional[str]) -> bool:
    return role in (GIST_ROLE, GIST_OTHER_ROLE)


def _entry(role: str, text: str, span: Span) -> str:
    """A fact as the summarizer reads it: its role, and its date when it was
    said on one day (``[user] [2023-05-22] ...``), so the gist can keep it."""
    if span and span[0] == span[1]:
        return f"[{role}] [{span[0]}] {text}"
    return f"[{role}] {text}"


def _gist_lines(text: str) -> List[str]:
    """The facts of a stored gist, one per line, without their bullets."""
    lines = []
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith(("- ", "* ")):
            line = line[2:].strip()
        if line:
            lines.append(line)
    return lines


def plan_compaction(
    texts: Sequence[str],
    roles: Sequence[str],
    token_budget: int,
    summarize: Optional[Summarize],
    gist_share: float = 0.5,
    gist_chunk_tokens: int = 32,
    max_gist_chunks: int = 4,
    fold_first: Collection[int] = (),
    spans: Optional[Sequence[Span]] = None,
    levels: Optional[Sequence[int]] = None,
    policy: str = "refold",
) -> Optional[CompactionPlan]:
    """Plan one compaction of a user's memories, given oldest first.

    ``roles`` holds each memory's source role; an earlier gist has
    ``GIST_ROLE`` or ``GIST_OTHER_ROLE``. Returns ``None`` when everything
    already fits ``token_budget``.

    ``gist_share`` of the budget is set aside for gists. The rest goes to the
    newest facts, user facts before the others. What does not survive, and
    every earlier gist, is summarized in up to ``max_gist_chunks`` chunks of
    about ``gist_chunk_tokens`` tokens: user history in order, with one chunk
    kept for the other roles once there are four. ``summarize(texts,
    max_tokens)`` writes each gist from lines labelled ``[user] ...`` /
    ``[assistant] ...``; without one (``None``) nothing is summarized and the
    whole budget goes to the newest facts.

    ``fold_first`` lists memories that must not survive as they are, such as
    facts a newer one has replaced: they are folded before anything else.

    ``spans`` gives each memory's first and last date (``("2023-05-22",
    "2023-05-22")`` for a fact said that day, an earlier gist's own span) or
    ``None``; each gist's span is then the span of what it was written from.
    Without dates a gist is undated, and a reader cannot place what it says
    in time. A fact said on one day reaches the summarizer with its date
    (``[user] [2023-05-22] ...``), so a gist can keep it on the line.

    ``policy`` decides what happens to earlier gists:

    - ``"refold"`` (default): every earlier gist is summarized again with
      what ages out, on every pass. A fact is rewritten once per pass, and
      compacting four times as a conversation arrived scored 0.411 where
      compacting once scored 0.464 (``benchmarks/PHASE_PROGRESS.md``).
    - ``"merge_oldest"``: earlier gists stay as they are. What ages out
      becomes new gists, and only while the gists exceed their share are the
      two oldest gists of one role and of the lowest level merged into one
      a level up, as a log-structured store compacts. A fact is then
      rewritten about log2(passes) times. ``levels`` gives each memory's
      level (an earlier gist's, from its payload; 0 otherwise).

    A summarizer that raises aborts the plan: nothing is planned, so nothing
    is removed.
    """
    if len(texts) != len(roles):
        raise ValueError("texts and roles must have the same length")
    if spans is not None and len(spans) != len(texts):
        raise ValueError("spans and texts must have the same length")
    if levels is not None and len(levels) != len(texts):
        raise ValueError("levels and texts must have the same length")
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    spans = list(spans) if spans is not None else [None] * len(texts)
    levels = [int(level or 0) for level in levels] if levels is not None else [0] * len(texts)
    if token_budget < 2:
        raise ValueError("token budget must be at least 2")
    if not 0.0 < gist_share < 1.0:
        raise ValueError("gist_share must be between 0 and 1")
    if gist_chunk_tokens < 1 or max_gist_chunks < 1:
        raise ValueError("gist chunk settings must be positive")
    if total_tokens(texts) <= token_budget:
        return None

    gist_limit = 0
    if summarize is not None:
        gist_limit = max(1, min(token_budget - 1, round(token_budget * gist_share)))
    barred = set(fold_first)
    candidates = [i for i, role in enumerate(roles) if not is_gist(role) and i not in barred]
    kept = pack_role_priority_recent_indexes(
        [texts[i] for i in candidates], [roles[i] for i in candidates], token_budget - gist_limit)
    keep = sorted(candidates[position] for position in kept)
    surviving = set(keep)
    fold = [i for i in range(len(texts)) if i not in surviving]
    if summarize is None:
        return CompactionPlan(keep=keep, fold=fold, gists=[], gist_roles=[])
    if policy == "merge_oldest":
        return _merge_oldest(texts, roles, spans, levels, keep, fold, summarize, gist_limit,
                             gist_chunk_tokens)

    # What the gists are written from, oldest first. An earlier gist stands
    # for history older than any fact still stored, so its lines lead.
    user_tail: List[str] = []
    other_tail: List[str] = []
    user_spans: List[Span] = []  # the span of each entry of the tails
    other_spans: List[Span] = []
    for i in fold:
        if roles[i] == GIST_ROLE:
            lines = [f"[user] {line}" for line in _gist_lines(texts[i])]
            user_tail.extend(lines)
            user_spans.extend([spans[i]] * len(lines))
        elif roles[i] == GIST_OTHER_ROLE:
            lines = [f"[assistant] {line}" for line in _gist_lines(texts[i])]
            other_tail.extend(lines)
            other_spans.extend([spans[i]] * len(lines))
    for i in fold:
        if is_gist(roles[i]):
            continue
        entry = _entry(roles[i], texts[i], spans[i])
        if roles[i] == "user":
            user_tail.append(entry)
            user_spans.append(spans[i])
        else:
            other_tail.append(entry)
            other_spans.append(spans[i])

    chunk_count = min(max_gist_chunks, max(1, gist_limit // gist_chunk_tokens))
    # Tight budgets are reserved for user history. At four or more chunks, one
    # is kept for the other roles so their answerable facts are not erased.
    other_chunks = 1 if other_tail and chunk_count >= 4 else 0
    user_chunks = chunk_count - other_chunks if user_tail else 0
    if not user_chunks and other_tail:
        other_chunks = chunk_count
    def with_spans(parts, entry_spans):
        """Each chunk with the span of its entries (chunks are contiguous)."""
        out, start = [], 0
        for part in parts:
            out.append((part, _join(entry_spans[start:start + len(part)])))
            start += len(part)
        return out

    chunks = [(GIST_ROLE, chunk, span) for chunk, span in with_spans(
        partition_by_token_weight(user_tail, user_chunks) if user_chunks else [], user_spans)]
    if other_chunks:
        chunks.extend((GIST_OTHER_ROLE, chunk, span) for chunk, span in with_spans(
            partition_by_token_weight(other_tail, other_chunks), other_spans))

    gists: List[str] = []
    gist_roles: List[str] = []
    gist_spans: List[Span] = []
    if chunks:
        base, extra = divmod(gist_limit, len(chunks))
        for position, (role, chunk, span) in enumerate(chunks):
            limit = base + (1 if position < extra else 0)
            gist = fit_complete_facts_to_budget(summarize(chunk, limit), limit)
            if gist:
                gists.append(gist)
                gist_roles.append(role)
                gist_spans.append(span)
    return CompactionPlan(keep=keep, fold=fold, gists=gists, gist_roles=gist_roles,
                          gist_spans=gist_spans, gist_levels=[0] * len(gists))


def _merge_oldest(texts, roles, spans, levels, keep, fold, summarize, gist_limit,
                  chunk_tokens) -> CompactionPlan:
    """``policy="merge_oldest"``: see ``plan_compaction``."""
    earlier = [i for i in range(len(texts)) if is_gist(roles[i])]
    aged_out = [i for i in fold if not is_gist(roles[i])]
    # Every gist in play: the earlier ones as they are, then the new ones.
    pool = [{"role": roles[i], "text": texts[i], "span": spans[i], "level": levels[i],
             "index": i} for i in earlier]
    for gist_role, ours in ((GIST_ROLE, lambda role: role == "user"),
                            (GIST_OTHER_ROLE, lambda role: role != "user")):
        sources = [i for i in aged_out if ours(roles[i])]
        if not sources:
            continue
        lines = [_entry(roles[i], texts[i], spans[i]) for i in sources]
        gist = fit_complete_facts_to_budget(summarize(lines, chunk_tokens), chunk_tokens)
        if gist:
            pool.append({"role": gist_role, "text": gist, "level": 0, "index": None,
                         "span": _join([spans[i] for i in sources])})

    def age(gist):
        """Older first: by the start of its span, then by store order."""
        start = gist["span"][0] if gist["span"] else "9999-99-99"
        return (start, gist["index"] if gist["index"] is not None else len(texts))

    def oldest_pair():
        """The two oldest gists of one role at its lowest level holding two
        or more; across roles, the pair whose older gist is older."""
        best = None
        for role in (GIST_ROLE, GIST_OTHER_ROLE):
            ours = sorted((g for g in pool if g["role"] == role), key=age)
            by_level: Dict[int, List[dict]] = {}
            for gist in ours:
                by_level.setdefault(gist["level"], []).append(gist)
            paired = [group[:2] for _, group in sorted(by_level.items()) if len(group) >= 2]
            pair = paired[0] if paired else (ours[:2] if len(ours) >= 2 else None)
            if pair and (best is None or age(pair[0]) < age(best[0])):
                best = pair
        return best

    while total_tokens([g["text"] for g in pool]) > gist_limit:
        pair = oldest_pair()
        if pair is None:
            break  # one gist per role left: nothing more to merge
        label = "[user] " if pair[0]["role"] == GIST_ROLE else "[assistant] "
        lines = [label + line for gist in pair for line in _gist_lines(gist["text"])]
        merged = fit_complete_facts_to_budget(summarize(lines, chunk_tokens), chunk_tokens)
        for gist in pair:
            pool.remove(gist)
        if merged:
            pool.append({"role": pair[0]["role"], "text": merged,
                         "span": _join([g["span"] for g in pair]),
                         "level": max(g["level"] for g in pair) + 1, "index": None})

    still = {g["index"] for g in pool if g["index"] is not None}
    new = [g for g in pool if g["index"] is None]
    return CompactionPlan(
        keep=sorted(set(keep) | still),
        fold=sorted(set(aged_out) | {i for i in earlier if i not in still}),
        gists=[g["text"] for g in new], gist_roles=[g["role"] for g in new],
        gist_spans=[g["span"] for g in new], gist_levels=[g["level"] for g in new])
