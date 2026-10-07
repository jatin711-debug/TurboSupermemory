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

from dataclasses import dataclass
from typing import Callable, Collection, List, Optional, Sequence

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
    """

    keep: List[int]
    fold: List[int]
    gists: List[str]
    gist_roles: List[str]


def is_gist(role: Optional[str]) -> bool:
    return role in (GIST_ROLE, GIST_OTHER_ROLE)


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

    A summarizer that raises aborts the plan: nothing is planned, so nothing
    is removed.
    """
    if len(texts) != len(roles):
        raise ValueError("texts and roles must have the same length")
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

    # What the gists are written from, oldest first. An earlier gist stands
    # for history older than any fact still stored, so its lines lead.
    user_tail: List[str] = []
    other_tail: List[str] = []
    for i in fold:
        if roles[i] == GIST_ROLE:
            user_tail.extend(f"[user] {line}" for line in _gist_lines(texts[i]))
        elif roles[i] == GIST_OTHER_ROLE:
            other_tail.extend(f"[assistant] {line}" for line in _gist_lines(texts[i]))
    for i in fold:
        if is_gist(roles[i]):
            continue
        entry = f"[{roles[i]}] {texts[i]}"
        (user_tail if roles[i] == "user" else other_tail).append(entry)

    chunk_count = min(max_gist_chunks, max(1, gist_limit // gist_chunk_tokens))
    # Tight budgets are reserved for user history. At four or more chunks, one
    # is kept for the other roles so their answerable facts are not erased.
    other_chunks = 1 if other_tail and chunk_count >= 4 else 0
    user_chunks = chunk_count - other_chunks if user_tail else 0
    if not user_chunks and other_tail:
        other_chunks = chunk_count
    chunks = [(GIST_ROLE, chunk) for chunk in (
        partition_by_token_weight(user_tail, user_chunks) if user_chunks else [])]
    if other_chunks:
        chunks.extend((GIST_OTHER_ROLE, chunk)
                      for chunk in partition_by_token_weight(other_tail, other_chunks))

    gists: List[str] = []
    gist_roles: List[str] = []
    if chunks:
        base, extra = divmod(gist_limit, len(chunks))
        for position, (role, chunk) in enumerate(chunks):
            limit = base + (1 if position < extra else 0)
            gist = fit_complete_facts_to_budget(summarize(chunk, limit), limit)
            if gist:
                gists.append(gist)
                gist_roles.append(role)
    return CompactionPlan(keep=keep, fold=fold, gists=gists, gist_roles=gist_roles)
