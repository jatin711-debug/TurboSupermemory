"""Equal-budget comparison stores for the bounded-storage evaluations.

The token estimate and packing primitives are the shipped ones in
``tsm.budget`` (re-exported here for the eval scripts); only the eval-specific
construction of matched naive / gist-compressed stores lives in this module.
"""

from dataclasses import dataclass, field
from typing import Callable, Sequence

from tsm.budget import (  # noqa: F401  (re-exported for the eval scripts)
    estimate_tokens,
    fit_complete_facts_to_budget,
    fit_text_to_budget,
    pack_recent,
    pack_role_priority_recent,
    partition_by_token_weight,
    total_tokens,
    truncate_to_budget,
)
from tsm.compaction import plan_compaction


@dataclass(frozen=True)
class BoundedStores:
    naive: list[str]
    compressed: list[str]
    naive_overflow: list[str]
    compressed_tail: list[str]
    limit: int
    unit: str
    gist_limit: int = 0
    gists: list[str] = field(default_factory=list)


def build_token_bounded_stores(
    facts: Sequence[str],
    token_budget: int,
    summarize: Callable[[Sequence[str], int], str],
    gist_share: float = 0.25,
    roles: Sequence[str] | None = None,
    role_aware: bool = False,
    gist_chunk_tokens: int = 32,
    max_gist_chunks: int = 4,
) -> BoundedStores | None:
    """Build equal-token naive and gist-compression stores under recency pressure."""
    if token_budget < 2:
        raise ValueError("storage token budget must be at least 2")
    if not 0.0 < gist_share < 1.0:
        raise ValueError("gist_share must be between 0 and 1")
    if gist_chunk_tokens < 1 or max_gist_chunks < 1:
        raise ValueError("gist chunk settings must be positive")
    if total_tokens(facts) <= token_budget:
        return None

    naive, naive_overflow = pack_recent(facts, token_budget)
    gist_limit = max(1, min(token_budget - 1, round(token_budget * gist_share)))
    if role_aware:
        if roles is None or len(roles) != len(facts):
            raise ValueError("role-aware compression requires one role per fact")
        # The shipped policy: what Memory(max_user_tokens=...) does to a store.
        plan = plan_compaction(facts, roles, token_budget, summarize, gist_share=gist_share,
                               gist_chunk_tokens=gist_chunk_tokens,
                               max_gist_chunks=max_gist_chunks)
        survivors = [facts[index] for index in plan.keep]
        tail = [facts[index] for index in plan.fold]
        gists = list(plan.gists)
    else:
        survivors, tail = pack_recent(facts, token_budget - gist_limit)
        gist = fit_complete_facts_to_budget(summarize(list(tail), gist_limit), gist_limit)
        gists = [gist] if gist else []
    compressed = survivors + gists
    if total_tokens(compressed) > token_budget:
        raise AssertionError("compressed store exceeded its active-memory token budget")

    return BoundedStores(
        naive=naive,
        compressed=compressed,
        naive_overflow=naive_overflow,
        compressed_tail=tail,
        limit=token_budget,
        unit="tokens",
        gist_limit=gist_limit,
        gists=gists,
    )


def build_slot_bounded_stores(
    facts: Sequence[str],
    slot_budget: int,
    summarize: Callable[[Sequence[str], int], str],
    gist_token_limit: int = 120,
) -> BoundedStores | None:
    """Reproduce the historical equal-slot benchmark for comparison with old results."""
    if slot_budget < 2:
        raise ValueError("storage slot budget must be at least 2")
    if len(facts) <= slot_budget:
        return None

    survivors = list(facts[-(slot_budget - 1):])
    tail = list(facts[:-(slot_budget - 1)])
    naive = list(facts[-slot_budget:])
    # Slot mode reproduces the historical benchmark: generation is capped by the
    # gister, but the returned text is not clipped by approximate storage tokens.
    gist = summarize(tail, gist_token_limit).strip()
    compressed = survivors + ([gist] if gist else [])
    naive_set = set(naive)
    return BoundedStores(
        naive=naive,
        compressed=compressed,
        naive_overflow=[fact for fact in facts if fact not in naive_set],
        compressed_tail=tail,
        limit=slot_budget,
        unit="slots",
        gist_limit=gist_token_limit,
        gists=[gist] if gist else [],
    )
