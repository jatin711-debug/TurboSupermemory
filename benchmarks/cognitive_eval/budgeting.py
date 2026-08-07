"""Shared approximate-token budgeting for cognitive evaluations."""

import re
from dataclasses import dataclass, field
from typing import Callable, Sequence


def estimate_tokens(text: str) -> int:
    """Match the benchmark suite's established four-characters-per-token estimate."""
    return max(1, len(text) // 4) if text and text.strip() else 0


def total_tokens(texts: Sequence[str]) -> int:
    return sum(estimate_tokens(text) for text in texts)


def truncate_to_budget(texts: Sequence[str], token_budget: int) -> list[str]:
    """Greedily pack texts in input order, skipping entries that do not fit."""
    selected = []
    used = 0
    for text in texts:
        tokens = estimate_tokens(text)
        if tokens and used + tokens <= token_budget:
            selected.append(text)
            used += tokens
    return selected


def pack_recent(texts: Sequence[str], token_budget: int) -> tuple[list[str], list[str]]:
    """Keep the most-recent entries that fit and return `(kept, overflow)` in input order."""
    selected_indexes = _pack_indexes(texts, token_budget, range(len(texts) - 1, -1, -1))
    kept = [text for index, text in enumerate(texts) if index in selected_indexes]
    overflow = [text for index, text in enumerate(texts) if index not in selected_indexes]
    return kept, overflow


def _pack_indexes(texts, token_budget, candidate_indexes):
    selected_indexes = set()
    used = 0
    for index in candidate_indexes:
        tokens = estimate_tokens(texts[index])
        if tokens and used + tokens <= token_budget:
            selected_indexes.add(index)
            used += tokens
    return selected_indexes


def pack_role_priority_recent(texts, roles, token_budget):
    """Pack recent user facts first, then system/assistant facts, under one cap."""
    if len(texts) != len(roles):
        raise ValueError("texts and roles must have the same length")
    role_order = ("user", "system", "assistant")
    candidates = []
    for role in role_order:
        candidates.extend(
            index for index in range(len(texts) - 1, -1, -1) if roles[index] == role
        )
    candidates.extend(
        index
        for index in range(len(texts) - 1, -1, -1)
        if roles[index] not in role_order
    )
    selected_indexes = _pack_indexes(texts, token_budget, candidates)
    kept = [text for index, text in enumerate(texts) if index in selected_indexes]
    overflow = [text for index, text in enumerate(texts) if index not in selected_indexes]
    return kept, overflow


def fit_text_to_budget(text: str, token_budget: int) -> str:
    """Bound one generated text using the same approximation as storage accounting."""
    if not text or token_budget <= 0:
        return ""
    if estimate_tokens(text) <= token_budget:
        return text.strip()
    clipped = text[: token_budget * 4].strip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0].rstrip(" ,;:")
    return clipped


def fit_complete_facts_to_budget(text: str, token_budget: int) -> str:
    """Keep complete generated fact units; never hard-cut a fact mid-sentence."""
    if not text or token_budget <= 0:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        lines = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]

    selected = []
    for line in lines:
        fact = re.sub(r"^(?:[-*]\s*|\d+[.)]\s*)", "", line).strip()
        if not fact:
            continue
        candidate = "\n".join([*selected, f"- {fact}"])
        if estimate_tokens(candidate) <= token_budget:
            selected.append(f"- {fact}")
    return "\n".join(selected)


def partition_by_token_weight(texts: Sequence[str], chunk_count: int) -> list[list[str]]:
    """Split ordered texts into contiguous chunks with roughly equal token weight."""
    if chunk_count < 1:
        raise ValueError("chunk_count must be at least 1")
    if not texts:
        return []
    chunk_count = min(chunk_count, len(texts))
    total = sum(max(1, estimate_tokens(text)) for text in texts)
    chunks = []
    start = 0
    remaining_tokens = total
    for chunk_index in range(chunk_count):
        remaining_chunks = chunk_count - chunk_index
        if remaining_chunks == 1:
            chunks.append(list(texts[start:]))
            break
        target = max(1, round(remaining_tokens / remaining_chunks))
        end = start
        used = 0
        max_end = len(texts) - (remaining_chunks - 1)
        while end < max_end:
            tokens = max(1, estimate_tokens(texts[end]))
            if end > start and used + tokens > target:
                break
            used += tokens
            end += 1
        if end == start:
            end += 1
            used = max(1, estimate_tokens(texts[start]))
        chunks.append(list(texts[start:end]))
        start = end
        remaining_tokens -= used
    return chunks


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
        survivors, tail = pack_role_priority_recent(facts, roles, token_budget - gist_limit)
        survivor_counts = {}
        for survivor in survivors:
            survivor_counts[survivor] = survivor_counts.get(survivor, 0) + 1
        tail_entries = []
        for fact, role in zip(facts, roles):
            remaining = survivor_counts.get(fact, 0)
            if remaining:
                survivor_counts[fact] = remaining - 1
            else:
                tail_entries.append((fact, role))
        labeled_tail = [f"[{role}] {fact}" for fact, role in tail_entries]
        user_tail = [fact for fact in labeled_tail if fact.lower().startswith("[user] ")]
        other_tail = [fact for fact in labeled_tail if not fact.lower().startswith("[user] ")]
        chunk_count = min(max_gist_chunks, max(1, gist_limit // gist_chunk_tokens))
        # Tight budgets are reserved for user history. At four or more chunks,
        # keep one chunk for assistant/system information so that answerable
        # assistant facts are not erased entirely.
        other_chunks = 1 if other_tail and chunk_count >= 4 else 0
        user_chunks = chunk_count - other_chunks if user_tail else 0
        if not user_chunks and other_tail:
            other_chunks = chunk_count
        summary_chunks = partition_by_token_weight(user_tail, user_chunks) if user_chunks else []
        if other_chunks:
            summary_chunks.extend(partition_by_token_weight(other_tail, other_chunks))
    else:
        survivors, tail = pack_recent(facts, token_budget - gist_limit)
        summary_chunks = [list(tail)]

    chunk_limits = []
    if summary_chunks:
        base_limit, remainder = divmod(gist_limit, len(summary_chunks))
        chunk_limits = [base_limit + (1 if index < remainder else 0)
                        for index in range(len(summary_chunks))]
    gists = []
    for summary_input, chunk_limit in zip(summary_chunks, chunk_limits):
        gist = fit_complete_facts_to_budget(summarize(summary_input, chunk_limit), chunk_limit)
        if gist:
            gists.append(gist)
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
