"""Token budgeting: choosing and packing memories under a context budget.

``select_under_budget`` picks the best *set* of retrieved memories that fits a
token budget (greedy submodular MMR). The packing helpers bound stored or
generated text with the same four-characters-per-token estimate, so storage
accounting and recall accounting agree.
"""

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# A candidate more similar than this to something already selected is a
# near-duplicate paraphrase: it is skipped rather than merely penalized.
MAX_REDUNDANCY = 0.72
# Marginal-gain bonus for the first memory taken from a not-yet-covered turn
# (cross-turn / cross-session coverage).
NEW_TURN_BONUS = 0.20


def estimate_tokens(text: str) -> int:
    """Approximate token count: four characters per token, 0 for blank text."""
    return max(1, len(text) // 4) if text and text.strip() else 0


def total_tokens(texts: Sequence[str]) -> int:
    return sum(estimate_tokens(text) for text in texts)


def select_under_budget(
    pool: Sequence[Dict],
    token_budget: int,
    embed: Optional[Callable[[List[str]], Sequence]] = None,
    method: str = "mmr",
    lam: float = 0.55,
    max_items: Optional[int] = None,
) -> List[Dict]:
    """Select the best set of ``pool`` items whose texts fit ``token_budget``.

    Each pool item is a dict with ``"text"`` and ``"score"`` (relevance) and
    optionally ``"turn_index"``. Returns the chosen items in selection order.

    ``method="mmr"`` (default): greedy submodular Maximal Marginal Relevance —
    each step adds the candidate maximizing
    ``lam * relevance - (1 - lam) * max_redundancy + new_turn_bonus`` against
    the already-selected set. Needs ``embed`` (texts -> vectors) for pairwise
    redundancy. ``method="truncate"``: relevance order only, skipping items
    that do not fit (the naive baseline; ``embed`` unused).

    The token budget is what bounds the set. ``max_items`` adds a limit on
    the number of items for a caller that wants one; by default there is
    none. (There used to be: ``min(10, max(4, budget // 35))``. With memories
    of about 18 tokens it filled 72 of a 150-token budget and cost judged
    answers, 0.487 against 0.565 without it.)
    """
    if not pool:
        return []
    texts = [p["text"] or "" for p in pool]
    rel = np.array([float(p["score"]) for p in pool], dtype=np.float32)
    toks = np.array([max(1, len(t) // 4) for t in texts], dtype=np.int64)
    cap = max_items if max_items else len(pool)

    if method == "truncate":
        sel, used = [], 0
        for i in np.argsort(-rel):
            if len(sel) >= cap:
                break
            if used + int(toks[i]) <= token_budget:
                sel.append(int(i))
                used += int(toks[i])
        return [pool[i] for i in sel]
    if method != "mmr":
        raise ValueError(f"unknown selection method: {method!r} (use 'mmr' or 'truncate')")
    if embed is None:
        raise ValueError("method='mmr' needs an embed callable for redundancy")

    # Pairwise redundancy from pool embeddings (unit-normed -> cosine).
    embs = np.asarray(embed(texts), dtype=np.float32)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    sim = embs @ embs.T

    selected, used, remaining = [], 0, list(range(len(pool)))
    selected_turns = set()
    while remaining and len(selected) < cap:
        best_i, best_gain = None, -1e9
        for i in remaining:
            if used + int(toks[i]) > token_budget:
                continue
            red = max((float(sim[i, j]) for j in selected), default=0.0)
            if red > MAX_REDUNDANCY:
                continue
            t_idx = pool[i].get("turn_index")
            bonus = NEW_TURN_BONUS if (t_idx is not None and t_idx not in selected_turns) else 0.0
            gain = lam * float(rel[i]) - (1.0 - lam) * red + bonus
            if gain > best_gain:
                best_gain, best_i = gain, i
        if best_i is None:
            break  # nothing else fits the budget or the redundancy threshold
        selected.append(best_i)
        used += int(toks[best_i])
        t_idx = pool[best_i].get("turn_index")
        if t_idx is not None:
            selected_turns.add(t_idx)
        remaining.remove(best_i)
    return [pool[i] for i in selected]


# --- packing ----------------------------------------------------------------


def truncate_to_budget(texts: Sequence[str], token_budget: int) -> List[str]:
    """Greedily pack texts in input order, skipping entries that do not fit."""
    selected = []
    used = 0
    for text in texts:
        tokens = estimate_tokens(text)
        if tokens and used + tokens <= token_budget:
            selected.append(text)
            used += tokens
    return selected


def pack_recent(texts: Sequence[str], token_budget: int) -> Tuple[List[str], List[str]]:
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


def partition_by_token_weight(texts: Sequence[str], chunk_count: int) -> List[List[str]]:
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
