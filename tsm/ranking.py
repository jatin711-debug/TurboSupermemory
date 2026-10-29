"""Role prior applied to recall scores.

A question phrased from the user's side ("what did I...", "how many ...") is
far more often answered by something the user said than by the assistant's own
reply, so user-sourced facts are boosted and assistant-sourced ones damped for
such queries.
"""

import re
from typing import Dict, List, Optional

# Tokens that mark a query as asking about the user's own facts. Deliberately
# broad (it includes common question words): this is the cue set the
# evaluations were run with, so change it together with a re-run.
FIRST_PERSON_CUES = frozenset({
    "i", "my", "me", "mine", "we", "our", "did", "have", "how", "what",
    "total", "number", "many",
})

USER_ROLE_BOOST = 1.30
ASSISTANT_ROLE_DAMP = 0.85

# A gist counts as its source: one written from what the user said is ranked
# like a user fact, the other kind like an assistant fact (the role names are
# `tsm.compaction.GIST_ROLE` / `GIST_OTHER_ROLE`). Without this a store that
# has been compacted ranks every raw fact above the history behind it. A
# stored passage of a message (`tsm.chunks.message_role`) counts as its
# speaker too.
USER_ROLES = frozenset({"user", "summary", "message"})
ASSISTANT_ROLES = frozenset({"assistant", "summary-assistant", "message-assistant"})


# A question about what the assistant itself said ("you told me", "did you
# say", "remind me", "we decided", "our previous chat") is mostly answered by
# an assistant reply, and it says "I" and "me" all the same. The first-person
# prior then held every assistant memory back: on 100 LongMemEval-S
# questions the contexts of the twelve such questions held 529 user passages
# and 4 of the assistant's. Past tense only: "can you suggest a hotel" asks
# for something new. On the 500 LongMemEval questions these cues match 54 of
# the 56 single-session-assistant questions and no question of another type.
_RECALLED = (r"told|said|mentioned|suggested|recommended|provided|gave|wrote|created|listed|"
             r"described|explained|shared|offered|proposed|came up with|generated|showed|"
             r"named|called|included|compiled")
_ASKED_AGAIN = (r"say|tell|mention|suggest|recommend|provide|give|write|create|list|describe|"
                r"explain|share|offer|propose|name|call|include")
ASSISTANT_RECALL_CUES = re.compile(
    r"\byou\s+(?:\w+\s+){0,2}?(?:" + _RECALLED + r")\b"
    r"|\bdid you\s+(?:\w+\s+)?(?:" + _ASKED_AGAIN + r")\b"
    r"|\bremind me\b"
    r"|\bwe\s+(?:\w+\s+)?(?:decided|discussed|talked|agreed|came up|chose|settled)\b"
    r"|\bdid we\s+(?:\w+\s+)?(?:decide|discuss|talk|agree|come up|choose|settle)\b"
    r"|\bour\s+(?:\w+\s+){0,2}(?:chat|conversation|discussion)\b",
    re.IGNORECASE)


def is_first_person_query(query: str) -> bool:
    """True when ``query`` contains any first-person / question cue token."""
    return any(w in FIRST_PERSON_CUES for w in query.lower().split())


def asks_what_assistant_said(query: str) -> bool:
    """True when ``query`` asks for something the assistant said earlier."""
    return bool(ASSISTANT_RECALL_CUES.search(query or ""))


# A question that counts or adds up ("how many", "how much", "in total")
# needs every mention of the thing, and the mentions read alike, which the
# packer's variety term counts against them. For such a question the packer
# weighs relevance at COUNTING_LAM instead of RECALL_LAM.
COUNTING_CUES = re.compile(r"\bhow (?:many|much)\b|\btotal\b|\bin all\b|\baltogether\b",
                           re.IGNORECASE)
RECALL_LAM = 0.7
COUNTING_LAM = 0.9

# The search's best few candidates are never damped by the role prior: an
# assistant reply that is the closest match of the whole store was ranked
# 95th after the prior (the answer to "what was on my jumpsuit" was in the
# reply, and the question has no cue that it asks what the assistant said).
PRIOR_SPARES_TOP = 3


def is_counting_query(query: str) -> bool:
    """True when ``query`` asks for a count or a sum."""
    return bool(COUNTING_CUES.search(query or ""))


def packing_lam(query: str, lam: Optional[float] = None) -> float:
    """The packer's relevance weight for ``query``: ``lam`` when given, else
    ``COUNTING_LAM`` for a counting question and ``RECALL_LAM`` otherwise."""
    if lam is not None:
        return float(lam)
    return COUNTING_LAM if is_counting_query(query) else RECALL_LAM


def apply_role_prior(pool: List[Dict], query: str) -> None:
    """Scale each candidate's ``"score"`` by ``role_prior`` for ``query``,
    except that the ``PRIOR_SPARES_TOP`` best-scored candidates are never
    scaled down."""
    first_person = is_first_person_query(query)
    assistant_said = asks_what_assistant_said(query)
    spared = {id(p) for p in sorted(pool, key=lambda p: -p["score"])[:PRIOR_SPARES_TOP]}
    for p in pool:
        factor = role_prior(first_person, p["role"], assistant_said)
        if id(p) in spared:
            factor = max(factor, 1.0)
        p["score"] = p["score"] * factor


def role_prior(first_person: bool, role: str, assistant_said: bool = False) -> float:
    """Score multiplier for a memory with source ``role``. A question about
    what the assistant said (``assistant_said``, ``asks_what_assistant_said``)
    gets no prior at all: its answer is in a reply as often as not in what the
    user said in answer to it ("the name we decided on"), so the search alone
    decides. On the eleven such questions of a 100-question LongMemEval-S
    slice it found all of the evidence for 3 with the prior, 8 with the roles
    swapped, 9 with only the assistant boosted and 10 without a prior."""
    if assistant_said:
        return 1.0
    if first_person and role in USER_ROLES:
        return USER_ROLE_BOOST
    if first_person and role in ASSISTANT_ROLES:
        return ASSISTANT_ROLE_DAMP
    return 1.0
