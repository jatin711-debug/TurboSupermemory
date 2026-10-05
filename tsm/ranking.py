"""Role prior applied to recall scores.

A question phrased from the user's side ("what did I...", "how many ...") is
far more often answered by something the user said than by the assistant's own
reply, so user-sourced facts are boosted and assistant-sourced ones damped for
such queries.
"""

# Tokens that mark a query as asking about the user's own facts. Deliberately
# broad (it includes common question words): this is the cue set the
# evaluations were run with, so change it together with a re-run.
FIRST_PERSON_CUES = frozenset({
    "i", "my", "me", "mine", "we", "our", "did", "have", "how", "what",
    "total", "number", "many",
})

USER_ROLE_BOOST = 1.30
ASSISTANT_ROLE_DAMP = 0.85


def is_first_person_query(query: str) -> bool:
    """True when ``query`` contains any first-person / question cue token."""
    return any(w in FIRST_PERSON_CUES for w in query.lower().split())


def role_prior(first_person: bool, role: str) -> float:
    """Score multiplier for a memory with source ``role``."""
    if first_person and role == "user":
        return USER_ROLE_BOOST
    if first_person and role == "assistant":
        return ASSISTANT_ROLE_DAMP
    return 1.0
