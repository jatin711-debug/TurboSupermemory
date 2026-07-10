"""Verification of proposed belief-revision supersessions (W3).

An engine `propose_supersessions()` call returns candidate (old, new) pairs from
the geometric detector (mutual-nearest-neighbour + text/opposition gates). Before
the destructive demotion is committed, a verifier vets each pair semantically.
`NLIVerifier` uses a local NLI cross-encoder — no LLM server required.
"""

from .nli import NLIVerifier

__all__ = ["NLIVerifier"]
