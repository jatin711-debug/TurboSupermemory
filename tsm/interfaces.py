"""Pluggable interfaces for the ``tsm`` memory layer.

Three small protocols decouple the memory engine from how text is embedded,
how facts are extracted, and how proposed supersessions are vetted. Any object
with the matching methods plugs in — local models, remote APIs, or fakes for
testing. ``tsm`` ships OpenAI-backed defaults (``tsm.embedders``,
``tsm.extractors``) and an NLI cross-encoder verifier (``tsm.verification``).
"""

from typing import Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

# A proposed supersession pair as returned by the engine:
#   (old_id, new_id, kind, cosine) with kind in {"refinement", "contradiction"}.
ProposedPair = Tuple[str, str, str, float]
# An accepted supersession triple passed back to the engine for commitment:
#   (old_id, new_id, kind).
CommitTriple = Tuple[str, str, str]


@runtime_checkable
class Embedder(Protocol):
    """Turns text into float vectors.

    ``encode`` accepts a single string (returns a 1-D array) or a list of
    strings (returns a 2-D array, one row per text), ``float32``. This mirrors
    the SentenceTransformer surface, so SentenceTransformer models plug in
    directly.
    """

    @property
    def dimension(self) -> int:
        """Output vector dimension."""
        ...

    def encode(self, texts):
        """Embed a string or a list of strings."""
        ...


@runtime_checkable
class Extractor(Protocol):
    """Extracts atomic facts from a conversation message."""

    def extract_facts(self, message: str, context: Optional[List[str]] = None) -> List[str]:
        """Return the atomic facts asserted in ``message``.

        ``context`` is the list of preceding message texts in the conversation
        (oldest first), useful for resolving anaphora. Return an empty list
        when the message contains no storable fact.
        """
        ...


@runtime_checkable
class Verifier(Protocol):
    """Semantically vets supersession pairs proposed by the engine."""

    def verify(
        self,
        proposals: Sequence[ProposedPair],
        id_to_text: Dict[str, str],
    ) -> List[CommitTriple]:
        """Return the subset of ``proposals`` that should be committed.

        ``id_to_text`` maps memory ids to their stored text. Each accepted
        proposal is returned as an ``(old_id, new_id, kind)`` triple.
        """
        ...
