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

    Two optional attributes: ``fingerprint`` (a string naming the model, see
    ``embedder_fingerprint``) and ``engine_defaults`` (a dict of
    ``MemoryEngine`` settings tuned to this model's cosine geometry, which
    ``Memory`` applies under its conversational profile and below any setting
    passed to ``Memory(...)`` itself: the profile's belief thresholds
    are tuned on OpenAI vectors and do not carry over to another model).
    """

    @property
    def dimension(self) -> int:
        """Output vector dimension."""
        ...

    def encode(self, texts):
        """Embed a string or a list of strings."""
        ...


@runtime_checkable
class AsymmetricEmbedder(Embedder, Protocol):
    """An ``Embedder`` whose model treats a search query differently from the
    text it searches (EmbeddingGemma prefixes each with its own task prompt).

    ``encode`` keeps meaning "text that is stored", so a plain ``Embedder``
    needs no change: ``Memory`` asks for ``encode_query`` only when it
    embeds a question, and falls back to ``encode`` when it is missing.
    An embedder may also expose ``fingerprint`` (see ``embedder_fingerprint``).
    """

    def encode_query(self, texts):
        """Embed one question, or a list of them."""
        ...

    def encode_document(self, texts):
        """Embed text to be stored (or compared with stored text)."""
        ...


def embed_query(embedder, text):
    """Vector for a search query: ``encode_query`` where the embedder has it."""
    return (getattr(embedder, "encode_query", None) or embedder.encode)(text)


def embed_documents(embedder, texts):
    """Vectors for text that is stored, or compared with stored text:
    ``encode_document`` where the embedder has it."""
    return (getattr(embedder, "encode_document", None) or embedder.encode)(texts)


def embedder_fingerprint(embedder, dimension: int) -> str:
    """What identifies the vectors an embedder produces.

    A database must only ever hold vectors from one embedder, and two models
    of the same dimension look alike to the engine. An embedder says which it
    is with a ``fingerprint`` attribute (the ones in ``tsm.embedders`` do:
    model, revision, dimension, task prompts). Any other falls back to its
    class name and the dimension, which tells apart two different backends
    but not two models behind one class: give such an embedder a
    ``fingerprint``. The module is left out on purpose: the same class is
    ``__main__.X`` in a script and ``pkg.mod.X`` when imported, and one
    embedder must not look like two.
    """
    declared = getattr(embedder, "fingerprint", None)
    if isinstance(declared, str) and declared:
        return declared
    return f"{type(embedder).__qualname__}|dim={int(dimension)}"


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


@runtime_checkable
class Reranker(Protocol):
    """Reranks candidate retrieved texts against a query using multi-vector or cross-encoder scoring."""

    def rerank(self, query: str, texts: Sequence[str]) -> Sequence[float]:
        """Compute relevance scores for candidate texts against query."""
        ...

