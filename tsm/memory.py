"""``tsm.Memory`` — the shipped conversational-memory facade.

A thin, pure-Python layer over the compiled ``turbomemory`` engine that
packages the mechanisms proven in the cognitive evaluations as a one-flag
preset (``profile="conversational"``):

  - role-tagged, scope-filtered fact storage (user-scoped memories),
  - belief revision with refinement/contradiction thresholds 0.85 / 0.75,
  - cognitive search (``cognitive_alpha = 0.5``) with superseded facts
    EXCLUDED from results (the B1 ghost-memory fix),
  - NLI-verified supersession: consolidation proposes, a ``Verifier`` vets,
    only accepted pairs are committed,
  - access-aware eviction and importance auto-scoring,
  - concept extraction (bigram ngrams) for the memory graph,
  - MMR best-set recall under a token budget.

Pluggable backends: pass any ``Embedder`` / ``Extractor`` / ``Verifier``
(see ``tsm.interfaces``) to use local models or other APIs. Defaults are
OpenAI-backed and read the key from ``OPENAI_API_KEY``.
"""

import json
import logging
import re
from typing import Callable, Dict, List, Optional

import numpy as np

from ._loader import load_turbomemory
from .interfaces import Embedder, Extractor, Verifier  # noqa: F401  (re-exported types)

logger = logging.getLogger("tsm.memory")

# The proven conversational configuration, from the evaluation wins. Every key
# is a MemoryEngine kwarg; explicit engine_kwargs passed to Memory() override
# these. `defer_supersession_commit` is added by Memory depending on whether a
# verifier is installed.
CONVERSATIONAL_PROFILE = {
    "exclude_superseded": True,          # B1: drop superseded facts from results
    "refinement_cosine_threshold": 0.85,
    "contradiction_cosine_threshold": 0.75,
    "cognitive_alpha": 0.5,
    "importance_auto_scoring": True,
    "concept_max_ngram_len": 2,
    "max_concepts": 10,
    "belief_source_roles": ["user"],     # only user-sourced facts supersede
    "access_aware_eviction": True,
    "auto_consolidation_secs": 0,        # manual consolidation (deterministic)
}

# Common single-word sentence starters that are capitalized for syntactic
# reasons rather than because they are proper nouns.
_SENTENCE_START_WORDS = {
    "the", "a", "an", "i", "it", "he", "she", "they", "we", "you",
    "this", "that", "these", "those", "there", "here", "what", "which",
    "when", "where", "why", "how", "if", "but", "and", "or", "so",
    "because", "although", "however", "therefore", "moreover", "furthermore",
    "actually", "basically", "honestly", "hopefully", "unfortunately",
    "fortunately", "interestingly", "surprisingly", "obviously", "clearly",
    "sure", "yes", "no", "maybe", "ok", "okay", "right", "wrong",
}

# Stop words used to filter content-word extraction.
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "each", "few",
    "more", "most", "other", "some", "such", "only", "own", "same", "than",
    "too", "very", "just", "and", "but", "if", "or", "because", "until",
    "while", "this", "that", "these", "those", "me", "my", "myself", "our",
    "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "him", "his", "himself", "her", "hers", "herself", "its", "itself",
    "them", "their", "theirs", "themselves", "what", "which", "who", "whom",
    "whose", "whoever", "whomever", "whatever", "whichever", "also",
    "about", "any", "both", "either", "neither", "nor", "not", "out",
    "over", "off", "down", "up", "now", "still", "even", "well", "back",
    "away", "around", "along", "since", "though", "unless", "whether",
})


def extract_concepts(text: str) -> List[str]:
    """Extract salient concepts from a fact for the memory graph.

    Multi-strategy: capitalized phrases (proper nouns), hyphenated compounds,
    then content words (4+ chars, not stop words). Returns at most 15
    deduplicated lowercase concepts, most salient first.
    """
    concepts: List[str] = []
    for m in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text):
        if len(m.split()) > 1 or m.lower() not in _SENTENCE_START_WORDS:
            concepts.append(m.lower())
    concepts.extend(re.findall(r"\b[a-z]+(?:-[a-z]+)+\b", text.lower()))
    for w in re.findall(r"\b[a-zA-Z]{4,}\b", text.lower()):
        if w not in _STOP_WORDS:
            concepts.append(w)

    seen = set()
    unique: List[str] = []
    for c in concepts:
        c = c.strip()
        if c and c not in seen and len(c) > 2:
            seen.add(c)
            unique.append(c)
    return unique[:15]


class Memory:
    """Scoped, self-correcting conversational memory.

    Usage::

        with Memory("./my_db") as mem:                # OPENAI_API_KEY required
            mem.add([{"role": "user", "content": "I moved to Lisbon."}],
                    user_id="alice")
            results = mem.recall("Where does Alice live?", user_id="alice")
            mem.consolidate()                          # verified belief revision

    The id->text map used to feed the verifier is kept in memory only
    (matching the proven eval adapter): after reopening a database, recall
    renders ``text`` via the engine's stored-text lookup (``get_text``)
    instead of the in-memory map, and the verifier only vets pairs whose
    texts are known in this process. Persist your own mapping if you need
    cross-process verified consolidation.

    Scope guard: the engine's exact-scan path treats an EMPTY scope bitmap
    (a scope with no records yet) as "unfiltered" and would leak other
    scopes' memories. ``recall`` therefore also drops hits whose scope is
    known in this process and does not match ``user_id``. Scope knowledge is
    in-memory only, like the text map, so the guard is exact for memories
    added by this process and defers to the engine for pre-existing ones.
    """

    def __init__(
        self,
        db_path: str,
        dimension: Optional[int] = None,
        profile: Optional[str] = "conversational",
        embedder: Optional[Embedder] = None,
        extractor: Optional[Extractor] = None,
        verifier: Optional[Verifier] = None,
        gist_summarizer: Optional[Callable[[List[str]], str]] = None,
        **engine_kwargs,
    ):
        """
        Args:
            db_path: Database directory for the engine.
            dimension: Embedding dimension. Defaults to the embedder's
                ``dimension``, else 1536 (OpenAI text-embedding-3-small).
            profile: ``"conversational"`` applies the proven preset;
                ``None`` leaves engine defaults (plain vector store).
                Explicit ``engine_kwargs`` override the profile.
            embedder: ``Embedder`` implementation. Default: ``OpenAIEmbedder``.
            extractor: ``Extractor`` implementation. Default: ``OpenAIExtractor``.
            verifier: ``Verifier`` implementation. When installed, the profile
                defers supersession commitment so ``consolidate()`` runs
                propose -> verify -> commit. ``NLIVerifier`` (local
                cross-encoder) is available in ``tsm.verification``.
            gist_summarizer: optional callable mapping a list of evicted fact
                texts to a single gist string (typically an LLM call). When
                provided, the engine's B4 gist-before-evict is enabled:
                eviction victims are compressed into searchable gist records
                (same scope, chronological chunks) instead of being dropped.
                The gist embedding comes from this Memory's ``embedder``.
            **engine_kwargs: forwarded to ``turbomemory.MemoryEngine``.
        """
        cache_dir = None
        if embedder is None or extractor is None:
            import os

            cache_dir = os.path.join(db_path, "tsm_cache")
        if embedder is None:
            from .embedders import OpenAIEmbedder

            embedder = OpenAIEmbedder(cache_dir=cache_dir)
        if extractor is None:
            from .extractors import OpenAIExtractor

            extractor = OpenAIExtractor(cache_dir=cache_dir)
        self.embedder = embedder
        self.extractor = extractor
        self.verifier = verifier

        self.dim = int(dimension or getattr(embedder, "dimension", None) or 1536)

        config: Dict = {}
        if profile == "conversational":
            config.update(CONVERSATIONAL_PROFILE)
            config["defer_supersession_commit"] = verifier is not None
        elif profile is not None:
            raise ValueError(f"unknown profile: {profile!r} (use 'conversational' or None)")
        if gist_summarizer is not None:
            config["gist_before_evict"] = True
        config.update(engine_kwargs)  # explicit kwargs win over the profile
        self.profile = profile
        self._gist_summarizer = gist_summarizer

        turbomemory = load_turbomemory()
        self.engine = turbomemory.MemoryEngine(
            db_path=db_path, dimension=self.dim, **config
        )
        if gist_summarizer is not None:
            set_compressor = getattr(self.engine, "set_gist_compressor", None)
            if set_compressor is None:
                raise ValueError(
                    "gist_summarizer requires an engine build with gist-before-evict "
                    "support (set_gist_compressor); rebuild the turbomemory extension"
                )
            set_compressor(self._compress_gist)

        # In-memory id -> fact text (see class docstring for the limitation).
        self._id_to_text: Dict[str, str] = {}
        # In-memory id -> scope, used by the recall scope guard.
        self._id_to_scope: Dict[str, Optional[str]] = {}
        self._insert_counter = 0
        self._closed = False

    # writes ----------------------------------------------------------------------
    def add(self, messages: List[Dict], user_id: str) -> int:
        """Extract facts from conversation messages and store them.

        Args:
            messages: list of ``{"role": ..., "content": ...}`` dicts
                (an optional ``"timestamp"`` is carried into the payload).
            user_id: scope the facts are stored under (recall is scoped too).

        Returns:
            The number of facts stored. Exact-text duplicates within the same
            batch are skipped (write gate); cross-batch near-duplicates are
            the engine's job (dedup config / belief revision).
        """
        facts: List[str] = []
        metas: List[Dict] = []
        seen_in_batch = set()
        context: List[str] = []
        for msg in messages:
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            role = msg.get("role", "user")
            for fact in self.extractor.extract_facts(content, context):
                norm = " ".join(fact.lower().split())
                if not norm or norm in seen_in_batch:
                    continue  # write gate: exact duplicate within this batch
                seen_in_batch.add(norm)
                facts.append(fact)
                metas.append({
                    "role": role,
                    "timestamp": msg.get("timestamp", ""),
                    "content": content,
                })
            context.append(content)

        if not facts:
            return 0

        embeddings = np.asarray(self.embedder.encode(facts), dtype=np.float32)
        for fact, meta, emb in zip(facts, metas, embeddings):
            self._insert_counter += 1
            memory_id = f"{user_id}_{self._insert_counter}" if user_id else f"mem_{self._insert_counter}"
            self._id_to_text[memory_id] = fact
            self._id_to_scope[memory_id] = user_id
            self.engine.insert(
                id=memory_id,
                text=fact,
                embedding=emb.astype(np.float32),
                importance_score=1.0,
                concepts=extract_concepts(fact),
                payload=json.dumps({
                    "timestamp": meta["timestamp"],
                    "role": meta["role"],
                    "user_id": user_id,
                    "original_message": meta["content"],
                }),
                scope=user_id,
                source_role=meta["role"],
            )
        return len(facts)

    def _compress_gist(self, texts: List[str]):
        """Gist-compressor callback handed to the engine (B4).

        Summarizes one chunk of eviction victims with the user-supplied
        ``gist_summarizer`` and embeds the gist with this Memory's embedder.
        Returns ``(gist_text, embedding)`` or ``None`` — any failure abstains
        so eviction is never blocked by the summarizer.
        """
        try:
            gist = (self._gist_summarizer(texts) or "").strip()
            if not gist:
                return None
            emb = np.asarray(self.embedder.encode([gist]), dtype=np.float32)[0]
            return gist, emb.tolist()
        except Exception as e:  # noqa: BLE001 — never block eviction
            logger.warning("gist summarizer failed for %d texts: %s", len(texts), e)
            return None

    def _engine_text(self, memory_id: str) -> str:
        """Engine-side text lookup for records this process did not mint
        (gist records, or facts from an earlier process). Empty string on
        older engines without ``get_text``."""
        get_text = getattr(self.engine, "get_text", None)
        if get_text is None:
            return ""
        try:
            return get_text(memory_id) or ""
        except Exception:  # noqa: BLE001 — text is best-effort rendering
            return ""

    # reads -----------------------------------------------------------------------
    def recall(
        self,
        query: str,
        user_id: str,
        token_budget: Optional[int] = None,
        top_k: int = 10,
        pool_k: int = 20,
        lam: float = 0.7,
        resolve_beliefs: bool = True,
    ) -> List[Dict]:
        """Search memories under ``user_id``'s scope.

        Returns a list of ``{"id", "text", "score"}`` dicts, best first.
        Superseded facts are excluded by the engine when the conversational
        profile is active.

        With ``resolve_beliefs`` (default True), results are ANNOTATED with
        belief lineage: any returned memory that has been superseded by a
        newer belief which is NOT itself in the result set gains
        ``"superseded_by"`` (the current belief's id) and ``"chain"`` (the
        full supersession chain, oldest first, head last). Nothing is dropped
        or re-ranked — an agent can present "current belief + history". Older
        engines without ``resolve_beliefs`` degrade to unannotated results.

        With ``token_budget`` set, a candidate pool of ``pool_k`` is retrieved
        and greedy Maximal-Marginal-Relevance selection (relevance weight
        ``lam=0.7`` vs. redundancy, ~4 chars per token) picks the best SET of
        memories whose estimated total tokens fit the budget, returned in
        selection order.
        """
        query_embedding = np.asarray(self.embedder.encode(query), dtype=np.float32)

        fetch_k = pool_k if token_budget is not None else top_k
        results = self.engine.search(
            query_text=query,
            query_embedding=query_embedding,
            top_k=fetch_k,
            scope=user_id,
        )
        if not results:  # empty, or the FOK gate rejected the query
            return []

        pool = []
        for mid, score in results:
            # Scope guard (see class docstring): drop known-foreign hits that
            # the engine's empty-bitmap quirk can leak into scoped searches.
            if user_id is not None:
                known_scope = self._id_to_scope.get(mid)
                if known_scope is not None and known_scope != user_id:
                    continue
            pool.append({"id": mid, "text": self._id_to_text.get(mid) or self._engine_text(mid),
                         "score": float(score)})
        if not pool:
            return []
        if token_budget is None:
            final = pool[:top_k]
        else:
            final = self._mmr_under_budget(pool, token_budget, lam)
        if resolve_beliefs:
            self._annotate_beliefs(final)
        return final

    def _annotate_beliefs(self, results: List[Dict]) -> None:
        """Attach ``superseded_by``/``chain`` lineage to superseded results.

        Only results whose current belief (chain head) is NOT itself in the
        result set are annotated — when the head is already present the agent
        sees the current belief directly. Older engines lacking
        ``resolve_beliefs`` silently leave results unannotated.
        """
        resolve = getattr(self.engine, "resolve_beliefs", None)
        if resolve is None or not results:
            return
        present = {r["id"] for r in results}
        by_id = {r["id"]: r for r in results}
        for res in resolve([r["id"] for r in results]):
            current = res["current_id"]
            if current != res["id"] and current not in present:
                by_id[res["id"]]["superseded_by"] = current
                by_id[res["id"]]["chain"] = list(res["chain"])

    def _mmr_under_budget(self, pool: List[Dict], token_budget: int, lam: float) -> List[Dict]:
        """Greedy submodular MMR best-set selection (proven recall_under_budget).

        Each step adds the candidate maximizing
        ``lam * relevance - (1 - lam) * max_redundancy`` against the selected
        set, skipping candidates that would overflow the token budget.
        """
        texts = [p["text"] or "" for p in pool]
        rel = np.array([p["score"] for p in pool], dtype=np.float32)
        toks = np.array([max(1, len(t) // 4) for t in texts], dtype=np.int64)

        # Pairwise redundancy from pool embeddings (unit-normed -> cosine).
        embs = np.asarray(self.embedder.encode(texts), dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        sim = (embs / norms) @ (embs / norms).T

        selected, used, remaining = [], 0, list(range(len(pool)))
        while remaining:
            best_i, best_gain = None, -1e9
            for i in remaining:
                if used + int(toks[i]) > token_budget:
                    continue
                red = max((float(sim[i, j]) for j in selected), default=0.0)
                gain = lam * float(rel[i]) - (1.0 - lam) * red
                if gain > best_gain:
                    best_gain, best_i = gain, i
            if best_i is None:
                break  # nothing else fits the budget
            selected.append(best_i)
            used += int(toks[best_i])
            remaining.remove(best_i)
        return [pool[i] for i in selected]

    # maintenance -----------------------------------------------------------------
    def consolidate(self) -> int:
        """Run consolidation; with a verifier installed, vet supersessions.

        The engine runs its consolidation cycle (dedup, importance, belief
        detection). When a ``Verifier`` is installed, supersession commitment
        is deferred: candidates are proposed, vetted against the id->text map
        (semantic gate — accept contradiction/entailment, reject neutral),
        and only accepted pairs are committed, after which the engine's
        superseded-exclusion hides the stale facts from recall.

        Returns:
            The number of supersession edges committed (0 without a verifier).
        """
        self.engine.trigger_consolidation()
        if self.verifier is None:
            return 0
        proposed = self.engine.propose_supersessions()  # (old, new, kind, cosine)
        if not proposed:
            return 0
        accepted = self.verifier.verify(proposed, self._id_to_text)
        if not accepted:
            return 0
        committed = self.engine.commit_supersessions(accepted)
        logger.info("verified supersession: proposed=%d accepted=%d committed=%d",
                    len(proposed), len(accepted), committed)
        return committed

    def flush(self) -> None:
        """Durably persist all pending writes."""
        self.engine.flush()

    def close(self) -> None:
        """Flush and shut down the engine."""
        if not self._closed:
            self._closed = True
            self.engine.close()

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
