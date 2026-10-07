"""``tsm.Memory`` — the shipped conversational-memory facade.

A thin, pure-Python layer over the compiled ``turbomemory`` engine that
packages the mechanisms proven in the cognitive evaluations as a one-flag
preset (``profile="conversational"``):

  - role-tagged, scope-filtered fact storage (user-scoped memories),
  - belief revision with refinement/contradiction thresholds 0.85 / 0.75,
  - cognitive search (``cognitive_alpha = 0.5``),
  - verified supersession: consolidation proposes, a ``Verifier`` vets, only
    accepted pairs are committed, and the superseded facts are then EXCLUDED
    from results (the B1 ghost-memory fix). Without a verifier the engine's
    own detection is not trusted to hide anything: superseded facts are
    ranked lower and flagged instead,
  - access-aware eviction and importance auto-scoring,
  - concept extraction (bigram ngrams) for the memory graph,
  - MMR best-set recall under a token budget.

Pluggable backends: pass any ``Embedder`` / ``Extractor`` / ``Verifier``
(see ``tsm.interfaces``) to use local models or other APIs. Defaults are
OpenAI-backed and read the key from ``OPENAI_API_KEY``.
"""

import json
import logging
import threading
from typing import Callable, Dict, List, Optional

import numpy as np

from ._loader import load_turbomemory
from .budget import select_under_budget
from .concepts import extract_concepts
from .interfaces import Embedder, Extractor, Verifier  # noqa: F401  (re-exported types)
from .ranking import is_first_person_query, role_prior

logger = logging.getLogger("tsm.memory")

# Engine methods this SDK cannot work without. The extension and the SDK ship
# together; this guards against a stale locally-built turbomemory.pyd/.so.
_REQUIRED_ENGINE_METHODS = ("get_records", "next_insert_seq", "recovery_report")

# The proven conversational configuration, from the evaluation wins. Every key
# is a MemoryEngine kwarg; explicit engine_kwargs passed to Memory() override
# these. `defer_supersession_commit` is added by Memory depending on whether a
# verifier is installed, and `exclude_superseded` only applies with one.
CONVERSATIONAL_PROFILE = {
    "exclude_superseded": True,          # B1: drop VERIFIED superseded facts from results
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

class Memory:
    """Scoped, self-correcting conversational memory.

    Usage::

        with Memory("./my_db") as mem:                # OPENAI_API_KEY required
            mem.add([{"role": "user", "content": "I moved to Lisbon."}],
                    user_id="alice")
            results = mem.recall("Where does Alice live?", user_id="alice")
            mem.consolidate()                          # verified belief revision

    Durability: the engine is the only store. Text, role, and scope are
    read back from it at recall time and ids are minted from its durable
    insert sequence, so a database behaves the same after it is reopened —
    in this process or another — as it did when it was written: ``add``
    keeps appending, the role prior still applies, and a verifier installed
    later can vet facts stored by an earlier session.

    Scope guard: ``recall`` drops any hit whose stored scope is neither
    ``user_id`` nor global (unscoped). This is belt-and-suspenders over the
    engine's own scope filter.

    ``close()`` releases the database (lock, mmaps, worker threads); the
    same ``db_path`` can be reopened immediately afterwards.
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
        reranker: Optional[object] = None,
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
            verifier: ``Verifier`` implementation, or ``"nli"`` /``"llm"``
                for the two in ``tsm.verification`` (``NLIVerifier``: a small
                local cross-encoder; ``LLMVerifier``: a chat model, far more
                accurate, one short request per few candidate pairs). When
                installed, ``consolidate()`` runs propose -> verify -> commit
                and superseded facts are excluded from recall. Without one,
                superseded facts stay in results, ranked lower and flagged
                (``superseded_by``): unverified detection hides true facts
                too often to be allowed to remove them.
            gist_summarizer: optional callable mapping a list of evicted fact
                texts to a single gist string (typically an LLM call).
            reranker: optional ``Reranker`` implementation or ``"colbert"``
                to enable Stage-2 MultiVector late-interaction MaxSim precision
                reranking (e.g. ``LFM2.5-ColBERT-350M``).
            **engine_kwargs: forwarded to ``turbomemory.MemoryEngine``.
        """
        # Backend names are resolved here; anything else that is a string is a
        # typo, and is reported now rather than at the first add().
        for kind, value, known in (
            ("embedder", embedder, ("openai", "sentence_transformer", "local", "minilm")),
            ("extractor", extractor, ("openai", "gliner", "passthrough")),
            ("reranker", reranker, ("colbert",)),
            ("verifier", verifier, ("nli", "llm")),
        ):
            if isinstance(value, str) and value not in known:
                raise ValueError(
                    f"unknown {kind} {value!r}: use one of {', '.join(known)} "
                    f"or pass an instance")
        if embedder == "openai":
            embedder = None
        cache_dir = None
        if embedder is None or extractor is None or extractor == "openai":
            import os

            cache_dir = os.path.join(db_path, "tsm_cache")
        if embedder is None:
            from .embedders import OpenAIEmbedder

            embedder = OpenAIEmbedder(cache_dir=cache_dir)
        elif embedder in ("sentence_transformer", "local", "minilm"):
            from .embedders import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder()
        if extractor is None or extractor == "openai":
            from .extractors import OpenAIExtractor

            extractor = OpenAIExtractor(cache_dir=cache_dir)
        elif extractor == "gliner":
            from .extractors import GlinerExtractor

            extractor = GlinerExtractor(cache_dir=cache_dir)
        elif extractor == "passthrough" or extractor is False:
            from .extractors import PassthroughExtractor

            extractor = PassthroughExtractor()
        if reranker == "colbert":
            from .rerankers import ColBertReranker

            reranker = ColBertReranker()
        if verifier == "nli":
            from .verification import NLIVerifier

            verifier = NLIVerifier()
        elif verifier == "llm":
            import os

            from .verification import LLMVerifier

            verifier = LLMVerifier(cache_dir=os.path.join(db_path, "tsm_cache"))
        self.embedder = embedder
        self.extractor = extractor
        self.verifier = verifier
        self.reranker = reranker

        self.dim = int(dimension or getattr(embedder, "dimension", None) or 1536)

        config: Dict = {}
        if profile == "conversational":
            config.update(CONVERSATIONAL_PROFILE)
        elif profile is not None:
            raise ValueError(f"unknown profile: {profile!r} (use 'conversational' or None)")
        # A verifier only gets to vet supersessions if the engine does not
        # commit them first, whatever the profile.
        config["defer_supersession_commit"] = verifier is not None
        # Hiding a fact is only safe once something has checked the pair: on
        # its own the detector also fires on facts that are both still true
        # ("my sister lives in Vancouver" / "my brother lives in Vancouver").
        # Unverified supersessions rank lower and are flagged in recall().
        if verifier is None and "exclude_superseded" in config:
            config["exclude_superseded"] = False
        if gist_summarizer is not None:
            config["gist_before_evict"] = True
        config.update(engine_kwargs)  # explicit kwargs win over the profile
        self.profile = profile
        self._gist_summarizer = gist_summarizer

        turbomemory = load_turbomemory()
        self.engine = turbomemory.MemoryEngine(
            db_path=db_path, dimension=self.dim, **config
        )
        self._closed = False
        # One writer at a time: id minting and the insert it feeds must not
        # interleave between threads sharing this Memory.
        self._write_lock = threading.Lock()
        missing = [m for m in _REQUIRED_ENGINE_METHODS if not hasattr(self.engine, m)]
        if missing:
            self.close()
            raise RuntimeError(
                "the compiled turbomemory extension is older than this tsm SDK "
                f"(missing: {', '.join(missing)}); rebuild it with 'make build-python'"
            )
        if gist_summarizer is not None:
            set_compressor = getattr(self.engine, "set_gist_compressor", None)
            if set_compressor is None:
                self.close()  # do not leave the database locked behind the error
                raise ValueError(
                    "gist_summarizer requires an engine build with gist-before-evict "
                    "support (set_gist_compressor); rebuild the turbomemory extension"
                )
            set_compressor(self._compress_gist)

        # Ids are `{user_id}_{n}`. Seeding n from the engine's durable insert
        # sequence (which starts at 1 and is never reused) keeps ids unique
        # when an existing database is reopened.
        self._insert_counter = max(0, int(self.engine.next_insert_seq()) - 1)

    # writes ----------------------------------------------------------------------
    def add(self, messages: List[Dict], user_id: str) -> int:
        """Extract facts from conversation messages and store them.

        Args:
            messages: list of ``{"role": ..., "content": ...}`` dicts
                (optional ``"timestamp"`` and ``"turn_index"`` are carried
                into the payload). Facts extracted from one message share a
                turn index, which budget recall uses to spread its selection
                across turns; when the caller gives none, one is assigned.
            user_id: scope the facts are stored under (recall is scoped too).

        Returns:
            The number of facts stored. Exact-text duplicates within the same
            batch are skipped (write gate); cross-batch near-duplicates are
            the engine's job (dedup config / belief revision).

        The facts of one call are stored as a single batch that the engine
        validates before writing anything, so a call either stores all of its
        facts or raises having stored none. Safe to call from several threads.
        """
        self._require_open()
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            messages = [messages]

        facts: List[str] = []
        metas: List[Dict] = []
        seen_in_batch = set()
        context: List[str] = []
        for turn, msg in enumerate(messages):
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
                    "turn": turn,
                    "turn_index": msg.get("turn_index"),
                })
            context.append(content)

        if not facts:
            return 0

        embeddings = np.ascontiguousarray(self.embedder.encode(facts), dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape != (len(facts), self.dim):
            raise ValueError(
                f"embedder returned shape {embeddings.shape} for {len(facts)} facts; "
                f"expected ({len(facts)}, {self.dim})"
            )
        with self._write_lock:
            assigned_turns: Dict[int, int] = {}
            ids: List[str] = []
            payloads: List[str] = []
            for meta in metas:
                ids.append(self._next_id(user_id))
                # A caller-supplied turn index wins. Otherwise the turn is keyed
                # by the sequence number of its first stored fact: unique per
                # message and durable across restarts, like the ids themselves.
                turn_index = meta["turn_index"]
                if turn_index is None:
                    turn_index = assigned_turns.setdefault(meta["turn"], self._insert_counter)
                payloads.append(json.dumps({
                    "timestamp": meta["timestamp"],
                    "role": meta["role"],
                    "user_id": user_id,
                    "original_message": meta["content"],
                    "turn_index": turn_index,
                }))
            self.engine.insert_batch(
                ids,
                facts,
                embeddings,
                [1.0] * len(facts),
                [extract_concepts(fact) for fact in facts],
                payloads,
                [user_id] * len(facts) if user_id is not None else None,
                [str(meta["role"]) for meta in metas],
            )
        return len(facts)

    def _compress_gist(self, texts: List[str]):
        """Gist-compressor callback handed to the engine (B4).

        Summarizes one chunk of eviction victims with the user-supplied
        ``gist_summarizer`` and embeds the gist with this Memory's embedder.
        Returns ``(gist_text, embedding)``, or ``None`` when the summarizer
        produced nothing (it found nothing worth keeping, so the chunk is
        dropped without a gist).

        A failure is NOT an abstention: if the summarizer or the embedder
        raises, the exception is passed on to the engine, which then keeps
        this chunk's memories and tries again on the next eviction. Swallowing
        it would delete them with no gist in their place.
        """
        try:
            gist = (self._gist_summarizer(texts) or "").strip()
            if not gist:
                return None
            emb = np.asarray(self.embedder.encode([gist]), dtype=np.float32)[0]
            return gist, emb.tolist()
        except Exception as e:  # noqa: BLE001 — logged here, handled by the engine
            logger.warning("gist summarizer failed for %d texts (kept, will retry): %s",
                           len(texts), e)
            raise

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Memory is closed")

    def _next_id(self, user_id: Optional[str]) -> str:
        """Mint a ``{user_id}_{n}`` id that is not live in the engine."""
        while True:
            self._insert_counter += 1
            memory_id = (f"{user_id}_{self._insert_counter}" if user_id
                         else f"mem_{self._insert_counter}")
            # Only reachable for ids this SDK did not mint (e.g. records
            # written straight through the engine); costs one lookup per fact.
            if not self.engine.contains_id(memory_id):
                return memory_id

    def _records(self, ids: List[str]) -> Dict[str, Dict]:
        """Stored record metadata for ``ids``, read back from the engine.

        Maps id -> ``{"text", "scope", "source_role", "payload", ...}`` with
        ``payload`` parsed to a dict. Ids that are no longer live are omitted.
        """
        out: Dict[str, Dict] = {}
        for rec in self.engine.get_records(list(ids)):
            if rec is None:
                continue
            try:
                rec["payload"] = json.loads(rec["payload"]) if rec["payload"] else {}
            except (TypeError, ValueError):
                rec["payload"] = {}
            out[rec["id"]] = rec
        return out

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
        rerank: bool = False,
        reranker: Optional[object] = None,
    ) -> List[Dict]:
        """Search memories under ``user_id``'s scope.

        Returns a list of dicts, best first, each with at least ``"id"``,
        ``"text"``, ``"score"``, ``"role"`` (the stored source role) and
        ``"turn_index"``. With ``token_budget`` set, the result is instead the
        best *set* that fits the budget (``tsm.budget.select_under_budget``:
        greedy MMR over a pool of ``pool_k`` candidates), in selection order.
        Superseded facts are excluded by the engine when the conversational
        profile is active.

        With ``resolve_beliefs`` (default True), results are ANNOTATED with
        belief lineage: any returned memory that has been superseded gains
        ``"superseded_by"`` (the current belief's id) and ``"chain"`` (the
        full supersession chain, oldest first, head last), whether or not the
        current belief is in the result set too. A result without
        ``"superseded_by"`` is current.

        With ``rerank=True`` or an active ``reranker`` (e.g. ``ColBertReranker``),
        retrieved candidate shortlists from TSM's cognitive graph are reranked
        using token-level MaxSim late interaction.
        """
        self._require_open()
        query_embedding = np.asarray(self.embedder.encode(query), dtype=np.float32)

        active_reranker = reranker or (self.reranker if rerank else None)
        if rerank and active_reranker is None:
            from .rerankers import ColBertReranker
            active_reranker = ColBertReranker()

        fetch_k = max(pool_k, 30) if token_budget is not None else (top_k * 3 if active_reranker else top_k)
        results = self.engine.search(
            query_text=query,
            query_embedding=query_embedding,
            top_k=fetch_k,
            scope=user_id,
        )
        if not results:  # nothing in this scope matched
            return []

        records = self._records([mid for mid, _ in results])
        pool = []
        for mid, score in results:
            rec = records.get(mid)
            if rec is None:
                continue  # deleted or evicted since the search returned it
            # Scope guard: only this user's memories and global (unscoped) ones.
            if user_id is not None and rec["scope"] not in (None, user_id):
                continue
            pool.append({"id": mid, "text": rec["text"], "score": float(score),
                         "role": rec["source_role"] or "",
                         "turn_index": rec["payload"].get("turn_index")})
        if not pool:
            return []

        first_person = is_first_person_query(query)
        for p in pool:
            p["score"] = p["score"] * role_prior(first_person, p["role"])

        # Stage-2 MultiVector / ColBERT Late-Interaction Precision Reranking
        if active_reranker is not None and len(pool) > 1:
            candidate_texts = [p["text"] or "" for p in pool]
            rerank_scores = np.asarray(active_reranker.rerank(query, candidate_texts), dtype=np.float32)
            if len(rerank_scores) == len(pool):
                exp_scores = np.exp(rerank_scores - np.max(rerank_scores))
                norm_sim = exp_scores / np.sum(exp_scores)
                for idx, p in enumerate(pool):
                    p["maxsim_score"] = float(rerank_scores[idx])
                    p["score"] = float(p["score"]) * (1.0 + float(norm_sim[idx]) * len(pool))
                pool.sort(key=lambda x: x["score"], reverse=True)

        if token_budget is None:
            final = pool[:top_k]
        else:
            final = select_under_budget(pool, token_budget,
                                        embed=self.embedder.encode, lam=lam)
        if resolve_beliefs:
            self._annotate_beliefs(final)
        return final

    def _annotate_beliefs(self, results: List[Dict]) -> None:
        """Attach ``superseded_by``/``chain`` lineage to superseded results.

        Every superseded result is annotated, also when its current belief is
        in the result set: with both in front of it, a reader still has to be
        told which of the two is the stale one. Older engines lacking
        ``resolve_beliefs`` silently leave results unannotated.
        """
        resolve = getattr(self.engine, "resolve_beliefs", None)
        if resolve is None or not results:
            return
        by_id = {r["id"]: r for r in results}
        for res in resolve([r["id"] for r in results]):
            current = res["current_id"]
            if current != res["id"]:
                by_id[res["id"]]["superseded_by"] = current
                by_id[res["id"]]["chain"] = list(res["chain"])

    # maintenance -----------------------------------------------------------------
    def consolidate(self) -> int:
        """Run consolidation; with a verifier installed, vet supersessions.

        The engine runs its consolidation cycle (dedup, importance, belief
        detection). When a ``Verifier`` is installed, supersession commitment
        is deferred: candidates are proposed, vetted against their stored
        texts, and only accepted pairs are committed, after which the engine's
        superseded-exclusion hides the stale facts from recall.

        Which candidates are proposed depends on the verifier. One that can
        judge meaning says so with a ``candidate_min_cosine`` attribute
        (``LLMVerifier``) and is given every new fact paired with its nearest
        older facts (the closest one and any about as close). Any other
        verifier (``NLIVerifier``) only sees the pairs that already passed the
        engine's own lexical gates.

        Returns:
            The number of supersession edges committed (0 without a verifier).
        """
        self._require_open()
        self.engine.trigger_consolidation()
        if self.verifier is None:
            return 0
        min_cosine = getattr(self.verifier, "candidate_min_cosine", None)
        wide = getattr(self.engine, "propose_supersession_candidates", None)
        if min_cosine is not None and wide is not None:
            proposed = wide(float(min_cosine),
                            int(getattr(self.verifier, "candidates_per_record", 2)),
                            getattr(self.verifier, "candidate_margin", 0.1))
        else:
            proposed = self.engine.propose_supersessions()  # (old, new, kind, cosine)
        if not proposed:
            return 0
        pair_ids = sorted({mid for old, new, *_ in proposed for mid in (old, new)})
        id_to_text = {mid: rec["text"] for mid, rec in self._records(pair_ids).items()}
        accepted = self.verifier.verify(proposed, id_to_text)
        if not accepted:
            return 0
        committed = self.engine.commit_supersessions(accepted)
        logger.info("verified supersession: proposed=%d accepted=%d committed=%d",
                    len(proposed), len(accepted), committed)
        return committed

    def flush(self) -> None:
        """Durably persist all pending writes."""
        self._require_open()
        self.engine.flush()

    def close(self) -> None:
        """Flush and release the engine. Idempotent.

        Releases the database lock, mmaps, and worker threads, so ``db_path``
        can be reopened right away. Any later call on this object raises
        ``RuntimeError``.
        """
        if not self._closed:
            try:
                self.engine.close()
            finally:
                # The engine handle is released even when its final flush
                # raises, so this object is closed either way.
                self._closed = True
                # Persist the backends' paid-for caches (extractions,
                # embeddings); they otherwise only reach disk every few
                # hundred entries or at interpreter exit.
                for backend, method in ((self.extractor, "flush_cache"),
                                        (self.embedder, "flush"),
                                        (self.verifier, "flush_cache")):
                    flush = getattr(backend, method, None)
                    if callable(flush):
                        try:
                            flush()
                        except Exception as e:  # noqa: BLE001 — never fail close()
                            logger.warning("cache flush failed on close: %s", e)

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
