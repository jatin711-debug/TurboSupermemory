"""TurboSuperMemory adapter for benchmark compatibility.

Makes TSM behave like Mem0 for benchmark evaluation:
- Mem0-style `add(messages, user_id)` API
- Mem0-style `search(query, user_id, top_k)` API
- Automatic fact extraction via LLM
- Temporal metadata tracking
"""

import json
import logging
import os
import shutil
import sys
import tempfile
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger("cognitive_eval.adapters.tsm")


# Common single-word sentence starters that are capitalized for syntactic
# reasons rather than because they are proper nouns. Multi-word capitalized
# spans are always kept (they are almost always named entities).
_SENTENCE_START_WORDS = {
    "the", "a", "an", "i", "it", "he", "she", "they", "we", "you",
    "this", "that", "these", "those", "there", "here", "what", "which",
    "when", "where", "why", "how", "if", "but", "and", "or", "so",
    "because", "although", "however", "therefore", "moreover", "furthermore",
    "actually", "basically", "honestly", "hopefully", "unfortunately",
    "fortunately", "interestingly", "surprisingly", "obviously", "clearly",
    "sure", "yes", "no", "maybe", "ok", "okay", "right", "wrong",
}

# Stop words used to filter content-word extraction. Kept as a frozenset for
# O(1) membership tests in the hot path.
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


def _setup_turbomemory():
    """Locate and load the compiled turbomemory extension."""
    # Find project root (2 levels up from this file: adapters/ -> cognitive_eval/ -> benchmarks/ -> project_root)
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    project_root = os.path.dirname(script_dir)
    ext = ".pyd" if sys.platform.startswith("win") else ".so"
    pyd = os.path.join(project_root, f"turbomemory{ext}")
    
    # Try to find and copy the DLL if needed
    if sys.platform.startswith("win") and not os.path.exists(pyd):
        dll = os.path.join(project_root, "target", "release", "turbomemory.dll")
        if os.path.exists(dll):
            shutil.copy2(dll, pyd)
    
    if not os.path.exists(pyd):
        raise RuntimeError(
            f"turbomemory extension not found at {pyd}. "
            "Run 'make build-python' first."
        )
    
    sys.path.insert(0, project_root)
    import turbomemory
    return turbomemory


class TSMAdapter:
    """Adapter that makes TSM behave like Mem0 for benchmark compatibility.
    
    Usage:
        adapter = TSMAdapter(
            db_path="./eval_db",
            embedding_model="BAAI/bge-large-en-v1.5",
            extractor="ollama",  # or "mock"
        )
        
        # Add conversation (Mem0-style)
        adapter.add(messages, user_id="user_123")
        
        # Search (Mem0-style)
        results = adapter.search("Where does the user live?", user_id="user_123", top_k=3)
    """
    
    def __init__(
        self,
        db_path: str,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        extractor: str = "mock",
        extractor_model: str = "llama3.2:3b",
        extractor_instance=None,   # W6: a prebuilt extractor to SHARE across
                                   # adapters/arms (reuses its cross-arm cache).
        cognitive_features: bool = False,  # Disabled by default for benchmarks
        belief_revision: bool = True,  # Build Contradicts/Refines edges + demotion
        dimension: Optional[int] = None,
        model=None,  # Preloaded embedding model to share across adapters
        store_roles=None,  # e.g. {"user"} to store only user facts; None = all roles
        belief_source_roles=None,  # e.g. ["user"]: store ALL roles but role-scope
                                   # belief detection in the engine (mode b). Every
                                   # memory stays retrievable; only supersession is
                                   # restricted to these roles.
        verify_demotions=False,    # W3: gate each supersession through an NLI
                                   # cross-encoder before the destructive demotion.
        verifier=None,             # preloaded NLIVerifier to share across adapters.
        concept_expansion=None,    # W4: enable/disable concept + abstraction graph
                                   # expansion in the augmenter (None = engine default=on).
        max_records=None,          # W5: bounded-storage cap (forces eviction).
        access_aware_eviction=None,# W5: cognitive retain-what-is-used eviction (True)
                                   # vs naive FIFO baseline (False). None = default(on).
        supersession_mode="demote",# B1: what to do with superseded facts at recall.
                                   # "demote" = rank only (current); "exclude" = drop
                                   # them from the answer context; "tag" = prefix
                                   # their text with [OUTDATED].
        **kwargs,
    ):
        """Initialize the TSM adapter.
        
        Args:
            db_path: Path to TSM database directory
            embedding_model: SentenceTransformer model name for embeddings
            extractor: Fact extractor to use ("ollama", "mock")
            extractor_model: Ollama model name (if using Ollama)
            cognitive_features: Enable cognitive layer features (slow, for production only)
            dimension: Embedding dimension (auto-detected if None)
        """
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self.cognitive_features = cognitive_features
        # Optional role filter: a memory of USER facts should not ingest the
        # assistant's own (verbose, repetitive) responses as revisable "facts".
        # (mode a: drop non-matching messages entirely.)
        self.store_roles = set(store_roles) if store_roles is not None else None
        # mode b: store every role but restrict belief-revision detection to
        # these roles inside the engine (first-class role-aware memory).
        self.belief_source_roles = list(belief_source_roles) if belief_source_roles else None
        # W3: verified demotion. When on, consolidation defers supersession
        # commitment and the adapter drives propose -> NLI-verify -> commit.
        self.verify_demotions = bool(verify_demotions)
        self.verifier = verifier
        if self.verify_demotions and self.verifier is None:
            from ..verification import NLIVerifier
            self.verifier = NLIVerifier()
        # W4: concept/abstraction graph expansion toggle (None = engine default).
        self.concept_expansion = concept_expansion
        # W5: retention/eviction knobs.
        self.max_records = max_records
        self.access_aware_eviction = access_aware_eviction
        # B1: supersession handling at recall.
        self.supersession_mode = supersession_mode
        self._superseded_cache = None  # set[str], invalidated on add/consolidate
        # Stop-word set used by _extract_concepts.
        self._stop_words = _STOP_WORDS

        # Store mapping of id -> text for retrieval
        self._id_to_text = {}
        
        # Load embedding model (with fallback for sentence-transformers issues)
        # We try sentence-transformers first, but on Windows it often fails
        # due to torchcodec/FFmpeg dependency issues
        self.model = model
        self.dim = dimension
        if self.model is not None and self.dim is None:
            dim_attr = self.model.get_sentence_embedding_dimension
            self.dim = dim_attr() if callable(dim_attr) else dim_attr

        # Try sentence-transformers first (only if not on Windows or if explicitly requested)
        if sys.platform != "win32":
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(embedding_model)
                self.dim = dimension or self.model.get_sentence_embedding_dimension()
                logger.info("Loaded embedding model: %s (dim=%d)", embedding_model, self.dim)
            except Exception as e:
                logger.warning("sentence-transformers failed: %s", e)
        
        # Fallback to transformers directly
        if self.model is None:
            logger.info("Using transformers fallback for embeddings")
            from ..embedding import create_embedding_provider
            # Pass batch_size if provided in kwargs
            batch_size = kwargs.get('batch_size', 32)
            self.model = create_embedding_provider(embedding_model, batch_size=batch_size)
            # Handle both property (SimpleEmbeddingProvider) and method (SentenceTransformer)
            dim_attr = self.model.get_sentence_embedding_dimension
            self.dim = dimension or (dim_attr() if callable(dim_attr) else dim_attr)
            logger.info("Loaded embedding model via fallback: %s (dim=%d)", embedding_model, self.dim)
        
        # Load turbomemory
        self.tsm = _setup_turbomemory()
        
        # Initialize extractor via the factory (auto/ollama/openai/mock). A
        # shared prebuilt instance short-circuits so its cross-arm cache is reused.
        from ..extraction import create_extractor
        self.extractor = create_extractor(extractor, ollama_model=extractor_model,
                                          shared=extractor_instance)
        
        # Initialize TSM engine with cognitive features
        config = {
            "db_path": db_path,
            "dimension": self.dim,
            "max_concepts": 10,
            "auto_consolidation_secs": 0,  # Manual consolidation for determinism
        }
        
        # Enable cognitive features by default for better recall
        # These thresholds are tuned for conversational memory retrieval
        config.update({
            "refinement_cosine_threshold": 0.5,
            "contradiction_cosine_threshold": 0.5,
            "contradiction_text_threshold": 0.3,
            "contradiction_weaken_factor": 0.5,
            "cognitive_alpha": 0.7,  # Bounded additive graph boost over cosine
            "spreading_iterations": 2,  # 2-hop bounded augmenter for better coverage
            "spreading_decay": 0.5,
            "seed_hops_from": 10,  # Expand from top-10 ANN seeds
            "expansion_max_candidates": 50,  # Cap added candidates
            "importance_auto_scoring": True,
            "concept_evolution_enabled": True,
            "abstraction_co_occurrence_threshold": 3,
            "edge_decay_half_life_secs": 0,  # No decay for persistent memories
        })
        
        if cognitive_features:
            config.update({
                "refinement_cosine_threshold": 0.5,
                "contradiction_cosine_threshold": 0.5,
                "contradiction_text_threshold": 0.3,
                "contradiction_weaken_factor": 0.5,
                "cognitive_alpha": 0.7,  # Bounded additive graph boost over cosine
                "spreading_iterations": 2,  # 2-hop bounded augmenter for better coverage
                "spreading_decay": 0.5,
                "seed_hops_from": 10,  # Expand from top-10 ANN seeds
                "expansion_max_candidates": 50,  # Cap added candidates
                "importance_auto_scoring": True,
                "concept_evolution_enabled": True,
                "abstraction_co_occurrence_threshold": 3,
            })
        
        # Belief-revision toggle: disabling nulls the refinement/contradiction
        # thresholds so NO Contradicts/Refines edges (and no supersession
        # demotion) are created. Everything else (concepts, abstraction, alpha,
        # spreading) stays identical, so an ON-vs-OFF delta is attributable to
        # belief revision specifically.
        if not belief_revision:
            config["refinement_cosine_threshold"] = None
            config["contradiction_cosine_threshold"] = None
        # Engine-level role scoping for belief revision (mode b). The engine
        # then only lets memories whose source_role is in this list create or
        # receive supersession edges.
        if self.belief_source_roles:
            config["belief_source_roles"] = self.belief_source_roles
        # Verified demotion: defer supersession commitment so the adapter can
        # vet each pair before it demotes anything.
        if self.verify_demotions:
            config["defer_supersession_commit"] = True
        if self.concept_expansion is not None:
            config["concept_expansion"] = bool(self.concept_expansion)
        # W5: bounded storage + eviction policy.
        if self.max_records is not None:
            config["max_records"] = int(self.max_records)
        if self.access_aware_eviction is not None:
            config["access_aware_eviction"] = bool(self.access_aware_eviction)
        self.engine = self.tsm.MemoryEngine(**config)
        logger.info("TSM engine initialized (cognitive=%s, belief_revision=%s)",
                    cognitive_features, belief_revision)
        
        # Counter for unique IDs
        self._insert_counter = 0
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract meaningful concepts from text for graph building.

        Uses a multi-strategy approach:
        1. Named entities (capitalized phrases) - high-value proper nouns
        2. Compound words with hyphens (high semantic value)
        3. Nouns and content words (>3 chars, not stop words)

        Prioritizes semantically meaningful concepts that help build the
        memory graph. Returns at most 15 deduplicated lowercase concepts.
        """
        import re

        concepts: List[str] = []

        # Strategy 1: Capitalized phrases (proper nouns, names, places, orgs).
        # These carry the highest discriminative value for retrieval.
        for m in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text):
            # Skip common sentence-start words unless multi-word.
            m_lower = m.lower()
            if len(m.split()) > 1 or m_lower not in _SENTENCE_START_WORDS:
                concepts.append(m_lower)

        # Strategy 2: Compound words with hyphens (e.g. "state-of-the-art").
        concepts.extend(re.findall(r"\b[a-z]+(?:-[a-z]+)+\b", text.lower()))

        # Strategy 3: Content words (nouns and important terms).
        # A 4-char minimum plus stop-word filter removes most function words
        # while keeping nouns, verbs, and domain terms.
        for w in re.findall(r"\b[a-zA-Z]{4,}\b", text.lower()):
            if w not in self._stop_words:
                concepts.append(w)

        # Deduplicate while preserving order of first occurrence.
        seen = set()
        unique: List[str] = []
        for c in concepts:
            c_clean = c.strip()
            if c_clean and c_clean not in seen and len(c_clean) > 2:
                seen.add(c_clean)
                unique.append(c_clean)

        # Cap the number of concepts to bound graph density. The engine's
        # max_concepts setting also caps this, but pre-filtering here keeps
        # the most salient (earliest) concepts.
        return unique[:15]
    
    def add(self, messages: List[Dict], user_id: Optional[str] = None, batch: bool = True) -> Dict:
        """Add conversation messages to memory (Mem0-compatible API).
        
        Args:
            messages: List of message dicts or Message objects with keys:
                - role: "user" or "assistant"
                - content: Message text
                - timestamp: ISO timestamp string
            user_id: Optional user/conversation ID for scoping
            batch: Whether to use batch embedding (faster, more memory)
            
        Returns:
            Dict with timing breakdown for profiling
        """
        import time
        total_start = time.perf_counter()
        
        # Normalize messages to dicts
        msg_dicts = []
        for msg in messages:
            if hasattr(msg, 'content'):
                msg_dicts.append({
                    'content': msg.content,
                    'role': getattr(msg, 'role', 'user'),
                    'timestamp': getattr(msg, 'timestamp', ''),
                })
            else:
                msg_dicts.append(msg)
        
        # Extract all facts first
        extract_start = time.perf_counter()
        context = []
        all_facts = []
        fact_metadata = []
        
        for msg in msg_dicts:
            content = msg.get("content", "")
            if not content or not content.strip():
                continue
            # Role filter: skip messages whose role is excluded (e.g. assistant).
            if self.store_roles is not None and msg.get("role", "user") not in self.store_roles:
                continue

            facts = self.extractor.extract_facts(content, context)
            context.append(content)
            
            for fact in facts:
                all_facts.append(fact)
                fact_metadata.append({
                    'role': msg.get("role", "user"),
                    'timestamp': msg.get("timestamp", ""),
                    'content': content,
                })
        extract_time = (time.perf_counter() - extract_start) * 1000
        
        # Batch embed all facts
        embed_start = time.perf_counter()
        if all_facts:
            if batch and len(all_facts) > 1:
                embeddings = self.model.encode(all_facts)
            else:
                embeddings = np.vstack([self.model.encode(f) for f in all_facts])
        else:
            embeddings = np.array([])
        embed_time = (time.perf_counter() - embed_start) * 1000
        
        # Insert all facts into TSM
        insert_start = time.perf_counter()
        if len(embeddings) > 0:
            for i, (fact, meta) in enumerate(zip(all_facts, fact_metadata)):
                self._insert_counter += 1
                memory_id = f"{user_id}_{self._insert_counter}" if user_id else f"mem_{self._insert_counter}"
                
                self._id_to_text[memory_id] = fact
                
                # Extract simple concepts from the fact (nouns and key phrases)
                # For now, use simple word extraction - in production this would use NLP
                concepts = self._extract_concepts(fact)
                
                self.engine.insert(
                    id=memory_id,
                    text=fact,
                    embedding=embeddings[i].astype(np.float32),
                    importance_score=1.0,
                    concepts=concepts,
                    payload=json.dumps({
                        "timestamp": meta['timestamp'],
                        "role": meta['role'],
                        "user_id": user_id,
                        "original_message": meta['content'],
                    }),
                    scope=user_id,
                    source_role=meta['role'],
                )
        insert_time = (time.perf_counter() - insert_start) * 1000
        total_time = (time.perf_counter() - total_start) * 1000
        self._superseded_cache = None  # store changed → recompute on next recall

        # Log timing breakdown
        logger.info("  Add() timing: extract=%.1fms (%.1fms/fact), embed=%.1fms (%.1fms/fact), insert=%.1fms, total=%.1fms",
                    extract_time, extract_time / max(len(all_facts), 1),
                    embed_time, embed_time / max(len(all_facts), 1),
                    insert_time, total_time)
        
        return {
            "num_facts": len(all_facts),
            "extract_ms": extract_time,
            "embed_ms": embed_time,
            "insert_ms": insert_time,
            "total_ms": total_time,
        }
    
    def search(self, query: str, user_id: Optional[str] = None, top_k: int = 3, use_cognitive: bool = False) -> List[Dict]:
        """Search memories (Mem0-compatible API).
        
        Args:
            query: Search query text
            user_id: Optional user/conversation ID for scoping
            top_k: Number of results to return
            use_cognitive: Whether to use cognitive graph (slower but smarter) or direct ANN (faster)
                
                - False (default): Direct ANN search. Fast (~25ms), pure vector similarity.
                  Use this for benchmarking retrieval speed and comparing with other systems.
                
                - True: Full cognitive search. Slower (~350ms), but includes:
                  - Spreading activation through memory graph
                  - Temporal reasoning (prefers recent facts)
                  - Contradiction handling (weakens outdated facts)
                  - Concept linking (related facts boost each other)
                  - FOK gate (returns None if confidence too low)
                  
                  Use this for production AI agents that need cognitive reasoning.
        """
        query_embedding = self.model.encode(query)

        # B1: in exclude/tag mode, over-fetch so that after dropping superseded
        # facts we still return top_k CURRENT ones (a fair fixed-context compare).
        fetch_k = top_k
        if self.supersession_mode in ("exclude", "tag"):
            fetch_k = top_k * 2 + 5

        if use_cognitive:
            # Full cognitive search (spreading activation, FOK gate, etc.)
            results = self.engine.search(
                query_text=query,
                query_embedding=query_embedding.astype(np.float32),
                top_k=fetch_k,
                scope=user_id,
            )
        else:
            # Direct ANN search (fast, no cognitive overhead)
            results = self.engine.search_ann(
                query_embedding.astype(np.float32),
                fetch_k,
            )

        if not results:
            return []

        superseded = self._superseded_set() if self.supersession_mode in ("exclude", "tag") else set()
        out = []
        for r in results:
            mid = r[0]
            if self.supersession_mode == "exclude" and mid in superseded:
                continue  # drop stale facts from the answer context (A-TMA)
            text = self._id_to_text.get(mid, "")
            if self.supersession_mode == "tag" and mid in superseded:
                text = "[OUTDATED] " + text
            out.append({"id": mid, "score": float(r[1]), "text": text})
            if len(out) >= top_k:
                break
        return out

    def _superseded_set(self):
        """Cached set of superseded memory ids (B1). Recomputed after any
        add()/consolidation, which is when the supersession graph can change."""
        if self._superseded_cache is None:
            try:
                self._superseded_cache = set(self.engine.superseded_ids())
            except Exception:  # engine without the method (older .pyd)
                self._superseded_cache = set()
        return self._superseded_cache

    def recall_under_budget(self, query, user_id=None, token_budget=100,
                            method="mmr", pool_k=20, lam=0.7, pool=None):
        """B2: return the best SET of memory texts whose total tokens fit
        `token_budget`, chosen from a retrieved candidate pool.

        `method="truncate"`: greedy by relevance only (score order, skip items
        that don't fit) — the naive baseline. `method="mmr"`: greedy submodular
        Maximal-Marginal-Relevance — each step adds the candidate maximizing
        `lam*relevance - (1-lam)*max_redundancy` to the already-selected set, so
        near-duplicates are penalized and coverage improves (PACMS, 2026). Both
        pack the same budget, so a difference isolates the diversity term.
        Returns list[str] of the selected memory texts (in selection order)."""
        # Retrieve a pool (supersession_mode is honored via search()). A
        # precomputed pool lets a caller select multiple ways without re-querying.
        if pool is None:
            pool = self.search(query, user_id=user_id, top_k=pool_k, use_cognitive=True)
        if not pool:
            return []
        texts = [p["text"] or "" for p in pool]
        rel = np.array([float(p["score"]) for p in pool], dtype=np.float32)
        toks = np.array([max(1, len(t) // 4) for t in texts], dtype=np.int32)

        if method == "truncate":
            order = list(np.argsort(-rel))
            sel, used = [], 0
            for i in order:
                if used + int(toks[i]) <= token_budget:
                    sel.append(i)
                    used += int(toks[i])
            return [texts[i] for i in sel]

        # MMR: needs pairwise similarity → embed the pool once (unit vectors).
        embs = self.model.encode(texts)
        embs = np.asarray(embs, dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        embs = embs / norms
        sim = embs @ embs.T  # cosine, since unit-normed

        selected, used, remaining = [], 0, list(range(len(texts)))
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
        return [texts[i] for i in selected]
    
    def search_ann(self, query: str, user_id: Optional[str] = None, top_k: int = 3) -> List[Dict]:
        """Pure ANN search without cognitive layer (for comparison).
        
        Args:
            query: Search query text
            user_id: Optional user/conversation ID for scoping
            top_k: Number of results to return
            
        Returns:
            List of result dicts with keys: id, score
        """
        query_embedding = self.model.encode(query)
        
        results = self.engine.search_ann(
            query_embedding=query_embedding.astype(np.float32),
            top_k=top_k,
            scope=user_id,
        )
        
        return [
            {
                "id": r[0],
                "score": float(r[1]),
            }
            for r in results
        ]
    
    def trigger_consolidation(self) -> None:
        """Trigger manual consolidation (for deterministic benchmarks).

        With `verify_demotions`, consolidation defers supersession commitment;
        the adapter then detects candidate supersessions, vets each through the
        NLI verifier, and commits only the survivors — so a demotion never
        happens until it is semantically confirmed.
        """
        self.engine.trigger_consolidation()
        self._superseded_cache = None  # supersession edges may have changed
        if not self.verify_demotions:
            return
        proposed = self.engine.propose_supersessions()  # (old, new, kind, cosine)
        if not proposed:
            return
        accepted = self.verifier.verify(proposed, self._id_to_text)
        if accepted:
            self.engine.commit_supersessions(accepted)
            self._superseded_cache = None
        logger.info("verified demotion: proposed=%d accepted=%d",
                    len(proposed), len(accepted))
    
    def close(self) -> None:
        """Close the TSM engine and clean up."""
        self.engine.close()
    
    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            "gpu_accelerated": self.engine.gpu_accelerated,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Quick test
    print("Testing TSMAdapter...")
    
    adapter = TSMAdapter(
        db_path=tempfile.mkdtemp(prefix="tsm_test_"),
        embedding_model="all-MiniLM-L6-v2",  # Small model for testing
        extractor="mock",
    )
    
    messages = [
        {"role": "user", "content": "I just moved to San Francisco.", "timestamp": "2024-01-15T10:00:00Z"},
        {"role": "assistant", "content": "Great! How do you like it?", "timestamp": "2024-01-15T10:01:00Z"},
        {"role": "user", "content": "I love the weather here.", "timestamp": "2024-01-15T10:02:00Z"},
    ]
    
    print("\nAdding messages...")
    adapter.add(messages, user_id="user_123")
    adapter.trigger_consolidation()
    
    print("\nSearching...")
    results = adapter.search("Where does the user live?", user_id="user_123", top_k=3)
    
    print(f"\nResults ({len(results)}):")
    for r in results:
        print(f"  {r['id']}: score={r['score']:.4f}")
    
    adapter.close()
    shutil.rmtree(adapter.db_path, ignore_errors=True)
    print("\nTest complete!")
