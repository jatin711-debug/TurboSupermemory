"""
TurboSuperMemory (TSM) Client for BEAM Benchmark Harness
========================================================

Implements the unified async client interface for TurboSuperMemory:
  - add(messages, user_id)
  - search(query, user_id, top_k=200)
  - delete_user(user_id)
  - reset()
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
from typing import Any

# Ensure root repo and benchmarks dir are on path to import turbomemory / cognitive_eval modules
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BENCHMARKS_DIR = os.path.join(REPO_ROOT, "benchmarks")
for p in [REPO_ROOT, BENCHMARKS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

logger = logging.getLogger(__name__)


class TSMClient:
    """Async TurboSuperMemory client for BEAM and memory-benchmarks."""

    def __init__(
        self,
        db_dir: str | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedder_type: str = "local",  # "local" or "openai"
        openai_embed_model: str = "text-embedding-3-small",
        extractor: str = "mock",        # "mock" or "openai"
        token_budget: int = 300,
        pool_k: int = 20,
        cognitive_features: bool = True,
        belief_revision: bool = True,
    ):
        self.base_dir = db_dir or os.path.join(REPO_ROOT, "benchmarks", "memory_benchmarks", "results", "beam_tsm_db")
        os.makedirs(self.base_dir, exist_ok=True)
        self.embedding_model = embedding_model
        self.embedder_type = embedder_type
        self.openai_embed_model = openai_embed_model
        self.extractor_type = extractor
        self.token_budget = token_budget
        self.pool_k = pool_k
        self.cognitive_features = cognitive_features
        self.belief_revision = belief_revision

        self._adapters: dict[str, Any] = {}
        self._shared_model = None
        self._shared_verifier = None
        self._shared_extractor = None

        logger.info(
            "TSMClient initialized at %s (embedder=%s, extractor=%s, cognitive=%s)",
            self.base_dir,
            self.embedder_type,
            self.extractor_type,
            self.cognitive_features,
        )

    def _get_or_create_adapter(self, user_id: str):
        if user_id in self._adapters:
            return self._adapters[user_id]

        from cognitive_eval.adapters.tsm_adapter import TSMAdapter
        from cognitive_eval.extraction import create_extractor

        user_db = os.path.join(self.base_dir, f"tsm_user_{user_id}")
        os.makedirs(user_db, exist_ok=True)

        if self._shared_model is None:
            if self.embedder_type == "openai":
                from cognitive_eval.openai_embedder import OpenAIEmbedder
                self._shared_model = OpenAIEmbedder(model=self.openai_embed_model)

        if self._shared_extractor is None:
            self._shared_extractor = create_extractor(self.extractor_type)

        adapter = TSMAdapter(
            db_path=user_db,
            embedding_model=self.embedding_model,
            extractor=self.extractor_type,
            extractor_instance=self._shared_extractor,
            cognitive_features=self.cognitive_features,
            belief_revision=self.belief_revision,
            model=self._shared_model,
            belief_source_roles=["user"],
            verify_demotions=True,
            verifier=self._shared_verifier,
            supersession_mode="exclude",
        )
        if self._shared_model is None:
            self._shared_model = adapter.model
        if self._shared_verifier is None:
            self._shared_verifier = adapter.verifier

        self._adapters[user_id] = adapter
        return adapter

    async def add(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        metadata: dict | None = None,
        filters: dict | None = None,
        timestamp: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Ingest conversational turns into TSM."""
        adapter = self._get_or_create_adapter(user_id)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, adapter.add, messages, user_id)
        return {"results": [{"status": "success", "count": len(messages)}]}

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 200,
        rerank: bool = False,
        score_debug: bool = False,
        **kwargs: Any,
    ) -> list[dict]:
        """Retrieve memories from TSM with cognitive scoring & MMR diversity."""
        adapter = self._get_or_create_adapter(user_id)
        loop = asyncio.get_event_loop()

        def _do_search():
            adapter.trigger_consolidation()
            
            # 1. Detect Query Intent
            q_lower = query.lower()
            timeline_triggers = (
                "order", "sequence", "chronological", "walk me through", "evolution",
                "phases", "timeline", "steps", "progression", "first", "then", "progress",
                "history", "what came first", "lifecycle"
            )
            update_triggers = (
                "current", "latest", "updated", "average", "how many", "response time",
                "commits", "status", "changed to", "now", "newest", "recently"
            )
            
            is_timeline = any(trigger in q_lower for trigger in timeline_triggers)
            is_update = any(trigger in q_lower for trigger in update_triggers)
            
            # 2. Fetch Expanded Candidate Pool
            fetch_k = max(top_k * 3, 60)
            raw_results = adapter.search(query, user_id=user_id, top_k=fetch_k, use_cognitive=self.cognitive_features)
            
            if not raw_results:
                return []
            
            items = []
            for r in raw_results:
                if isinstance(r, tuple) and len(r) >= 2:
                    fact, score = r[0], float(r[1])
                    t_idx = 0
                elif isinstance(r, dict):
                    fact = r.get("text", r.get("memory", ""))
                    score = float(r.get("score", 0.0))
                    t_idx = r.get("turn_index", 0) or 0
                else:
                    fact, score, t_idx = str(r), 1.0, 0
                items.append({"memory": fact, "score": score, "turn_index": t_idx})
            
            # Max turn for normalization
            max_turn = max([x["turn_index"] for x in items] + [1])
            
            if is_timeline and len(items) > 5:
                # 3. Chronological Stratified Sampling (Timeline Bucketing)
                # Split timeline into 5 distinct epochs (0-20%, 20-40%, 40-60%, 60-80%, 80-100%)
                num_buckets = 5
                bucket_size = max_turn / num_buckets if max_turn > 0 else 1
                buckets = [[] for _ in range(num_buckets)]
                
                for item in items:
                    b_idx = min(int(item["turn_index"] / bucket_size), num_buckets - 1)
                    buckets[b_idx].append(item)
                
                # Pick top items from each bucket proportionally
                per_bucket = max(top_k // num_buckets, 2)
                selected = []
                for b in buckets:
                    b.sort(key=lambda x: x["score"], reverse=True)
                    selected.extend(b[:per_bucket])
                
                # If still under top_k, fill with remaining highest scoring items
                if len(selected) < top_k:
                    seen = {id(x) for x in selected}
                    rem = [x for x in items if id(x) not in seen]
                    rem.sort(key=lambda x: x["score"], reverse=True)
                    selected.extend(rem[: top_k - len(selected)])
                
                # Format strictly in chronological sequence (ascending turn order)
                selected.sort(key=lambda x: x["turn_index"])
                final_items = selected[:top_k]
            
            elif is_update:
                # 4. Temporal Recency Re-ranking for Dynamic Fact Updates
                # Boost newer turns: score' = score * (1 + 0.40 * (turn / max_turn))
                for item in items:
                    recency_ratio = item["turn_index"] / max(max_turn, 1)
                    item["score"] = item["score"] * (1.0 + 0.40 * recency_ratio)
                
                items.sort(key=lambda x: x["score"], reverse=True)
                final_items = items[:top_k]
            
            else:
                # General query: gentle recency preference + semantic ranking
                for item in items:
                    recency_ratio = item["turn_index"] / max(max_turn, 1)
                    item["score"] = item["score"] * (1.0 + 0.15 * recency_ratio)
                items.sort(key=lambda x: x["score"], reverse=True)
                final_items = items[:top_k]
            
            return [{"memory": x["memory"], "score": float(x["score"]), "id": ""} for x in final_items]

        return await loop.run_in_executor(None, _do_search)

    async def delete_user(self, user_id: str) -> None:
        """Delete user memory store."""
        if user_id in self._adapters:
            del self._adapters[user_id]

    async def reset(self) -> None:
        """Reset all adapters."""
        self._adapters.clear()

    async def close(self) -> None:
        """Close client."""
        await self.reset()

    async def __aenter__(self) -> TSMClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
