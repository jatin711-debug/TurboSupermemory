"""Unit tests for the tsm SDK.

Run from the repo root with plain Python (no pytest, no API keys, no model
downloads):

    python -m unittest tsm.tests.test_memory -v

Uses deterministic fake Embedder/Extractor/Verifier implementations backed by
hash-based vectors, but a REAL turbomemory engine (temp dir per test) — so the
supersession-exclusion path exercised here is the engine's, not a mock's.
"""

import hashlib
import os
import shutil
import sys
import tempfile
import unittest

# Repo root = two levels up from this file (tsm/tests/ -> tsm/ -> root). Makes
# `import tsm` and the repo-root `turbomemory.pyd` importable from anywhere.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from tsm import CONVERSATIONAL_PROFILE, Memory


class FakeEmbedder:
    """Deterministic hash-based count-vector embedder (unit-normalized).

    Cosine similarity between two texts approximates their token overlap, so
    tests control similarity purely by wording — e.g. repeating shared tokens
    drives the cosine above the profile's 0.85 refinement threshold.
    """

    def __init__(self, dim=1024):
        self.dim = dim

    @property
    def dimension(self):
        return self.dim

    def encode(self, texts):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        out = np.stack([self._vec(t) for t in items]).astype(np.float32)
        return out[0] if single else out

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v


class FakeExtractor:
    """Sentence splitter: each non-empty sentence is one 'fact'."""

    def extract_facts(self, message, context=None):
        return [s.strip() for s in message.split(".") if s.strip()]


class AcceptAllVerifier:
    """Fake verifier: accepts every proposed supersession."""

    def __init__(self):
        self.calls = 0

    def verify(self, proposals, id_to_text):
        self.calls += 1
        return [(old, new, kind) for (old, new, kind, _c) in proposals]


def _cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


class MemoryTestBase(unittest.TestCase):
    def setUp(self):
        self.db_path = tempfile.mkdtemp(prefix="tsm_test_")
        self.addCleanup(self._cleanup)
        self.mem = None

    def _cleanup(self):
        if self.mem is not None:
            self.mem.close()
        shutil.rmtree(self.db_path, ignore_errors=True)

    def make_memory(self, verifier=None, **kwargs):
        self.mem = Memory(
            db_path=self.db_path,
            embedder=FakeEmbedder(),
            extractor=FakeExtractor(),
            verifier=verifier,
            **kwargs,
        )
        return self.mem


class TestAddRecallRoundTrip(MemoryTestBase):
    def test_add_then_recall_returns_stored_facts(self):
        mem = self.make_memory(verifier=AcceptAllVerifier())
        n = mem.add(
            [
                {"role": "user", "content": "I adopted a dog named Rex."},
                {"role": "assistant", "content": "Nice, how old is Rex?"},
                {"role": "user", "content": "He is three years old."},
            ],
            user_id="alice",
        )
        self.assertEqual(n, 3)  # one sentence-fact per message

        results = mem.recall("dog Rex", user_id="alice")
        self.assertTrue(results, "recall returned no results")
        texts = [r["text"] for r in results]
        self.assertIn("I adopted a dog named Rex", texts)
        for r in results:
            self.assertEqual(set(r), {"id", "text", "score"})
            self.assertIsInstance(r["score"], float)

    def test_recall_is_scoped_per_user(self):
        mem = self.make_memory()
        mem.add([{"role": "user", "content": "I adopted a dog named Rex."}],
                user_id="alice")
        other = mem.recall("dog Rex", user_id="bob")
        self.assertEqual(other, [], "scope leak: bob saw alice's memory")

    def test_exact_duplicates_within_batch_skipped(self):
        mem = self.make_memory()
        n = mem.add(
            [{"role": "user", "content": "I like tea. I like tea. I like tea."}],
            user_id="alice",
        )
        self.assertEqual(n, 1)

    def test_dimension_inferred_from_embedder(self):
        mem = self.make_memory()
        self.assertEqual(mem.dim, 1024)


class TestProfileConfig(MemoryTestBase):
    def test_conversational_profile_kwargs_accepted_by_engine(self):
        # Constructing the engine must not raise: every profile key is a real
        # MemoryEngine kwarg.
        mem = self.make_memory(verifier=AcceptAllVerifier())
        self.assertEqual(mem.engine.record_count(), 0)

    def test_profile_contents_match_proven_preset(self):
        self.assertEqual(CONVERSATIONAL_PROFILE["refinement_cosine_threshold"], 0.85)
        self.assertEqual(CONVERSATIONAL_PROFILE["contradiction_cosine_threshold"], 0.75)
        self.assertEqual(CONVERSATIONAL_PROFILE["cognitive_alpha"], 0.5)
        self.assertIs(CONVERSATIONAL_PROFILE["exclude_superseded"], True)
        self.assertIs(CONVERSATIONAL_PROFILE["access_aware_eviction"], True)
        self.assertEqual(CONVERSATIONAL_PROFILE["belief_source_roles"], ["user"])
        self.assertEqual(CONVERSATIONAL_PROFILE["concept_max_ngram_len"], 2)
        self.assertEqual(CONVERSATIONAL_PROFILE["max_concepts"], 10)

    def test_profile_none_is_plain_vector_store(self):
        mem = self.make_memory(profile=None)
        mem.add([{"role": "user", "content": "Plain fact one."}], user_id="alice")
        self.assertEqual(mem.engine.record_count(), 1)

    def test_explicit_engine_kwarg_overrides_profile(self):
        mem = self.make_memory(cognitive_alpha=0.9)
        self.assertEqual(mem.profile, "conversational")
        # Engine accepted the override (construction would raise otherwise).
        self.assertEqual(mem.engine.record_count(), 0)


class TestSupersessionFlow(MemoryTestBase):
    def test_correction_supersedes_stale_fact(self):
        verifier = AcceptAllVerifier()
        mem = self.make_memory(verifier=verifier)
        # Repeated shared tokens push the pair's cosine above the 0.85
        # refinement threshold so the engine proposes a supersession.
        old_fact = "user user user user lives in paris"
        new_fact = "user user user user lives in london"
        emb = FakeEmbedder()
        self.assertGreaterEqual(_cosine(emb.encode(old_fact), emb.encode(new_fact)),
                                0.85, "test facts not similar enough to be proposed")

        mem.add([{"role": "user", "content": old_fact + "."}], user_id="alice")
        mem.add([{"role": "user", "content": new_fact + "."}], user_id="alice")

        before = [r["text"] for r in mem.recall("user lives", user_id="alice")]
        self.assertIn(old_fact, before)

        committed = mem.consolidate()
        self.assertGreaterEqual(verifier.calls, 1)
        self.assertGreaterEqual(committed, 1, "no supersession was committed")

        after = [r["text"] for r in mem.recall("user lives", user_id="alice", top_k=10)]
        self.assertIn(new_fact, after)
        self.assertNotIn(old_fact, after, "stale fact not excluded after supersession")


class _EngineWithoutResolve:
    """Proxy that hides ``resolve_beliefs``, simulating an older pyd."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        if name == "resolve_beliefs":
            raise AttributeError(name)
        return getattr(self._inner, name)


class TestBeliefResolution(MemoryTestBase):
    """The annotation contract: recall tags stale results whose CURRENT
    belief is not itself in the result set with superseded_by + chain."""

    OLD_FACT = "user user user user lives in paris"
    NEW_FACT = "user user user user lives in london"

    def _memory_with_correction(self):
        # exclude_superseded=False keeps the stale fact recallable (annotation
        # instead of exclusion); demotion factor 1.0 + alpha 1.0 make ranking
        # pure cosine so the exact-query stale fact deterministically tops the
        # result while the chain head falls outside top_k.
        mem = self.make_memory(
            verifier=AcceptAllVerifier(),
            exclude_superseded=False,
            supersession_demotion_factor=1.0,
            cognitive_alpha=1.0,
        )
        mem.add([{"role": "user", "content": self.OLD_FACT + "."}], user_id="alice")
        mem.add([{"role": "user", "content": self.NEW_FACT + "."}], user_id="alice")
        committed = mem.consolidate()
        self.assertGreaterEqual(committed, 1, "no supersession was committed")
        return mem

    def test_recall_annotates_stale_result_with_lineage(self):
        mem = self._memory_with_correction()
        old_id, new_id = "alice_1", "alice_2"

        results = mem.recall(self.OLD_FACT, user_id="alice", top_k=1)
        self.assertEqual(len(results), 1)
        stale = results[0]
        self.assertEqual(stale["id"], old_id)
        self.assertEqual(stale["text"], self.OLD_FACT)
        self.assertEqual(stale["superseded_by"], new_id,
                         "stale result must point at the current belief")
        self.assertEqual(stale["chain"], [old_id, new_id],
                         "chain is the full lineage, oldest first, head last")

    def test_head_in_result_set_means_no_annotation(self):
        mem = self._memory_with_correction()
        # top_k=2: both the stale fact and its current belief are returned,
        # so per the contract nothing is annotated.
        results = mem.recall(self.OLD_FACT, user_id="alice", top_k=2)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertNotIn("superseded_by", r)
            self.assertNotIn("chain", r)

    def test_mmr_budget_recall_annotates_stale_result(self):
        mem = self._memory_with_correction()
        # Budget fits only one of the two facts; the exact-query stale fact
        # has the highest relevance, so it is selected and annotated.
        results = mem.recall(self.OLD_FACT, user_id="alice", token_budget=9)
        self.assertEqual(len(results), 1)
        stale = results[0]
        self.assertEqual(stale["id"], "alice_1")
        self.assertEqual(stale["superseded_by"], "alice_2")
        self.assertEqual(stale["chain"], ["alice_1", "alice_2"])

    def test_resolve_beliefs_false_skips_annotation(self):
        mem = self._memory_with_correction()
        results = mem.recall(self.OLD_FACT, user_id="alice", top_k=1,
                             resolve_beliefs=False)
        self.assertEqual(len(results), 1)
        self.assertNotIn("superseded_by", results[0])

    def test_recall_degrades_gracefully_without_engine_support(self):
        mem = self.make_memory()
        mem.add([{"role": "user", "content": "I adopted a dog named Rex."}],
                user_id="alice")
        mem.engine = _EngineWithoutResolve(mem.engine)
        results = mem.recall("dog Rex", user_id="alice")
        self.assertTrue(results, "recall should still return results")
        for r in results:
            self.assertEqual(set(r), {"id", "text", "score"})


class TestMmrBudgetRecall(MemoryTestBase):
    def test_best_set_fits_token_budget(self):
        mem = self.make_memory()
        facts = [
            "alice enjoys hiking in the alps during summer",
            "bob plays chess every weekend with his club",
            "carol bakes sourdough bread every morning",
            "dave runs marathons twice a year abroad",
            "erin paints watercolor landscapes on sundays",
            "frank collects vintage vinyl jazz records",
        ]
        mem.add([{"role": "user", "content": f + "."} for f in facts], user_id="alice")

        budget = 12  # ~48 chars -> at most one of these facts fits
        results = mem.recall("hobby", user_id="alice", token_budget=budget)
        self.assertTrue(results, "MMR recall returned no results")
        used = sum(max(1, len(r["text"]) // 4) for r in results)
        self.assertLessEqual(used, budget, f"token budget exceeded: {used} > {budget}")

    def test_no_budget_returns_top_k_dicts(self):
        mem = self.make_memory()
        facts = [f"fact number {i} about topic {i}" for i in range(5)]
        mem.add([{"role": "user", "content": f + "."} for f in facts], user_id="alice")
        results = mem.recall("fact topic", user_id="alice", top_k=3)
        self.assertLessEqual(len(results), 3)
        self.assertTrue(all({"id", "text", "score"} == set(r) for r in results))


if __name__ == "__main__":
    unittest.main(verbosity=2)
