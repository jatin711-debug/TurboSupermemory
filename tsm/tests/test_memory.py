"""Unit tests for the tsm SDK.

Run from the repo root with plain Python (no pytest, no API keys, no model
downloads):

    python -m unittest tsm.tests.test_memory -v

Uses deterministic fake Embedder/Extractor/Verifier implementations backed by
hash-based vectors, but a REAL turbomemory engine (temp dir per test) — so the
supersession-exclusion path exercised here is the engine's, not a mock's.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
from types import SimpleNamespace

# Repo root = two levels up from this file (tsm/tests/ -> tsm/ -> root). Makes
# `import tsm` and the repo-root `turbomemory.pyd` importable from anywhere.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

import tsm
from tsm import CONVERSATIONAL_PROFILE, Memory
from tsm.gist import ExtractiveGistSummarizer
from tsm.memory import STALE_MARK
from tsm.verification import LLMVerifier


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
            self.assertTrue({"id", "text", "score"}.issubset(set(r)))
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
        # Superseded facts stay in recall unless a caller asks otherwise.
        self.assertNotIn("exclude_superseded", CONVERSATIONAL_PROFILE)
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

        after = mem.recall("user lives", user_id="alice", top_k=10)
        texts = [r["text"] for r in after]
        # The correction is served first; the fact it replaced is still
        # there (a question about the past needs it), flagged and marked.
        self.assertLess(texts.index(new_fact), texts.index(old_fact))
        by_text = {r["text"]: r for r in after}
        self.assertEqual(by_text[old_fact]["superseded_by"], "alice_2")
        self.assertEqual(by_text[old_fact]["context"], STALE_MARK + old_fact)
        self.assertEqual(by_text[new_fact]["context"], new_fact)
        self.assertNotIn("superseded_by", by_text[new_fact])

    def test_superseded_facts_can_be_excluded_instead(self):
        mem = self.make_memory(verifier=AcceptAllVerifier(), exclude_superseded=True)
        old_fact = "user user user user lives in paris"
        new_fact = "user user user user lives in london"
        mem.add([{"role": "user", "content": old_fact + "."}], user_id="alice")
        mem.add([{"role": "user", "content": new_fact + "."}], user_id="alice")
        self.assertGreaterEqual(mem.consolidate(), 1)
        after = [r["text"] for r in mem.recall("user lives", user_id="alice", top_k=10)]
        self.assertIn(new_fact, after)
        self.assertNotIn(old_fact, after)


class _EngineWithoutResolve:
    """Proxy that hides ``resolve_beliefs``, simulating an older pyd."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        if name == "resolve_beliefs":
            raise AttributeError(name)
        return getattr(self._inner, name)


class TestBeliefResolution(MemoryTestBase):
    """The annotation contract: recall tags every stale result with
    superseded_by + chain, so a result without them is a current belief."""

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

        results = mem.recall(self.OLD_FACT, user_id="alice", top_k=2)
        self.assertEqual([r["id"] for r in results], [new_id, old_id])
        stale = results[1]
        self.assertEqual(stale["text"], self.OLD_FACT)
        self.assertEqual(stale["context"], STALE_MARK + self.OLD_FACT)
        self.assertEqual(stale["superseded_by"], new_id,
                         "stale result must point at the current belief")
        self.assertEqual(stale["chain"], [old_id, new_id],
                         "chain is the full lineage, oldest first, head last")

    def test_current_belief_is_brought_in_when_the_search_missed_it(self):
        mem = self._memory_with_correction()
        # top_k=1 and a query in the old wording: the search returns only the
        # stale fact. What is believed now is what comes back.
        hits = mem.engine.search(query_text=self.OLD_FACT,
                                 query_embedding=FakeEmbedder().encode(self.OLD_FACT),
                                 top_k=1, scope="alice")
        self.assertEqual([mid for mid, _ in hits], ["alice_1"])
        results = mem.recall(self.OLD_FACT, user_id="alice", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "alice_2")
        self.assertEqual(results[0]["text"], self.NEW_FACT)
        self.assertEqual(results[0]["replaces"], ["alice_1"])
        self.assertNotIn("superseded_by", results[0])

    def test_stale_result_is_flagged_next_to_its_current_belief_too(self):
        mem = self._memory_with_correction()
        # top_k=2: both the stale fact and its current belief are returned.
        # The reader has to be told which is which.
        results = {r["id"]: r for r in mem.recall(self.OLD_FACT, user_id="alice", top_k=2)}
        self.assertEqual(set(results), {"alice_1", "alice_2"})
        self.assertEqual(results["alice_1"]["superseded_by"], "alice_2")
        self.assertEqual(results["alice_1"]["chain"], ["alice_1", "alice_2"])
        self.assertNotIn("superseded_by", results["alice_2"])
        self.assertNotIn("chain", results["alice_2"])

    def test_mmr_budget_recall_serves_the_current_belief_first(self):
        mem = self._memory_with_correction()
        # A budget with room for one fact: the current belief takes it, even
        # though the query is worded like the stale one.
        results = mem.recall(self.OLD_FACT, user_id="alice", token_budget=9)
        self.assertEqual([r["id"] for r in results], ["alice_2"])
        # With room for more, the stale fact follows, marked. Its marker
        # counts against the budget.
        results = mem.recall(self.OLD_FACT, user_id="alice", token_budget=40)
        self.assertEqual([r["id"] for r in results], ["alice_2", "alice_1"])
        stale = results[1]
        self.assertEqual(stale["superseded_by"], "alice_2")
        self.assertEqual(stale["chain"], ["alice_1", "alice_2"])
        self.assertTrue(stale["context"].startswith(STALE_MARK))
        self.assertLessEqual(sum(len(r["context"]) // 4 for r in results), 40)

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
            self.assertTrue({"id", "text", "score"}.issubset(set(r)))
            self.assertIsInstance(r["score"], float)


class TestMmrBudgetRecall(MemoryTestBase):
    def test_best_set_fits_token_budget(self):
        mem = self.make_memory()
        facts = [
            "alice adopted a golden retriever puppy named charlie in june",
            "bob bought a red bicycle with ten speeds for commuting",
            "carol visited the botanical garden and saw the orchids",
            "dave learned to bake sourdough with a three-year-old starter",
        ]
        mem.add([{"role": "user", "content": f + "."} for f in facts], user_id="alice")

        # 4 facts ~ 40 words ~ 50 tokens. Budget 25 tokens should return at most 2.
        results = mem.recall("alice bob carol dave", user_id="alice", token_budget=25)
        self.assertTrue(results, "recall returned empty result set")
        self.assertLessEqual(len(results), 2, "budget constraint violated")
        total_toks = sum(max(1, len(r["text"]) // 4) for r in results)
        self.assertLessEqual(total_toks, 25)

    def test_no_budget_returns_top_k_dicts(self):
        mem = self.make_memory()
        mem.add([{"role": "user", "content": "the sky is blue."}], user_id="alice")
        results = mem.recall("sky", user_id="alice", top_k=5)
        self.assertEqual(len(results), 1)
        self.assertTrue(all({"id", "text", "score"}.issubset(set(r)) for r in results))


class TestGistBeforeEvict(MemoryTestBase):
    def test_eviction_victims_become_searchable_gists(self):
        captured = []

        def summarizer(texts):
            captured.append(list(texts))
            return " ; ".join(texts)

        mem = self.make_memory(gist_summarizer=summarizer, max_records=2)
        facts = [
            "alice adopted a beagle named rex",
            "bob bakes sourdough every morning",
            "carol moved to lisbon in spring",
            "dave runs marathons twice yearly",
        ]
        mem.add([{"role": "user", "content": f + "."} for f in facts], user_id="alice")

        evicted = mem.engine.evict()
        self.assertEqual(evicted, 2, "max_records=2 over 4 facts evicts exactly 2")
        # 2 survivors + 1 gist record.
        self.assertEqual(mem.engine.record_count(), 3)
        # The summarizer saw exactly the two victim texts, in one chunk.
        self.assertEqual(len(captured), 1)
        self.assertEqual(len(captured[0]), 2)
        self.assertTrue(all(any(t in f for f in facts) for t in captured[0]))

        # The gist is retrievable through the scoped recall path, and its text
        # is read back from the engine (this process never minted its id).
        results = mem.recall("beagle rex", user_id="alice", top_k=5)
        gists = [r for r in results if r["id"].startswith("gist:")]
        self.assertTrue(gists, f"no gist record in recall results: {results}")
        self.assertTrue(gists[0]["text"], "gist text did not render")
        self.assertIn(" ; ", gists[0]["text"])

    def test_gist_disabled_without_summarizer(self):
        mem = self.make_memory(max_records=2)
        mem.add(
            [{"role": "user", "content": f"fact number {i} about topic {i}."} for i in range(4)],
            user_id="alice",
        )
        evicted = mem.engine.evict()
        self.assertEqual(evicted, 2)
        self.assertEqual(mem.engine.record_count(), 2)
        results = mem.recall("fact topic", user_id="alice", top_k=10)
        self.assertFalse(any(r["id"].startswith("gist:") for r in results))

    def test_context_manager_auto_close_and_flush(self):
        embedder = FakeEmbedder()
        extractor = FakeExtractor()

        with Memory(db_path=self.db_path, embedder=embedder, extractor=extractor) as mem:
            mem.add([{"role": "user", "content": "alice prefers python programming."}], user_id="alice")
            results = mem.recall("python programming", user_id="alice")
            self.assertTrue(results)
            self.assertFalse(mem._closed)

        self.assertTrue(mem._closed)


class TestPackaging(unittest.TestCase):
    def test_version_matches_pyproject(self):
        pyproject = os.path.join(_ROOT, "pyproject.toml")
        if not os.path.exists(pyproject):
            self.skipTest("not running from a source checkout")
        import tomllib

        with open(pyproject, "rb") as f:
            declared = tomllib.load(f)["project"]["version"]
        self.assertEqual(tsm.__version__, declared)


class RecordingVerifier(AcceptAllVerifier):
    """Accepts everything and remembers the id -> text map it was handed."""

    def __init__(self):
        super().__init__()
        self.seen_texts = {}

    def verify(self, proposals, id_to_text):
        self.seen_texts.update(id_to_text)
        return super().verify(proposals, id_to_text)


class TestDurability(MemoryTestBase):
    """A database must behave the same after it is closed and reopened."""

    OLD_FACT = "user user user user lives in paris"
    NEW_FACT = "user user user user lives in london"

    def reopen(self, **kwargs):
        self.mem.close()
        return self.make_memory(**kwargs)

    def test_add_after_reopen_appends_with_fresh_ids(self):
        mem = self.make_memory()
        mem.add([{"role": "user", "content": "I adopted a dog named Rex."}], user_id="alice")

        mem = self.reopen()
        self.assertEqual(mem.engine.record_count(), 1)
        n = mem.add([{"role": "user", "content": "I moved to Lisbon last year."}],
                    user_id="alice")
        self.assertEqual(n, 1)
        self.assertEqual(mem.engine.record_count(), 2)

        results = mem.recall("dog Rex Lisbon", user_id="alice", top_k=5)
        self.assertEqual({r["id"] for r in results}, {"alice_1", "alice_2"})
        self.assertEqual({r["text"] for r in results},
                         {"I adopted a dog named Rex", "I moved to Lisbon last year"})

    def test_ids_are_not_reused_after_eviction_and_reopen(self):
        mem = self.make_memory(max_records=2)
        mem.add(
            [{"role": "user", "content": f"fact number {i} about topic {i}."} for i in range(4)],
            user_id="alice",
        )
        self.assertEqual(mem.engine.evict(), 2)

        mem = self.reopen(max_records=2)
        self.assertEqual(mem.engine.record_count(), 2)
        mem.add([{"role": "user", "content": "a brand new fact."}], user_id="alice")
        # Four ids were handed out before the restart, two of them evicted
        # since; the next one must still be the fifth.
        self.assertTrue(mem.engine.contains_id("alice_5"))

    def test_close_releases_database_for_reopen(self):
        first = self.make_memory()
        first.add([{"role": "user", "content": "I adopted a dog named Rex."}], user_id="alice")
        first.close()
        first.close()  # idempotent

        # `first` is still referenced, so only a real release (not garbage
        # collection) can let the same path be opened again.
        second = self.make_memory()
        self.assertEqual(second.engine.record_count(), 1)

        for call in (
            lambda: first.recall("dog Rex", user_id="alice"),
            lambda: first.add([{"role": "user", "content": "More."}], user_id="alice"),
            first.consolidate,
            first.flush,
        ):
            with self.assertRaises(RuntimeError):
                call()

    def test_role_and_scope_survive_reopen(self):
        mem = self.make_memory()
        mem.add(
            [
                {"role": "user", "content": "I adopted a dog named Rex."},
                {"role": "assistant", "content": "Rex is a lovely dog name."},
            ],
            user_id="alice",
        )
        before = mem.recall("what is my dog named", user_id="alice", top_k=5)

        mem = self.reopen()
        after = mem.recall("what is my dog named", user_id="alice", top_k=5)
        self.assertEqual({r["id"]: r["role"] for r in after},
                         {"alice_1": "user", "alice_2": "assistant"})
        # Same role prior on both sides of the restart, so the same ranking.
        self.assertEqual([r["id"] for r in after], [r["id"] for r in before])
        self.assertEqual(mem.recall("dog Rex", user_id="bob"), [],
                         "scope leak after reopen: bob saw alice's memory")

    def test_turn_index_groups_facts_by_message_across_reopen(self):
        mem = self.make_memory()
        mem.add(
            [
                {"role": "user", "content": "I adopted a dog. His name is Rex."},
                {"role": "user", "content": "I moved to Lisbon."},
            ],
            user_id="alice",
        )
        mem = self.reopen()
        mem.add([{"role": "user", "content": "I started a pottery class."}], user_id="alice")

        results = mem.recall("dog Rex Lisbon pottery", user_id="alice", top_k=10)
        turn = {r["text"]: r["turn_index"] for r in results}
        self.assertEqual(len(turn), 4)
        # Two facts from one message share a turn; every message is its own turn,
        # including the one added after the restart.
        self.assertEqual(turn["I adopted a dog"], turn["His name is Rex"])
        self.assertEqual(len(set(turn.values())), 3)

    def test_caller_supplied_turn_index_is_kept(self):
        mem = self.make_memory()
        mem.add([{"role": "user", "content": "I adopted a dog named Rex.", "turn_index": 41}],
                user_id="alice")
        results = mem.recall("dog Rex", user_id="alice")
        self.assertEqual(results[0]["turn_index"], 41)

    def test_verifier_vets_facts_from_an_earlier_session(self):
        mem = self.make_memory()  # no verifier: nothing is committed yet
        mem.add([{"role": "user", "content": self.OLD_FACT + "."}], user_id="alice")
        mem.add([{"role": "user", "content": self.NEW_FACT + "."}], user_id="alice")

        verifier = RecordingVerifier()
        mem = self.reopen(verifier=verifier)
        committed = mem.consolidate()
        self.assertGreaterEqual(committed, 1, "no supersession was committed")
        self.assertEqual(verifier.seen_texts,
                         {"alice_1": self.OLD_FACT, "alice_2": self.NEW_FACT})

        after = {r["text"]: r for r in mem.recall("user lives", user_id="alice", top_k=10)}
        self.assertNotIn("superseded_by", after[self.NEW_FACT])
        self.assertEqual(after[self.OLD_FACT]["superseded_by"], "alice_2")


class PoisonableEmbedder(FakeEmbedder):
    """Returns a NaN vector for any text containing "poison" while armed."""

    armed = True

    def _vec(self, text):
        v = super()._vec(text)
        if self.armed and "poison" in text:
            v[0] = np.nan
        return v


class TestRobustness(MemoryTestBase):
    """Failures that were reproduced against the SDK: writes lost when the
    process died, one user's facts crowding out another's, an add() that
    stored half its facts, and a summarizer outage that erased memories."""

    def test_adds_survive_a_process_that_dies_without_close(self):
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        child = textwrap.dedent(f"""
            import os, sys
            sys.path.insert(0, {_ROOT!r})
            sys.path.insert(0, {tests_dir!r})
            from test_memory import FakeEmbedder, FakeExtractor
            from tsm import Memory
            mem = Memory(db_path={self.db_path!r}, embedder=FakeEmbedder(),
                         extractor=FakeExtractor())
            for i in range(25):
                mem.add([{{"role": "user", "content": f"zebra fact number {{i}}."}}],
                        user_id="alice")
            os._exit(0)  # no close(), no flush(): what a crash leaves behind
        """)
        subprocess.run([sys.executable, "-c", child], check=True)

        mem = self.make_memory()
        self.assertEqual(mem.engine.record_count(), 25)
        self.assertEqual(mem.engine.recovery_report()["wal_ops_replayed"], 25)
        results = mem.recall("zebra fact", user_id="alice", top_k=25)
        self.assertEqual(len(results), 25)
        # New ids continue after the recovered ones.
        mem.add([{"role": "user", "content": "one more zebra fact."}], user_id="alice")
        self.assertEqual(mem.engine.record_count(), 26)
        self.assertTrue(mem.engine.contains_id("alice_26"))

    def test_clean_close_leaves_nothing_to_recover(self):
        mem = self.make_memory()
        mem.add([{"role": "user", "content": "I adopted a dog named Rex."}], user_id="alice")
        mem.close()
        mem = self.make_memory()
        self.assertFalse(any(mem.engine.recovery_report().values()))
        self.assertEqual(mem.engine.record_count(), 1)

    def test_other_users_matching_facts_do_not_starve_recall(self):
        mem = self.make_memory()
        for i in range(40):
            mem.add([{"role": "user",
                      "content": f"salary payroll bonus detail number {i}."}], user_id="bob")
        mem.add([{"role": "user", "content": "my salary is 185000."}], user_id="alice")
        mem.add([{"role": "user", "content": "I live in Lisbon."}], user_id="alice")
        mem.add([{"role": "user", "content": "I play guitar."}], user_id="alice")

        mine = mem.recall("salary payroll", user_id="alice", top_k=3)
        self.assertEqual(len(mine), 3, f"alice has three memories, got {mine}")
        self.assertTrue(all(r["id"].startswith("alice_") for r in mine))
        theirs = mem.recall("salary lisbon guitar", user_id="bob", top_k=40)
        self.assertEqual(len(theirs), 40)
        self.assertTrue(all(r["id"].startswith("bob_") for r in theirs))
        # The same holds under a token budget (a different engine pool size).
        budgeted = mem.recall("salary payroll", user_id="alice", token_budget=200)
        self.assertTrue(budgeted)
        self.assertTrue(all(r["id"].startswith("alice_") for r in budgeted))

    def test_failed_add_stores_nothing_and_can_be_retried(self):
        embedder = PoisonableEmbedder()
        self.mem = mem = Memory(db_path=self.db_path, embedder=embedder,
                                extractor=FakeExtractor())
        message = [{"role": "user",
                    "content": "good fact one. poison fact two. good fact three."}]
        with self.assertRaises(ValueError):
            mem.add(message, user_id="alice")
        self.assertEqual(mem.engine.record_count(), 0,
                         "a rejected add must not leave some of its facts behind")

        embedder.armed = False
        self.assertEqual(mem.add(message, user_id="alice"), 3)
        self.assertEqual(mem.engine.record_count(), 3)

    def test_summarizer_outage_keeps_the_memories(self):
        outage = {"down": True}

        def summarizer(texts):
            if outage["down"]:
                raise RuntimeError("summarizer unreachable")
            return " ; ".join(texts)

        mem = self.make_memory(gist_summarizer=summarizer, max_records=2)
        mem.add([{"role": "user", "content": f"fact number {i} about topic {i}."}
                 for i in range(4)], user_id="alice")

        with self.assertLogs("tsm.memory", level="WARNING"):
            evicted = mem.engine.evict()
        self.assertEqual(evicted, 0, "nothing may be deleted without its gist")
        self.assertEqual(mem.engine.record_count(), 4)

        outage["down"] = False
        self.assertEqual(mem.engine.evict(), 2)
        self.assertEqual(mem.engine.record_count(), 3)  # 2 survivors + 1 gist

    def test_concurrent_adds_all_land_with_distinct_ids(self):
        mem = self.make_memory()
        errors = []

        def worker(t):
            try:
                for i in range(10):
                    mem.add([{"role": "user", "content": f"thread {t} fact {i}."}],
                            user_id="alice")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(mem.engine.record_count(), 80)

    def test_verifier_decides_even_without_the_conversational_profile(self):
        class RejectAll(AcceptAllVerifier):
            def verify(self, proposals, id_to_text):
                self.calls += 1
                self.proposed = list(proposals)
                return []

        verifier = RejectAll()
        mem = self.make_memory(verifier=verifier, profile=None,
                               refinement_cosine_threshold=0.85, exclude_superseded=True,
                               concept_max_ngram_len=2, max_concepts=10)
        mem.add([{"role": "user", "content": TestDurability.OLD_FACT + "."}], user_id="alice")
        mem.add([{"role": "user", "content": TestDurability.NEW_FACT + "."}], user_id="alice")

        self.assertEqual(mem.consolidate(), 0)
        self.assertEqual(verifier.calls, 1)
        self.assertEqual(len(verifier.proposed), 1, "the pair reached the verifier")
        # The verifier said no, so nothing was retired behind its back.
        self.assertEqual(mem.engine.superseded_ids(), [])
        texts = [r["text"] for r in mem.recall("user lives", user_id="alice", top_k=10)]
        self.assertIn(TestDurability.OLD_FACT, texts)
        self.assertIn(TestDurability.NEW_FACT, texts)

    def test_backend_names_are_checked_at_construction(self):
        for kwargs in ({"embedder": "opnai"}, {"extractor": "gpt"}, {"reranker": "bert"}):
            args = {"embedder": FakeEmbedder(), "extractor": FakeExtractor(), **kwargs}
            with self.assertRaises(ValueError) as ctx:
                Memory(db_path=self.db_path, **args)
            self.assertIn("unknown", str(ctx.exception))
        # Nothing was opened by the failed attempts: the path is free.
        saved = os.environ.pop("OPENAI_API_KEY", None)
        try:
            # "openai" is a known name: it gets as far as needing the key.
            with self.assertRaises(RuntimeError) as ctx:
                Memory(db_path=self.db_path, embedder="openai", extractor=FakeExtractor())
            self.assertIn("OPENAI_API_KEY", str(ctx.exception))
        finally:
            if saved is not None:
                os.environ["OPENAI_API_KEY"] = saved
        self.make_memory().add([{"role": "user", "content": "still usable."}], user_id="alice")

    def test_bad_embeddings_and_queries_are_rejected(self):
        mem = self.make_memory()
        mem.add([{"role": "user", "content": "I adopted a dog named Rex."}], user_id="alice")
        bad = np.full(mem.dim, np.nan, dtype=np.float32)
        with self.assertRaises(ValueError):
            mem.engine.insert("x", "t", bad, 1.0, [])
        with self.assertRaises(ValueError):
            mem.engine.search_ann(bad, 5)
        # An absurd top_k is clamped, not allocated.
        self.assertEqual(len(mem.engine.search_ann(mem.embedder.encode("dog Rex"), 2 ** 40)), 1)


class FakeJudge:
    """An OpenAI-compatible chat client that gives every pair one verdict."""

    def __init__(self, verdict):
        self.verdict = verdict
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        pairs = kwargs["messages"][-1]["content"].count("OLDER:")
        text = "\n".join(f"{i + 1}: {self.verdict}" for i in range(pairs))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class TestUnverifiedRevision(MemoryTestBase):
    """Without a verifier the engine's own detection still runs, but it is
    not allowed to remove a fact from recall: on real text it also fires on
    facts that are both still true."""

    OLD_FACT = "user user user user lives in paris"
    NEW_FACT = "user user user user lives in london"

    def _add_both(self, mem):
        mem.add([{"role": "user", "content": self.OLD_FACT + "."}], user_id="alice")
        mem.add([{"role": "user", "content": self.NEW_FACT + "."}], user_id="alice")

    def test_superseded_fact_is_flagged_and_ranked_lower_not_hidden(self):
        mem = self.make_memory()
        self._add_both(mem)
        self.assertEqual(mem.consolidate(), 0)
        self.assertEqual(mem.engine.superseded_ids(), ["alice_1"], "detection did not run")

        results = mem.recall("user lives", user_id="alice", top_k=10)
        order = [r["text"] for r in results]
        self.assertIn(self.OLD_FACT, order, "an unverified supersession hid a fact")
        self.assertLess(order.index(self.NEW_FACT), order.index(self.OLD_FACT))
        by_text = {r["text"]: r for r in results}
        self.assertEqual(by_text[self.OLD_FACT]["superseded_by"], "alice_2")
        self.assertNotIn("superseded_by", by_text[self.NEW_FACT])

    def test_exclusion_can_still_be_asked_for_explicitly(self):
        mem = self.make_memory(exclude_superseded=True)
        self._add_both(mem)
        mem.consolidate()
        texts = [r["text"] for r in mem.recall("user lives", user_id="alice", top_k=10)]
        self.assertIn(self.NEW_FACT, texts)
        self.assertNotIn(self.OLD_FACT, texts)


class TestLLMVerifiedRevision(MemoryTestBase):
    """An update that shares almost no words with the fact it replaces. The
    engine's lexical gates never propose it; a verifier that can judge
    meaning is given every new fact with its nearest older ones instead."""

    OLD_FACT = "user user user works at shopify as a backend developer"
    NEW_FACT = "user user user started at stripe on monday"

    def _add_both(self, mem):
        emb = FakeEmbedder()
        cosine = _cosine(emb.encode(self.OLD_FACT), emb.encode(self.NEW_FACT))
        self.assertTrue(0.45 <= cosine < 0.75, f"cosine {cosine} is outside the tested band")
        mem.add([{"role": "user", "content": self.OLD_FACT + "."}], user_id="alice")
        mem.add([{"role": "user", "content": "the cello is fun to play."}], user_id="alice")
        mem.add([{"role": "user", "content": self.NEW_FACT + "."}], user_id="alice")

    def _recalled(self, mem):
        return [r["text"] for r in mem.recall("user works", user_id="alice", top_k=10)]

    def test_the_lexical_gates_never_propose_it(self):
        verifier = AcceptAllVerifier()
        mem = self.make_memory(verifier=verifier)
        self._add_both(mem)
        self.assertEqual(mem.consolidate(), 0)
        self.assertEqual(verifier.calls, 0, "nothing was proposed, so nothing was vetted")
        self.assertIn(self.OLD_FACT, self._recalled(mem))

    def test_a_judging_verifier_sees_it_and_retires_the_old_fact(self):
        judge = FakeJudge("REPLACES")
        mem = self.make_memory(verifier=LLMVerifier(client=judge, candidates_per_record=1))
        self._add_both(mem)
        self.assertEqual(mem.consolidate(), 1)
        self.assertEqual(mem.engine.superseded_ids(), ["alice_1"])
        asked = judge.requests[0]["messages"][-1]["content"]
        self.assertIn(f"OLDER: {self.OLD_FACT}\nNEWER: {self.NEW_FACT}", asked)
        recalled = self._recalled(mem)
        self.assertLess(recalled.index(self.NEW_FACT), recalled.index(self.OLD_FACT),
                        "the fact that replaced it must be served first")
        stale = [r for r in mem.recall("user works", user_id="alice", top_k=10)
                 if r["text"] == self.OLD_FACT][0]
        self.assertEqual(stale["context"], STALE_MARK + self.OLD_FACT)
        # A second pass has nothing left to ask: the pair is settled.
        requests = len(judge.requests)
        self.assertEqual(mem.consolidate(), 0)
        self.assertEqual(len(judge.requests), requests)

    def test_a_judge_that_says_keep_changes_nothing(self):
        judge = FakeJudge("KEEPS")
        mem = self.make_memory(verifier=LLMVerifier(client=judge))
        self._add_both(mem)
        self.assertEqual(mem.consolidate(), 0)
        self.assertGreaterEqual(len(judge.requests), 1)
        self.assertEqual(mem.engine.superseded_ids(), [])
        results = mem.recall("user works", user_id="alice", top_k=10)
        self.assertIn(self.OLD_FACT, [r["text"] for r in results])
        for r in results:
            self.assertNotIn("superseded_by", r)

    # On the same topic as NEW_FACT, but clearly further from it than the
    # fact it updates.
    ON_TOPIC = "user user likes tea and long walks by the river on sunday mornings"

    def _add_three(self, mem):
        emb = FakeEmbedder()
        new = emb.encode(self.NEW_FACT)
        closest = _cosine(emb.encode(self.OLD_FACT), new)
        further = _cosine(emb.encode(self.ON_TOPIC), new)
        self.assertTrue(0.45 <= further < closest - 0.1, f"cosines {further} and {closest}")
        for fact in (self.OLD_FACT, self.ON_TOPIC, self.NEW_FACT):
            mem.add([{"role": "user", "content": fact + "."}], user_id="alice")

    def test_only_the_closest_older_facts_are_judged(self):
        # This judge would retire anything it is asked about.
        judge = FakeJudge("REPLACES")
        mem = self.make_memory(verifier=LLMVerifier(client=judge))
        self._add_three(mem)
        self.assertEqual(mem.consolidate(), 1)
        self.assertEqual(mem.engine.superseded_ids(), ["alice_1"])
        # Still so on the next pass, when the closest fact is already retired.
        self.assertEqual(mem.consolidate(), 0)
        self.assertEqual(mem.engine.superseded_ids(), ["alice_1"])
        self.assertIn(self.ON_TOPIC, [r["text"] for r in mem.recall("user", user_id="alice", top_k=10)])

    def test_the_margin_can_be_turned_off(self):
        judge = FakeJudge("REPLACES")
        mem = self.make_memory(verifier=LLMVerifier(client=judge, candidate_margin=None))
        self._add_three(mem)
        self.assertEqual(mem.consolidate(), 2)
        self.assertEqual(sorted(mem.engine.superseded_ids()), ["alice_1", "alice_2"])

    def test_verifier_names_are_resolved(self):
        with self.assertRaises(ValueError):
            self.make_memory(verifier="gpt")
        saved = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with self.assertRaises(RuntimeError) as ctx:  # "llm" is known: it needs a key
                self.make_memory(verifier="llm")
            self.assertIn("OPENAI_API_KEY", str(ctx.exception))
        finally:
            if saved is not None:
                os.environ["OPENAI_API_KEY"] = saved


class TestUserBudget(MemoryTestBase):
    """Memory(max_user_tokens=...): a user's memory is kept under a token
    budget by keeping the newest facts and folding the rest into gists."""

    BUDGET = 60

    @staticmethod
    def _tokens(mem, user):
        records = [r for r in mem.engine.get_records(mem.engine.scope_ids(user)) if r]
        return sum(max(1, len(r["text"]) // 4) for r in records), records

    def _fill(self, mem, user, first, count, role="user"):
        # Each stored fact is 7 tokens by the four-characters estimate.
        for i in range(first, first + count):
            mem.add([{"role": role, "content": f"{role} fact number {i:03d} is here."}], user_id=user)

    def test_store_is_brought_under_the_budget_newest_user_facts_first(self):
        mem = self.make_memory(max_user_tokens=self.BUDGET,
                               gist_summarizer=ExtractiveGistSummarizer())
        self._fill(mem, "alice", 0, 12)
        self._fill(mem, "alice", 0, 6, role="assistant")
        self._fill(mem, "bob", 0, 3)
        before, _ = self._tokens(mem, "alice")
        self.assertGreater(before, self.BUDGET)

        mem.consolidate()
        after, records = self._tokens(mem, "alice")
        self.assertLessEqual(after, self.BUDGET)
        kept = [r["text"] for r in records if r["source_role"] == "user"]
        # Half the budget holds the newest facts as they were, the user's
        # before the assistant's (which were written later).
        self.assertEqual(kept, [f"user fact number {i:03d} is here" for i in (8, 9, 10, 11)])
        self.assertFalse([r for r in records if r["source_role"] == "assistant"])
        gists = [r for r in records if r["source_role"] == "summary"]
        self.assertEqual(len(gists), 1)
        self.assertIn("user fact number", gists[0]["text"])
        self.assertEqual(gists[0]["scope"], "alice")
        # A gist is a turn of its own, unlike any fact's.
        payload = json.loads(gists[0]["payload"])
        self.assertEqual(payload["kind"], "gist")
        fact_turns = {json.loads(r["payload"])["turn_index"] for r in records
                      if r["source_role"] == "user"}
        self.assertNotIn(payload["turn_index"], fact_turns)
        # Another user's memory is not touched.
        self.assertEqual(self._tokens(mem, "bob")[0], 21)

        # The gist is an ordinary memory: recall finds it.
        found = mem.recall(gists[0]["text"].splitlines()[0], user_id="alice", top_k=5)
        self.assertIn(gists[0]["id"], [r["id"] for r in found])

    def test_later_passes_fold_the_earlier_gist_again(self):
        mem = self.make_memory(max_user_tokens=self.BUDGET,
                               gist_summarizer=ExtractiveGistSummarizer())
        self._fill(mem, "alice", 0, 12)
        mem.consolidate()
        first_gist = [r["id"] for r in self._tokens(mem, "alice")[1]
                      if r["source_role"] == "summary"]
        self.assertEqual(len(first_gist), 1)
        for start in (12, 24, 36):
            self._fill(mem, "alice", start, 12)
            mem.consolidate()
            tokens, records = self._tokens(mem, "alice")
            self.assertLessEqual(tokens, self.BUDGET)
            gists = [r for r in records if r["source_role"] == "summary"]
            self.assertEqual(len(gists), 1, "gists must be folded, not piled up")
        self.assertFalse(mem.engine.contains_id(first_gist[0]))
        # Nothing to do when the store already fits.
        self.assertEqual(mem.compact("alice"), 0)

    def test_a_failing_summarizer_removes_nothing(self):
        def broken(texts):
            raise RuntimeError("summarizer is down")

        mem = self.make_memory(max_user_tokens=self.BUDGET, gist_summarizer=broken)
        self._fill(mem, "alice", 0, 12)
        with self.assertLogs("tsm.memory", level="WARNING"):
            mem.consolidate()  # maintenance carries on
        self.assertEqual(len(mem.engine.scope_ids("alice")), 12)
        with self.assertRaises(RuntimeError):
            mem.compact("alice")  # asked for directly, the failure is reported
        self.assertEqual(len(mem.engine.scope_ids("alice")), 12)

    def test_without_a_summarizer_the_older_facts_are_dropped(self):
        mem = self.make_memory(max_user_tokens=self.BUDGET)
        self._fill(mem, "alice", 0, 12)
        self.assertEqual(mem.compact("alice"), 4)
        tokens, records = self._tokens(mem, "alice")
        self.assertLessEqual(tokens, self.BUDGET)
        # The whole budget goes to the newest facts: eight of seven tokens.
        self.assertEqual([r["text"] for r in records],
                         [f"user fact number {i:03d} is here" for i in range(4, 12)])

    def test_a_replaced_fact_is_folded_before_a_current_one(self):
        old_fact = "user user user user lives in paris"
        new_fact = "user user user user lives in london"
        mem = self.make_memory(verifier=AcceptAllVerifier(), max_user_tokens=self.BUDGET)
        self._fill(mem, "alice", 0, 8)
        # The pair is written last, so by age both would survive.
        mem.add([{"role": "user", "content": old_fact + "."}], user_id="alice")
        mem.add([{"role": "user", "content": new_fact + "."}], user_id="alice")
        self.assertEqual(mem.consolidate(), 1)
        texts = [r["text"] for r in self._tokens(mem, "alice")[1]]
        self.assertIn(new_fact, texts)
        self.assertNotIn(old_fact, texts)
        # Its place went to one more of the facts that are still true.
        self.assertEqual(len(texts), 8)

    def test_compact_with_no_user_covers_every_user(self):
        mem = self.make_memory(max_user_tokens=self.BUDGET)
        self._fill(mem, "alice", 0, 12)
        self._fill(mem, "bob", 0, 12)
        mem.close()  # a new process: nothing is marked as written to
        mem = self.make_memory(max_user_tokens=self.BUDGET)
        self.assertEqual(mem.consolidate(), 0)
        self.assertEqual(len(mem.engine.scope_ids("alice")), 12)
        self.assertEqual(mem.compact(), 8)
        for user in ("alice", "bob"):
            self.assertLessEqual(self._tokens(mem, user)[0], self.BUDGET)

    def test_budget_is_validated(self):
        with self.assertRaises(ValueError):
            self.make_memory(max_user_tokens=8)
        mem = self.make_memory()
        with self.assertRaises(ValueError):
            mem.compact("alice")


if __name__ == "__main__":
    unittest.main(verbosity=2)

