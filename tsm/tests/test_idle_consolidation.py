"""Unit tests for ``Memory(idle_consolidation_secs=...)``.

A real engine and fake embedder / extractor / verifiers, as in the rest of the
suite. The worker polls, so these wait for a condition with a generous ceiling
instead of sleeping a fixed time; the ceiling is only ever reached on failure.
"""

import os
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import tsm.memory
from tsm import Memory
from tsm.tests.test_memory import AcceptAllVerifier, FakeEmbedder, FakeExtractor, MemoryTestBase
from tsm.verification import LLMVerifier

OLD = "user user user user lives in paris"
NEW = "user user user user lives in london"


def wait_for(predicate, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def worker_threads():
    return [t for t in threading.enumerate() if t.name == "tsm-idle-consolidation"]


class IdleTestBase(MemoryTestBase):
    def tearDown(self):
        super().tearDown()

    def two_facts(self, mem):
        mem.add([{"role": "user", "content": OLD + "."}], user_id="alice")
        mem.add([{"role": "user", "content": NEW + "."}], user_id="alice")


class TestIdleWorker(IdleTestBase):
    def test_consolidates_once_the_memory_has_been_idle(self):
        verifier = AcceptAllVerifier()
        mem = self.make_memory(verifier=verifier, idle_consolidation_secs=0.3)
        self.two_facts(mem)
        self.assertTrue(wait_for(lambda: verifier.calls >= 1), "the worker never ran")
        self.assertTrue(wait_for(lambda: mem.engine.superseded_ids()),
                        "the pass ran but committed nothing")

    def test_does_not_run_while_the_memory_is_in_use(self):
        verifier = AcceptAllVerifier()
        # A call must stall for the whole idle period before the worker may run, so
        # the period is long against what a busy CI runner can stall a recall for.
        mem = self.make_memory(verifier=verifier, idle_consolidation_secs=1.5)
        self.two_facts(mem)
        end = time.monotonic() + 3.2  # more than two idle periods of steady use
        while time.monotonic() < end:
            mem.recall("user lives", user_id="alice")
            time.sleep(0.1)
        self.assertEqual(verifier.calls, 0)
        self.assertTrue(wait_for(lambda: verifier.calls >= 1), "never ran once use stopped")

    def test_nothing_written_nothing_run(self):
        verifier = AcceptAllVerifier()
        self.make_memory(verifier=verifier, idle_consolidation_secs=0.2)
        time.sleep(1.0)
        self.assertEqual(verifier.calls, 0)

    def test_a_store_with_memories_is_consolidated_after_it_is_opened(self):
        first = self.make_memory(verifier=AcceptAllVerifier())
        self.two_facts(first)
        first.close()

        verifier = AcceptAllVerifier()
        self.make_memory(verifier=verifier, idle_consolidation_secs=0.3)
        self.assertTrue(wait_for(lambda: verifier.calls >= 1))
        self.assertTrue(wait_for(lambda: self.mem.engine.superseded_ids()))

    def test_a_pass_runs_again_only_after_new_writes(self):
        verifier = AcceptAllVerifier()
        mem = self.make_memory(verifier=verifier, idle_consolidation_secs=0.2)
        self.two_facts(mem)
        self.assertTrue(wait_for(lambda: verifier.calls >= 1))
        time.sleep(1.0)
        calls = verifier.calls
        time.sleep(1.0)
        self.assertEqual(verifier.calls, calls, "idle passes kept running with nothing new")
        mem.add([{"role": "user", "content": "I have a dog named Rex."}], user_id="alice")
        self.assertTrue(wait_for(lambda: verifier.calls > calls), "a new write was not picked up")

    def test_a_failed_pass_loses_nothing_and_is_retried(self):
        class FailsOnce(AcceptAllVerifier):
            def verify(self, proposals, id_to_text):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("provider unavailable")
                return [(old, new, kind) for (old, new, kind, _c) in proposals]

        verifier = FailsOnce()
        with mock.patch.object(tsm.memory, "IDLE_RETRY_FLOOR_SECS", 0.2):
            mem = self.make_memory(verifier=verifier, idle_consolidation_secs=0.2)
            self.two_facts(mem)
            self.assertTrue(wait_for(lambda: verifier.calls >= 2), "the pass was not retried")
            self.assertTrue(wait_for(lambda: mem.engine.superseded_ids()))
        self.assertEqual(mem.engine.record_count(), 2)
        self.assertEqual(len(worker_threads()), 1, "the failure ended the worker")

    def test_a_verifier_that_could_not_finish_is_asked_again(self):
        class Unfinished(AcceptAllVerifier):
            unresolved = 0

            def verify(self, proposals, id_to_text):
                self.calls += 1
                if self.calls == 1:
                    self.unresolved = 1  # e.g. the request failed after its retries
                    return []
                self.unresolved = 0
                return [(old, new, kind) for (old, new, kind, _c) in proposals]

        verifier = Unfinished()
        with mock.patch.object(tsm.memory, "IDLE_RETRY_FLOOR_SECS", 0.2):
            mem = self.make_memory(verifier=verifier, idle_consolidation_secs=0.2)
            self.two_facts(mem)
            self.assertTrue(wait_for(lambda: verifier.calls >= 2),
                            "an unresolved verification was not retried")
            self.assertTrue(wait_for(lambda: mem.engine.superseded_ids()))

    def test_a_manual_consolidate_works_beside_the_worker(self):
        verifier = AcceptAllVerifier()
        mem = self.make_memory(verifier=verifier, idle_consolidation_secs=60)
        self.two_facts(mem)
        self.assertGreaterEqual(mem.consolidate(), 1)


class TestIdleWorkerLifecycle(IdleTestBase):
    def test_close_stops_the_worker_and_releases_the_database(self):
        before = len(worker_threads())
        mem = self.make_memory(verifier=AcceptAllVerifier(), idle_consolidation_secs=0.2)
        self.assertEqual(len(worker_threads()), before + 1)
        mem.close()
        self.assertEqual(len(worker_threads()), before)
        reopened = self.make_memory(verifier=AcceptAllVerifier())  # the lock is free at once
        self.assertEqual(reopened.engine.record_count(), 0)

    def test_close_during_a_running_pass_waits_for_it(self):
        started, release = threading.Event(), threading.Event()

        class Slow(AcceptAllVerifier):
            def verify(self, proposals, id_to_text):
                self.calls += 1
                started.set()
                release.wait(10)
                return [(old, new, kind) for (old, new, kind, _c) in proposals]

        mem = self.make_memory(verifier=Slow(), idle_consolidation_secs=0.2)
        self.two_facts(mem)
        self.assertTrue(started.wait(15), "the pass never started")
        closer = threading.Thread(target=mem.close)
        closer.start()
        time.sleep(0.3)
        self.assertTrue(closer.is_alive(), "close() did not wait for the running pass")
        release.set()
        closer.join(15)
        self.assertFalse(closer.is_alive())
        self.assertEqual(worker_threads(), [])

    def test_a_non_positive_period_is_refused_before_anything_is_opened(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                Memory(self.db_path, embedder=FakeEmbedder(), extractor=FakeExtractor(),
                       idle_consolidation_secs=bad)
        self.make_memory()  # nothing was left locked by the refusals

    def test_off_by_default(self):
        before = len(worker_threads())
        self.make_memory(verifier=AcceptAllVerifier())
        self.assertEqual(len(worker_threads()), before)


class _Chat:
    """An OpenAI-style client that answers with the queued replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_kwargs):
        content = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class TestVerifierUnresolved(unittest.TestCase):
    def test_counts_the_pairs_it_got_no_verdict_for(self):
        verifier = LLMVerifier(client=_Chat(["no verdicts in this reply", "1: moved. REPLACES"]))
        pairs = [("I live in Paris", "I live in London")]
        self.assertEqual(verifier.judge(pairs), [None])
        self.assertEqual(verifier.unresolved, 1)
        self.assertEqual(verifier.judge(pairs), ["replaces"])
        self.assertEqual(verifier.unresolved, 0)
        self.assertEqual(verifier.unresolved_total, 1)  # summed over both calls


if __name__ == "__main__":
    unittest.main()
