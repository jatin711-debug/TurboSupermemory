"""Observations (``tsm.observe``, ``Memory(observer=...)``): a fake observer
and a fake Claude client, a real engine. Run with
``python -m unittest tsm.tests.test_observe -v``."""

import os
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tsm import Memory
from tsm.observe import (AnthropicObserver, conversations, observation_context,
                         parse_observations)
from tsm.tests.test_memory import FakeEmbedder, MemoryTestBase


class FakeObserver:
    """One observation per user message: "User said: <message>"."""

    def __init__(self, fail=False):
        self.seen = []
        self.fail = fail

    def observe(self, messages):
        if self.fail:
            raise RuntimeError("observer down")
        self.seen.append([dict(m) for m in messages])
        return [f"User said: {m['content']}" for m in messages if m["role"] == "user"]


DAY1 = [
    {"role": "user", "content": "I bought a red bike helmet last Saturday.",
     "timestamp": "2023/05/22 (Mon) 10:00"},
    {"role": "assistant", "content": "Nice. Make sure it fits snugly.",
     "timestamp": "2023/05/22 (Mon) 10:01"},
]
DAY2 = [
    {"role": "user", "content": "I adopted a cat named Luna.",
     "timestamp": "2023/05/25 (Thu) 09:00"},
]


class TestHelpers(unittest.TestCase):
    def test_parse(self):
        reply = "- User owns a dog.\n* User likes tea.\n2. Assistant listed 3 parks.\n" \
                "- (nothing)\nHeading\n"
        self.assertEqual(parse_observations(reply),
                         ["User owns a dog.", "User likes tea.", "Assistant listed 3 parks."])
        self.assertEqual(parse_observations("- (nothing)"), [])

    def test_conversations_by_day(self):
        got = conversations(DAY1 + DAY2 + [{"role": "user", "content": "undated"}])
        self.assertEqual([len(c) for c in got], [2, 2])

    def test_context_is_dated_and_resolved(self):
        self.assertEqual(observation_context("User bought a helmet last Saturday.",
                                             "2023/05/22 (Mon) 10:00"),
                         "[2023-05-22] User bought a helmet last Saturday [2023-05-20].")


class TestMemoryObservations(MemoryTestBase):
    def make(self, observer, **kwargs):
        return self.make_memory(observer=observer, keep_messages=True, extractor="none",
                                **kwargs)

    def test_consolidate_observes_each_day_once(self):
        obs = FakeObserver()
        mem = self.make(obs)
        mem.add(DAY1 + DAY2, user_id="alice")
        mem.consolidate()
        self.assertEqual(len(obs.seen), 2)  # one conversation per day
        self.assertEqual([m["content"] for m in obs.seen[0]],
                         [DAY1[0]["content"], DAY1[1]["content"]])
        got = mem.observations("alice")
        self.assertEqual([o["context"] for o in got], [
            "[2023-05-22] User said: I bought a red bike helmet last Saturday [2023-05-20].",
            "[2023-05-25] User said: I adopted a cat named Luna."])
        mem.consolidate()
        self.assertEqual(len(obs.seen), 2)  # nothing new, nothing asked
        mem.add([{"role": "user", "content": "I sold my old bike.",
                  "timestamp": "2023/05/27 (Sat) 09:00"}], user_id="alice")
        mem.consolidate()
        self.assertEqual(len(obs.seen), 3)
        self.assertEqual([m["content"] for m in obs.seen[2]], ["I sold my old bike."])
        self.assertEqual(len(mem.observations("alice")), 3)
        self.assertEqual(mem.observations("bob"), [])

    def test_reopened_store_does_not_observe_again(self):
        mem = self.make(FakeObserver())
        mem.add(DAY1, user_id="alice")
        mem.consolidate()
        mem.close()
        obs = FakeObserver()
        self.mem = Memory(db_path=self.db_path, embedder=FakeEmbedder(), extractor="none",
                          keep_messages=True, observer=obs)
        self.mem.consolidate()
        self.assertEqual(obs.seen, [])
        self.assertEqual(len(self.mem.observations("alice")), 1)

    def test_a_failed_pass_stores_nothing_and_is_retried(self):
        obs = FakeObserver(fail=True)
        mem = self.make(obs)
        mem.add(DAY1, user_id="alice")
        mem.consolidate()  # logged, not raised
        self.assertEqual(mem.observations("alice"), [])
        obs.fail = False
        mem.consolidate()
        self.assertEqual(len(mem.observations("alice")), 1)

    def test_query_and_budget_select_then_order(self):
        mem = self.make(FakeObserver())
        mem.add(DAY1 + DAY2, user_id="alice")
        mem.consolidate()
        got = mem.observations("alice", query="cat Luna adopted", token_budget=20)
        self.assertEqual([o["text"] for o in got], ["User said: I adopted a cat named Luna."])

    def test_observer_needs_the_messages(self):
        with self.assertRaises(ValueError):
            Memory(db_path=self.db_path, embedder=FakeEmbedder(), observer=FakeObserver(),
                   extractor="passthrough", keep_messages=False)


class FakeClaude:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        text, stop = self.replies.pop(0)
        return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=text)],
                               usage=SimpleNamespace(input_tokens=10, output_tokens=5))


class TestAnthropicObserver(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="tsm_obs_")
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_cut_off_reply_gets_more_room_and_is_cached(self):
        client = FakeClaude([("- User", "max_tokens"), ("- User owns a dog.", "end_turn")])
        obs = AnthropicObserver(client=client, cache_dir=self.dir, max_tokens=100, workers=1)
        self.assertEqual(obs.observe(DAY1), ["User owns a dog."])
        self.assertEqual([r["max_tokens"] for r in client.requests], [100, 300])
        self.assertEqual(client.requests[0]["thinking"], {"type": "disabled"})
        self.assertEqual(obs.observe(DAY1), ["User owns a dog."])
        self.assertEqual(len(client.requests), 2)
        obs.flush_cache()
        again = AnthropicObserver(client=FakeClaude([]), cache_dir=self.dir)
        self.assertEqual(again.observe(DAY1), ["User owns a dog."])


if __name__ == "__main__":
    unittest.main()
