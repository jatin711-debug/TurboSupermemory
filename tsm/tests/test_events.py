"""Events (``tsm.events``, ``Memory(events=...)``): fake extractors and a
fake chat client, a real engine. Run with
``python -m unittest tsm.tests.test_events -v``."""

import json
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
from tsm.events import EVENT_ROLE, OpenAIEventExtractor, event_note, event_text, normalize
from tsm.tests.test_memory import FakeEmbedder, FakeExtractor, MemoryTestBase


class FakeEvents:
    """One event per sentence that says "bought"; records what it was asked."""

    def __init__(self):
        self.asked = []

    def extract_events(self, message, context=None):
        self.asked.append((message, list(context or [])))
        return [{"what": s.strip(), "kind": "purchase", "amount": None, "unit": None,
                 "when": "last week", "status": "done"}
                for s in message.split(".") if "bought" in s]


TALK = [
    {"role": "user", "content": "I bought a red bike helmet. The weather was nice.",
     "timestamp": "2023/05/01 (Mon) 10:00"},
    {"role": "assistant", "content": "Great, I bought nothing but I can help.",
     "timestamp": "2023/05/01 (Mon) 10:01"},
    {"role": "user", "content": "I bought a red bike helmet.",
     "timestamp": "2023/05/20 (Sat) 09:00"},
]


class TestMemoryEvents(MemoryTestBase):
    def test_only_user_messages_are_asked_with_what_came_before(self):
        events = FakeEvents()
        mem = self.make_memory(events=events)
        mem.add(TALK, user_id="alice")
        self.assertEqual([m for m, _ in events.asked],
                         [TALK[0]["content"], TALK[2]["content"]])
        self.assertEqual(events.asked[1][1], [TALK[0]["content"], TALK[1]["content"]])

    def test_events_live_in_an_engine_of_their_own(self):
        mem = self.make_memory(events=FakeEvents())
        mem.add(TALK, user_id="alice")
        ids = mem._event_engine.scope_ids("alice")
        # The same purchase said in two messages is two records: a count needs both.
        self.assertEqual(len(ids), 2)
        recs = mem._event_engine.get_records(ids)
        self.assertTrue(all(r["source_role"] == EVENT_ROLE for r in recs))
        self.assertTrue(all(json.loads(r["payload"])["event"]["what"] for r in recs))
        self.assertFalse([r for r in mem.engine.get_records(mem.engine.scope_ids("alice"))
                          if r["source_role"] == EVENT_ROLE])
        # Ordinary recall never sees them.
        got = mem.recall("bike helmet", user_id="alice")
        self.assertTrue(got)
        self.assertFalse([r for r in got if r["role"] == EVENT_ROLE])

    def test_ordinary_recall_is_the_same_as_without_events(self):
        mem = self.make_memory(events=FakeEvents(), keep_messages=True)
        mem.add(TALK, user_id="alice")
        with_events = [r["context"] for r in mem.recall(
            "What colour is the bike helmet?", user_id="alice", token_budget=300)]
        mem.close()
        shutil.rmtree(self.db_path, ignore_errors=True)
        mem = self.make_memory(keep_messages=True)
        mem.add(TALK, user_id="alice")
        without = [r["context"] for r in mem.recall(
            "What colour is the bike helmet?", user_id="alice", token_budget=300)]
        self.assertEqual(with_events, without)

    def test_a_counting_question_gets_dated_event_notes(self):
        mem = self.make_memory(events=FakeEvents())
        mem.add(TALK, user_id="alice")
        got = mem.recall("How many bike helmets have I bought?", user_id="alice",
                         token_budget=400)
        notes = [r for r in got if r["role"] == EVENT_ROLE]
        self.assertEqual(len(notes), 2)
        for note in notes:
            self.assertTrue(note["context"].startswith("[2023-05-"), note["context"])
            self.assertIn("event: I bought a red bike helmet", note["context"])
            self.assertIn("last week [week of 2023-0", note["context"])
            self.assertIsNotNone(note["turn_index"])
            self.assertIn("bike helmet", note["source"])
        # Within the budget, events included.
        used = sum(max(1, len(r["context"]) // 4) for r in got)
        self.assertLessEqual(used, 400 + 8)

    def test_events_zero_and_other_questions_leave_recall_alone(self):
        mem = self.make_memory(events=FakeEvents())
        mem.add(TALK, user_id="alice")
        off = mem.recall("How many bike helmets have I bought?", user_id="alice",
                         token_budget=400, events=0)
        self.assertFalse([r for r in off if r["role"] == EVENT_ROLE])
        other = mem.recall("What colour is my bike helmet?", user_id="alice",
                           token_budget=400)
        self.assertFalse([r for r in other if r["role"] == EVENT_ROLE])

    def test_a_store_without_events_recalls_as_before(self):
        mem = self.make_memory()
        mem.add(TALK, user_id="alice")
        with_share = mem.recall("How many bike helmets have I bought?", user_id="alice",
                                token_budget=400)
        without = mem.recall("How many bike helmets have I bought?", user_id="alice",
                             token_budget=400, events=0)
        self.assertEqual([r["id"] for r in with_share], [r["id"] for r in without])

    def test_reopened_store_uses_its_events_without_an_extractor(self):
        mem = self.make_memory(events=FakeEvents())
        mem.add(TALK, user_id="alice")
        mem.close()
        self.mem = Memory(db_path=self.db_path, embedder=FakeEmbedder(),
                          extractor=FakeExtractor(), keep_messages=False)
        got = self.mem.recall("How many bike helmets have I bought?", user_id="alice",
                              token_budget=400)
        self.assertEqual(len([r for r in got if r["role"] == EVENT_ROLE]), 2)

    def test_unknown_events_name_is_rejected(self):
        with self.assertRaises(ValueError):
            Memory(db_path=self.db_path, embedder=FakeEmbedder(),
                   extractor=FakeExtractor(), events="gpt")


class FakeChat:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        content, finish = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason=finish)])


class TestOpenAIEventExtractor(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="tsm_events_")
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_parses_normalizes_and_caches(self):
        reply = json.dumps({"events": [
            {"what": "bought a chain", "kind": "bike part", "amount": "$1,200",
             "unit": "USD", "when": "null", "status": "bought"},
            {"what": "  ", "kind": "x"},
            "not an object",
        ]})
        client = FakeChat([(reply, "stop")])
        ex = OpenAIEventExtractor(client=client, cache_dir=self.dir, workers=1)
        got = ex.extract_events("I bought a chain for $1,200.", ["hi"])
        self.assertEqual(got, [{"what": "bought a chain", "kind": "bike part", "amount": 1200,
                                "unit": "USD", "when": None, "status": "done"}])
        self.assertEqual(ex.extract_events("I bought a chain for $1,200.", ["hi"]), got)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(client.requests[0]["response_format"], {"type": "json_object"})
        ex.flush_cache()
        again = OpenAIEventExtractor(client=FakeChat([]), cache_dir=self.dir)
        self.assertTrue(again.is_cached("I bought a chain for $1,200.", ["hi"]))

    def test_cut_off_reply_is_asked_again_then_given_up_uncached(self):
        client = FakeChat([('{"events": [', "length"), ("not json", "stop")])
        ex = OpenAIEventExtractor(client=client, cache_dir=self.dir, max_tokens=50)
        self.assertEqual(ex.extract_events("I ran 5 miles today."), [])
        self.assertEqual([r["max_tokens"] for r in client.requests], [50, 200])
        self.assertFalse(ex.is_cached("I ran 5 miles today."))

    def test_a_message_not_about_the_speaker_is_not_asked(self):
        client = FakeChat([])
        ex = OpenAIEventExtractor(client=client, cache_dir=self.dir)
        self.assertEqual(ex.extract_events("Write a poem about autumn leaves."), [])
        self.assertEqual(client.requests, [])

    def test_context_is_the_end_of_the_last_two_messages(self):
        client = FakeChat([('{"events": []}', "stop")])
        ex = OpenAIEventExtractor(client=client, cache_dir=self.dir)
        ex.extract_events("I did that.", ["first", "x" * 1000, "y" * 50])
        user = client.requests[0]["messages"][1]["content"]
        self.assertNotIn("first", user)
        self.assertIn("- " + "x" * 300 + "\n", user)
        self.assertNotIn("x" * 301, user)

    def test_extract_many_keeps_order(self):
        replies = [(json.dumps({"events": [{"what": f"ran {n} miles"}]}), "stop")
                   for n in range(5)]
        client = FakeChat(replies)
        ex = OpenAIEventExtractor(client=client, cache_dir=self.dir, workers=1)
        got = ex.extract_many([(f"I said {n}", []) for n in range(5)])
        self.assertEqual([g[0]["what"] for g in got], [f"ran {n} miles" for n in range(5)])


class TestEventText(unittest.TestCase):
    def test_note_and_text(self):
        event = normalize({"what": "went to a jazz concert", "kind": "concert",
                           "amount": 2, "unit": "hours", "when": "last Saturday",
                           "status": "done"})
        self.assertEqual(event_text(event), "concert: went to a jazz concert")
        self.assertEqual(event_note(event, "2023/05/22 (Mon) 10:00"),
                         "event: went to a jazz concert (concert; 2 hours; "
                         "last Saturday [2023-05-20])")
        planned = normalize({"what": "a trip to Rome", "status": "planned"})
        self.assertEqual(event_note(planned), "event (planned): a trip to Rome")


if __name__ == "__main__":
    unittest.main()
