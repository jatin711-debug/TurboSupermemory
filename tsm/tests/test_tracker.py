"""Unit tests for the model-backed intention tracker
(``tsm.intentions.LLMIntentionTracker``, ``Memory(intention_tracker=...)``).

Same rules as ``test_memory``: plain ``unittest``, a real engine in a temp
directory, a fake embedder and fake trackers / chat clients, no keys.
"""

import json
import os
import sys
import unittest
from types import SimpleNamespace

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tsm.intentions import LLMIntentionTracker, parse_track
from tsm.tests.test_memory import MemoryTestBase


class TestParseTrack(unittest.TestCase):
    def test_reads_checked_operations(self):
        reply = "Sure:\n" + json.dumps({"ops": [
            {"op": "create", "message": 1, "what": " dentist  check-up ",
             "due_from": "2024-03-07", "due_until": None, "situation": None},
            {"op": "create", "message": 2, "what": "picture hooks", "due_from": None,
             "situation": "at the hardware store"},
            {"op": "close", "id": "P1", "as": "cancelled", "message": 2},
            {"op": "close", "id": "N2", "message": 3},
            {"op": "reschedule", "id": "N1", "due_from": "2024-03-21", "message": 3},
            {"op": "close", "id": "P9", "message": 1},          # no such open item
            {"op": "close", "id": "N7", "message": 1},          # not created
            {"op": "create", "message": 4, "what": "x"},        # no such message
            {"op": "reschedule", "id": "P1", "due_from": "soon", "message": 1},
            {"op": "delete", "id": "P1", "message": 1},
        ]})
        ops = parse_track(reply, ["int_1"], 3)
        self.assertEqual(ops, [
            {"op": "create", "message": 1, "what": "dentist check-up",
             "due_from": "2024-03-07", "due_until": "2024-03-07", "situation": None},
            {"op": "create", "message": 2, "what": "picture hooks", "due_from": None,
             "due_until": None, "situation": "at the hardware store"},
            {"op": "close", "message": 2, "target": ("open", "int_1"), "as": "cancelled"},
            {"op": "close", "message": 3, "target": ("new", 2), "as": "done"},
            {"op": "reschedule", "message": 3, "target": ("new", 1),
             "due_from": "2024-03-21", "due_until": "2024-03-21"},
        ])

    def test_unreadable_reply_changes_nothing(self):
        self.assertEqual(parse_track("no idea", ["int_1"], 2), [])
        self.assertEqual(parse_track('{"ops": "none"}', [], 1), [])


class TestLLMIntentionTracker(unittest.TestCase):
    def test_request_shows_open_items_and_numbered_messages(self):
        asked = []

        def create(**kwargs):
            asked.append(kwargs)
            reply = json.dumps({"ops": [{"op": "close", "id": "P1", "message": 1}]})
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=reply))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=create)))
        tracker = LLMIntentionTracker(client=client)
        ops = tracker.track([{"id": "int_4", "text": "dentist", "due_from": "2024-03-07",
                              "due_until": "2024-03-07", "situation": None}],
                            [{"text": "dentist done", "said_on": "2024-03-07"}])
        self.assertEqual(ops, [{"op": "close", "message": 1, "target": ("open", "int_4"),
                                "as": "done"}])
        user = asked[0]["messages"][-1]["content"]
        self.assertIn("P1. dentist (2024-03-07)", user)
        self.assertIn("1. [2024-03-07] dentist done", user)
        self.assertEqual(asked[0]["response_format"], {"type": "json_object"})
        tracker.track([{"id": "int_4", "text": "dentist", "due_from": "2024-03-07",
                        "due_until": "2024-03-07", "situation": None}],
                      [{"text": "dentist done", "said_on": "2024-03-07"}])
        self.assertEqual(tracker.calls, 1, "the same request is answered from the cache")

    def test_an_empty_reply_is_not_cached(self):
        replies = ["", json.dumps({"ops": []})]

        def create(**_kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=replies.pop(0)))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=create)))
        tracker = LLMIntentionTracker(client=client)
        message = [{"text": "dentist on Thursday", "said_on": "2024-03-04"}]
        self.assertEqual(tracker.track([], message), [])
        self.assertEqual(tracker.track([], message), [])
        self.assertEqual(tracker.calls, 2, "the empty reply was asked again")


class ScriptedTracker:
    """Creates, closes and reschedules by message text."""

    batch_size = 8
    max_open = 60

    def __init__(self):
        self.seen = []
        self.fail = False

    def track(self, open_items, messages):
        if self.fail:
            raise RuntimeError("tracker down")
        self.seen.extend(m["text"] for m in messages)
        refs = {item["text"]: f"P{k + 1}" for k, item in enumerate(open_items)}
        ops, created = [], []
        for n, m in enumerate(messages, 1):
            text = m["text"]
            if text.startswith("dentist on"):
                ops.append({"op": "create", "message": n, "what": "dentist",
                            "due_from": "2024-03-07", "due_until": "2024-03-07",
                            "situation": None})
                created.append("dentist")
            elif text.startswith("hooks"):
                ops.append({"op": "create", "message": n, "what": "buy picture hooks",
                            "due_from": None, "due_until": None,
                            "situation": "at the hardware store"})
                created.append("hooks")
            elif text.startswith("dentist moved"):
                ref = refs.get("dentist") or f"N{created.index('dentist') + 1}"
                ops.append({"op": "reschedule", "id": ref, "message": n,
                            "due_from": "2024-03-21", "due_until": "2024-03-21"})
            elif text.startswith("dentist done"):
                ops.append({"op": "close", "id": refs["dentist"], "message": n, "as": "done"})
        reply = json.dumps({"ops": ops})
        return parse_track(reply, [item["id"] for item in open_items], len(messages))


class TestMemoryTracker(MemoryTestBase):
    def _say(self, mem, text, day):
        mem.add([{"role": "user", "content": text, "timestamp": f"{day}T09:00:00"}],
                user_id="u")

    def test_forms_reschedules_and_closes_at_consolidation_only(self):
        tracker = ScriptedTracker()
        mem = self.make_memory(intentions=True, intention_tracker=tracker)
        self._say(mem, "dentist on Thursday at 3", "2024-03-04")
        self.assertEqual(mem.intentions("u"), [], "nothing is formed on add()")
        mem.consolidate()
        (item,) = mem.intentions("u")
        self.assertEqual((item["text"], item["due_from"]), ("dentist", "2024-03-07"))
        self.assertEqual([i["text"] for i in mem.due("u", now="2024-03-07")], ["dentist"])
        self._say(mem, "dentist moved to the 21st", "2024-03-05")
        mem.consolidate()
        self.assertEqual(mem.due("u", now="2024-03-07"), [], "moved")
        self.assertEqual([i["text"] for i in mem.due("u", now="2024-03-21")], ["dentist"])
        self._say(mem, "dentist done, all fine", "2024-03-21")
        mem.consolidate()
        self.assertEqual(mem.intentions("u"), [])
        self.assertEqual(len(tracker.seen), 3, "each message is read once")
        mem.consolidate()
        self.assertEqual(len(tracker.seen), 3)

    def test_a_situation_is_brought_to_mind_by_the_context(self):
        mem = self.make_memory(intentions=True, intention_tracker=ScriptedTracker())
        self._say(mem, "hooks next time I pass the shop", "2024-03-04")
        mem.consolidate()
        self.assertEqual(mem.due("u", now="2024-03-05"), [], "no date")
        hits = mem.due("u", now="2024-03-05", context="buy picture hooks at the hardware store",
                       cue_min_cosine=0.3)
        self.assertEqual([(i["text"], i["verdict"]) for i in hits],
                         [("buy picture hooks (at the hardware store)", "observe")])

    def test_a_failed_pass_leaves_the_messages_unread(self):
        tracker = ScriptedTracker()
        tracker.fail = True
        mem = self.make_memory(intentions=True, intention_tracker=tracker)
        self._say(mem, "dentist on Thursday at 3", "2024-03-04")
        mem.consolidate()
        self.assertTrue(mem._retry)
        self.assertEqual(mem.intentions("u"), [])
        tracker.fail = False
        mem.consolidate()
        self.assertEqual(len(mem.intentions("u")), 1)

    def test_forget_removes_the_message_and_what_it_formed(self):
        mem = self.make_memory(intentions=True, intention_tracker=ScriptedTracker(),
                               keep_messages=True, extractor="none")
        self._say(mem, "dentist on Thursday at 3", "2024-03-04")
        mem.consolidate()
        (passage,) = mem.recall("dentist", user_id="u")
        report = mem.forget([passage["id"]], user_id="u")
        self.assertTrue(report["verified"])
        self.assertEqual(report["intentions_removed"], 2, "the intention and its inbox copy")
        self.assertEqual(mem.intentions("u"), [])

    def test_needs_intentions(self):
        with self.assertRaises(ValueError):
            self.make_memory(intention_tracker=ScriptedTracker())


if __name__ == "__main__":
    unittest.main()
