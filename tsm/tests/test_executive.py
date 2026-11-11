"""Unit tests for the memory-side executive: intentions (``due``),
experience (``feedback``, ``lessons``) and the belief ledger (``retract``,
``history``, ``forget``), plus the engine calls they stand on (payload
patches, lookups by payload, filtered vector search, retraction).

Same rules as ``test_memory``: plain ``unittest``, a real engine in a temp
directory, fake embedder / extractor / judges, no keys and no downloads.
"""

import json
import os
import sys
import unittest
from datetime import date

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


from tsm import intentions as intents
from tsm.experience import (ExperienceLedger, lesson_block, normalize_outcome, status_for)
from tsm.intentions import detect_closure, detect_intentions, future_window
from tsm.tests.test_memory import AcceptAllVerifier, FakeEmbedder, MemoryTestBase, _cosine


# ------------------------------------------------------------------ intentions
class TestIntentionRules(unittest.TestCase):
    NOW = date(2024, 3, 1)  # a Friday

    def test_future_window(self):
        self.assertEqual(future_window("call her tomorrow", self.NOW),
                         (date(2024, 3, 2), date(2024, 3, 2)))
        self.assertEqual(future_window("next Friday", self.NOW),
                         (date(2024, 3, 8), date(2024, 3, 8)))
        self.assertEqual(future_window("renew it by March 3", self.NOW),
                         (self.NOW, date(2024, 3, 3)))
        self.assertEqual(future_window("in two weeks", self.NOW),
                         (date(2024, 3, 15), date(2024, 3, 15)))
        self.assertEqual(future_window("next week", self.NOW),
                         (date(2024, 3, 4), date(2024, 3, 10)))
        self.assertEqual(future_window("on 2024-02-01", self.NOW), None, "in the past")
        self.assertEqual(future_window("on January 5", self.NOW),
                         (date(2025, 1, 5), date(2025, 1, 5)), "next year's")
        self.assertIsNone(future_window("I like tea", self.NOW))
        self.assertIsNone(future_window("tomorrow", None))

    def test_detect_intentions(self):
        found = detect_intentions("Remind me to water the plants.", "2024-03-01")
        self.assertEqual([(d.cue, d.due) for d in found], [("reminder", None)])
        found = detect_intentions("Nice day. I need to renew my passport by March 3.",
                                  "2024-03-01")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].cue, "commitment")
        self.assertEqual(found[0].objective, "I need to renew my passport by March 3.")
        self.assertEqual(found[0].due, (self.NOW, date(2024, 3, 3)))
        self.assertEqual(detect_intentions("I need to renew my passport.", "2024-03-01"), [],
                         "a commitment without a time is not an intention")
        self.assertEqual(detect_intentions("Should I call her tomorrow?", "2024-03-01"), [])
        self.assertEqual(detect_intentions("No need to remind me to call her.", "2024-03-01"),
                         [], "taking one back forms none")

    def test_detect_closure(self):
        self.assertEqual(detect_closure("Good news. I already renewed my passport."),
                         ("done", "I already renewed my passport."))
        self.assertEqual(detect_closure("Never mind about the dentist."),
                         ("cancelled", "Never mind about the dentist."))
        self.assertIsNone(detect_closure("I renewed my passport in 2019 and liked it"))


class TestIntentions(MemoryTestBase):
    def test_off_by_default(self):
        mem = self.make_memory()
        with self.assertRaises(RuntimeError):
            mem.due("u")

    def test_a_reminder_comes_due_with_its_evidence(self):
        mem = self.make_memory(intentions=True)
        mem.add([{"role": "user", "content": "Remind me to renew my passport next week.",
                  "timestamp": "2024-03-01"},
                 {"role": "assistant", "content": "Remind you to do what? Sure."}],
                user_id="u")
        pending = mem.intentions("u")
        self.assertEqual(len(pending), 1, "only the user's message forms one")
        self.assertEqual(pending[0]["due_from"], "2024-03-04")
        self.assertEqual(mem.intentions("someone else"), [])
        self.assertEqual(mem.due("u", now="2024-03-02"), [], "not yet")
        due = mem.due("u", now="2024-03-05")
        self.assertEqual([(d["verdict"], d["overdue"]) for d in due], [("act", False)])
        self.assertEqual([e["text"] for e in due[0]["evidence"]],
                         ["Remind me to renew my passport next week"])
        self.assertTrue(mem.due("u", now="2024-03-20")[0]["overdue"])
        # Before its time, the context can bring it to mind.
        cued = mem.due("u", now="2024-03-02", context="about to renew my passport")
        self.assertEqual([d["verdict"] for d in cued], ["observe"])
        self.assertGreaterEqual(cued[0]["cue_score"], intents.CUE_MIN_COSINE)

    def test_dates_as_objects(self):
        from datetime import datetime

        seen = []

        class Judge:
            def judge(self, items, context, today):
                seen.append(today)
                return {0: "wait"}

        mem = self.make_memory(intentions=True, readiness_judge=Judge())
        mem.add([{"role": "user", "content": "Remind me to water the plants tomorrow.",
                  "timestamp": "2024-03-01"}], user_id="u")
        due = mem.due("u", now=datetime(2024, 3, 2, 9, 30))
        self.assertEqual([(d["verdict"], d.get("judged")) for d in due], [("wait", True)])
        self.assertEqual(seen, ["2024-03-02"])
        (item,) = mem.intentions("u")
        self.assertTrue(mem.resolve_intention(item["id"], snooze_until=date(2024, 5, 1)))
        self.assertEqual(mem.intentions("u")[0]["due_from"], "2024-05-01")

    def test_resolve_and_snooze(self):
        mem = self.make_memory(intentions=True)
        mem.add([{"role": "user", "content": "Remind me to water the plants tomorrow.",
                  "timestamp": "2024-03-01"}], user_id="u")
        (item,) = mem.intentions("u")
        self.assertTrue(mem.resolve_intention(item["id"], snooze_until="2024-03-10"))
        self.assertEqual(mem.due("u", now="2024-03-05"), [])
        self.assertEqual(len(mem.due("u", now="2024-03-10")), 1)
        self.assertTrue(mem.resolve_intention(item["id"], "done"))
        self.assertEqual(mem.intentions("u"), [])
        self.assertFalse(mem.resolve_intention("int_missing", "done"))
        with self.assertRaises(ValueError):
            mem.resolve_intention(item["id"], "maybe")

    def test_consolidate_cancels_and_replaces(self):
        mem = self.make_memory(intentions=True)
        mem.add([{"role": "user", "content": "Remind me to renew my passport next week.",
                  "timestamp": "2024-03-01"}], user_id="u")
        mem.add([{"role": "user", "content": "Remind me to call the dentist on Friday",
                  "timestamp": "2024-03-01"}], user_id="u")
        mem.add([{"role": "user", "content": "Remind me to call the dentist on Monday instead",
                  "timestamp": "2024-03-02"}], user_id="u")
        mem.add([{"role": "user",
                  "content": "Never mind, no need to remind me to renew my passport next week.",
                  "timestamp": "2024-03-02"}], user_id="u")
        self.assertEqual(len(mem.intentions("u")), 3, "nothing changes before consolidation")
        mem.consolidate()
        texts = [i["text"] for i in mem.intentions("u")]
        self.assertEqual(texts, ["Remind me to call the dentist on Monday instead"])
        records = mem._intentions._items(mem._intentions.engine.ids_where(None, None))
        by_text = {r["text"]: r for r in records.values()}
        passport = by_text["Remind me to renew my passport next week."]
        self.assertEqual(passport["status"], "cancelled")
        self.assertEqual(passport["closed_on"], "2024-03-02")
        friday = by_text["Remind me to call the dentist on Friday"]
        self.assertEqual(friday["status"], "superseded")
        self.assertEqual(friday["superseded_by"],
                         by_text["Remind me to call the dentist on Monday instead"]["id"])
        # A second pass changes nothing.
        mem.consolidate()
        self.assertEqual(len(mem.intentions("u")), 1)

    def test_intentions_change_no_recall(self):
        messages = [{"role": "user", "content": "Remind me to renew my passport next week.",
                     "timestamp": "2024-03-01"},
                    {"role": "user", "content": "My passport expires in April.",
                     "timestamp": "2024-03-01"}]
        results = []
        for on in (False, True):
            self.db_path = self.db_path + ("_on" if on else "")
            os.makedirs(self.db_path, exist_ok=True)
            mem = self.make_memory(intentions=on)
            mem.add(messages, user_id="u")
            results.append([(p["id"], round(p["score"], 6)) for p in
                            mem.recall("passport renewal", user_id="u", token_budget=200,
                                       now=False)])
            mem.close()
            self.mem = None
        self.assertEqual(results[0], results[1])

    def test_intentions_survive_a_reopen(self):
        mem = self.make_memory(intentions=True)
        mem.add([{"role": "user", "content": "Remind me to water the plants tomorrow.",
                  "timestamp": "2024-03-01"}], user_id="u")
        (item,) = mem.intentions("u")
        mem.resolve_intention(item["id"], snooze_until="2024-04-01")
        mem.close()
        self.mem = None
        mem = self.make_memory(intentions=True)
        (again,) = mem.intentions("u")
        self.assertEqual(again["due_from"], "2024-04-01")


# ------------------------------------------------------------------ experience
class TestExperienceRules(unittest.TestCase):
    def test_outcomes_and_statuses(self):
        self.assertEqual(normalize_outcome(True), "success")
        self.assertEqual(normalize_outcome(0.2), "failure")
        self.assertEqual(normalize_outcome("Wrong"), "failure")
        with self.assertRaises(ValueError):
            normalize_outcome("meh")
        self.assertEqual(status_for(2, 0, "probation"), "active")
        self.assertEqual(status_for(1, 0, "probation"), "probation")
        self.assertEqual(status_for(1, 2, "active"), "deprecated")
        self.assertEqual(status_for(5, 0, "deprecated"), "deprecated")
        block = lesson_block([{"content": "a" * 40}, {"content": "b" * 400}], token_cap=20)
        self.assertEqual(block, "Lessons from earlier tasks:\n- " + "a" * 40)
        self.assertEqual(lesson_block([]), "")


class TestExperience(MemoryTestBase):
    TASK = "convert the invoice dates to ISO format"

    def test_a_lesson_is_learned_and_promoted_by_use(self):
        mem = self.make_memory(experience=True)
        mem.feedback(self.TASK, "success", user_id="u",
                     lesson="Read day-first dates when the vendor is European.")
        self.assertEqual(mem.lessons(self.TASK, "u", include_probation=True), [],
                         "curated at consolidation, not on feedback")
        mem.consolidate()
        (item,) = mem.lessons(self.TASK, "u", include_probation=True)
        self.assertEqual((item["status"], item["helpful"], item["harmful"]),
                         ("probation", 1, 0))
        self.assertEqual(mem.lessons(self.TASK, "u"), [], "not active yet")
        self.assertEqual(mem.lessons(self.TASK, "other user", include_probation=True), [])
        mem.feedback(self.TASK, True, memory_ids=[item["id"]], user_id="u")
        mem.consolidate()
        (item,) = mem.lessons(self.TASK, "u")
        self.assertEqual((item["status"], item["helpful"]), ("active", 2))
        self.assertEqual(len(item["provenance"]), 2)

    def test_a_used_lesson_that_fails_is_deprecated(self):
        mem = self.make_memory(experience=True)
        mem.feedback(self.TASK, "success", user_id="u", lesson="Assume US dates.")
        mem.consolidate()
        (item,) = mem.lessons(self.TASK, "u", include_probation=True)
        for _ in range(2):
            mem.feedback(self.TASK, "failure", memory_ids=[item["id"]], user_id="u")
        mem.consolidate()
        self.assertEqual(mem.lessons(self.TASK, "u", include_probation=True), [])
        stored = mem._experience._items([item["id"]])[item["id"]]
        self.assertEqual((stored["status"], stored["harmful"]), ("deprecated", 2))

    def test_an_unused_lesson_is_not_blamed(self):
        mem = self.make_memory(experience=True)
        mem.feedback(self.TASK, "success", user_id="u", lesson="Assume US dates.")
        mem.consolidate()
        mem.feedback(self.TASK, "failure", user_id="u")  # the lesson was not shown
        mem.consolidate()
        (item,) = mem.lessons(self.TASK, "u", include_probation=True)
        self.assertEqual(item["harmful"], 0)

    def test_self_reported_outcomes_teach_nothing(self):
        mem = self.make_memory(experience=True)
        mem.feedback(self.TASK, "success", user_id="u", source="self",
                     lesson="Always trust my own output.")
        mem.consolidate()
        self.assertEqual(mem.lessons(self.TASK, "u", include_probation=True), [])
        (episode,) = mem.episodes(self.TASK, "u")
        self.assertEqual((episode["source"], episode["outcome"]), ("self", "success"))
        with self.assertRaises(ValueError):
            mem.feedback(self.TASK, "success", source="the internet")

    def test_a_distiller_writes_the_missing_lesson(self):
        calls = []

        def distiller(task, outcome, note):
            calls.append((task, outcome))
            return "Check the vendor country first."

        mem = self.make_memory(experience=True, distiller=distiller)
        mem.feedback(self.TASK, "failure", user_id="u")
        mem.consolidate()
        self.assertEqual(calls, [(self.TASK, "failure")])
        (item,) = mem.lessons(self.TASK, "u", include_probation=True)
        self.assertEqual((item["content"], item["item_kind"]),
                         ("Check the vendor country first.", "warning"))

    def test_the_same_lesson_twice_supports_one_item(self):
        mem = self.make_memory(experience=True)
        mem.feedback(self.TASK, "success", user_id="u", lesson="Read day-first dates.")
        mem.feedback(self.TASK + " again", "success", user_id="u",
                     lesson="Read day-first dates.")
        mem.consolidate()
        items = mem.lessons(self.TASK, "u", include_probation=True)
        self.assertEqual(len(items), 1)
        self.assertEqual((items[0]["helpful"], items[0]["status"]), (2, "active"))


# ---------------------------------------------------------------- belief ledger
OLD_FACT = "user user user user lives in paris"
NEW_FACT = "user user user user lives in london"


class TestBeliefLedger(MemoryTestBase):
    def _superseded(self):
        mem = self.make_memory(verifier=AcceptAllVerifier())
        mem.add([{"role": "user", "content": OLD_FACT + ".", "timestamp": "2024-01-01"}],
                user_id="u")
        mem.add([{"role": "user", "content": NEW_FACT + ".", "timestamp": "2024-02-01"}],
                user_id="u")
        self.assertGreaterEqual(mem.consolidate(), 1)
        return mem

    def test_history_has_validity_dates(self):
        mem = self._superseded()
        chain = mem.history("u_1")
        self.assertEqual([(c["text"], c["valid_from"], c["valid_until"], c["current"])
                          for c in chain],
                         [(OLD_FACT, "2024-01-01", "2024-02-01", False),
                          (NEW_FACT, "2024-02-01", None, True)])

    def test_retract_undoes_a_supersession_for_good(self):
        mem = self._superseded()
        by_text = {r["text"]: r for r in mem.recall("user lives", user_id="u")}
        self.assertEqual(by_text[OLD_FACT]["superseded_by"], "u_2")
        self.assertTrue(mem.retract("u_1", "u_2"))
        self.assertFalse(mem.retract("u_1", "u_2"), "nothing left to take back")
        by_text = {r["text"]: r for r in mem.recall("user lives", user_id="u")}
        self.assertNotIn("superseded_by", by_text[OLD_FACT])
        mem.consolidate()  # the verifier would accept the pair again
        by_text = {r["text"]: r for r in mem.recall("user lives", user_id="u")}
        self.assertNotIn("superseded_by", by_text[OLD_FACT], "stays retracted")

    def test_deleting_the_replacement_restores_the_old_fact(self):
        mem = self._superseded()
        mem.engine.delete("u_2")
        (old,) = mem.recall("user lives", user_id="u")
        self.assertNotIn("superseded_by", old)
        self.assertEqual(mem.engine.resolve_beliefs(["u_1"])[0]["demotion"], 1.0)

    def test_forget_cascades_and_verifies(self):
        mem = self.make_memory(keep_messages=True, intentions=True)
        mem.add([{"role": "user",
                  "content": "My locker code is 4512. Remind me to change it next week.",
                  "timestamp": "2024-03-01"}], user_id="u")
        mem.add([{"role": "user", "content": "I like green tea.", "timestamp": "2024-03-01"}],
                user_id="u")
        mem.add([{"role": "user", "content": "I like green tea too."}], user_id="v")
        self.assertEqual(len(mem.intentions("u")), 1)
        locker = [r for r in mem.recall("locker code", user_id="u")
                  if "4512" in r["text"] and r["role"] == "user"]
        self.assertEqual(len(locker), 1, "one extracted fact holds the code")
        report = mem.forget([locker[0]["id"]], user_id="u")
        self.assertTrue(report["verified"])
        self.assertGreaterEqual(len(report["forgotten"]), 3, "its facts and its passage")
        self.assertEqual(report["intentions_removed"], 1)
        self.assertEqual(mem.intentions("u"), [])
        left = " ".join(r["text"] for r in mem.recall("locker code", user_id="u"))
        self.assertNotIn("4512", left)
        self.assertIn("green tea", left, "other messages stay")
        # Another user's memory is refused when a user is named.
        other = mem.recall("green tea", user_id="v")[0]["id"]
        self.assertEqual(mem.forget([other], user_id="u")["skipped"], [other])
        self.assertTrue(mem.engine.contains_id(other))


class TestForgettingRules(unittest.TestCase):
    def test_salient_terms_drop_function_question_and_time_words(self):
        from tsm.forgetting import salient_terms

        self.assertEqual(salient_terms("How many weddings did I attend last month?"),
                         ["weddings", "attend"])
        self.assertEqual(salient_terms("What did my sister's dog eat 3 days ago?"),
                         ["sister", "dog", "eat"])

    def test_what_carries_forgotten_content(self):
        from tsm.forgetting import carries, distinctive_tokens

        gone = ["My locker code is 4512", "my sister Ana lives in Porto with two cats"]
        self.assertTrue(carries("The code 4512 opens it.", gone), "a code")
        self.assertTrue(carries("Ana lives in Porto now, her cats too.", gone), "most words")
        self.assertTrue(carries("[2024-01-01] MY LOCKER   CODE IS 4512!", gone), "verbatim")
        self.assertFalse(carries("I visited Porto in 2024.", gone), "one word, a year")
        self.assertFalse(carries("I like green tea.", gone))
        self.assertEqual(distinctive_tokens("call 555-0199 or a@b.co in 2023, room 12"),
                         {"555-0199", "a@b.co"})
        self.assertFalse(carries("yes, sure.", ["Yes."]), "a short text only as a whole")
        self.assertTrue(carries("Yes.", ["Yes."]))

    def test_redact_keeps_the_other_sentences(self):
        from tsm.forgetting import redact

        kept, removed = redact("Alice likes green tea. Her locker code is 4512. She runs.",
                               ["My locker code is 4512"])
        self.assertEqual(kept, "Alice likes green tea. She runs.")
        self.assertEqual(removed, ["Her locker code is 4512."])


class TestForgetDerived(MemoryTestBase):
    GIST = "Alice likes green tea. Her locker code is 4512. She runs on Sundays."

    def _store(self, **kwargs):
        mem = self.make_memory(**kwargs)
        mem.add([{"role": "user", "content": "My locker code is 4512.",
                  "timestamp": "2024-03-01"}], user_id="u")
        payload = {"kind": "gist", "user_id": "u", "folded": 3, "turn_index": 99, "level": 0,
                   "span": ["2024-02-01", "2024-03-02"]}
        emb = FakeEmbedder()
        mem.engine.insert_batch(["u_gist"], [self.GIST], emb.encode([self.GIST]), [1.0], [[]],
                                [json.dumps(payload)], ["u"], ["summary"])
        mem.engine.insert_batch(["u_gist2"], ["She runs on Sundays."],
                                emb.encode(["She runs on Sundays."]), [1.0], [[]],
                                [json.dumps(payload)], ["u"], ["summary"])
        return mem

    def _locker_fact(self, mem):
        return next(r["id"] for r in mem.recall("locker code", user_id="u")
                    if r["role"] == "user")

    def test_a_gist_is_redacted_not_lost(self):
        mem = self._store()
        report = mem.forget([self._locker_fact(mem)], user_id="u")
        self.assertTrue(report["verified"])
        self.assertEqual(report["gists_deleted"], [])
        (new_id,) = report["gists_redacted"]
        self.assertFalse(mem.engine.contains_id("u_gist"))
        rec = mem.engine.get_records([new_id])[0]
        self.assertEqual(rec["text"], "Alice likes green tea. She runs on Sundays.")
        self.assertEqual(rec["source_role"], "summary")
        payload = json.loads(rec["payload"])
        self.assertEqual((payload["span"], payload["redacted"]),
                         (["2024-02-01", "2024-03-02"], 1))
        self.assertEqual(report["gists_may_hold"], ["u_gist2"], "same days, nothing in it")
        self.assertTrue(mem.engine.contains_id("u_gist2"))
        everything = " ".join(r["text"] for r in mem.recall("locker code 4512", user_id="u"))
        self.assertNotIn("4512", everything)

    def test_a_gist_with_nothing_left_is_deleted(self):
        mem = self.make_memory()
        mem.add([{"role": "user", "content": "My locker code is 4512.",
                  "timestamp": "2024-03-01"}], user_id="u")
        mem.engine.insert_batch(["u_g"], ["The locker code is 4512."],
                                FakeEmbedder().encode(["The locker code is 4512."]), [1.0],
                                [[]], [json.dumps({"kind": "gist"})], ["u"], ["summary"])
        report = mem.forget([self._locker_fact(mem)], user_id="u")
        self.assertEqual(report["gists_deleted"], ["u_g"])
        self.assertFalse(mem.engine.contains_id("u_g"))

    def test_keep_mode_lists_and_keeps(self):
        mem = self._store()
        fact = self._locker_fact(mem)
        with self.assertRaises(ValueError):
            mem.forget([fact], gists="refold")
        report = mem.forget([fact], user_id="u", gists="keep")
        self.assertIn("u_gist", report["gists_may_hold"])
        self.assertTrue(mem.engine.contains_id("u_gist"))

    def test_caches_are_purged(self):
        from tsm.embedders import OpenAIEmbedder
        from tsm.extractors import OpenAIExtractor
        from tsm.tests.test_backends import FakeEmbeddings, ScriptedChat, chat_reply

        cache = os.path.join(self.db_path, "cache")
        extractor = OpenAIExtractor(client=ScriptedChat(
            chat_reply('{"facts": ["My locker code is 4512"]}'),
            chat_reply('{"facts": ["I like green tea"]}')), cache_dir=cache)
        embedder = OpenAIEmbedder(client=FakeEmbeddings(1536), cache_dir=cache)
        embedder.encode(["My locker code is 4512", "I like green tea", "unrelated query"])
        extractor.extract_facts("My locker code is 4512.")
        extractor.extract_facts("I like green tea.")
        mem = self.make_memory()
        mem.add([{"role": "user", "content": "My locker code is 4512."}], user_id="u")
        # Only the caches of these backends are looked at; the store's
        # vectors came from the fake embedder.
        mem.extractor, mem.embedder = extractor, embedder
        report = mem.forget(mem.engine.ids_where(None, "u"), user_id="u")
        self.assertEqual(report["cache_entries_purged"], 2, "one extraction, one embedding")
        self.assertEqual(set(embedder._cache), {"I like green tea", "unrelated query"})
        self.assertEqual(len(extractor._cache), 1)
        reloaded = OpenAIEmbedder(client=FakeEmbeddings(1536), cache_dir=cache)
        self.assertNotIn("My locker code is 4512", reloaded._cache, "rewritten on disk")
        mem.extractor = mem.embedder = None  # not real backends: keep close() simple

    def test_lessons_that_carry_it_go(self):
        mem = self._store(experience=True)
        mem.feedback("open the locker", "success", user_id="u",
                     lesson="The locker opens with 4512.")
        mem.feedback("make tea", "success", user_id="u", lesson="Green tea at 80 degrees.")
        mem.consolidate()
        report = mem.forget([self._locker_fact(mem)], user_id="u")
        self.assertEqual(report["experience_removed"], 2, "the lesson and its episode")
        (left,) = mem.lessons("open the locker", "u", include_probation=True) or [None]
        self.assertTrue(left is None or "4512" not in left["content"])


# ------------------------------------------------------------ engine primitives
class TestEnginePrimitives(MemoryTestBase):
    def test_payload_patches_keep_the_record_in_place(self):
        mem = self.make_memory()
        mem.add([{"role": "user", "content": "First fact here."},
                 {"role": "user", "content": "Second fact here."}], user_id="u")
        engine = mem.engine
        before = {r["id"]: r for r in engine.get_records(["u_1", "u_2"])}
        payload = json.loads(before["u_1"]["payload"])
        payload["status"] = "checked"
        self.assertEqual(engine.set_payloads(["u_1", "nope"], [json.dumps(payload), "{}"]), 1)
        after = engine.get_records(["u_1"])[0]
        self.assertEqual(after["insert_seq"], before["u_1"]["insert_seq"])
        self.assertEqual(json.loads(after["payload"])["status"], "checked")
        flt = json.dumps({"op": "eq", "field": "status", "value": "checked"})
        self.assertEqual(engine.ids_where(flt), ["u_1"])
        self.assertEqual(engine.ids_where(flt, "someone else"), [])
        self.assertEqual(engine.ids_where(None, "u"), ["u_1", "u_2"])
        vec = FakeEmbedder().encode("Second fact here")
        self.assertEqual([h for h, _ in engine.search_ann(vec, 2, filter=flt)], ["u_1"])
        with self.assertRaises(ValueError):
            engine.set_payloads(["u_1"], ["{bad"])
        with self.assertRaises(ValueError):
            engine.ids_where('{"op": "nope"}')
        with self.assertRaises(ValueError):
            engine.set_payloads(["u_1"], [])


class TestLedgerDirect(MemoryTestBase):
    def test_experience_ledger_on_its_own_engine(self):
        from tsm._loader import load_turbomemory

        engine = load_turbomemory().MemoryEngine(
            db_path=os.path.join(self.db_path, "x"), dimension=64)
        emb = FakeEmbedder(dim=64)
        ledger = ExperienceLedger(engine, lambda texts: emb.encode(list(texts)))
        ledger.record("task one", "success", lesson="Do the thing.", user_id=None)
        self.assertEqual(ledger.curate()["added"], 1)
        self.assertEqual(ledger.curate()["episodes"], 0, "each episode is curated once")
        (item,) = ledger.lessons(emb.encode("task one"), include_probation=True)
        self.assertEqual(item["content"], "Do the thing.")
        self.assertGreater(_cosine(emb.encode("task one"), emb.encode("task one")), 0.99)
        engine.close()


if __name__ == "__main__":
    unittest.main()
