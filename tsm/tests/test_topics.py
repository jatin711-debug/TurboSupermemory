"""Unit tests for topic-keyed supersession candidates (``tsm.topics``,
``Memory(topics=...)``).

Same rules as ``test_memory``: plain ``unittest``, a real engine in a temp
directory, fake embedder / keyer / verifier, no keys and no downloads.
"""

import json
import os
import sys
import tempfile
import shutil
import unittest
from types import SimpleNamespace

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tsm.tests.test_memory import MemoryTestBase
from tsm.topics import LLMTopicKeyer, normalize_key, parse_topics, topic_pairs

KREUZBERG = "I live in a small flat in Kreuzberg"
LISBON = "I moved to Lisbon last week"
CAT = "We have a tabby cat named Otis"
DOG = "We adopted a puppy called Juno"


class JudgingVerifier:
    """Judges meaning (has ``candidate_min_cosine``, so it is given the wide
    candidates) and replaces only the text pairs it is told to."""

    candidate_min_cosine = 0.45
    candidates_per_record = 2
    candidate_margin = 0.1

    def __init__(self, replaces=()):
        self.replaces = set(replaces)
        self.seen = []

    def verify(self, proposals, id_to_text):
        out = []
        for old, new, kind, _cosine in proposals:
            pair = (id_to_text[old], id_to_text[new])
            self.seen.append(pair)
            if pair in self.replaces:
                out.append((old, new, kind))
        return out


class FakeKeyer:
    def __init__(self, keys, fail=False):
        self.keys = keys
        self.fail = fail
        self.asked = []
        self.known = []
        self.flushed = 0

    def topics(self, texts, known=()):
        if self.fail:
            raise RuntimeError("keying service down")
        self.asked.extend(texts)
        self.known.append(list(known))
        return [self.keys.get(t) for t in texts]

    def flush_cache(self):
        self.flushed += 1


def fake_chat(replies):
    """An OpenAI-shaped client that answers from ``replies`` in order and
    records each request's user message."""
    asked = []

    def create(**kwargs):
        asked.append(kwargs["messages"][-1]["content"])
        content = replies[len(asked) - 1]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, asked


class TestTopicRules(unittest.TestCase):
    def test_parse_reads_keys_cardinality_and_none(self):
        reply = ("1 | one | user.residence\n"
                 "2. ONE | User Sister.Residence\n"
                 "3 | none\n"
                 "4 | user.pets | many\n"
                 "5 | one | not a key\n"
                 "6 | one | user.employer | Stripe\n"
                 "7 | user.age | 30\n"
                 "9 | one | user.age\n"
                 "1 | one | user.employer\n"
                 "chatter")
        self.assertEqual(parse_topics(reply, 7), {
            0: ("user.residence", True),
            1: ("user_sister.residence", True),
            2: ("", False),
            3: ("user.pets", False),
            5: ("user.employer", True),
        })

    def test_unnumbered_lines_are_read_in_order(self):
        reply = "N | none\nN | one | user.age\nN | user.pets | many"
        self.assertEqual(parse_topics(reply, 3),
                         {0: ("", False), 1: ("user.age", True), 2: ("user.pets", False)})
        self.assertEqual(parse_topics(reply, 4), {}, "not one line per statement")

    def test_normalize_key(self):
        self.assertEqual(normalize_key(" `User.Job Title` "), "user.job_title")
        self.assertEqual(normalize_key("residence"), "", "a key needs a subject")

    def test_pairs_only_single_valued_keys_in_one_scope(self):
        def rec(mid, seq, key, one=True, scope="u"):
            return {"id": mid, "insert_seq": seq, "scope": scope,
                    "payload": {"topic": key, "topic_one": one}}

        records = [rec("a", 1, "user.residence"), rec("b", 2, "user.pets", one=False),
                   rec("c", 3, "user.residence"), rec("d", 4, "user.pets", one=False),
                   rec("e", 5, "user.residence", scope="v"), rec("f", 6, ""),
                   rec("g", 7, "user.residence"), rec("h", 8, "user.residence"),
                   rec("i", 9, "user.residence"), rec("j", 10, "user.residence")]
        pairs = topic_pairs(records, per_record=3)
        self.assertIn(("a", "c"), pairs)
        self.assertNotIn(("b", "d"), pairs, "several pets can hold at once")
        self.assertFalse([p for p in pairs if "e" in p], "another user's fact")
        self.assertEqual([p for p in pairs if p[1] == "j"], [("i", "j"), ("h", "j"), ("g", "j")],
                         "the latest older facts first, at most per_record")


class TestLLMTopicKeyer(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="tsm_topics_")
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_batches_reuse_known_keys_and_cache_per_fact(self):
        client, asked = fake_chat(["1 | one | user.residence\n2 | none",
                                   "1 | one | user.residence"])
        keyer = LLMTopicKeyer(client=client, cache_dir=self.dir, batch_size=2)
        out = keyer.topics([KREUZBERG, "I watched a film", LISBON], known=["user.age"])
        self.assertEqual(out, [("user.residence", True), ("", False), ("user.residence", True)])
        self.assertEqual(len(asked), 2)
        self.assertIn("KNOWN KEYS: user.age\n", asked[0])
        self.assertIn("user.residence", asked[1].splitlines()[0],
                      "a key handed out in this call is offered to the next batch")
        keyer.flush_cache()
        again = LLMTopicKeyer(client=fake_chat([])[0], cache_dir=self.dir)
        self.assertEqual(again.topics([LISBON]), [("user.residence", True)], "from the cache")
        self.assertEqual(again.calls, 0)
        self.assertEqual(again.forget_texts([LISBON, "never keyed"]), 1)
        with open(os.path.join(self.dir, "topics_gpt-4.1-mini.json"), encoding="utf-8") as fh:
            self.assertEqual(len(json.load(fh)), 2)

    def test_an_unreadable_line_counts_as_no_property(self):
        client, asked = fake_chat(["1 | one | user.residence\n2 | user.residence | Lisbon"])
        keyer = LLMTopicKeyer(client=client, cache_dir=self.dir)
        self.assertEqual(keyer.topics([KREUZBERG, LISBON]), [("user.residence", True), ("", False)])
        self.assertEqual(keyer.topics([LISBON]), [("", False)], "not asked again")
        self.assertEqual(len(asked), 1)


class TestMemoryTopics(MemoryTestBase):
    KEYS = {KREUZBERG: ("user.residence", True), LISBON: ("user.residence", True),
            CAT: ("user.pets", False), DOG: ("user.pets", False)}

    def _store(self, verifier, keyer, **kwargs):
        mem = self.make_memory(verifier=verifier, topics=keyer, **kwargs)
        for text in (KREUZBERG, CAT, LISBON, DOG):
            mem.add([{"role": "user", "content": text + "."}], user_id="u")
        mem.add([{"role": "user", "content": KREUZBERG + "."}], user_id="v")
        return mem

    def test_a_reworded_update_reaches_the_verifier(self):
        verifier = JudgingVerifier(replaces={(KREUZBERG, LISBON)})
        keyer = FakeKeyer(self.KEYS)
        mem = self._store(verifier, keyer)
        self.assertEqual(mem.consolidate(), 1)
        self.assertEqual(verifier.seen.count((KREUZBERG, LISBON)), 1)
        self.assertNotIn((CAT, DOG), verifier.seen, "several pets can hold at once")
        by_text = {r["text"]: r for r in mem.recall("where do I live", user_id="u")}
        self.assertEqual(by_text[KREUZBERG]["superseded_by"], "u_3")
        (other,) = mem.recall("where do I live", user_id="v")
        self.assertNotIn("superseded_by", other, "another user's fact is untouched")
        payload = mem._records(["u_1"])["u_1"]["payload"]
        self.assertEqual((payload["topic"], payload["topic_one"]), ("user.residence", True))

    def test_the_same_pair_is_not_offered_without_topics(self):
        verifier = JudgingVerifier(replaces={(KREUZBERG, LISBON)})
        mem = self.make_memory(verifier=verifier)
        for text in (KREUZBERG, LISBON):
            mem.add([{"role": "user", "content": text + "."}], user_id="u")
        self.assertEqual(mem.consolidate(), 0, "too far apart to be a candidate by cosine")

    def test_facts_are_keyed_once_and_survive_a_reopen(self):
        keyer = FakeKeyer(self.KEYS)
        mem = self._store(JudgingVerifier(), keyer)
        mem.consolidate()
        self.assertEqual(sorted(keyer.asked), sorted([KREUZBERG, CAT, LISBON, DOG, KREUZBERG]))
        mem.consolidate()
        self.assertEqual(len(keyer.asked), 5, "nothing new to key")
        mem.close()
        keyer = FakeKeyer(self.KEYS)
        verifier = JudgingVerifier(replaces={(KREUZBERG, LISBON)})
        mem = self.make_memory(verifier=verifier, topics=keyer)
        self.assertEqual(mem.consolidate(), 1, "keys read back from the store")
        self.assertEqual(keyer.asked, [])

    def test_unanswered_and_failed_keying_is_retried(self):
        keyer = FakeKeyer({KREUZBERG: ("user.residence", True)})
        mem = self._store(JudgingVerifier(), keyer)
        mem.consolidate()
        self.assertTrue(mem._retry, "unanswered facts are asked again")
        keyer.keys = self.KEYS
        mem.consolidate()
        self.assertIn(LISBON, keyer.asked[5:])
        broken = FakeKeyer({}, fail=True)
        mem.topic_keyer = broken
        mem.add([{"role": "user", "content": "I work at a bakery."}], user_id="u")
        mem.consolidate()  # logged, nothing raised
        self.assertIn("u", mem._topic_scopes)

    def test_retracted_pairs_stay_retracted(self):
        verifier = JudgingVerifier(replaces={(KREUZBERG, LISBON)})
        mem = self._store(verifier, FakeKeyer(self.KEYS))
        self.assertEqual(mem.consolidate(), 1)
        self.assertTrue(mem.retract("u_1", "u_3"))
        self.assertEqual(mem.consolidate(), 0)

    def test_needs_a_verifier_that_judges_meaning(self):
        from tsm.tests.test_memory import AcceptAllVerifier

        with self.assertRaises(ValueError):
            self.make_memory(verifier=AcceptAllVerifier(), topics=FakeKeyer({}))
        with self.assertRaises(ValueError):
            self.make_memory(topics="bogus", verifier=JudgingVerifier())

    def test_forget_drops_the_cached_key(self):
        client, _ = fake_chat(["\n".join(f"{i} | one | user.residence" for i in range(1, 6))])
        keyer = LLMTopicKeyer(client=client, cache_dir=os.path.join(self.db_path, "k"))
        mem = self._store(JudgingVerifier(), keyer)
        mem.consolidate()
        report = mem.forget(["u_3"], user_id="u")
        self.assertTrue(report["verified"])
        self.assertGreaterEqual(report["cache_entries_purged"], 1)


if __name__ == "__main__":
    unittest.main()
