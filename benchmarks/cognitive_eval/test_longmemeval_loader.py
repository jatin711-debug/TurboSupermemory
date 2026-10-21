#!/usr/bin/env python3
"""Deterministic tests for the LongMemEval loader (no dataset, no network)."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognitive_eval.benchmark_datasets.longmemeval import (  # noqa: E402
    Conversation,
    Query,
    attach_official_labels,
    chronological,
    conversation_from_official,
    iter_json_array,
    load_official,
    parse_date,
    parse_documents,
)

DOCS = [
    "[Date: 2023/05/22 (Mon) 09:10]\nUser: I adopted a dog.\n\n  His name is Rex.\n"
    "Assistant: Congratulations!",
    "[Date: 2023/05/20 (Sat) 11:19]\nUser: I moved to Lisbon.\nAssistant: Lovely city.",
]
OFFICIAL = {
    "question_id": "q1",
    "question_type": "temporal-reasoning",
    "question": "Which happened first?",
    "answer": "The move",
    "question_date": "2023/05/30 (Tue) 08:00",
    "haystack_dates": ["2023/05/22 (Mon) 09:10", "2023/05/20 (Sat) 11:19"],
    "haystack_session_ids": ["answer_a", "filler_b"],
    "answer_session_ids": ["answer_a"],
    "haystack_sessions": [
        [{"role": "user", "content": "I adopted a dog.\n\n  His name is Rex.", "has_answer": True},
         {"role": "assistant", "content": "Congratulations!", "has_answer": False}],
        [{"role": "user", "content": "I moved to Lisbon.", "has_answer": False},
         {"role": "assistant", "content": "Lovely city.", "has_answer": False}],
    ],
}


class TestParquetDocuments(unittest.TestCase):
    def test_sessions_keep_their_date_and_index(self):
        messages = parse_documents(DOCS)
        self.assertEqual([(m.role, m.session) for m in messages],
                         [("user", 0), ("assistant", 0), ("user", 1), ("assistant", 1)])
        self.assertEqual({m.date for m in messages if m.session == 0}, {"2023/05/22 (Mon) 09:10"})
        self.assertEqual({m.date for m in messages if m.session == 1}, {"2023/05/20 (Sat) 11:19"})
        # Lines are stripped and blank lines dropped; the date line is not content.
        self.assertEqual(messages[0].content, "I adopted a dog.\nHis name is Rex.")
        self.assertEqual([m.timestamp for m in messages], [0.0, 1.0, 2.0, 3.0])


class TestDates(unittest.TestCase):
    def test_parse_date(self):
        self.assertEqual(parse_date("2023/05/20 (Sat) 11:19").isoformat(), "2023-05-20T11:19:00")
        self.assertEqual(parse_date("2023/05/20").isoformat(), "2023-05-20T00:00:00")
        self.assertIsNone(parse_date("last tuesday"))
        self.assertIsNone(parse_date(""))
        self.assertIsNone(parse_date("2023/13/40 (Sat) 11:19"))

    def test_chronological_orders_sessions_not_turns(self):
        messages = parse_documents(DOCS)
        ordered = chronological(messages)
        self.assertEqual([m.session for m in ordered], [1, 1, 0, 0])
        self.assertEqual([m.role for m in ordered], ["user", "assistant", "user", "assistant"])
        # Two sessions at the same minute keep their file order.
        same = parse_documents([DOCS[1], DOCS[1].replace("Lisbon", "Porto")])
        self.assertEqual([m.session for m in chronological(same)], [0, 0, 1, 1])

    def test_an_undated_session_leaves_the_file_order(self):
        messages = parse_documents([DOCS[0], "User: no date here.\nAssistant: none."])
        self.assertEqual([m.timestamp for m in chronological(messages)],
                         [m.timestamp for m in messages])


class TestOfficialFiles(unittest.TestCase):
    def test_a_question_becomes_the_same_conversation_as_the_parquet_row(self):
        conv = conversation_from_official(OFFICIAL)
        parquet = parse_documents(DOCS)
        self.assertEqual([(m.role, m.content, m.session, m.date) for m in conv.messages],
                         [(m.role, m.content, m.session, m.date) for m in parquet])
        self.assertEqual([m.has_answer for m in conv.messages], [True, False, False, False])
        query = conv.queries[0]
        self.assertEqual((conv.conv_id, query.query_id, query.question_type), ("q1", "q1", "temporal-reasoning"))
        self.assertEqual(query.question_date, "2023/05/30 (Tue) 08:00")
        self.assertEqual(query.answer_sessions, (0,))
        self.assertFalse(query.is_abstention)
        self.assertTrue(conversation_from_official(dict(OFFICIAL, question_id="q1_abs"))
                        .queries[0].is_abstention)

    def test_files_are_read_one_question_at_a_time(self):
        items = [dict(OFFICIAL, question_id=f"q{i}") for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "official.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(items, fh, indent=1)
            # A tiny read size splits every item across many reads.
            self.assertEqual([x["question_id"] for x in iter_json_array(path, chunk=64)],
                             ["q0", "q1", "q2", "q3", "q4"])
            self.assertEqual([c.conv_id for c in load_official(path)], ["q0", "q1", "q2", "q3", "q4"])
            self.assertEqual([c.conv_id for c in load_official(path, ["q3", "q1", "missing"])],
                             ["q3", "q1"])

    def test_labels_are_attached_to_matching_parquet_conversations_only(self):
        def parquet_conversation(question_id, docs):
            messages = parse_documents(docs)
            return Conversation(conv_id="7", messages=messages, queries=[
                Query(query_id=question_id, query_text="?", answer_text="!", message_index=0)])

        matching = parquet_conversation("q1", DOCS)
        different = parquet_conversation("q2", [DOCS[0], DOCS[1].replace("Lisbon", "Porto")])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "oracle.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump([OFFICIAL, dict(OFFICIAL, question_id="q2")], fh)
            self.assertEqual(attach_official_labels([matching, different], path), 1)
        self.assertEqual([m.has_answer for m in matching.messages], [True, False, False, False])
        self.assertEqual(matching.queries[0].question_date, "2023/05/30 (Tue) 08:00")
        self.assertEqual(matching.queries[0].answer_sessions, (0,))
        self.assertEqual(matching.conv_id, "7")
        self.assertFalse(any(m.has_answer for m in different.messages))
        self.assertEqual(different.queries[0].question_date, "")


if __name__ == "__main__":
    unittest.main()
