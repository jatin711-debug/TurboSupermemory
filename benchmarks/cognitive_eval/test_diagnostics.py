#!/usr/bin/env python3
"""Deterministic tests for bounded-memory diagnostic classification."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognitive_eval.diagnostics import (
    classify_disagreement,
    classify_failure,
    summarize_diagnostics,
    write_diagnostics,
)


class DiagnosticTests(unittest.TestCase):
    def test_failure_stages(self):
        self.assertEqual("correct", classify_failure(True, True, True, True, True))
        self.assertEqual(
            "unclassified_no_gold_token",
            classify_failure(False, False, False, False, False),
        )
        self.assertEqual(
            "no_lexical_signal_in_source",
            classify_failure(False, True, False, False, False),
        )
        self.assertEqual(
            "budget_removed_lexical_signal",
            classify_failure(False, True, True, False, False),
        )
        self.assertEqual(
            "retrieval_missed_lexical_signal",
            classify_failure(False, True, True, True, False),
        )
        self.assertEqual(
            "answer_failed_with_lexical_signal",
            classify_failure(False, True, True, True, True),
        )

    def test_tsm_mem0_disagreement_labels(self):
        self.assertEqual("both", classify_disagreement({"tsm": True, "mem0": True}))
        self.assertEqual("tsm_only", classify_disagreement({"tsm": True, "mem0": False}))
        self.assertEqual("mem0_only", classify_disagreement({"tsm": False, "mem0": True}))
        self.assertEqual("neither", classify_disagreement({"tsm": False, "mem0": False}))

    def test_summary_counts_failures_and_query_level_disagreements(self):
        records = [
            self._record("tsm", True, "correct"),
            self._record("mem0", False, "budget_removed_lexical_signal"),
            self._record("naive", False, "retrieval_missed_lexical_signal"),
        ]
        summary = summarize_diagnostics(records)
        self.assertEqual({"tsm_only": 1}, summary["disagreements"]["64-tokens"])
        self.assertEqual(
            {"budget_removed_lexical_signal": 1},
            summary["failure_stages"]["64-tokens"]["mem0"],
        )
        self.assertEqual(
            [{
                "budget": "64-tokens",
                "conversation_id": "conv-1",
                "query_id": "query-1",
                "label": "tsm_only",
            }],
            summary["query_labels"],
        )

    def test_diagnostics_are_written_as_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "diagnostics.json"
            summary = summarize_diagnostics([
                self._record("tsm", True, "correct"),
                self._record("mem0", False, "no_lexical_signal_in_source"),
            ])
            write_diagnostics(path, {"summary": summary})
            self.assertIn('"query_labels"', path.read_text(encoding="utf-8"))
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    @staticmethod
    def _record(system, correct, failure_stage):
        return {
            "budget": "64-tokens",
            "conversation_id": "conv-1",
            "query_id": "query-1",
            "system": system,
            "correct": correct,
            "failure_stage": failure_stage,
        }


if __name__ == "__main__":
    unittest.main()
