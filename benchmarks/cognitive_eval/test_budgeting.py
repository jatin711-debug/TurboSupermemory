#!/usr/bin/env python3
"""Deterministic tests for fair active-memory budget accounting."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognitive_eval.budgeting import (
    build_slot_bounded_stores,
    build_token_bounded_stores,
    estimate_tokens,
    fit_complete_facts_to_budget,
    fit_text_to_budget,
    pack_recent,
    pack_role_priority_recent,
    partition_by_token_weight,
    total_tokens,
    truncate_to_budget,
)
from cognitive_eval.gist import create_gister


class BudgetingTests(unittest.TestCase):
    def test_estimate_and_truncate_use_one_accounting_rule(self):
        texts = ["a" * 8, "b" * 16, "c" * 4]
        self.assertEqual([2, 4, 1], [estimate_tokens(text) for text in texts])
        self.assertEqual([texts[0], texts[2]], truncate_to_budget(texts, 3))
        self.assertEqual(3, total_tokens(truncate_to_budget(texts, 3)))

    def test_pack_recent_prefers_newer_entries_but_can_skip_one_that_does_not_fit(self):
        texts = ["a" * 8, "b" * 12, "c" * 16]
        kept, overflow = pack_recent(texts, 6)
        self.assertEqual([texts[0], texts[2]], kept)
        self.assertEqual([texts[1]], overflow)
        self.assertLessEqual(total_tokens(kept), 6)

    def test_fit_text_never_exceeds_budget(self):
        fitted = fit_text_to_budget("alpha beta gamma delta epsilon", 4)
        self.assertTrue(fitted)
        self.assertLessEqual(estimate_tokens(fitted), 4)

    def test_complete_fact_fitter_drops_whole_units_instead_of_cutting_them(self):
        generated = "- First complete fact.\n- Second fact is much too long for this budget."
        fitted = fit_complete_facts_to_budget(generated, 6)
        self.assertEqual("- First complete fact.", fitted)
        self.assertNotIn("Second", fitted)
        self.assertLessEqual(estimate_tokens(fitted), 6)

    def test_role_priority_keeps_user_facts_before_recent_assistant_advice(self):
        texts = [
            "User bought a camera.",
            "Assistant recommends a tripod.",
            "User prefers a zoom lens.",
            "Assistant recommends a bag.",
        ]
        roles = ["user", "assistant", "user", "assistant"]
        kept, overflow = pack_role_priority_recent(texts, roles, 12)
        self.assertEqual([texts[0], texts[2]], kept)
        self.assertEqual([texts[1], texts[3]], overflow)

    def test_token_weight_partition_preserves_order_and_covers_all_items(self):
        texts = ["a" * 8, "b" * 40, "c" * 8, "d" * 40, "e" * 8]
        chunks = partition_by_token_weight(texts, 3)
        self.assertEqual(texts, [text for chunk in chunks for text in chunk])
        self.assertEqual(3, len(chunks))

    def test_token_bounded_stores_enforce_same_cap(self):
        facts = [
            "old account number is 12345",
            "user moved to Lisbon in 2022",
            "user likes Ethiopian coffee",
            "current project is called Atlas",
            "preferred editor is Neovim",
        ]
        calls = []

        def summarize(tail, max_tokens):
            calls.append((list(tail), max_tokens))
            return "- ID: 12345."

        stores = build_token_bounded_stores(facts, 20, summarize, gist_share=0.25)
        self.assertIsNotNone(stores)
        self.assertEqual("tokens", stores.unit)
        self.assertEqual(5, stores.gist_limit)
        self.assertTrue(stores.naive_overflow)
        self.assertTrue(stores.compressed_tail)
        self.assertEqual(stores.compressed_tail, calls[0][0])
        self.assertEqual(stores.gist_limit, calls[0][1])
        self.assertLessEqual(total_tokens(stores.naive), stores.limit)
        self.assertLessEqual(total_tokens(stores.compressed), stores.limit)
        self.assertLessEqual(estimate_tokens(stores.compressed[-1]), stores.gist_limit)

    def test_token_builder_skips_unpressured_store(self):
        stores = build_token_bounded_stores(
            ["short fact"], 20, lambda _tail, _limit: "unused"
        )
        self.assertIsNone(stores)

    def test_role_aware_builder_reserves_tight_gist_for_user_facts(self):
        facts = [
            "User bought a camera.",
            "Assistant recommends a tripod.",
            "User prefers a zoom lens.",
            "Assistant recommends a bag.",
        ]
        roles = ["user", "assistant", "user", "assistant"]
        captured = []

        def summarize(tail, _max_tokens):
            captured.extend(tail)
            return "- User bought a camera."

        stores = build_token_bounded_stores(
            facts,
            18,
            summarize,
            gist_share=0.5,
            roles=roles,
            role_aware=True,
        )
        self.assertTrue(captured)
        self.assertTrue(all(item.startswith("[user]") for item in captured))
        survivors = stores.compressed[:-len(stores.gists)] if stores.gists else stores.compressed
        self.assertTrue(all("Assistant recommends" not in item for item in survivors))
        self.assertLessEqual(total_tokens(stores.compressed), 18)

    def test_role_aware_builder_creates_separate_timeline_gists(self):
        facts = [
            f"User event {index:02d} happened on March {index + 1:02d} with item {index:02d}."
            for index in range(24)
        ] + [
            f"Assistant recommendation {index:02d} contains generic advice for later."
            for index in range(6)
        ]
        roles = ["user"] * 24 + ["assistant"] * 6
        captured = []

        def summarize(chunk, _max_tokens):
            captured.append(list(chunk))
            return "- " + chunk[0].split("] ", 1)[-1]

        stores = build_token_bounded_stores(
            facts,
            256,
            summarize,
            gist_share=0.5,
            roles=roles,
            role_aware=True,
            gist_chunk_tokens=32,
            max_gist_chunks=4,
        )
        self.assertEqual(4, len(captured))
        self.assertEqual(4, len(stores.gists))
        self.assertTrue(all(item.startswith("[user]") for chunk in captured[:3] for item in chunk))
        self.assertTrue(all(item.startswith("[assistant]") for item in captured[3]))
        self.assertLessEqual(total_tokens(stores.compressed), 256)

    def test_slot_builder_preserves_historical_shape(self):
        facts = ["one", "two", "three", "four"]
        long_gist = "gist " * 20
        stores = build_slot_bounded_stores(
            facts, 3, lambda _tail, _limit: long_gist, gist_token_limit=1
        )
        self.assertEqual(facts[-3:], stores.naive)
        self.assertEqual(facts[-2:], stores.compressed[:-1])
        self.assertEqual(["one", "two"], stores.compressed_tail)
        self.assertEqual(long_gist.strip(), stores.compressed[-1])
        self.assertEqual(3, len(stores.compressed))

    def test_extractive_gister_is_offline_and_bounded(self):
        gister = create_gister("extractive")
        gist = gister.summarize(["alpha beta gamma", "delta epsilon zeta"], max_tokens=4)
        self.assertLessEqual(estimate_tokens(gist), 4)
        self.assertEqual(0, gister.calls)

    def test_invalid_token_budget_configuration_fails_fast(self):
        with self.assertRaises(ValueError):
            build_token_bounded_stores(["a", "b"], 1, lambda _tail, _limit: "gist")
        with self.assertRaises(ValueError):
            build_token_bounded_stores(
                ["a" * 20, "b" * 20], 10, lambda _tail, _limit: "gist", gist_share=1.0
            )


if __name__ == "__main__":
    unittest.main()
