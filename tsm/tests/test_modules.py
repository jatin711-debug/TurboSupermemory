"""Unit tests for the engine-free tsm modules: budget selection and packing,
gist summarizers, concept extraction, and the role prior.

Pure Python + numpy (no engine, no API keys, no model downloads)::

    python -m unittest tsm.tests.test_modules -v
"""

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from tsm.budget import (
    estimate_tokens,
    fit_complete_facts_to_budget,
    pack_recent,
    select_under_budget,
)
from tsm.concepts import extract_concepts
from tsm.gist import ExtractiveGistSummarizer, OpenAIGistSummarizer, single_fact, strip_role
from tsm.ranking import is_first_person_query, role_prior


def _item(text, score, vec, turn_index=None):
    return {"text": text, "score": score, "vec": vec, "turn_index": turn_index}


def _embedder(pool):
    """Embed callable that returns each pool item's fixed test vector."""
    by_text = {p["text"]: p["vec"] for p in pool}
    return lambda texts: np.array([by_text[t] for t in texts], dtype=np.float32)


# Three mutually orthogonal directions plus one near-duplicate of the first.
_A, _B, _C = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
_A_DUP = [0.99, 0.14, 0.0]  # cosine 0.99 with _A, above the redundancy cutoff


class TestSelectUnderBudget(unittest.TestCase):
    def test_empty_pool(self):
        self.assertEqual(select_under_budget([], 100, embed=lambda t: []), [])

    def test_respects_token_budget(self):
        pool = [_item("a" * 40, 0.9, _A), _item("b" * 40, 0.8, _B), _item("c" * 40, 0.7, _C)]
        chosen = select_under_budget(pool, 25, embed=_embedder(pool))  # 10 tokens each
        self.assertEqual([p["text"][0] for p in chosen], ["a", "b"])
        self.assertLessEqual(sum(estimate_tokens(p["text"]) for p in chosen), 25)

    def test_near_duplicate_is_skipped_not_just_penalized(self):
        pool = [_item("first", 0.9, _A), _item("paraphrase", 0.89, _A_DUP),
                _item("other", 0.2, _B)]
        chosen = select_under_budget(pool, 1000, embed=_embedder(pool))
        self.assertEqual([p["text"] for p in chosen], ["first", "other"])

    def test_a_memory_and_the_one_it_replaced_are_not_duplicates(self):
        # They read alike, but one is the current value and one the earlier.
        pool = [dict(_item("lives in london", 0.9, _A), id="new"),
                dict(_item("lives in paris", 0.5, _A_DUP), id="old", superseded_by="new"),
                dict(_item("lives in a flat", 0.4, _A_DUP), id="other")]
        chosen = select_under_budget(pool, 1000, embed=_embedder(pool))
        self.assertEqual([p["id"] for p in chosen], ["new", "old"])

    def test_the_shown_text_is_what_counts_against_the_budget(self):
        pool = [dict(_item("a" * 40, 0.9, _A), context="[marked] " + "a" * 40),
                _item("b" * 40, 0.8, _B)]
        # 10 tokens of text each; the first is shown with 2 more.
        chosen = select_under_budget(pool, 21, embed=_embedder(pool))
        self.assertEqual([p["text"][0] for p in chosen], ["a"])
        chosen = select_under_budget(pool, 22, embed=_embedder(pool))
        self.assertEqual([p["text"][0] for p in chosen], ["a", "b"])

    def test_new_turn_bonus_prefers_uncovered_turn(self):
        # Same turn as the first pick but more relevant, vs. a new turn.
        pool = [_item("turn1 top", 0.9, _A, turn_index=1),
                _item("turn1 more", 0.7, _B, turn_index=1),
                _item("turn2 only", 0.6, _C, turn_index=2)]
        with_turns = select_under_budget(pool, 1000, embed=_embedder(pool))
        self.assertEqual([p["text"] for p in with_turns],
                         ["turn1 top", "turn2 only", "turn1 more"])

        # Without turn information the order is relevance alone.
        for p in pool:
            p["turn_index"] = None
        without = select_under_budget(pool, 1000, embed=_embedder(pool))
        self.assertEqual([p["text"] for p in without],
                         ["turn1 top", "turn1 more", "turn2 only"])

    def test_the_token_budget_is_the_only_default_limit(self):
        vecs = np.eye(8, dtype=np.float32)
        # "fact N" is one token by the four-characters estimate.
        pool = [_item(f"fact {i}", 1.0 - i * 0.01, vecs[i].tolist()) for i in range(8)]
        for method, kwargs in (("mmr", {"embed": _embedder(pool)}), ("truncate", {})):
            # A roomy budget takes everything: nothing caps the item count.
            self.assertEqual(len(select_under_budget(pool, 1000, method=method, **kwargs)), 8)
            # A tight one is filled, not half used.
            self.assertEqual(len(select_under_budget(pool, 6, method=method, **kwargs)), 6)
            # A caller can still ask for fewer.
            self.assertEqual(
                len(select_under_budget(pool, 1000, method=method, max_items=3, **kwargs)), 3)

    def test_truncate_is_relevance_order_and_needs_no_embedder(self):
        pool = [_item("low", 0.1, _A), _item("high", 0.9, _A_DUP), _item("mid", 0.5, _B)]
        chosen = select_under_budget(pool, 1000, method="truncate")
        self.assertEqual([p["text"] for p in chosen], ["high", "mid", "low"])

    def test_invalid_arguments(self):
        pool = [_item("x", 1.0, _A)]
        with self.assertRaises(ValueError):
            select_under_budget(pool, 10, method="nope")
        with self.assertRaises(ValueError):
            select_under_budget(pool, 10)  # mmr without an embedder


class TestPacking(unittest.TestCase):
    def test_estimate_tokens(self):
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("   "), 0)
        self.assertEqual(estimate_tokens("abc"), 1)
        self.assertEqual(estimate_tokens("a" * 40), 10)

    def test_pack_recent_keeps_newest_in_input_order(self):
        texts = ["a" * 40, "b" * 40, "c" * 40]
        kept, overflow = pack_recent(texts, 20)
        self.assertEqual(kept, ["b" * 40, "c" * 40])
        self.assertEqual(overflow, ["a" * 40])

    def test_fit_complete_facts_never_cuts_a_fact(self):
        text = "- alice adopted a beagle\n- bob moved to lisbon in the spring\n- carol runs"
        fitted = fit_complete_facts_to_budget(text, 9)
        self.assertEqual(fitted, "- alice adopted a beagle\n- carol runs")
        self.assertEqual(fit_complete_facts_to_budget(text, 0), "")


class TestGist(unittest.TestCase):
    def test_role_helpers(self):
        self.assertEqual(strip_role("[user] I like tea"), "I like tea")
        self.assertEqual(strip_role("plain"), "plain")
        self.assertEqual(single_fact("[assistant] Try green tea."), "")
        self.assertEqual(single_fact("[user] I like tea"), "I like tea")

    def test_extractive_prefers_user_facts_and_drops_assistant(self):
        gister = ExtractiveGistSummarizer(max_tokens=120)
        gist = gister([
            "[assistant] You could try a tripod.",
            "[system] Session started.",
            "[user] I bought a Suica card in Tokyo.",
        ])
        self.assertEqual(gist, "- I bought a Suica card in Tokyo.\n- Session started.")
        self.assertEqual(gister([]), "")

    def test_extractive_respects_budget(self):
        gister = ExtractiveGistSummarizer()
        gist = gister.summarize(["[user] " + "x" * 60, "[user] short fact"], max_tokens=6)
        self.assertEqual(gist, "- short fact")

    def test_openai_summarizer_shortcuts_and_truncation(self):
        class _Msg:
            def __init__(self, content):
                self.message = type("M", (), {"content": content})()
                self.finish_reason = "length"

        class _Client:
            def __init__(self):
                self.requests = []
                self.chat = type("C", (), {})()
                self.chat.completions = type("K", (), {"create": self._create})()

            def _create(self, **kwargs):
                self.requests.append(kwargs)
                usage = type("U", (), {"prompt_tokens": 11, "completion_tokens": 7})()
                return type("R", (), {"usage": usage,
                                      "choices": [_Msg("- complete fact\n- cut off mid")]})()

        client = _Client()
        gister = OpenAIGistSummarizer(client=client, max_tokens=40)
        # Zero or one fact never costs an API call.
        self.assertEqual(gister([]), "")
        self.assertEqual(gister(["[user] only fact"]), "only fact")
        self.assertEqual(gister.calls, 0)
        # A length-truncated reply keeps only its complete lines.
        self.assertEqual(gister(["[user] one", "[user] two"]), "- complete fact")
        self.assertEqual((gister.calls, gister.input_tokens, gister.output_tokens), (1, 11, 7))
        self.assertEqual(client.requests[0]["max_tokens"], 40)

    def test_openai_summarizer_requires_a_key_or_client(self):
        saved = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with self.assertRaises(RuntimeError):
                OpenAIGistSummarizer()
        finally:
            if saved is not None:
                os.environ["OPENAI_API_KEY"] = saved


class TestConceptsAndRanking(unittest.TestCase):
    def test_extract_concepts(self):
        concepts = extract_concepts("However, I moved to San Francisco with a state-of-the-art lab.")
        # Proper-noun phrases first, then hyphenated compounds, then content
        # words. The capitalized sentence starter is not taken for a proper
        # noun, and stop words never appear.
        self.assertEqual(concepts[:2], ["san francisco", "state-of-the-art"])
        self.assertIn("moved", concepts)
        self.assertNotIn("with", concepts)
        self.assertLessEqual(len(extract_concepts(" ".join(f"word{i:02d}x" for i in range(40)))), 15)

    def test_role_prior(self):
        self.assertTrue(is_first_person_query("What did I buy?"))
        self.assertFalse(is_first_person_query("capital of France"))
        self.assertEqual(role_prior(True, "user"), 1.30)
        self.assertEqual(role_prior(True, "assistant"), 0.85)
        self.assertEqual(role_prior(True, "gist"), 1.0)
        self.assertEqual(role_prior(False, "user"), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
