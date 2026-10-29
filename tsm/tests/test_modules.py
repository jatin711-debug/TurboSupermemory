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
from tsm.chunks import MESSAGE_ROLE, is_message, message_role, occasion, passages, spoken_by
from tsm.compaction import GIST_OTHER_ROLE, GIST_ROLE, plan_compaction
from tsm.concepts import extract_concepts
from tsm.gist import ExtractiveGistSummarizer, OpenAIGistSummarizer, single_fact, strip_role
from tsm.ranking import (ASSISTANT_ROLE_DAMP, COUNTING_LAM, PRIOR_SPARES_TOP, RECALL_LAM,
                         USER_ROLE_BOOST, apply_role_prior, asks_what_assistant_said,
                         is_counting_query, is_first_person_query, packing_lam, role_prior)
from tsm.reading import ANSWER_SYSTEM, answer_prompt, final_answer, in_order
from tsm.answering import (add_days, answer_with_tools, calculate, count_distinct,
                           days_between, run_tool)
from tsm.sources import excerpt, turn_units
from tsm.when import annotate, said_on


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

    def test_a_tie_within_rounding_goes_to_the_one_ranked_first(self):
        # The second is ahead by less than GAIN_TIE: rounding, not relevance.
        pool = [_item("a" * 40, 0.5, _A), _item("b" * 40, 0.5 + 1e-7, _B)]
        chosen = select_under_budget(pool, 10, embed=_embedder(pool))
        self.assertEqual([p["text"][0] for p in chosen], ["a"])

    def test_the_same_words_on_another_day_are_another_occurrence(self):
        pool = [_item("first", 0.9, _A), _item("paraphrase", 0.89, _A_DUP),
                _item("other", 0.2, _B)]
        pool[0]["occasion"], pool[1]["occasion"] = 738000, 738000
        chosen = select_under_budget(pool, 1000, embed=_embedder(pool))
        self.assertEqual([p["text"] for p in chosen], ["first", "other"])
        pool[1]["occasion"] = 738001  # said a day later: a second occurrence
        chosen = select_under_budget(pool, 1000, embed=_embedder(pool))
        # Kept, though still weighed as redundant: after the unrelated one.
        self.assertEqual([p["text"] for p in chosen], ["first", "other", "paraphrase"])

    def test_near_duplicate_is_skipped_not_just_penalized(self):
        pool = [_item("first", 0.9, _A), _item("paraphrase", 0.89, _A_DUP),
                _item("other", 0.2, _B)]
        chosen = select_under_budget(pool, 1000, embed=_embedder(pool))
        self.assertEqual([p["text"] for p in chosen], ["first", "other"])

    def test_memories_that_read_alike_are_kept_together(self):
        # Two instances of one kind of event: cosine 0.87, well short of the
        # same text twice. A question that counts them needs both.
        alike = [0.87, 0.493, 0.0]
        pool = [_item("yoga class on tuesday", 0.9, _A), _item("yoga class on thursday", 0.88, alike),
                _item("bought a kettle", 0.3, _C)]
        chosen = select_under_budget(pool, 1000, embed=_embedder(pool))
        # The second one is penalized (the unrelated memory is taken before
        # it) but not thrown away.
        self.assertEqual([p["text"] for p in chosen],
                         ["yoga class on tuesday", "bought a kettle", "yoga class on thursday"])

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

    def test_a_question_about_what_the_assistant_said_has_no_role_prior(self):
        for q in ("I remember you told me about the refineries. Which processes?",
                  "How many eggs did you say we need for the omelette?",
                  "Can you remind me of the deli near the Vatican?",
                  "What did we finally decide to name it?",
                  "what we finally decided to name the zombie",
                  "I was going through our previous chat about binaural beats.",
                  "the two companies you mentioned that prioritize safety"):
            self.assertTrue(asks_what_assistant_said(q), q)
        for q in ("Can you suggest a hotel for my upcoming trip to Miami?",
                  "Can you recommend a show for me to watch tonight?",
                  "What did I buy for my sister's birthday?",
                  "How many model kits have I worked on?", ""):
            self.assertFalse(asks_what_assistant_said(q), q)
        for role in ("assistant", "message-assistant", "user", "message", "summary"):
            self.assertEqual(role_prior(True, role, True), 1.0, role)

    def test_a_counting_question_packs_for_relevance(self):
        for q in ("How many tanks do I have?", "How much did I spend on workshops?",
                  "What is the total I paid?", "How many in all?"):
            self.assertTrue(is_counting_query(q), q)
            self.assertEqual(packing_lam(q), COUNTING_LAM)
        for q in ("I totally forgot", "What degree did I get?", ""):
            self.assertFalse(is_counting_query(q), q)
            self.assertEqual(packing_lam(q), RECALL_LAM)
        self.assertEqual(packing_lam("How many?", lam=0.3), 0.3)

    def test_the_best_matches_are_never_damped(self):
        pool = [{"score": 0.9 - k / 100, "role": "assistant"} for k in range(5)]
        apply_role_prior(pool, "what is on my jumpsuit")  # first person, not about replies
        kept = [round(p["score"], 4) for p in pool]
        self.assertEqual(kept[:PRIOR_SPARES_TOP], [0.9, 0.89, 0.88][:PRIOR_SPARES_TOP])
        self.assertEqual(kept[PRIOR_SPARES_TOP:],
                         [round((0.9 - k / 100) * ASSISTANT_ROLE_DAMP, 4)
                          for k in range(PRIOR_SPARES_TOP, 5)])
        pool = [{"score": 0.5, "role": "user"}]
        apply_role_prior(pool, "what did I buy")
        self.assertAlmostEqual(pool[0]["score"], 0.5 * USER_ROLE_BOOST)  # boosts still apply


class TestGistRanking(unittest.TestCase):
    def test_a_gist_is_ranked_like_the_facts_it_was_written_from(self):
        self.assertEqual(role_prior(True, GIST_ROLE), USER_ROLE_BOOST)
        self.assertEqual(role_prior(True, GIST_ROLE), role_prior(True, "user"))
        self.assertEqual(role_prior(True, GIST_OTHER_ROLE), ASSISTANT_ROLE_DAMP)
        self.assertEqual(role_prior(False, GIST_ROLE), 1.0)


class TestPassages(unittest.TestCase):
    def test_a_short_message_is_one_passage(self):
        self.assertEqual(passages("  I moved to Lisbon.  "), ["I moved to Lisbon."])
        self.assertEqual(passages("   "), [])
        self.assertEqual(passages(None), [])

    def test_a_long_message_is_cut_between_sentences_and_nothing_is_lost(self):
        text = " ".join(f"Sentence number {i} says something about topic {i}." for i in range(40))
        parts = passages(text, max_tokens=30)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(estimate_tokens(p) <= 30 for p in parts))
        self.assertTrue(all(p.endswith(".") for p in parts), "cuts fall between sentences")
        self.assertEqual(" ".join(parts), text)

    def test_an_overlong_sentence_is_cut_between_words(self):
        text = " ".join(f"word{i}" for i in range(200))
        parts = passages(text, max_tokens=20)
        self.assertTrue(all(estimate_tokens(p) <= 20 for p in parts))
        self.assertEqual(" ".join(parts).split(), text.split())

    def test_lines_are_cut_too(self):
        text = "\n".join("- item %d of a long list that goes on for a while" % i for i in range(30))
        parts = passages(text, max_tokens=40)
        self.assertGreater(len(parts), 1)
        self.assertEqual(" ".join(parts).split(), text.split())

    def test_a_passage_knows_the_day_it_was_said(self):
        self.assertEqual(occasion("message", "2023/05/29 (Mon) 08:33"),
                         occasion("message-assistant", "2023-05-29"))
        self.assertNotEqual(occasion("message", "2023-05-29"), occasion("message", "2023-05-30"))
        self.assertIsNone(occasion("user", "2023-05-29"))  # a fact: no occasion
        self.assertIsNone(occasion("message", ""))

    def test_roles(self):
        self.assertEqual(message_role("user"), MESSAGE_ROLE)
        self.assertEqual(message_role(None), MESSAGE_ROLE)
        self.assertEqual(message_role("assistant"), "message-assistant")
        for role in ("user", "assistant", "system"):
            self.assertTrue(is_message(message_role(role)))
            self.assertEqual(spoken_by(message_role(role)), role)
        self.assertFalse(is_message("user"))
        self.assertFalse(is_message("messages"))
        self.assertFalse(is_message(None))
        self.assertEqual(spoken_by("assistant"), "assistant")

    def test_a_passage_is_ranked_like_its_speaker(self):
        self.assertEqual(role_prior(True, message_role("user")), role_prior(True, "user"))
        self.assertEqual(role_prior(True, message_role("assistant")),
                         role_prior(True, "assistant"))


class TestReading(unittest.TestCase):
    """answer_prompt / final_answer: what a model is asked, and what it said."""

    def test_memories_are_shown_in_the_order_they_were_said(self):
        recalled = [
            {"context": "[2023-05-20] Bought the necklace.", "turn_index": 9},
            {"context": "[2023-03-02] Ordered the photo album.", "turn_index": 4},
            {"context": "[2023-05-20 08:15] Had breakfast.", "turn_index": 7},
            {"context": "A summary of older conversations.", "role": "summary"},
            {"context": "[2023-05-20] Wrapped the necklace.", "turn_index": 11},
        ]
        self.assertEqual([m["context"] for m in in_order(recalled)],
                         ["[2023-03-02] Ordered the photo album.", "[2023-05-20] Bought the necklace.",
                          "[2023-05-20] Wrapped the necklace.", "[2023-05-20 08:15] Had breakfast.",
                          "A summary of older conversations."])
        # Strings work too (the "context" of each result), and turns order undated ones.
        self.assertEqual(in_order(["[turn 12] later", "[turn 3] earlier"]),
                         ["[turn 3] earlier", "[turn 12] later"])
        # A dated gist goes where what it summarizes starts.
        self.assertEqual(in_order(["[2023-03-01] a fact", "[2023-01-05 to 2023-02-20] a gist"]),
                         ["[2023-01-05 to 2023-02-20] a gist", "[2023-03-01] a fact"])
        # Without tags in the text, the result's own fields are used.
        untagged = [{"text": "second", "timestamp": "2024/02/01 (Thu) 10:00", "turn_index": 2},
                    {"text": "first", "timestamp": "2024-01-15", "turn_index": 9}]
        self.assertEqual([m["text"] for m in in_order(untagged)], ["first", "second"])

    def test_the_prompt_carries_the_notes_the_question_and_its_date(self):
        messages = answer_prompt("Which did I buy first?",
                                 [{"context": "[2023-05-20] Bought the necklace."},
                                  {"context": "[2023-03-02] Ordered the photo album."}],
                                 today="2023/05/22 (Mon) 12:21")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], ANSWER_SYSTEM)
        self.assertEqual(messages[1]["content"],
                         "Notes:\n1. [2023-03-02] Ordered the photo album.\n"
                         "2. [2023-05-20] Bought the necklace.\n\n"
                         "Current date: 2023/05/22 (Mon) 12:21\nQuestion: Which did I buy first?")
        self.assertNotIn("Current date", answer_prompt("q", ["a note"])[1]["content"])
        self.assertIn("(none)", answer_prompt("q", [])[1]["content"])

    def test_the_answer_is_what_follows_the_last_answer_mark(self):
        reply = ("1. Relevant notes: ...\n2. Reasoning: the album came first.\n"
                 "Answer: the photo album for your mom")
        self.assertEqual(final_answer(reply), "the photo album for your mom")
        self.assertEqual(final_answer("Answer: 3\n\nanswer: 4 plants"), "4 plants")
        self.assertEqual(final_answer("It was the album."), "It was the album.")
        self.assertEqual(final_answer(None), "")


class TestCompactionPlan(unittest.TestCase):
    """plan_compaction: what stays, what is folded, what the gists are written from."""

    @staticmethod
    def _capture():
        seen = []

        def summarize(texts, max_tokens):
            seen.append((list(texts), max_tokens))
            return "\n".join("- " + t.split("] ", 1)[1] for t in texts)
        return seen, summarize

    def test_nothing_to_do_when_everything_fits(self):
        self.assertIsNone(plan_compaction(["a" * 40, "b" * 40], ["user", "user"], 20, None))

    def test_merge_oldest_keeps_earlier_gists_while_they_fit(self):
        # One earlier gist (8 tokens) and ten facts of 10 tokens; budget 60:
        # 30 for gists, 30 for the three newest facts.
        texts = ["- adopted a dog\n- moved to Porto"] + [f"u{i} " + "x" * 36 for i in range(10)]
        roles = [GIST_ROLE] + ["user"] * 10
        spans = [("2023-01-01", "2023-01-09")] + [(f"2023-02-{i + 1:02d}",) * 2 for i in range(10)]
        seen = []

        def summarize(lines, max_tokens):
            seen.append((list(lines), max_tokens))
            return "- seven facts in short"  # 5 tokens: 8 + 5 fit the 30 of gists

        plan = plan_compaction(texts, roles, 60, summarize, spans=spans, levels=[0] * 11,
                               policy="merge_oldest")
        self.assertIn(0, plan.keep, "the earlier gist stays as it is")
        self.assertNotIn(0, plan.fold)
        self.assertEqual(plan.keep, [0, 8, 9, 10])
        self.assertEqual(len(seen), 1, "one new gist from what aged out, nothing refolded")
        self.assertEqual(seen[0][0][0], f"[user] [2023-02-01] {texts[1]}")
        self.assertEqual(plan.gist_levels, [0])
        self.assertEqual(plan.gist_spans, [("2023-02-01", "2023-02-07")])

    def test_merge_oldest_merges_the_two_oldest_gists_of_one_level(self):
        gist = "- " + "y" * 54  # 14 tokens
        texts = [gist, gist, gist] + [f"u{i} " + "x" * 36 for i in range(6)]
        roles = [GIST_ROLE] * 3 + ["user"] * 6
        spans = [("2023-01-01",) * 2, ("2023-01-05",) * 2, ("2023-01-09",) * 2] + [None] * 6
        merges = []

        def summarize(lines, max_tokens):
            merges.append(list(lines))
            # The new gist is 10 tokens, a merge 4.
            return "- " + "z" * (40 if len(merges) == 1 else 14)

        plan = plan_compaction(texts, roles, 60, summarize, spans=spans, levels=[0, 0, 1] + [0] * 6,
                               policy="merge_oldest")
        # Three earlier gists (42 tokens) and a new one (10) exceed the 30 of
        # gists: the two oldest at level 0 are merged (not the level-1 one),
        # and the 28 tokens left fit.
        self.assertEqual(sorted(i for i in plan.fold if i < 3), [0, 1])
        self.assertIn(2, plan.keep)
        self.assertEqual(sorted(plan.gist_levels), [0, 1])
        self.assertIn(("2023-01-01", "2023-01-05"), plan.gist_spans)
        self.assertEqual(len(merges), 2)  # the new gist, then one merge
        with self.assertRaises(ValueError):
            plan_compaction(texts, roles, 60, summarize, policy="forget")

    def test_each_gist_spans_the_dates_of_what_it_was_written_from(self):
        texts = ["- old summary line"] + [f"u{i} " + "x" * 36 for i in range(9)]
        roles = [GIST_ROLE] + ["user"] * 9
        spans = [("2023-01-01", "2023-01-20")] + [(f"2023-02-{i + 1:02d}",) * 2 for i in range(8)]
        spans.append(None)  # an undated fact
        _, summarize = self._capture()
        plan = plan_compaction(texts, roles, 60, summarize, spans=spans)
        self.assertEqual(plan.keep, [7, 8, 9])
        # One gist from the earlier gist and facts 0-5 (Feb 1-6).
        self.assertEqual(plan.gist_spans, [("2023-01-01", "2023-02-06")])
        undated = plan_compaction(texts, roles, 60, summarize)
        self.assertEqual(undated.gist_spans, [None])
        with self.assertRaises(ValueError):
            plan_compaction(texts, roles, 60, summarize, spans=spans[:3])

    def test_newest_user_facts_survive_and_the_rest_is_summarized(self):
        texts = [f"{'u' if i % 2 == 0 else 'a'}{i} " + "x" * 36 for i in range(10)]  # 10 tokens each
        roles = ["user" if i % 2 == 0 else "assistant" for i in range(10)]
        seen, summarize = self._capture()
        plan = plan_compaction(texts, roles, 60, summarize)
        # 30 tokens for facts: the three newest user facts, though assistant
        # facts are newer still.
        self.assertEqual(plan.keep, [4, 6, 8])
        self.assertEqual(plan.fold, [0, 1, 2, 3, 5, 7, 9])
        self.assertEqual(sorted(plan.keep + plan.fold), list(range(10)))
        # 30 tokens of gist is one chunk, and a single chunk is the user's.
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0], [f"[user] {texts[0]}", f"[user] {texts[2]}"])
        self.assertEqual(seen[0][1], 30)
        self.assertEqual(plan.gist_roles, [GIST_ROLE])
        kept_tokens = sum(estimate_tokens(texts[i]) for i in plan.keep)
        self.assertLessEqual(kept_tokens + sum(estimate_tokens(g) for g in plan.gists), 60)

    def test_an_earlier_gist_is_folded_again_and_leads_the_new_one(self):
        texts = ["- old fact one\n- old fact two", "u1 " + "x" * 36, "u2 " + "x" * 36,
                 "u3 " + "x" * 36, "u4 " + "x" * 36]
        roles = [GIST_ROLE, "user", "user", "user", "user"]
        seen, summarize = self._capture()
        plan = plan_compaction(texts, roles, 40, summarize)
        self.assertEqual(plan.keep, [3, 4])
        self.assertIn(0, plan.fold, "a gist never survives as it is")
        self.assertEqual(seen[0][0][:2], ["[user] old fact one", "[user] old fact two"])
        self.assertEqual(seen[0][0][2:], [f"[user] {texts[1]}", f"[user] {texts[2]}"])

    def test_assistant_history_gets_a_chunk_once_there_are_four(self):
        texts = [f"{role[0]}{i} " + "x" * 36 for i, role in
                 enumerate(["user", "assistant"] * 20)]
        roles = ["user", "assistant"] * 20
        seen, summarize = self._capture()
        plan = plan_compaction(texts, roles, 256, summarize)
        self.assertEqual(len(seen), 4)
        self.assertTrue(all(t.startswith("[user] ") for chunk, _ in seen[:3] for t in chunk))
        self.assertTrue(all(t.startswith("[assistant] ") for t in seen[3][0]))
        self.assertEqual(plan.gist_roles, [GIST_ROLE] * 3 + [GIST_OTHER_ROLE])

    def test_fold_first_keeps_a_memory_from_surviving(self):
        texts = ["u0 " + "x" * 36, "u1 " + "x" * 36, "u2 " + "x" * 36]
        plan = plan_compaction(texts, ["user"] * 3, 20, None, fold_first=[2])
        self.assertEqual(plan.keep, [0, 1])
        self.assertEqual(plan.fold, [2])

    def test_without_a_summarizer_the_whole_budget_holds_facts(self):
        texts = ["u%d " % i + "x" * 36 for i in range(6)]
        plan = plan_compaction(texts, ["user"] * 6, 40, None)
        self.assertEqual(plan.keep, [2, 3, 4, 5])
        self.assertEqual(plan.gists, [])

    def test_a_failing_summarizer_plans_nothing(self):
        def broken(texts, max_tokens):
            raise RuntimeError("down")

        with self.assertRaises(RuntimeError):
            plan_compaction(["u " + "x" * 80] * 3, ["user"] * 3, 30, broken)


class TestSourceUnits(unittest.TestCase):
    def test_a_short_message_is_kept_whole(self):
        self.assertEqual(excerpt("  I moved to Lisbon.  ", ["moved"], 20), "I moved to Lisbon.")

    def test_a_long_message_keeps_the_sentences_its_facts_share_words_with(self):
        message = ("We talked about many things. The weather was grey all week. "
                   "My sister Ana moved to Porto in May. We also cooked pasta. "
                   "Ana now teaches chemistry there.")
        cut = excerpt(message, ["Ana moved to Porto", "Ana teaches chemistry"], 20)
        self.assertEqual(cut, "My sister Ana moved to Porto in May. ... "
                              "Ana now teaches chemistry there.")

    def test_one_overlong_sentence_is_cut_not_dropped(self):
        cut = excerpt("word " * 200, ["word"], 10)
        self.assertTrue(cut)
        self.assertLessEqual(len(cut) // 4, 10)

    def test_facts_of_one_turn_become_one_unit(self):
        pool = [
            {"id": "a", "text": "Ana moved to Porto", "score": 0.9, "turn_index": 3,
             "source": "Ana moved to Porto and teaches there.", "scope": "u"},
            {"id": "b", "text": "Ana teaches", "score": 0.5, "turn_index": 3,
             "source": "Ana moved to Porto and teaches there.", "scope": "u"},
            {"id": "c", "text": "a gist", "score": 0.7, "turn_index": None, "source": None},
            {"id": "d", "text": "I like tea", "score": 0.6, "turn_index": 4,
             "source": "I like tea.", "scope": "u", "superseded_by": "x", "chain": ["d", "x"]},
        ]
        units = turn_units(pool, "where does Ana live")
        self.assertEqual([u["id"] for u in units], ["a", "c", "d"])
        self.assertEqual(units[0]["ids"], ["a", "b"])
        self.assertEqual(units[0]["facts"], ["Ana moved to Porto", "Ana teaches"])
        self.assertEqual(units[0]["text"], "Ana moved to Porto and teaches there.")
        self.assertNotIn("source", units[0])
        self.assertEqual(units[1]["text"], "a gist")  # no source: as it was
        self.assertEqual(units[2]["superseded_by"], "x")  # every fact of it is stale

    def test_a_turn_is_stale_only_when_all_its_facts_are(self):
        pool = [
            {"id": "a", "text": "I live in Rome", "score": 0.9, "turn_index": 1,
             "source": "I live in Rome and work at a bakery.", "superseded_by": "z"},
            {"id": "b", "text": "I work at a bakery", "score": 0.4, "turn_index": 1,
             "source": "I live in Rome and work at a bakery."},
        ]
        [unit] = turn_units(pool)
        self.assertNotIn("superseded_by", unit)


class TestEventDates(unittest.TestCase):
    SAID = said_on("2023/05/22 (Mon) 14:05")  # a Monday

    def check(self, text, expected):
        self.assertEqual(annotate(text, self.SAID), expected)

    def test_the_date_a_message_was_said(self):
        self.assertEqual(str(self.SAID), "2023-05-22")
        self.assertEqual(str(said_on("2023-05-22T10:00:00Z")), "2023-05-22")
        self.assertEqual(str(said_on(1684749600)), "2023-05-22")
        self.assertIsNone(said_on("last week"))
        self.assertIsNone(said_on(None))

    def test_relative_times_get_the_date_they_point_to(self):
        self.check("I saw the dentist two weeks ago.",
                   "I saw the dentist two weeks ago [week of 2023-05-08].")
        self.check("We moved 3 days ago", "We moved 3 days ago [2023-05-19]")
        self.check("I started a month ago", "I started a month ago [2023-04]")
        self.check("I quit two years ago", "I quit two years ago [2021]")
        self.check("I ran last Saturday", "I ran last Saturday [2023-05-20]")
        self.check("I ran last Monday", "I ran last Monday [2023-05-15]")
        self.check("Party next Friday", "Party next Friday [2023-05-26]")
        self.check("I flew yesterday", "I flew yesterday [2023-05-21]")
        self.check("I went last weekend", "I went last weekend [weekend of 2023-05-20]")
        self.check("It was busy last month", "It was busy last month [2023-04]")
        self.check("It started this year", "It started this year [2023]")
        self.check("I got it on the 15th of last month",
                   "I got it on the 15th of last month [2023-04-15]")

    def test_a_date_without_a_year_gets_the_nearest_year(self):
        self.check("I saw Dr. Smith on March 3rd.", "I saw Dr. Smith on March 3rd [2023-03-03].")
        self.check("My birthday is on the 15th of December",
                   "My birthday is on the 15th of December [2022-12-15]")
        self.check("Symptoms started in late January", "Symptoms started in late January [2023-01]")
        self.check("We go in August", "We go in August [2023-08]")
        self.check("It snowed in December", "It snowed in December [2022-12]")
        self.check("It rained in May", "It rained in May [2023-05]")
        self.assertEqual(annotate("in February", said_on("2023-11-20")), "in February [2024-02]")
        self.assertEqual(annotate("in May", said_on("2023-11-20")), "in May [2023-05]")

    def test_what_is_already_a_date_or_cannot_be_placed_is_left_alone(self):
        for text in ("I earned it in May 2016.", "On 2023-02-10 we met.", "I did it today.",
                     "I learned it in two weeks.", "a few weeks ago", "March 3, 2021",
                     "I may go there", "back in 2019"):
            self.check(text, text)
        self.assertEqual(annotate("two weeks ago", None), "two weeks ago")


class _Call:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = type("F", (), {"name": name, "arguments": arguments})()


class _ToolClient:
    """Asks for one calculation, then answers with its result."""

    def __init__(self):
        self.requests = []
        self.chat = type("C", (), {"completions": self})()

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if len(self.requests) == 1:
            message = type("M", (), {"content": "", "tool_calls": [
                _Call("c1", "calculate", '{"expression": "120 + 85 + 1,200"}')]})()
        else:
            result = kwargs["messages"][-1]["content"]
            message = type("M", (), {"content": f"1. ...\nAnswer: ${result}",
                                     "tool_calls": None})()
        usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 3})()
        return type("R", (), {"choices": [type("Ch", (), {"message": message})()],
                              "usage": usage})()


class TestAnsweringTools(unittest.TestCase):
    def test_calculate(self):
        self.assertEqual(calculate("(120 + 85) * 2"), "410")
        self.assertEqual(calculate("1,200 + 300"), "1500")
        self.assertEqual(calculate("max(3, 5) + round(2.456, 1)"), "7.5")
        self.assertEqual(calculate("sum([10, 20, 30]) / 3"), "20")
        self.assertEqual(calculate("10 / 4"), "2.5")
        self.assertEqual(calculate("$45 * 3"), "135")
        self.assertTrue(calculate("1 / 0").startswith("error"))
        self.assertTrue(calculate("__import__('os')").startswith("error"))
        self.assertTrue(calculate("2 ** 1000").startswith("error"))
        self.assertTrue(calculate("x + 1").startswith("error"))

    def test_dates_and_counts(self):
        self.assertEqual(days_between("2023-03-01", "2023-05-22"),
                         "82 days (11 weeks and 5 days; 2 whole months; 0.22 years)")
        self.assertTrue(days_between("2023-05-22", "2023-03-01").startswith("-82 days"))
        self.assertTrue(days_between("soon", "2023-03-01").startswith("error"))
        self.assertEqual(add_days("2023-05-22", -14), "2023-05-08 (Monday)")
        self.assertEqual(count_distinct(["Yoga", "yoga!", "Pilates", " spin "]),
                         "3 distinct: Yoga; Pilates; spin")
        self.assertEqual(run_tool("calculate", '{"expression": "2+2"}'), "4")
        self.assertTrue(run_tool("calculate", "not json").startswith("error"))
        self.assertTrue(run_tool("delete_everything", "{}").startswith("error"))

    def test_the_model_calls_a_tool_then_answers(self):
        client = _ToolClient()
        out = answer_with_tools(client, "m", "How much did I spend?",
                                ["[2023-05-01] I paid $120", "[2023-05-02] I paid $85"])
        self.assertEqual(out["answer"], "$1405")
        self.assertEqual(out["requests"], 2)
        self.assertEqual(out["tool_calls"],
                         [("calculate", '{"expression": "120 + 85 + 1,200"}', "1405")])
        self.assertEqual(out["input_tokens"], 20)
        first, second = client.requests
        self.assertEqual(first["tool_choice"], "auto")
        self.assertIn("calculate", first["messages"][0]["content"])
        self.assertEqual(second["messages"][-1], {"role": "tool", "tool_call_id": "c1",
                                                  "content": "1405"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
