"""Unit tests for the OpenAI-backed backends (extractor, embedder, summarizer,
supersession verifier) and the retry policy they share.

No network and no API key: each backend is given a fake client. The tests pin
behaviour under failure, which is where these backends used to lose data or
hang::

    python -m unittest tsm.tests.test_backends -v
"""

import json
import logging
import os
import pickle
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from tsm._retry import QuotaExhausted, call_with_retries, is_permanent, is_quota_exhausted
from tsm.budget import estimate_tokens
from tsm.embedders import OpenAIEmbedder
from tsm.extractors import OpenAIExtractor, message_pieces, statements
from tsm.gist import OpenAIGistSummarizer
from tsm.verification import LLMVerifier

logging.getLogger("tsm").setLevel(logging.CRITICAL)  # expected warnings stay quiet


class ApiError(Exception):
    """Stands in for an OpenAI SDK error: carries an HTTP status."""

    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code} with details that must not be echoed")
        self.status_code = status_code


def chat_reply(content, finish_reason="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content),
                                 finish_reason=finish_reason)],
        usage=None,
    )


class ScriptedChat:
    """A fake chat client that plays back replies (or raises exceptions) in order."""

    def __init__(self, *script):
        self.script = list(script)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


class FakeEmbeddings:
    """A fake embeddings client returning a deterministic vector per text."""

    def __init__(self, dim):
        self.dim = dim
        self.requests = []
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        size = kwargs.get("dimensions", self.dim)
        data = [SimpleNamespace(embedding=[float(len(text))] + [0.0] * (size - 1))
                for text in kwargs["input"]]
        return SimpleNamespace(data=data)


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="tsm_backend_")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)


class TestRetryPolicy(unittest.TestCase):
    def setUp(self):
        self.log = logging.getLogger("tsm.test")
        self.slept = []

    def run_with(self, *outcomes, max_retries=4):
        script = list(outcomes)

        def fn():
            step = script.pop(0)
            if isinstance(step, Exception):
                raise step
            return step

        return call_with_retries(fn, "the call", max_retries, self.log, sleep=self.slept.append)

    def test_transient_errors_are_retried_with_backoff(self):
        self.assertEqual(self.run_with(ApiError(429), ApiError(503), TimeoutError(), "ok"), "ok")
        self.assertEqual(self.slept, [5.0, 10.0, 20.0])

    def test_permanent_errors_fail_at_once(self):
        for status in (400, 401, 403, 404, 413, 422):
            self.slept.clear()
            with self.assertRaises(RuntimeError) as ctx:
                self.run_with(ApiError(status), "never reached")
            self.assertEqual(self.slept, [], f"HTTP {status} must not be retried")
            self.assertIn(f"HTTP {status}", str(ctx.exception))
            self.assertIsInstance(ctx.exception.__cause__, ApiError)
            # The provider's message text (which can quote the request) is not repeated.
            self.assertNotIn("details that must not be echoed", str(ctx.exception))
        self.assertTrue(is_permanent(ApiError(401)))
        self.assertFalse(is_permanent(ApiError(429)))
        self.assertFalse(is_permanent(ValueError("no status")))

    def test_an_account_out_of_credit_fails_at_once(self):
        empty = ApiError(429)
        empty.code = "credit_balance_exhausted"
        empty.body = {"code": "credit_balance_exhausted", "type": "insufficient_quota"}
        self.assertTrue(is_quota_exhausted(empty))
        with self.assertRaises(QuotaExhausted) as ctx:
            self.run_with(empty, "never reached")
        self.assertEqual(self.slept, [], "an empty account is not a rate limit")
        self.assertIn("out of credit", str(ctx.exception))
        self.assertNotIn("details that must not be echoed", str(ctx.exception))
        nested = ApiError(429)
        nested.body = {"error": {"type": "insufficient_quota"}}
        self.assertTrue(is_quota_exhausted(nested))
        self.assertFalse(is_quota_exhausted(ApiError(429)))  # a plain rate limit

    def test_the_verifier_stops_instead_of_leaving_pairs_unverified(self):
        empty = ApiError(429)
        empty.code = "insufficient_quota"
        verifier = LLMVerifier(client=ScriptedChat(empty), batch_size=1)
        with self.assertRaises(QuotaExhausted):
            verifier.judge([("I live in Lisbon.", "I live in Porto."),
                            ("I like tea.", "I like coffee.")])

    def test_verifier_batches_run_in_parallel_and_land_in_order(self):
        import threading

        lock, seen = threading.Lock(), []

        def create(**kwargs):
            with lock:
                seen.append(kwargs["messages"][1]["content"])
            return chat_reply("1: moved REPLACES\n2: second item KEEPS")

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        verifier = LLMVerifier(client=client, batch_size=2, workers=4)
        pairs = [(f"old {i}", f"new {i}") for i in range(10)]
        self.assertEqual(verifier.judge(pairs), ["replaces", "keeps"] * 5)
        self.assertEqual(verifier.calls, 5)
        self.assertEqual(len(seen), 5)
        # A second ask is served from the cache, in parallel or not.
        self.assertEqual(verifier.judge(pairs), ["replaces", "keeps"] * 5)
        self.assertEqual(verifier.calls, 5)

    def test_gives_up_after_the_last_attempt_without_a_trailing_sleep(self):
        with self.assertRaises(RuntimeError) as ctx:
            self.run_with(ApiError(500), ApiError(500), ApiError(500), max_retries=3)
        self.assertEqual(self.slept, [5.0, 10.0], "no sleep after the final failure")
        self.assertIn("after 3 attempts", str(ctx.exception))


class TestExtractor(TempDirTest):
    def extractor(self, *script, **kwargs):
        self.client = ScriptedChat(*script)
        return OpenAIExtractor(client=self.client, cache_dir=self.dir, **kwargs)

    def test_well_formed_reply_is_returned_and_cached(self):
        ex = self.extractor(chat_reply('{"facts": ["I live in Lisbon", " ", "I play guitar"]}'))
        self.assertEqual(ex.extract_facts("hello"), ["I live in Lisbon", "I play guitar"])
        self.assertEqual(ex.extract_facts("hello"), ["I live in Lisbon", "I play guitar"])
        self.assertEqual(len(self.client.requests), 1, "second call is served from the cache")

    def test_a_message_with_no_facts_is_cached_as_empty(self):
        ex = self.extractor(chat_reply('{"facts": []}'))
        self.assertEqual(ex.extract_facts("thanks!"), [])
        self.assertEqual(ex.extract_facts("thanks!"), [])
        self.assertEqual(len(self.client.requests), 1)

    def test_cut_off_reply_is_retried_with_a_larger_budget(self):
        ex = self.extractor(
            chat_reply('{"facts": ["fact one", "fact tw', finish_reason="length"),
            chat_reply('{"facts": ["fact one", "fact two", "fact three"]}'),
            max_tokens=400,
        )
        self.assertEqual(ex.extract_facts("a long message"),
                         ["fact one", "fact two", "fact three"])
        self.assertEqual([r["max_tokens"] for r in self.client.requests], [400, 1600])

    def test_message_is_kept_when_extraction_cannot_be_parsed(self):
        cases = {
            "cut off twice": [chat_reply('{"facts": ["a', "length"), chat_reply('{"facts": ["a', "length")],
            "not JSON": [chat_reply("Sure! Here are the facts:"), chat_reply("Sure!")],
            "refusal (no content)": [chat_reply(None), chat_reply(None)],
            "wrong shape": [chat_reply('{"facts": "one string"}'), chat_reply('[1, 2]')],
        }
        for name, script in cases.items():
            ex = self.extractor(*script)
            message = f"  the user said something important ({name})  "
            self.assertEqual(ex.extract_facts(message), [message.strip()],
                             f"{name}: the message itself must be stored")
            # Not cached: a later run gets another chance.
            self.client.script.append(chat_reply('{"facts": ["recovered"]}'))
            self.assertEqual(ex.extract_facts(message), ["recovered"], name)

    def test_by_default_only_the_first_prompt_is_asked(self):
        # The two recovery steps are opt-in: they did not pay when judged.
        message = "Any tips for a half marathon? I ran my first 10K in 58 minutes last month."
        ex = self.extractor(chat_reply('{"facts": []}'))
        self.assertEqual(ex.extract_facts(message), [])
        self.assertEqual(ex.extract_facts(message), [])
        self.assertEqual(len(self.client.requests), 1)
        self.assertTrue(ex.is_cached(message))

    def test_a_message_the_first_prompt_finds_nothing_in_gets_a_second_look(self):
        # A statement made while asking for something else: a model told to
        # extract asserted facts often returns none, and the memory was lost.
        message = "Any tips for a half marathon? I ran my first 10K in 58 minutes last month."
        ex = self.extractor(chat_reply('{"facts": []}'),
                            chat_reply('{"facts": ["The speaker ran a 10K in 58 minutes."]}'),
                            second_look=True)
        self.assertEqual(ex.extract_facts(message), ["The speaker ran a 10K in 58 minutes."])
        self.assertEqual(ex.extract_facts(message), ["The speaker ran a 10K in 58 minutes."])
        self.assertEqual(len(self.client.requests), 2, "both replies are cached")
        first, second = (r["messages"][0]["content"] for r in self.client.requests)
        self.assertNotEqual(first, second, "the second request uses the other prompt")
        self.assertTrue(ex.is_cached(message))
        self.assertFalse(ex.is_cached("Another message that was never extracted."))
        # A message with facts is asked about once.
        ex.client = None
        once = self.extractor(chat_reply('{"facts": ["The speaker lives in Lisbon."]}'))
        self.assertEqual(once.extract_facts("I live in Lisbon."), ["The speaker lives in Lisbon."])
        self.assertEqual(len(self.client.requests), 1)

    def test_what_a_message_states_is_kept_when_both_prompts_find_nothing(self):
        message = "Any tips for a half marathon? I ran my first 10K in 58 minutes last month."
        ex = self.extractor(chat_reply('{"facts": []}'), chat_reply('{"facts": []}'),
                            second_look=True, keep_unextracted=True)
        self.assertEqual(ex.extract_facts(message),
                         ["I ran my first 10K in 58 minutes last month."],
                         "what was said, without what was asked")
        self.assertEqual(ex.extract_facts(message),
                         ["I ran my first 10K in 58 minutes last month."])
        self.assertEqual(len(self.client.requests), 2, "the empty replies are cached")
        ex.flush_cache()
        # The same cache, no request: each step is a switch of its own.
        for switches, stored in (({"second_look": True}, []),
                                 ({"keep_unextracted": True},
                                  ["I ran my first 10K in 58 minutes last month."]),
                                 ({}, [])):
            self.assertEqual(self.extractor(**switches).extract_facts(message), stored, switches)
            self.assertEqual(self.client.requests, [], switches)
        # A message that only asks states nothing, so nothing is stored for it.
        asks = self.extractor(chat_reply('{"facts": []}'), chat_reply('{"facts": []}'),
                              second_look=True, keep_unextracted=True)
        self.assertEqual(asks.extract_facts("What is the capital of France?"), [])
        self.assertEqual(statements("Good to know. Can you say more?\nI use a Yamaha P-125."),
                         ["Good to know.\nI use a Yamaha P-125."])
        self.assertEqual(statements("Is it far? How long?"), [])
        # A numbered question goes with its number: no "1." is left behind.
        self.assertEqual(
            statements("I can help with that. A few details first:\n"
                       "1. Which city do you fly from?\n2. Which dates?\n"
                       "I will look for deals after that."),
            ["I can help with that. A few details first:\nI will look for deals after that."])

    def test_a_long_message_is_kept_in_pieces_that_fit(self):
        lines = [f"{n}. Parameter {n} sets one aspect of the reply, with an example of its use"
                 for n in range(1, 41)]
        message = "Sure, here are the parameters:\n" + "\n".join(lines)
        pieces = message_pieces(message, max_tokens=60)
        self.assertGreater(len(pieces), 5)
        self.assertTrue(all(estimate_tokens(piece) <= 60 for piece in pieces))
        self.assertEqual("\n".join(pieces).split(), message.split(),
                         "nothing left out, nothing reworded, in order")
        for line in lines:  # a line that fits is never cut in two
            self.assertEqual(sum(line in piece for piece in pieces), 1, line)
        # One sentence longer than a piece is cut between words.
        endless = " ".join(f"word{n}" for n in range(300))
        cut = message_pieces(endless, max_tokens=40)
        self.assertTrue(all(estimate_tokens(piece) <= 40 for piece in cut))
        self.assertEqual(" ".join(cut), endless)
        # A list number is not cut off from the item it numbers.
        item = "3. Regular training: " + " ".join(f"word{n}" for n in range(30)) + ". It helps."
        numbered = message_pieces("Some examples:\n" + item, max_tokens=40)
        self.assertTrue(all(estimate_tokens(piece) <= 40 for piece in numbered))
        self.assertEqual(numbered[0], "Some examples:")
        self.assertTrue(numbered[1].startswith("3. Regular training: word0"), numbered[1])
        self.assertEqual("\n".join(numbered).split(), ("Some examples:\n" + item).split())
        self.assertEqual(message_pieces("  short enough  "), ["short enough"])
        self.assertEqual(message_pieces("   "), [])

    def test_a_reply_to_another_prompt_is_not_served(self):
        # Caches written before the prompt changed were keyed by the message
        # alone; what that prompt found (here: nothing) must not be reused.
        message = "I moved to Lisbon last week, can you suggest a dentist?"
        with open(os.path.join(self.dir, "extract_gpt-4o-mini.json"), "w", encoding="utf-8") as f:
            json.dump({message: []}, f)
        ex = self.extractor(chat_reply('{"facts": ["The speaker moved to Lisbon last week."]}'))
        self.assertEqual(ex.extract_facts(message), ["The speaker moved to Lisbon last week."])
        self.assertEqual(len(self.client.requests), 1)
        self.assertNotEqual(ex.cache_key(message), message)
        self.assertNotEqual(ex.cache_key(message), ex.cache_key(message, ["earlier message"]))

    def test_cache_survives_a_reload_and_flush_is_idempotent(self):
        ex = self.extractor(chat_reply('{"facts": ["kept"]}'))
        ex.extract_facts("message")
        ex.flush_cache()
        ex.flush_cache()
        again = OpenAIExtractor(client=ScriptedChat(), cache_dir=self.dir)
        self.assertEqual(again.extract_facts("message"), ["kept"])

    def test_rejected_key_fails_fast(self):
        ex = self.extractor(ApiError(401))
        with self.assertRaises(RuntimeError) as ctx:
            ex.extract_facts("message")
        self.assertIn("HTTP 401", str(ctx.exception))
        self.assertEqual(len(self.client.requests), 1)


class TestEmbedder(TempDirTest):
    def test_encode_caches_and_survives_reload(self):
        client = FakeEmbeddings(1536)
        emb = OpenAIEmbedder(client=client, cache_dir=self.dir)
        out = emb.encode(["aa", "bbbb", "aa"])
        self.assertEqual(out.shape, (3, 1536))
        self.assertEqual([row[0] for row in out], [2.0, 4.0, 2.0])
        self.assertEqual(client.requests[0]["input"], ["aa", "bbbb"], "duplicates embedded once")
        self.assertNotIn("dimensions", client.requests[0])
        emb.flush()

        reloaded = FakeEmbeddings(1536)
        again = OpenAIEmbedder(client=reloaded, cache_dir=self.dir)
        self.assertEqual(again.encode("bbbb")[0], 4.0)
        self.assertEqual(reloaded.requests, [], "served from the cache file")

    def test_custom_dimension_is_requested_from_the_api(self):
        client = FakeEmbeddings(1536)
        emb = OpenAIEmbedder(dim=512, client=client, cache_dir=self.dir)
        self.assertEqual(emb.dimension, 512)
        self.assertEqual(emb.encode(["hello"]).shape, (1, 512))
        self.assertEqual(client.requests[0]["dimensions"], 512)
        # Shortened vectors do not share a cache file with full-size ones.
        emb.flush()
        full = OpenAIEmbedder(client=FakeEmbeddings(1536), cache_dir=self.dir)
        self.assertEqual(full.encode(["hello"]).shape, (1, 1536))

    def test_wrong_size_from_the_api_is_reported(self):
        emb = OpenAIEmbedder(model="some-other-model", client=FakeEmbeddings(1024),
                             cache_dir=self.dir)
        with self.assertRaises(RuntimeError) as ctx:
            emb.encode(["hello"])
        self.assertIn("1536", str(ctx.exception))
        declared = OpenAIEmbedder(model="some-other-model", dim=1024,
                                  client=FakeEmbeddings(1024), cache_dir=self.dir)
        self.assertEqual(declared.encode(["hello"]).shape, (1, 1024))

    def test_cache_file_cannot_run_code(self):
        marker = os.path.join(self.dir, "executed")

        class Payload:
            def __reduce__(self):
                return (open, (marker, "w"))

        path = os.path.join(self.dir, "emb_text-embedding-3-small.pkl")
        with open(path, "wb") as f:
            pickle.dump({"hello": Payload()}, f)

        client = FakeEmbeddings(1536)
        emb = OpenAIEmbedder(client=client, cache_dir=self.dir)
        self.assertFalse(os.path.exists(marker), "loading the cache executed its payload")
        # The untrusted file is ignored and the embedder still works.
        self.assertEqual(emb.encode("hello")[0], 5.0)
        self.assertEqual(len(client.requests), 1)

    def test_a_real_cache_file_still_loads(self):
        path = os.path.join(self.dir, "emb_text-embedding-3-small.pkl")
        vec = np.arange(1536, dtype=np.float32)
        with open(path, "wb") as f:
            pickle.dump({"hello": vec, "bye": vec * 2}, f)
        client = FakeEmbeddings(1536)
        emb = OpenAIEmbedder(client=client, cache_dir=self.dir)
        np.testing.assert_array_equal(emb.encode("bye"), vec * 2)
        self.assertEqual(client.requests, [])


class TestSummarizerRetries(unittest.TestCase):
    def test_rejected_key_fails_fast(self):
        client = ScriptedChat(ApiError(401))
        summarizer = OpenAIGistSummarizer(client=client)
        with self.assertRaises(RuntimeError) as ctx:
            summarizer(["[user] fact one", "[user] fact two"])
        self.assertIn("HTTP 401", str(ctx.exception))
        self.assertEqual(len(client.requests), 1)


class TestLLMVerifier(TempDirTest):
    """The chat-model supersession verifier: how replies are read, what is
    cached, and that a failure never commits (hides) anything."""

    TEXTS = {
        "o1": "I live in Lisbon.", "n1": "I live in Porto.",
        "o2": "I'm allergic to cats.", "n2": "Cats trigger my allergies.",
        "o3": "My sister lives in Vancouver.", "n3": "My brother lives in Vancouver.",
    }
    PAIRS = [("o1", "n1", "contradiction", 0.9), ("o2", "n2", "refinement", 0.8),
             ("o3", "n3", "contradiction", 0.7)]

    def verifier(self, *script, **kwargs):
        self.client = ScriptedChat(*script)
        kwargs.setdefault("max_retries", 1)
        return LLMVerifier(client=self.client, **kwargs)

    def test_reads_verdicts_however_the_model_formats_them(self):
        reply = ("<think>pair 1: KEEPS? no, they moved.</think>\n"
                 "1: REPLACES\n2. **same**\n3) Keeps")
        verifier = self.verifier(chat_reply(reply), accept=("replaces", "same"))
        self.assertEqual(verifier.verify(self.PAIRS, self.TEXTS),
                         [("o1", "n1", "contradiction"), ("o2", "n2", "refinement")])
        request = self.client.requests[0]
        self.assertEqual(request["temperature"], 0.0)
        asked = request["messages"][-1]["content"]
        self.assertIn("OLDER: I live in Lisbon.\nNEWER: I live in Porto.", asked)
        # By default a reworded repeat ("same") retires nothing.
        default = self.verifier(chat_reply(reply))
        self.assertEqual(default.verify(self.PAIRS, self.TEXTS), [("o1", "n1", "contradiction")])
        # The verdict is the last one on a line that starts with a pair
        # number; reasoning before it may mention the other words.
        wordy = self.verifier(chat_reply(
            "Pair 1: not the SAME city, they moved. REPLACES\n"
            "2 - both sentences describe one allergy: same\n"
            "Note: KEEPS is the safe answer.\n"
            "3: different people, both true. KEEPS"), accept=("replaces", "same"))
        self.assertEqual(wordy.verify(self.PAIRS, self.TEXTS),
                         [("o1", "n1", "contradiction"), ("o2", "n2", "refinement")])

    def test_unanswered_pairs_are_not_committed_and_are_asked_again(self):
        verifier = self.verifier(chat_reply("1: REPLACES\nI am not sure about the others."),
                                 chat_reply("1: KEEPS\n2: KEEPS"))
        self.assertEqual(verifier.verify(self.PAIRS, self.TEXTS), [("o1", "n1", "contradiction")])
        # Second round: the answered pair comes from the cache, only the two
        # the model skipped are sent.
        self.assertEqual(verifier.verify(self.PAIRS, self.TEXTS), [("o1", "n1", "contradiction")])
        second = self.client.requests[1]["messages"][-1]["content"]
        self.assertNotIn("Lisbon", second)
        self.assertEqual(second.count("OLDER:"), 2)

    def test_requests_are_batched_and_verdicts_cached_on_disk(self):
        verifier = self.verifier(chat_reply("1: REPLACES\n2: SAME"), chat_reply("1: KEEPS"),
                                 batch_size=2, cache_dir=self.dir, accept=("replaces", "same"))
        first = verifier.verify(self.PAIRS, self.TEXTS)
        self.assertEqual(verifier.calls, 2)
        self.assertEqual(len(first), 2)
        # A later process: nothing is asked (or paid for) twice.
        again = self.verifier(cache_dir=self.dir,  # empty script: any request would fail
                              accept=("replaces", "same"))
        self.assertEqual(again.verify(self.PAIRS, self.TEXTS), first)
        self.assertEqual(again.calls, 0)

    def test_an_outage_commits_nothing_and_caches_nothing(self):
        for failure in (ApiError(503), ApiError(401), TimeoutError()):
            verifier = self.verifier(failure, chat_reply("1: REPLACES\n2: SAME\n3: KEEPS"),
                                     cache_dir=self.dir)
            self.assertEqual(verifier.verify(self.PAIRS, self.TEXTS), [],
                             "a failed check must not retire a fact")
            # Nothing was remembered as a verdict: the next attempt asks again.
            self.assertEqual(verifier.verify(self.PAIRS, self.TEXTS),
                             [("o1", "n1", "contradiction")])
            os.remove(os.path.join(self.dir, "verdicts_gpt-4o-mini.json"))

    def test_an_older_fact_is_superseded_by_one_newer_fact_only(self):
        texts = dict(self.TEXTS, n9="I live in Madrid.")
        pairs = [("o1", "n1", "contradiction", 0.6), ("o1", "n9", "contradiction", 0.9)]
        verifier = self.verifier(chat_reply("1: REPLACES\n2: REPLACES"))
        self.assertEqual(verifier.verify(pairs, texts), [("o1", "n9", "contradiction")])

    def test_pairs_without_stored_text_are_skipped(self):
        verifier = self.verifier(chat_reply("1: REPLACES"))
        pairs = [("gone", "n1", "contradiction", 0.9), ("o1", "n1", "contradiction", 0.8)]
        self.assertEqual(verifier.verify(pairs, self.TEXTS), [("o1", "n1", "contradiction")])

    def test_needs_a_key_or_an_endpoint(self):
        saved = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with self.assertRaises(RuntimeError) as ctx:
                LLMVerifier()
            self.assertIn("OPENAI_API_KEY", str(ctx.exception))
        finally:
            if saved is not None:
                os.environ["OPENAI_API_KEY"] = saved


if __name__ == "__main__":
    unittest.main(verbosity=2)
