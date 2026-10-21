"""Unit tests for embedder handling: query/document encoding and database identity.

Like the rest of the suite these use fakes: no model downloads, no torch, no
keys. A stub stands in for the ``SentenceTransformer`` object, and the engine
is a real one in a temp directory.
"""

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

import numpy as np

from tsm import EmbedderMismatchError, Memory
from tsm.embedders import (OpenAIEmbedder, SentenceTransformerEmbedder, _model_defaults,
                           _resolve_dtype)
from tsm.gist import ExtractiveGistSummarizer
from tsm.identity import IDENTITY_FILE, read_identity
from tsm.interfaces import embed_documents, embed_query, embedder_fingerprint
from tsm.tests.test_memory import FakeEmbedder, FakeExtractor, MemoryTestBase


class RecordingEmbedder(FakeEmbedder):
    """Asymmetric: queries and documents go through different methods."""

    def __init__(self, dim=1024):
        super().__init__(dim)
        self.calls = []

    def encode(self, texts):
        self.calls.append(("encode", texts))
        return super().encode(texts)

    def encode_document(self, texts):
        self.calls.append(("document", texts))
        return super().encode(texts)

    def encode_query(self, texts):
        self.calls.append(("query", texts))
        return super().encode(texts)

    def kinds(self):
        return [kind for kind, _ in self.calls]


class OtherEmbedder(FakeEmbedder):
    """Same dimension as FakeEmbedder, a different embedder."""


class TestQueryDocumentRouting(MemoryTestBase):
    def make(self, **kwargs):
        self.embedder = RecordingEmbedder()
        self.mem = Memory(self.db_path, embedder=self.embedder, extractor=FakeExtractor(),
                          **kwargs)
        return self.mem

    def test_helpers_fall_back_to_encode(self):
        plain = FakeEmbedder()
        self.assertEqual(embed_query(plain, "a b").shape, (plain.dimension,))
        self.assertEqual(embed_documents(plain, ["a b", "c"]).shape, (2, plain.dimension))

    def test_stored_facts_are_documents_and_questions_are_queries(self):
        mem = self.make()
        mem.add([{"role": "user", "content": "I live in Lisbon. I have a cat named Miso"}],
                user_id="u")
        self.assertEqual(self.embedder.kinds(), ["document"])
        self.assertEqual(len(self.embedder.calls[0][1]), 2)

        self.embedder.calls.clear()
        mem.recall("where do I live", user_id="u")
        self.assertEqual(self.embedder.kinds(), ["query"])

    def test_the_packer_compares_stored_text_as_documents(self):
        mem = self.make()
        mem.add([{"role": "user", "content": "I live in Lisbon. I have a cat named Miso"}],
                user_id="u")
        self.embedder.calls.clear()
        results = mem.recall("where do I live", user_id="u", token_budget=100)
        self.assertTrue(results)
        self.assertEqual(self.embedder.kinds(), ["query", "document"])

    def test_gists_are_documents(self):
        mem = self.make(gist_summarizer=ExtractiveGistSummarizer())
        self.embedder.calls.clear()
        self.assertIsNotNone(mem._compress_gist(["I live in Lisbon", "I have a cat"]))
        self.assertEqual(self.embedder.kinds(), ["document"])


class TestEngineDefaults(MemoryTestBase):
    """An embedder can say which engine settings fit its cosine geometry."""

    class Tuned(FakeEmbedder):
        engine_defaults = {"refinement_cosine_threshold": 0.93,
                           "contradiction_cosine_threshold": 0.88}

    def open(self, embedder, **kwargs):
        mem = Memory(self.db_path, embedder=embedder, extractor=FakeExtractor(), **kwargs)
        self.addCleanup(mem.close)
        return mem

    def test_applied_over_the_profile(self):
        config = self.open(self.Tuned()).engine_config
        self.assertEqual(config["refinement_cosine_threshold"], 0.93)
        self.assertEqual(config["contradiction_cosine_threshold"], 0.88)
        self.assertEqual(config["graph_weight"], 0.5)  # the rest of the profile stays

    def test_an_explicit_setting_wins(self):
        mem = self.open(self.Tuned(), refinement_cosine_threshold=0.9)
        self.assertEqual(mem.engine_config["refinement_cosine_threshold"], 0.9)
        self.assertEqual(mem.engine_config["contradiction_cosine_threshold"], 0.88)

    def test_not_applied_without_the_profile(self):
        mem = self.open(self.Tuned(), profile=None)
        self.assertNotEqual(mem.engine_config.get("refinement_cosine_threshold"), 0.93)

    def test_an_embedder_without_them_leaves_the_profile_alone(self):
        config = self.open(FakeEmbedder()).engine_config
        self.assertEqual(config["refinement_cosine_threshold"], 0.85)
        self.assertEqual(config["contradiction_cosine_threshold"], 0.75)

    def test_embedding_gemma_declares_its_own_belief_thresholds(self):
        gemma = SentenceTransformerEmbedder("google/embeddinggemma-2", model=StubWithQueryMethods())
        self.assertEqual(gemma.engine_defaults, {"refinement_cosine_threshold": 0.92,
                                                 "contradiction_cosine_threshold": 0.86})
        other = SentenceTransformerEmbedder("some/model", model=StubWithQueryMethods())
        self.assertEqual(other.engine_defaults, {})


class TestEmbedderIdentity(MemoryTestBase):
    def open(self, embedder, **kwargs):
        mem = Memory(self.db_path, embedder=embedder, extractor=FakeExtractor(), **kwargs)
        self.addCleanup(mem.close)
        return mem

    def test_a_new_database_records_its_embedder(self):
        self.open(FakeEmbedder()).close()
        record = read_identity(self.db_path)
        self.assertEqual(record["fingerprint"], embedder_fingerprint(FakeEmbedder(), 1024))
        self.assertEqual(record["dimension"], 1024)

    def test_the_fallback_fingerprint_ignores_the_module(self):
        # One class is `__main__.X` in a script and `pkg.mod.X` when imported.
        as_script = type("X", (FakeEmbedder,), {"__module__": "__main__"})
        imported = type("X", (FakeEmbedder,), {"__module__": "pkg.mod"})
        self.assertEqual(embedder_fingerprint(as_script(), 1024),
                         embedder_fingerprint(imported(), 1024))
        self.assertNotEqual(embedder_fingerprint(as_script(), 1024),
                            embedder_fingerprint(as_script(), 768))

    def test_the_same_embedder_reopens(self):
        mem = self.open(FakeEmbedder())
        mem.add("I live in Lisbon", user_id="u")
        mem.close()
        again = self.open(FakeEmbedder())
        self.assertTrue(again.recall("where", user_id="u"))

    def test_another_embedder_is_refused_and_the_database_stays_usable(self):
        mem = self.open(FakeEmbedder())
        mem.add("I live in Lisbon", user_id="u")
        mem.close()
        with self.assertRaises(EmbedderMismatchError) as caught:
            self.open(OtherEmbedder())
        self.assertIn("allow_embedder_change", str(caught.exception))
        self.assertIsInstance(caught.exception, ValueError)
        # The refused open released the database: the right embedder opens it at once.
        again = self.open(FakeEmbedder())
        self.assertEqual(again.engine.record_count(), 1)

    def test_a_declared_fingerprint_is_what_counts(self):
        class Declared(FakeEmbedder):
            def __init__(self, fingerprint):
                super().__init__()
                self.fingerprint = fingerprint

        mem = self.open(Declared("model-a|dim=1024"))
        mem.add("I live in Lisbon", user_id="u")
        mem.close()
        self.open(Declared("model-a|dim=1024")).close()
        with self.assertRaises(EmbedderMismatchError):
            self.open(Declared("model-b|dim=1024"))

    def test_allow_embedder_change_replaces_the_record(self):
        mem = self.open(FakeEmbedder())
        mem.add("I live in Lisbon", user_id="u")
        mem.close()
        self.open(OtherEmbedder(), allow_embedder_change=True).close()
        self.assertEqual(read_identity(self.db_path)["fingerprint"],
                         embedder_fingerprint(OtherEmbedder(), 1024))
        self.open(OtherEmbedder()).close()  # and it is now the one that is checked

    def test_an_empty_database_takes_any_embedder(self):
        self.open(FakeEmbedder()).close()
        self.open(OtherEmbedder()).close()
        self.assertEqual(read_identity(self.db_path)["fingerprint"],
                         embedder_fingerprint(OtherEmbedder(), 1024))

    def test_a_database_without_a_record_is_adopted(self):
        mem = self.open(FakeEmbedder())
        mem.add("I live in Lisbon", user_id="u")
        mem.close()
        os.remove(os.path.join(self.db_path, IDENTITY_FILE))  # written before tsm kept one
        self.open(OtherEmbedder()).close()
        self.assertEqual(read_identity(self.db_path)["fingerprint"],
                         embedder_fingerprint(OtherEmbedder(), 1024))

    def test_an_unreadable_record_is_ignored_and_rewritten(self):
        mem = self.open(FakeEmbedder())
        mem.add("I live in Lisbon", user_id="u")
        mem.close()
        with open(os.path.join(self.db_path, IDENTITY_FILE), "w", encoding="utf-8") as fh:
            fh.write("{ not json")
        self.open(FakeEmbedder()).close()
        self.assertEqual(read_identity(self.db_path)["fingerprint"],
                         embedder_fingerprint(FakeEmbedder(), 1024))

    def test_the_record_is_plain_json_next_to_the_engine_files(self):
        self.open(FakeEmbedder()).close()
        with open(os.path.join(self.db_path, IDENTITY_FILE), encoding="utf-8") as fh:
            self.assertEqual(set(json.load(fh)),
                             {"fingerprint", "dimension", "written_by", "written_at"})
        self.assertFalse(os.path.exists(os.path.join(self.db_path, IDENTITY_FILE + ".tmp")))


class StubSentenceTransformer:
    """What SentenceTransformerEmbedder needs from a SentenceTransformer."""

    def __init__(self, dim=8, prompts=None, nan=False):
        self.dim = dim
        self.prompts = prompts or {}
        self.nan = nan
        self.max_seq_length = 256
        self.calls = []
        self.device = "cpu"

    def get_embedding_dimension(self):
        return self.dim

    def _embed(self, kind, texts, kwargs):
        self.calls.append((kind, list(texts), kwargs))
        out = np.full((len(texts), self.dim), 3.0, dtype=np.float32)  # not unit length
        out[:, 0] = [(len(t) % 5) + 1 for t in texts]
        if self.nan:
            out[:] = np.nan
        return out

    def encode(self, texts, **kwargs):
        return self._embed("encode", texts, kwargs)


class StubWithQueryMethods(StubSentenceTransformer):
    """A sentence-transformers 5+ model: it has encode_query / encode_document."""

    def encode_document(self, texts, **kwargs):
        return self._embed("document", texts, kwargs)

    def encode_query(self, texts, **kwargs):
        return self._embed("query", texts, kwargs)


class TestSentenceTransformerEmbedder(unittest.TestCase):
    def make(self, **kwargs):
        stub = kwargs.pop("stub", None) or StubWithQueryMethods()
        return SentenceTransformerEmbedder("some/model", model=stub, **kwargs), stub

    def test_vectors_are_float32_and_unit_length(self):
        embedder, _ = self.make()
        out = embedder.encode(["a", "bb", "ccc"])
        self.assertEqual(out.shape, (3, 8))
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)
        self.assertEqual(embedder.encode("a").shape, (8,))

    def test_no_texts_give_an_empty_matrix_without_calling_the_model(self):
        embedder, stub = self.make()
        stub.calls.clear()
        self.assertEqual(embedder.encode([]).shape, (0, 8))
        self.assertEqual(embedder.encode_query([]).shape, (0, 8))
        self.assertEqual(stub.calls, [])

    def test_documents_and_queries_use_their_own_methods(self):
        embedder, stub = self.make()
        stub.calls.clear()
        embedder.encode(["x"])
        embedder.encode_document(["x"])
        embedder.encode_query("x")
        self.assertEqual([c[0] for c in stub.calls], ["document", "document", "query"])
        self.assertTrue(all(c[2]["normalize_embeddings"] for c in stub.calls))

    def test_a_model_without_the_query_methods_uses_encode(self):
        embedder, stub = self.make(stub=StubSentenceTransformer())
        stub.calls.clear()
        embedder.encode_query("x")
        embedder.encode_document(["x"])
        self.assertEqual([c[0] for c in stub.calls], ["encode", "encode"])

    def test_fingerprint_follows_dimension_and_prompts(self):
        base, _ = self.make()
        again, _ = self.make()
        self.assertEqual(base.fingerprint, again.fingerprint)
        self.assertIn("some/model", base.fingerprint)
        self.assertIn("dim=8", base.fingerprint)
        prompted, _ = self.make(stub=StubWithQueryMethods(
            prompts={"query": "task: search | query: ", "document": "title: none | text: "}))
        self.assertNotEqual(base.fingerprint, prompted.fingerprint)
        wider, _ = self.make(stub=StubWithQueryMethods(dim=16))
        self.assertNotEqual(base.fingerprint, wider.fingerprint)

    def test_truncate_dim_sets_the_dimension(self):
        embedder, _ = self.make(stub=StubWithQueryMethods(dim=8), truncate_dim=8)
        self.assertEqual(embedder.dimension, 8)

    def test_max_seq_length_is_applied(self):
        _, stub = self.make(max_seq_length=64)
        self.assertEqual(stub.max_seq_length, 64)

    def test_non_finite_vectors_fail_at_load(self):
        with self.assertRaises(RuntimeError) as caught:
            self.make(stub=StubSentenceTransformer(nan=True))
        self.assertIn("float16", str(caught.exception))

    def test_a_declared_dimension_the_model_does_not_return_fails_at_load(self):
        with self.assertRaises(RuntimeError):
            self.make(stub=StubWithQueryMethods(dim=8), truncate_dim=4)

    def test_a_chosen_prompt_name_is_passed_and_changes_the_identity(self):
        prompts = {"query": "q: ", "document": "d: ", "QuestionAnswering": "qa: "}
        plain, _ = self.make(stub=StubWithQueryMethods(prompts=prompts))
        chosen, stub = self.make(stub=StubWithQueryMethods(prompts=prompts),
                                 query_prompt_name="QuestionAnswering")
        stub.calls.clear()
        chosen.encode_query("x")
        chosen.encode_document(["x"])
        (_, _, query_kwargs), (_, _, document_kwargs) = stub.calls
        self.assertEqual(query_kwargs["prompt_name"], "QuestionAnswering")
        self.assertNotIn("prompt_name", document_kwargs)
        self.assertNotEqual(plain.fingerprint, chosen.fingerprint)

    def test_an_unknown_prompt_name_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self.make(stub=StubWithQueryMethods(prompts={"query": "q: "}),
                      document_prompt_name="Nope")
        self.assertIn("Nope", str(caught.exception))

    def test_the_gemma_preset_bounds_the_input_length(self):
        stub = StubWithQueryMethods()
        SentenceTransformerEmbedder("google/embeddinggemma-2", model=stub)
        self.assertEqual(stub.max_seq_length, 8192)
        stub = StubWithQueryMethods()
        SentenceTransformerEmbedder("google/embeddinggemma-2", model=stub, max_seq_length=512)
        self.assertEqual(stub.max_seq_length, 512)

    def test_embedding_gemma_defaults(self):
        defaults = _model_defaults("google/embeddinggemma-2")
        self.assertEqual(defaults["config_kwargs"], {"vision_config": None, "audio_config": None})
        self.assertEqual(defaults["dtype"], "auto")
        self.assertEqual(defaults["max_seq_length"], 8192)
        self.assertEqual(_model_defaults("/models/embeddinggemma-2/"), defaults)
        self.assertEqual(_model_defaults("sentence-transformers/all-MiniLM-L6-v2"), {})


class _DType:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"torch.{self.name}"


class TestResolveDtype(unittest.TestCase):
    def torch(self, bf16):
        return SimpleNamespace(
            dtype=_DType, float32=_DType("float32"), float16=_DType("float16"),
            bfloat16=_DType("bfloat16"),
            cuda=SimpleNamespace(is_bf16_supported=lambda: bf16))

    def test_none_leaves_the_models_own(self):
        self.assertIsNone(_resolve_dtype(self.torch(True), None, "cuda"))

    def test_auto_is_bfloat16_on_a_gpu_that_has_it_and_float32_otherwise(self):
        torch = self.torch(True)
        self.assertIs(_resolve_dtype(torch, "auto", "cuda"), torch.bfloat16)
        self.assertIs(_resolve_dtype(torch, "auto", "cpu"), torch.float32)
        self.assertEqual(str(_resolve_dtype(self.torch(False), "auto", "cuda")),
                         "torch.float32")

    def test_float16_is_refused_for_a_model_that_overflows_it(self):
        with self.assertRaises(ValueError):
            _resolve_dtype(self.torch(True), "float16", "cuda", forbidden=("float16",))
        self.assertEqual(str(_resolve_dtype(self.torch(True), "float16", "cuda")),
                         "torch.float16")

    def test_unknown_name(self):
        with self.assertRaises(ValueError):
            _resolve_dtype(self.torch(True), "float7", "cpu")


class TestOpenAIFingerprint(unittest.TestCase):
    def test_follows_model_and_dimension(self):
        cache = tempfile.mkdtemp(prefix="tsm_fp_")
        self.addCleanup(shutil.rmtree, cache, ignore_errors=True)
        small = OpenAIEmbedder(client=object(), cache_dir=cache)
        large = OpenAIEmbedder(model="text-embedding-3-large", client=object(), cache_dir=cache)
        short = OpenAIEmbedder(dim=256, client=object(), cache_dir=cache)
        self.assertEqual(small.fingerprint, "openai:text-embedding-3-small|dim=1536")
        self.assertEqual(len({small.fingerprint, large.fingerprint, short.fingerprint}), 3)


if __name__ == "__main__":
    unittest.main()
