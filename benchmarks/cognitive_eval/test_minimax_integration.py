import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MiniMaxIntegrationTests(unittest.TestCase):
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_factories_create_minimax_components(self):
        from cognitive_eval.extraction import create_extractor
        from cognitive_eval.gist import create_gister
        from cognitive_eval.judge import create_judge

        extractor = create_extractor("minimax", minimax_model="MiniMax-M3")
        gister = create_gister("minimax", model="MiniMax-M3")
        judge = create_judge("minimax", minimax_model="MiniMax-M3")

        self.assertEqual("MiniMax-M3", extractor.model)
        self.assertEqual("MiniMax-M3", gister.model)
        self.assertEqual("MiniMax-M3", judge.model)
        self.assertEqual({"type": "disabled"}, gister.extra_body["thinking"])
        self.assertEqual({"type": "disabled"}, judge.extra_body["thinking"])

    def test_mem0_provider_disables_thinking(self):
        from cognitive_eval.minimax_provider import MiniMaxMem0LLM
        from mem0.llms.openai import OpenAILLM

        provider = object.__new__(MiniMaxMem0LLM)
        with patch.object(OpenAILLM, "generate_response", return_value="ok") as generate:
            self.assertEqual("ok", provider.generate_response([]))
        self.assertEqual(
            {"thinking": {"type": "disabled"}, "reasoning_split": True},
            generate.call_args.kwargs["extra_body"],
        )

    def test_ollama_embedder_matches_sentence_transformer_surface(self):
        response = SimpleNamespace(embeddings=[[1.0, 0.0], [0.0, 1.0]])
        client = Mock()
        client.embed.return_value = response
        with patch("ollama.Client", return_value=client):
            from cognitive_eval.ollama_embedder import OllamaEmbedder

            embedder = OllamaEmbedder(model="test", dim=2)
            vectors = embedder.encode(["one", "two"])
            repeated = embedder.encode("one")

        np.testing.assert_array_equal(np.eye(2, dtype=np.float32), vectors)
        np.testing.assert_array_equal(np.array([1.0, 0.0], dtype=np.float32), repeated)
        self.assertEqual(1, client.embed.call_count)


if __name__ == "__main__":
    unittest.main()
