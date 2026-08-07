#!/usr/bin/env python3
"""Tests for role attribution and contextual extraction setup."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognitive_eval.extraction.openai_extractor import extraction_cache_key
from cognitive_eval.gist import create_gister
from cognitive_eval.head_to_head_eval import conv_facts_with_roles
from cognitive_eval.run_belief_longmemeval import prewarm_extraction


class _Extractor:
    def __init__(self, outputs=None):
        self._cache = {}
        self.outputs = outputs or {}
        self.requests = []
        self.calls = 0

    def extract_facts(self, message, context=None):
        self.requests.append((message, list(context or [])))
        return list(self.outputs.get(message, [message]))

    def flush_cache(self):
        pass


class _Conversation:
    def __init__(self, messages):
        self.messages = messages


class RoleAwareTests(unittest.TestCase):
    def test_context_changes_extraction_cache_identity(self):
        self.assertEqual("yes", extraction_cache_key(" yes "))
        first = extraction_cache_key("yes", ["Use option A?"])
        second = extraction_cache_key("yes", ["Use option B?"])
        self.assertNotEqual(first, second)

    def test_contextual_prewarm_passes_recent_conversation_context(self):
        extractor = _Extractor()
        conversation = _Conversation([
            {"role": "assistant", "content": "Use option A?"},
            {"role": "user", "content": "Yes, I will use that one."},
        ])
        prewarm_extraction(extractor, [conversation], workers=1, contextual=True)
        self.assertEqual([], extractor.requests[0][1])
        self.assertEqual(["Use option A?"], extractor.requests[1][1])

    def test_user_reassertion_owns_duplicate_and_refreshes_recency(self):
        extractor = _Extractor({
            "assistant first": ["The project uses Rust."],
            "middle": ["An unrelated fact."],
            "user repeats": ["The project uses Rust."],
        })
        conversation = _Conversation([
            {"role": "assistant", "content": "assistant first"},
            {"role": "user", "content": "middle"},
            {"role": "user", "content": "user repeats"},
        ])
        facts = conv_facts_with_roles(extractor, conversation)
        self.assertEqual(
            [
                {"text": "An unrelated fact.", "role": "user"},
                {"text": "The project uses Rust.", "role": "user"},
            ],
            facts,
        )

    def test_extractive_gister_drops_lone_assistant_advice(self):
        gister = create_gister("extractive")
        self.assertEqual("", gister.summarize(["[assistant] Buy a tripod."], max_tokens=20))


if __name__ == "__main__":
    unittest.main()
