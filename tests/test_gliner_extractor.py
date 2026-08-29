#!/usr/bin/env python3
"""
Unit test for GLiNER Extractor & TSM Integration
=================================================

Tests zero-shot entity extraction, fact structuring, and memory insertion
using local Fastino GLiNER models without any external API dependencies.
"""

import os
import shutil
import sys
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tsm import Memory, SentenceTransformerEmbedder, GlinerExtractor


def test_gliner_extractor_basic():
    extractor = GlinerExtractor(
        model_name="urchade/gliner_multi-v2.1",
        schema=["database", "version", "port", "host", "medication", "dosage"],
    )
    
    text = "Sparky was prescribed 5mg of Apoquel daily, and Postgres 16 runs on port 5433."
    facts = extractor.extract_facts(text)
    
    assert len(facts) > 0
    assert "5433" in facts[0] or "Apoquel" in facts[0]
    print("\nExtracted facts with GLiNER:", facts)


def test_memory_with_gliner():
    db_path = os.path.join(project_root, "test_gliner_mem_db")
    if os.path.exists(db_path):
        shutil.rmtree(db_path, ignore_errors=True)

    embedder = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    extractor = GlinerExtractor("urchade/gliner_multi-v2.1", device="cpu")

    mem = Memory(
        db_path=db_path,
        embedder=embedder,
        extractor=extractor,
    )

    mem.add("User's primary database cluster is PostgreSQL 16 on port 5433", user_id="alice")
    mem.add("User updated staging database to port 5434 yesterday", user_id="alice")

    results = mem.recall("What port is PostgreSQL running on?", user_id="alice", top_k=2)
    assert len(results) > 0
    print("\nRecall results with GLiNER:", results)

    mem.close()
    shutil.rmtree(db_path, ignore_errors=True)


if __name__ == "__main__":
    test_gliner_extractor_basic()
    test_memory_with_gliner()
    print("\nAll GLiNER tests passed!")
