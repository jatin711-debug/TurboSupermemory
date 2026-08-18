import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from tsm import Memory
from tsm.embedders import SentenceTransformerEmbedder

class SimpleExtractor:
    def extract_facts(self, message: str, context=None):
        return [message]

def test_memory_colbert_reranking():
    test_db = "./test_colbert_mem_db"
    if os.path.exists(test_db):
        shutil.rmtree(test_db, ignore_errors=True)

    try:
        embedder = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
        extractor = SimpleExtractor()
        mem = Memory(
            db_path=test_db,
            embedder=embedder,
            extractor=extractor,
            profile="conversational",
            reranker="colbert",
        )

        user_id = "test_user"
        mem.add("Sparky is a 3-year old golden retriever who loves running in the park.", user_id=user_id)
        mem.add("Sparky was prescribed 5mg of Apoquel daily by the vet for his seasonal allergies.", user_id=user_id)
        mem.add("Sparky weighs 32kg and takes flea medication quarterly.", user_id=user_id)

        results = mem.recall("What exact dosage of medication does Sparky take daily?", user_id=user_id, rerank=True, top_k=3)
        assert len(results) > 0
        assert "5mg of Apoquel" in results[0]["text"]
        assert "maxsim_score" in results[0]
        print("Test passed! Top result:", results[0])
    finally:
        if os.path.exists(test_db):
            shutil.rmtree(test_db, ignore_errors=True)

if __name__ == "__main__":
    test_memory_colbert_reranking()
