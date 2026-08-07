"""Scratch check: process-level shared embedding provider / NLI verifier singletons.

Constructs two TSMAdapters and two shared verifiers and asserts the underlying
model objects are IDENTICAL (``is``, not ``==``). Runs fully offline against the
local HF cache:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python check_shared_singletons.py
"""
import os
import shutil
import sys
import tempfile

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognitive_eval.embedding import (  # noqa: E402
    create_embedding_provider,
    get_shared_embedding_provider,
)
from cognitive_eval.verification import NLIVerifier, get_shared_verifier  # noqa: E402
from cognitive_eval.adapters.tsm_adapter import TSMAdapter  # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, locally cached


def main():
    # 1. Embedding provider factory: identity of the shared instance AND its
    #    underlying HF model; the non-shared factory must stay isolated.
    p1 = get_shared_embedding_provider(MODEL)
    p2 = get_shared_embedding_provider(MODEL)
    assert p1 is p2, "get_shared_embedding_provider returned different instances"
    assert p1.model is p2.model, "underlying HF model objects differ"
    p_priv = create_embedding_provider(MODEL)
    assert p_priv is not p1 and p_priv.model is not p1.model, \
        "create_embedding_provider must return a private instance"
    print("[ok] get_shared_embedding_provider: same provider + same HF model object")

    # 2. Adapter level: two default adapters share the cached provider; the
    #    shared_embedding=False opt-out loads a private model.
    dbs = [tempfile.mkdtemp(prefix="tsm_singleton_") for _ in range(3)]
    try:
        a1 = TSMAdapter(db_path=dbs[0], embedding_model=MODEL, extractor="mock",
                        verify_demotions=True)
        a2 = TSMAdapter(db_path=dbs[1], embedding_model=MODEL, extractor="mock",
                        verify_demotions=True)
        assert a1.model is a2.model, "two default adapters must share one provider"
        assert a1.model is p1, "adapter provider should be the shared cache instance"
        assert a1.verifier is a2.verifier, \
            "two default adapters must share one NLI verifier"
        a3 = TSMAdapter(db_path=dbs[2], embedding_model=MODEL, extractor="mock",
                        shared_embedding=False)
        assert a3.model is not a1.model, "shared_embedding=False must give a private model"
        print("[ok] TSMAdapter: default shares provider + verifier; opt-out is private")
        for a in (a1, a2, a3):
            a.close()
    finally:
        for d in dbs:
            shutil.rmtree(d, ignore_errors=True)

    # 3. Verifier factory: identity of the instance AND its lazily-loaded
    #    cross-encoder; direct construction and different semantics stay isolated.
    v1 = get_shared_verifier()
    v2 = get_shared_verifier()
    assert v1 is v2, "get_shared_verifier returned different instances"
    v1._load()
    assert v1._model is v2._model, "underlying cross-encoder objects differ"
    v_priv = NLIVerifier()
    assert v_priv is not v1, "direct NLIVerifier() construction must stay isolated"
    v_other = get_shared_verifier(min_margin=0.5)
    assert v_other is not v1, "different semantics must not share an instance"
    print("[ok] get_shared_verifier: same verifier + same cross-encoder object")

    print("ALL SINGLETON CHECKS PASSED")


if __name__ == "__main__":
    main()
