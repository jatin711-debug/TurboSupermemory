"""OpenAI embedding backend (default embedder for ``tsm.Memory``).

Wraps the OpenAI embeddings API behind the ``Embedder`` protocol
(``encode()`` + ``dimension``). Vectors are cached to disk (keyed by
model+text) so repeated runs cost nothing. The API key is read from the
``OPENAI_API_KEY`` environment variable only — it is never handled or logged.
The ``openai`` package is imported lazily, so ``tsm`` imports fine without it.

text-embedding-3 vectors are already unit-normalized, so cosine == dot.
"""

import atexit
import logging
import os
import pickle
from typing import List, Optional

import numpy as np

from ._retry import call_with_retries

logger = logging.getLogger("tsm.embedders")


class _ArrayCacheUnpickler(pickle.Unpickler):
    """Loads the embedding cache (a dict of text -> float array) and nothing else.

    The cache lives next to the database when ``tsm.Memory`` creates the
    embedder, and a database directory can come from someone else. A plain
    ``pickle.load`` runs whatever the file tells it to; this loader only ever
    resolves the handful of numpy names an array needs and refuses the rest.
    """

    _ALLOWED = frozenset({
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy.core.numeric", "_frombuffer"),
        ("numpy._core.numeric", "_frombuffer"),
    })

    def find_class(self, module, name):
        if (module, name) in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"embedding cache may only contain arrays, found {module}.{name}")


def _load_cache(path: str) -> dict:
    with open(path, "rb") as f:
        cache = _ArrayCacheUnpickler(f).load()
    if not isinstance(cache, dict):
        raise pickle.UnpicklingError("embedding cache is not a dict")
    return cache

# Native output dimensions per model (used to size the engine's index).
_MODEL_DIM = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder:
    def __init__(self, model="text-embedding-3-small", dim=None, batch=256,
                 max_retries=6, request_timeout=30.0, cache_dir=None, client=None):
        if client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. The default OpenAIEmbedder needs "
                    "it; pass client=... for another OpenAI-compatible endpoint, or "
                    "a custom embedder to tsm.Memory to use another backend "
                    "(see tsm.interfaces.Embedder)."
                )
            from openai import OpenAI

            client = OpenAI(timeout=request_timeout)
        self._client = client
        self.model = model
        native = _MODEL_DIM.get(model)
        self._dim = dim or native or 1536
        # A dimension other than the model's native one has to be requested
        # from the API (text-embedding-3 models shorten on request). Reporting
        # it without requesting it made every insert fail on a size mismatch.
        # For any other model `dim` only declares the size it returns.
        shortens = native is not None and model.startswith("text-embedding-3")
        self._request_dim = self._dim if (shortens and dim and dim != native) else None
        self.batch = batch
        self.max_retries = max_retries
        self.calls = 0
        self._cache = {}
        self._dirty = 0
        cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "tsm")
        os.makedirs(cache_dir, exist_ok=True)
        # Shortened vectors get their own cache file: the cache is keyed by text.
        suffix = f"_{self._request_dim}d" if self._request_dim else ""
        self._cache_path = os.path.join(cache_dir, f"emb_{model}{suffix}.pkl")
        if os.path.exists(self._cache_path):
            try:
                self._cache = _load_cache(self._cache_path)
                logger.info("Loaded %d cached embeddings from %s",
                            len(self._cache), os.path.basename(self._cache_path))
            except Exception as e:  # noqa: BLE001
                logger.warning("embed cache ignored (could not be loaded safely): %s", e)
                self._cache = {}
        atexit.register(self.flush)

    # Embedder protocol -----------------------------------------------------------
    @property
    def dimension(self):
        return self._dim

    def get_sentence_embedding_dimension(self):
        """SentenceTransformer-compatible alias."""
        return self._dim

    def encode(self, texts, **_kwargs):
        """Embed a single string (-> 1-D vec) or a list (-> 2-D array). Cache-backed;
        only uncached, de-duplicated texts hit the API."""
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        # OpenAI rejects empty input; map blanks to a single space (stable key).
        norm = [t if (t and t.strip()) else " " for t in items]

        missing, seen = [], set()
        for t in norm:
            if t not in self._cache and t not in seen:
                missing.append(t)
                seen.add(t)
        for i in range(0, len(missing), self.batch):
            chunk = missing[i:i + self.batch]
            for t, v in zip(chunk, self._embed_batch(chunk)):
                self._cache[t] = v
                self._dirty += 1
        if self._dirty >= 300:
            self.flush()

        out = np.vstack([self._cache[t] for t in norm]).astype(np.float32)
        return out[0] if single else out

    # internals ------------------------------------------------------------------
    def _embed_batch(self, chunk):
        def request():
            self.calls += 1
            kwargs = {"model": self.model, "input": chunk}
            if self._request_dim:
                kwargs["dimensions"] = self._request_dim
            r = self._client.embeddings.create(**kwargs)
            return [np.asarray(d.embedding, dtype=np.float32) for d in r.data]

        vectors = call_with_retries(request, "OpenAI embedding", self.max_retries, logger)
        if len(vectors) != len(chunk) or any(v.shape != (self._dim,) for v in vectors):
            got = sorted({v.shape for v in vectors})
            raise RuntimeError(
                f"embedding API returned {len(vectors)} vectors of shape {got} for "
                f"{len(chunk)} texts; expected dimension {self._dim} "
                f"(pass dim= to OpenAIEmbedder for a model whose size is not known)")
        return vectors

    def flush(self):
        if not self._dirty:
            return
        try:
            tmp = self._cache_path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(self._cache, f)
            os.replace(tmp, self._cache_path)
            self._dirty = 0
        except Exception as e:  # noqa: BLE001
            logger.warning("embed cache flush failed: %s", e)


class SentenceTransformerEmbedder:
    """Local open-source embedding backend using ``sentence-transformers`` models."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        get_dim = getattr(self.model, "get_embedding_dimension", getattr(self.model, "get_sentence_embedding_dimension", None))
        self._dim = get_dim() if get_dim else 384

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, texts):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        embs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embs = np.asarray(embs, dtype=np.float32)
        return embs[0] if single else embs
