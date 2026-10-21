"""OpenAI embedding backend (default embedder for ``tsm.Memory``).

Wraps the OpenAI embeddings API behind the ``Embedder`` protocol
(``encode()`` + ``dimension``). Vectors are cached to disk (keyed by
model+text) so repeated runs cost nothing. The API key is read from the
``OPENAI_API_KEY`` environment variable only — it is never handled or logged.
The ``openai`` package is imported lazily, so ``tsm`` imports fine without it.

text-embedding-3 vectors are already unit-normalized, so cosine == dot.
"""

import atexit
import hashlib
import logging
import os
import pickle
import threading
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

    @property
    def fingerprint(self) -> str:
        """Identifies the vectors this embedder returns (see ``tsm.identity``)."""
        return f"openai:{self.model}|dim={self._dim}"

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


# Settings a model's own files do not carry but its model card requires. They
# apply to any SentenceTransformerEmbedder built for that model unless the
# caller passes something else.
_MODEL_DEFAULTS = {
    # EmbeddingGemma 2 (740M with its image and audio encoders, 270M for text).
    # Text memories need only the text model. Its activations overflow float16,
    # which returns NaN or quietly degraded vectors instead of an error.
    "embeddinggemma-2": {
        "config_kwargs": {"vision_config": None, "audio_config": None},
        "dtype": "auto",
        "forbid_dtypes": ("float16",),
        # sentence-transformers reports no limit (1e30) for it; the card says
        # 8,192 tokens, and an input past that must not reach a 4 GB GPU.
        "max_seq_length": 8192,
        # The conversational profile's belief thresholds (0.85 / 0.75) were
        # tuned on other embedders' vectors. On this one's, detection without a
        # verifier marks 30 of 58 still-true facts stale (8 of 58 on MiniLM) and
        # cost 10 of 77 questions their evidence at one history (p = 0.012); at
        # 0.92 / 0.86 it costs 1 (benchmarks/PHASE_PROGRESS.md, 2026-10-19).
        "engine_defaults": {
            "refinement_cosine_threshold": 0.92,
            "contradiction_cosine_threshold": 0.86,
        },
    },
}


def _model_defaults(model_name: str) -> dict:
    lowered = model_name.lower().rstrip("/\\")
    for key, defaults in _MODEL_DEFAULTS.items():
        if lowered.endswith(key):
            return defaults
    return {}


def _resolve_dtype(torch, dtype, device: str, forbidden=()):
    """A torch dtype from ``None`` (leave the model's own), ``"auto"``
    (bfloat16 where the GPU supports it, else float32) or a name."""
    if dtype is None:
        return None
    if dtype == "auto":
        bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
        return torch.bfloat16 if bf16 else torch.float32
    if isinstance(dtype, str):
        name = dtype.removeprefix("torch.")
        resolved = getattr(torch, name, None)
        if not isinstance(resolved, torch.dtype):
            raise ValueError(f"unknown dtype {dtype!r}")
    else:
        resolved = dtype
    if str(resolved).removeprefix("torch.") in forbidden:
        raise ValueError(
            f"{resolved} is not safe for this model (its activations overflow it, "
            f"giving NaN or degraded vectors); use bfloat16 or float32")
    return resolved


def _model_revision(model_name: str, revision: Optional[str]) -> Optional[str]:
    """The commit of the model that was loaded: weights re-uploaded under the
    same name give different vectors, so the commit belongs in the identity."""
    if revision:
        return revision[:12]
    if os.path.isdir(model_name):
        return None
    try:
        from huggingface_hub import snapshot_download

        return os.path.basename(snapshot_download(model_name, local_files_only=True))[:12]
    except Exception:  # noqa: BLE001 — identity then says "unknown", it does not fail
        return None


def _short_hash(text: Optional[str]) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:8]


class SentenceTransformerEmbedder:
    """Local embedding backend over ``sentence-transformers`` models.

    Runs offline once the model is cached. Models with task prompts are
    handled the way their card asks: ``encode_document`` (and ``encode``, which
    is the same) and ``encode_query`` each add their own prompt, which
    ``tsm.Memory`` uses for stored facts and for questions. For a model
    without prompts (the default MiniLM) the two are identical.

    ``SentenceTransformerEmbedder("google/embeddinggemma-2")`` loads the text
    model only, in bfloat16 where the GPU has it and float32 otherwise (see
    ``_MODEL_DEFAULTS``). ``truncate_dim`` keeps the first dimensions of a
    Matryoshka model (128, 256 or 512 of Gemma's 768) and re-normalizes.

    Vectors are unit length (``Memory`` relies on cosine == dot). A probe at
    load time checks the model returns finite vectors of the declared size.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        *,
        truncate_dim: Optional[int] = None,
        max_seq_length: Optional[int] = None,
        dtype=None,
        batch_size: int = 32,
        query_prompt_name: Optional[str] = None,
        document_prompt_name: Optional[str] = None,
        revision: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        config_kwargs: Optional[dict] = None,
        model=None,
    ):
        """
        Args:
            model_name: Hugging Face id or local path.
            device: ``"cuda"``, ``"cpu"``, ...; default CUDA when available.
            truncate_dim: keep this many leading dimensions (Matryoshka models).
            max_seq_length: replaces the model's own input limit, in tokens.
            dtype: ``None`` (the model's own), ``"auto"``, or a torch dtype / name.
            batch_size: texts per forward pass.
            query_prompt_name / document_prompt_name: which of the model's
                named prompts goes in front of a question / a stored text
                (default: its ``"query"`` and ``"document"``). EmbeddingGemma
                also has ``"QuestionAnswering"``, ``"FactChecking"`` and others.
            revision: model commit to load.
            model_kwargs / config_kwargs: passed to ``SentenceTransformer``;
                a key given here replaces the model's entry in ``_MODEL_DEFAULTS``.
            model: an already-built ``SentenceTransformer``-like object (tests).
        """
        defaults = _model_defaults(model_name)
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self._lock = threading.Lock()  # HF tokenizers raise "Already borrowed" when shared across threads
        injected = model is not None
        if not injected:
            import torch
            from sentence_transformers import SentenceTransformer

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            kwargs = dict(model_kwargs or {})
            resolved = _resolve_dtype(torch, dtype if dtype is not None else defaults.get("dtype"),
                                      device, defaults.get("forbid_dtypes", ()))
            if resolved is not None and "dtype" not in kwargs:
                kwargs["dtype"] = resolved
            config = defaults.get("config_kwargs", {}) if config_kwargs is None else config_kwargs
            options = {"device": device, "truncate_dim": truncate_dim}
            if revision:
                options["revision"] = revision
            if kwargs:
                options["model_kwargs"] = kwargs
            if config:
                options["config_kwargs"] = dict(config)
            model = SentenceTransformer(model_name, **options)
        self.device = device or str(getattr(model, "device", "cpu"))
        self.model = model
        limit = max_seq_length or defaults.get("max_seq_length")
        if limit:
            self.model.max_seq_length = int(limit)
        self._prompt_names = {"encode_query": query_prompt_name,
                              "encode_document": document_prompt_name}
        # Engine settings that depend on this model's cosine geometry; Memory
        # applies them under its conversational profile (see Memory.__init__).
        self.engine_defaults = dict(defaults.get("engine_defaults") or {})
        get_dim = (getattr(model, "get_embedding_dimension", None)
                   or getattr(model, "get_sentence_embedding_dimension", None))
        self._dim = int(truncate_dim or (get_dim() if get_dim else 384))
        prompts = getattr(model, "prompts", None) or {}
        for name in (query_prompt_name, document_prompt_name):
            if name is not None and name not in prompts:
                raise ValueError(f"{model_name} has no prompt named {name!r} "
                                 f"(it has: {', '.join(sorted(prompts)) or 'none'})")
        self._revision = None if injected else _model_revision(model_name, revision)
        self.fingerprint = (
            f"st:{model_name}@{self._revision or 'unknown'}|dim={self._dim}"
            f"|doc={_short_hash(prompts.get(document_prompt_name or 'document'))}"
            f"|query={_short_hash(prompts.get(query_prompt_name or 'query'))}")
        self._probe()

    @property
    def dimension(self) -> int:
        return self._dim

    def _probe(self) -> None:
        """One real encode: the declared size, and finite numbers."""
        probe = self.encode_document(["tsm embedder probe"])
        if probe.shape != (1, self._dim) or not np.isfinite(probe).all():
            raise RuntimeError(
                f"{self.model_name} returned {probe.shape} (finite: "
                f"{bool(np.isfinite(probe).all())}) for a probe; expected (1, {self._dim}) "
                f"finite values. Check dtype (float16 breaks some models) and truncate_dim.")

    def _run(self, method: str, texts):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        if not items:
            return np.zeros((0, self._dim), dtype=np.float32)
        encode = getattr(self.model, method, None) or self.model.encode
        options = {}
        if self._prompt_names[method] is not None:
            options["prompt_name"] = self._prompt_names[method]
        with self._lock:
            embs = encode(items, batch_size=self.batch_size, normalize_embeddings=True,
                          show_progress_bar=False, convert_to_numpy=True, **options)
        embs = np.asarray(embs, dtype=np.float32)
        # Normalized by the model in its own precision (bfloat16 is good to ~3
        # digits); once more in float32 so stored vectors are unit length.
        embs = embs / np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12)
        return embs[0] if single else embs

    def encode(self, texts):
        """Embed text that will be stored: the same as ``encode_document``."""
        return self._run("encode_document", texts)

    def encode_document(self, texts):
        return self._run("encode_document", texts)

    def encode_query(self, texts):
        return self._run("encode_query", texts)
