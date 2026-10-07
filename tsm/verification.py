"""NLI cross-encoder verification of proposed supersessions.

The geometric detector (``engine.propose_supersessions``) is high-recall but
imperfect: two *coexisting* facts about the same topic can be mutual-nearest
neighbours and slip through the text/opposition gates. Committing a
supersession there wrongly demotes a still-true memory. This verifier adds a
semantic gate BEFORE the destructive demotion, using a small
natural-language-inference cross-encoder that runs locally on CPU/GPU — no
LLM server or API key required.

For a candidate where ``new`` supersedes ``old``, NLI runs with
premise=``new``, hypothesis=``old``:

  - **entailment**  — the new memory implies the old one: an update / more
    complete version. Supersession is valid -> commit (demote old).
  - **contradiction** — the new memory opposes the old one: the belief
    changed. Supersession is valid -> commit (demote old).
  - **neutral** — the two are independent (coexisting facts). This is exactly
    the false positive to stop -> reject (do NOT demote).

So the default rule is simply *reject NEUTRAL*. ``torch``/``transformers``
are imported lazily on first use, so ``tsm`` imports fine without them.
"""

import hashlib
import json
import logging
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

from ._retry import call_with_retries
from .interfaces import CommitTriple, ProposedPair

logger = logging.getLogger("tsm.verification")

_DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-xsmall"
# Fallback label order for cross-encoder/nli-* models when the model config does
# not expose id2label. (These models are trained with this exact order.)
_FALLBACK_LABELS = ["contradiction", "entailment", "neutral"]


class NLIVerifier:
    """Vets proposed supersessions with a local NLI cross-encoder."""

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        accept_labels: Sequence[str] = ("contradiction", "entailment"),
        min_margin: float = 0.0,
        batch_size: int = 32,
        allow_download: bool = True,
    ):
        """
        Args:
            model_name: HF cross-encoder NLI model.
            accept_labels: NLI labels whose pairs are committed (demoted). The
                default accepts contradiction+entailment and rejects neutral.
            min_margin: require (P(top) - P(neutral)) >= this to accept, so a
                barely-neutral pair is not demoted. 0.0 = decide by argmax only.
            batch_size: cross-encoder batch size.
            allow_download: if True, temporarily lift HF offline flags so the
                ~70MB model can be fetched on first use (then cached).
        """
        self.model_name = model_name
        self.accept_labels = {s.lower() for s in accept_labels}
        self.min_margin = min_margin
        self.batch_size = batch_size
        self._allow_download = allow_download
        self._model = None
        self._labels: List[str] = _FALLBACK_LABELS

    def _load(self):
        if self._model is not None:
            return
        # NOTE: `transformers` is used directly rather than
        # `sentence_transformers.CrossEncoder`: an NLI cross-encoder is just an
        # AutoModelForSequenceClassification, and this avoids the heavier
        # sentence-transformers dependency chain. The model may need a
        # one-time fetch, so lift the offline flags just for this load if
        # requested.
        saved = {}
        if self._allow_download:
            for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
                saved[k] = os.environ.pop(k, None)
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._torch = torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading NLI model %s on %s", self.model_name, self._device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = (
                AutoModelForSequenceClassification.from_pretrained(self.model_name)
                .to(self._device)
                .eval()
            )
            # Recover the true label order from the HF config (deberta-v3 NLI
            # uses a different order than the sentence-transformers convention).
            id2label = self._model.config.id2label
            self._labels = [id2label[i].lower() for i in sorted(id2label)]
            logger.info("NLI labels: %s", self._labels)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def score_pairs(
        self, pairs: Sequence[ProposedPair], id_to_text: Dict[str, str]
    ) -> List[dict]:
        """Return one row per pair with the NLI label + per-label probabilities.
        Rows for pairs whose text is missing are dropped."""
        rows = [
            p for p in pairs
            if id_to_text.get(p[1]) and id_to_text.get(p[0])
        ]
        if not rows:
            return []
        self._load()
        torch = self._torch
        # premise = new memory, hypothesis = old memory.
        premises = [id_to_text[n] for (o, n, _k, _c) in rows]
        hypotheses = [id_to_text[o] for (o, n, _k, _c) in rows]
        out: List[dict] = []
        for i in range(0, len(rows), self.batch_size):
            bp = premises[i:i + self.batch_size]
            bh = hypotheses[i:i + self.batch_size]
            enc = self._tokenizer(
                bp, bh, padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().tolist()
            for j, prob in enumerate(probs):
                old_id, new_id, kind, cosine = rows[i + j]
                pmap = {lbl: float(prob[k]) for k, lbl in enumerate(self._labels)}
                label = max(pmap, key=pmap.get)
                out.append({
                    "old_id": old_id, "new_id": new_id, "kind": kind, "cosine": cosine,
                    "label": label, "probs": pmap,
                    "margin": pmap[label] - pmap.get("neutral", 0.0),
                })
        return out

    # Verifier protocol -----------------------------------------------------------
    def verify(
        self, pairs: Sequence[ProposedPair], id_to_text: Dict[str, str]
    ) -> List[CommitTriple]:
        """Return the (old_id, new_id, kind) triples that pass verification and
        should be committed (demoted)."""
        accepted: List[CommitTriple] = []
        for r in self.score_pairs(pairs, id_to_text):
            if r["label"] in self.accept_labels and r["margin"] >= self.min_margin:
                accepted.append((r["old_id"], r["new_id"], r["kind"]))
        return accepted


# --- language-model verification --------------------------------------------

_LLM_SYSTEM = """You maintain a memory of facts one person has told you over time.
For each numbered pair you get an OLDER statement and a NEWER statement, both by that person.

Ask yourself: can both statements be true at the same time?
- If they can, the older one stays. Typical cases: they are about different people, pets, things or occasions; the person simply has, likes, owns, uses or did more than one thing; the newer one adds detail to the older one; they describe different periods of the past.
- If they cannot, the newer one replaces the older one: the person's situation changed, a plan or date moved, they stopped or started doing something, or they corrected themselves. The older statement is now wrong.

Write one line per pair and nothing else: the pair number, a colon, a few words of reasoning, then your verdict in capitals:
REPLACES (they cannot both be true now), SAME (the same fact in other words) or KEEPS (both are true).
When unsure, answer KEEPS."""

_VERDICT_LINE = re.compile(r"^\W*(?:pair\s*)?(\d+)\s*[:.)\-](.*)$", re.IGNORECASE | re.MULTILINE)
_VERDICT_WORD = re.compile(r"\b(REPLACES|SAME|KEEPS)\b")
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _read_verdicts(reply: str, count: int) -> Dict[int, str]:
    """Verdicts in a reply, keyed by 0-based pair position. A line counts when
    it starts with a pair number; its verdict is the last verdict word on it
    (capitals preferred, since the reasoning before it may use the same
    words in passing)."""
    verdicts: Dict[int, str] = {}
    for number, rest in _VERDICT_LINE.findall(_THINK_BLOCK.sub("", reply)):
        index = int(number) - 1
        if not 0 <= index < count or index in verdicts:
            continue
        words = _VERDICT_WORD.findall(rest) or _VERDICT_WORD.findall(rest.upper())
        if words:
            verdicts[index] = words[-1].lower()
    return verdicts


class LLMVerifier:
    """Vets supersessions with a chat model.

    Telling "I moved to Porto" (replaces "I live in Lisbon") from "my brother
    lives in Vancouver" (does not replace "my sister lives in Vancouver")
    takes knowing what the statements mean; a small NLI model labels both
    pairs a contradiction. A language model can make the call, so this
    verifier also asks the engine for wide candidates (every new fact against
    its nearest older facts, see ``candidate_min_cosine``) instead of only the
    pairs that pass the engine's lexical gates.

    Works with any OpenAI-compatible chat endpoint: OpenAI itself (default,
    ``OPENAI_API_KEY``), or a local server through ``base_url`` (for example
    Ollama at ``http://localhost:11434/v1``). Cost is one short request per
    ``batch_size`` candidate pairs; verdicts are cached on disk when
    ``cache_dir`` is given, so a pair is only ever judged once.

    A request that fails after its retries, or a reply that cannot be read,
    leaves those pairs uncommitted (nothing is hidden) and uncached, so the
    next ``consolidate()`` asks again.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        client=None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        accept: Sequence[str] = ("replaces",),
        batch_size: int = 8,
        max_retries: int = 4,
        request_timeout: float = 60.0,
        cache_dir: Optional[str] = None,
        candidate_min_cosine: float = 0.45,
        candidates_per_record: int = 2,
        candidate_margin: Optional[float] = 0.1,
        request_kwargs: Optional[dict] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Args:
            model: chat model name.
            client: an OpenAI-compatible client; built from ``base_url`` /
                ``api_key`` / ``OPENAI_API_KEY`` when omitted.
            accept: verdicts that commit a supersession. Add ``"same"`` to
                also retire the older wording of a repeated fact; a wrong
                "same" then costs a fact, so it is off by default.
            batch_size: candidate pairs per request.
            cache_dir: directory for the verdict cache (one JSON file per model).
            candidate_min_cosine / candidates_per_record: how widely
                ``Memory.consolidate`` asks the engine for candidates.
            candidate_margin: of a new fact's older neighbours, only those
                within this cosine distance of the closest one are judged.
                A fact further down is a different fact on the same topic,
                and judging it mostly adds wrong answers. ``None`` judges
                every neighbour above ``candidate_min_cosine``.
            request_kwargs: extra arguments for ``chat.completions.create``
                (for example ``{"extra_body": {...}}`` for a local server).
            system_prompt: replaces the built-in instructions. Replies must
                still be one line per pair that starts with the pair number
                and contains REPLACES, SAME or KEEPS.
        """
        if client is None:
            if base_url is None and api_key is None and not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. LLMVerifier needs it, or pass "
                    "base_url=... (an OpenAI-compatible server) or client=...")
            from openai import OpenAI

            kwargs = {"timeout": request_timeout}
            if base_url is not None:
                kwargs["base_url"] = base_url
                # Local servers ignore the key but the client requires one.
                kwargs["api_key"] = api_key or os.environ.get("OPENAI_API_KEY") or "unused"
            elif api_key is not None:
                kwargs["api_key"] = api_key
            client = OpenAI(**kwargs)
        self._client = client
        self.model = model
        self.accept = {v.lower() for v in accept}
        self.batch_size = max(1, int(batch_size))
        self.max_retries = max_retries
        self.candidate_min_cosine = float(candidate_min_cosine)
        self.candidates_per_record = int(candidates_per_record)
        self.candidate_margin = None if candidate_margin is None else float(candidate_margin)
        self._request_kwargs = dict(request_kwargs or {})
        self._system = system_prompt or _LLM_SYSTEM
        # Verdicts are only reused for the instructions that produced them.
        self._prompt_id = hashlib.sha256(self._system.encode("utf-8")).hexdigest()[:12]
        self.calls = 0
        self._cache: Dict[str, str] = {}
        self._cache_path = None
        self._dirty = False
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            safe = model.replace("/", "_").replace(":", "_")
            self._cache_path = os.path.join(cache_dir, f"verdicts_{safe}.json")
            if os.path.exists(self._cache_path):
                try:
                    with open(self._cache_path, encoding="utf-8") as fh:
                        loaded = json.load(fh)
                    if isinstance(loaded, dict):
                        self._cache = {k: v for k, v in loaded.items()
                                       if v in ("replaces", "same", "keeps")}
                except (OSError, ValueError):
                    self._cache = {}

    def _key(self, older: str, newer: str) -> str:
        raw = "\x1f".join((self._prompt_id, self.model, older, newer))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _ask(self, batch: Sequence[Tuple[str, str]]) -> Dict[int, str]:
        """Verdicts for one request, keyed by position in ``batch``. Pairs the
        reply does not mention are simply absent."""
        user = "\n\n".join(
            f"Pair {i + 1}\nOLDER: {older}\nNEWER: {newer}"
            for i, (older, newer) in enumerate(batch))

        def request():
            self.calls += 1
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self._system},
                          {"role": "user", "content": user}],
                temperature=0.0,
                **self._request_kwargs,
            )
            return resp.choices[0].message.content or ""

        reply = call_with_retries(request, "supersession verification", self.max_retries, logger)
        return _read_verdicts(reply, len(batch))

    def judge(self, pairs: Sequence[Tuple[str, str]]) -> List[Optional[str]]:
        """One verdict per ``(older, newer)`` text pair: ``"replaces"``,
        ``"same"``, ``"keeps"``, or ``None`` when the model could not be
        asked or did not answer for that pair."""
        out: List[Optional[str]] = [self._cache.get(self._key(o, n)) for o, n in pairs]
        todo = [i for i, verdict in enumerate(out) if verdict is None]
        for start in range(0, len(todo), self.batch_size):
            chunk = todo[start:start + self.batch_size]
            try:
                verdicts = self._ask([pairs[i] for i in chunk])
            except RuntimeError as e:
                logger.warning("%s; %d pairs left unverified (not committed)", e, len(chunk))
                continue
            for position, verdict in verdicts.items():
                index = chunk[position]
                out[index] = verdict
                self._cache[self._key(*pairs[index])] = verdict
                self._dirty = True
        return out

    def flush_cache(self) -> None:
        """Write the verdict cache to disk (also done by ``Memory.close``)."""
        if not (self._dirty and self._cache_path):
            return
        try:
            tmp = self._cache_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._cache, fh)
            os.replace(tmp, self._cache_path)
            self._dirty = False
        except OSError as e:
            logger.warning("verdict cache write failed: %s", e)

    # Verifier protocol -----------------------------------------------------------
    def verify(
        self, pairs: Sequence[ProposedPair], id_to_text: Dict[str, str]
    ) -> List[CommitTriple]:
        rows = [p for p in pairs if id_to_text.get(p[0]) and id_to_text.get(p[1])]
        verdicts = self.judge([(id_to_text[p[0]], id_to_text[p[1]]) for p in rows])
        accepted: List[CommitTriple] = []
        superseded = set()
        # Best match first, so a new fact that could replace two older ones
        # is tried against its closest one first.
        order = sorted(range(len(rows)), key=lambda i: -float(rows[i][3]))
        for i in order:
            old_id, new_id, kind, _cosine = rows[i]
            if verdicts[i] in self.accept and old_id not in superseded:
                superseded.add(old_id)
                accepted.append((old_id, new_id, kind))
        self.flush_cache()
        return accepted
