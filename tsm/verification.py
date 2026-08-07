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

import logging
import os
from typing import Dict, List, Sequence

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
