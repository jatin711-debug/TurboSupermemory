#!/usr/bin/env python3
"""Inspect / calibrate NLI verification of proposed supersessions (W3).

Builds role-filtered, verification-deferred engines over a LongMemEval sample,
runs the geometric detector (`propose_supersessions`), scores every proposed
pair with the NLI cross-encoder, and prints the per-pair decision:

    [ACCEPT|reject] kind  label (p_contra/p_entail/p_neutral)
        old: <text>
        new: <text>

Use it to eyeball precision (are the accepted pairs genuine supersessions? are
the rejected ones genuine coexisting facts?) and to hand-label a calibration set
before tuning the accept rule / margin. Reports the label distribution and the
accept rate so the aggregate effect of verification is visible without a full run.

Usage:
    python benchmarks/cognitive_eval/verify_supersessions.py --limit 40
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
from collections import Counter

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cognitive_eval.adapters.tsm_adapter import TSMAdapter
from cognitive_eval.benchmark_datasets.longmemeval import load_longmemeval
from cognitive_eval.verification import get_shared_verifier

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("verify_supersessions")


def main():
    ap = argparse.ArgumentParser(description="Inspect NLI verification of proposed supersessions")
    ap.add_argument("--limit", type=int, default=40, help="#conversations to sample")
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--max-show", type=int, default=60, help="cap printed pairs")
    args = ap.parse_args()

    for name in ("cognitive_eval.adapters.tsm", "httpx", "httpcore", "urllib3",
                 "huggingface_hub", "sentence_transformers", "transformers"):
        logging.getLogger(name).setLevel(logging.ERROR)

    convs = load_longmemeval()[:args.limit]
    verifier = get_shared_verifier()
    model = None
    labels = Counter()
    accepted_n = proposed_n = shown = 0

    for conv in convs:
        db = tempfile.mkdtemp(prefix="tsm_vsup_")
        # defer commitment (belief detection runs, but nothing is committed) so
        # we can inspect the raw proposals + verifier decisions.
        adapter = TSMAdapter(db_path=db, embedding_model=args.model, extractor="mock",
                             cognitive_features=True, belief_revision=True, model=model,
                             belief_source_roles=["user"], verify_demotions=True,
                             verifier=verifier)
        model = adapter.model
        try:
            adapter.add(conv.messages, user_id=conv.conv_id)
            adapter.engine.trigger_consolidation()  # deferred: no commit
            proposed = adapter.engine.propose_supersessions()
            if not proposed:
                continue
            rows = verifier.score_pairs(proposed, adapter._id_to_text)
            accept_set = verifier.accept_labels
            for r in rows:
                proposed_n += 1
                labels[r["label"]] += 1
                ok = r["label"] in accept_set and r["margin"] >= verifier.min_margin
                accepted_n += int(ok)
                if shown < args.max_show:
                    shown += 1
                    p = r["probs"]
                    logger.info("[%s] %-13s %-13s  (c=%.2f e=%.2f n=%.2f)",
                                "ACCEPT" if ok else "reject", r["kind"], r["label"],
                                p.get("contradiction", 0), p.get("entailment", 0), p.get("neutral", 0))
                    logger.info("    old: %s", adapter._id_to_text.get(r["old_id"], "")[:96])
                    logger.info("    new: %s", adapter._id_to_text.get(r["new_id"], "")[:96])
        finally:
            adapter.close()
            shutil.rmtree(db, ignore_errors=True)

    logger.info("=" * 78)
    logger.info("Proposed pairs: %d | accepted (committed): %d (%.0f%%) | rejected: %d",
                proposed_n, accepted_n, 100.0 * accepted_n / max(proposed_n, 1),
                proposed_n - accepted_n)
    logger.info("NLI label distribution: %s", dict(labels))
    logger.info("Accept rule: labels=%s min_margin=%.2f",
                sorted(verifier.accept_labels), verifier.min_margin)


if __name__ == "__main__":
    main()
