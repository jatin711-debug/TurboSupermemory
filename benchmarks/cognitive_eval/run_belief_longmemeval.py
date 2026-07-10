#!/usr/bin/env python3
"""LongMemEval — belief-revision isolation on REAL conversational data.

The stock `run_longmemeval.py` measures only a loose word-overlap recall and
throws away `question_type`, so it cannot see belief revision. This runner:

  1. Breaks results out by `question_type` (knowledge-update is the belief-
     revision subset: "what camera do I have NOW", "PREVIOUS status", ...).
  2. Scores answer-containment with a strict distinctive-token match (does a
     top-k memory contain the gold answer's distinctive value, e.g. "f-150",
     "70-200mm", "premier"), at ranks 1 / 3 / k.
  3. Runs belief revision ON vs OFF (identical corpus + config; only the
     Contradicts/Refines edges + supersession demotion differ), so the
     per-type lift (ON - OFF) isolates belief revision's contribution on real
     data. abstention questions are excluded (they test refusal, not recall).

This is a RETRIEVAL proxy (no LLM): it measures whether the current fact is
SURFACED, not whether an LLM answers correctly. The absolute numbers are noisy;
the ON-vs-OFF *lift* is the trustworthy signal (the proxy noise cancels).

Usage:
    python benchmarks/cognitive_eval/run_belief_longmemeval.py --limit 20   # quick pipeline check
    python benchmarks/cognitive_eval/run_belief_longmemeval.py               # full test set
"""

import argparse
import logging
import os
import re
import shutil
import sys
import tempfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cognitive_eval.adapters.tsm_adapter import TSMAdapter
from cognitive_eval.benchmark_datasets.longmemeval import load_longmemeval

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("belief_longmemeval")

_STOP = set((
    "a an the of to in on at for and or but is are was were be been am i you my your his her "
    "our their it its this that with as by from about into over under how many much do does did "
    "have has had not no yes previously currently now new old most recent recently type kind"
).split())


def key_tokens(answer, n=2):
    """Distinctive tokens from a gold answer, most-discriminative first.
    Prefers tokens with digits or hyphens (model numbers), then long words."""
    raw = re.findall(r"[a-z0-9][a-z0-9\-]*", answer.lower())
    toks = [t for t in raw if t not in _STOP and (any(c.isdigit() for c in t) or "-" in t or len(t) >= 4)]
    if not toks:
        toks = [t for t in raw if len(t) >= 3 and t not in _STOP]
    toks.sort(key=lambda t: (any(c.isdigit() for c in t) or "-" in t, len(t)), reverse=True)
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:n]


def hit_at(answer, texts, k):
    """True if the gold answer's most-distinctive token appears in the top-k texts."""
    kts = key_tokens(answer)
    if not kts:
        return False
    joined = " || ".join((t or "").lower() for t in texts[:k])
    return kts[0] in joined


def run_arm(belief_on, conversations, model_name, top_k, extractor, shared_model=None,
            store_roles=None, belief_source_roles=None):
    """One fresh engine PER conversation (belief detection stays within a user,
    matching a properly-scoped store and isolating the mechanism). The embedding
    model is loaded once and shared across all engines."""
    by_type = defaultdict(lambda: {"n": 0, "h1": 0, "h3": 0, "hk": 0})
    edges = {"refine": 0, "contra": 0}
    model = shared_model
    for conv in conversations:
        db_path = tempfile.mkdtemp(prefix="tsm_lme_")
        adapter = TSMAdapter(db_path=db_path, embedding_model=model_name, extractor=extractor,
                             cognitive_features=True, belief_revision=belief_on, model=model,
                             store_roles=store_roles, belief_source_roles=belief_source_roles)
        model = adapter.model  # reuse the loaded model for the next conversation
        try:
            adapter.add(conv.messages, user_id=conv.conv_id)
            adapter.trigger_consolidation()
            g = adapter.engine.graph_stats()
            edges["refine"] += g[4]
            edges["contra"] += g[5]
            for q in conv.queries:
                if q.is_abstention:
                    continue
                res = adapter.search(q.query_text, user_id=conv.conv_id, top_k=top_k, use_cognitive=True)
                texts = [r.get("text", "") for r in res]
                s = by_type[q.question_type or "?"]
                s["n"] += 1
                s["h1"] += int(hit_at(q.answer_text, texts, 1))
                s["h3"] += int(hit_at(q.answer_text, texts, 3))
                s["hk"] += int(hit_at(q.answer_text, texts, top_k))
        finally:
            adapter.close()
            shutil.rmtree(db_path, ignore_errors=True)
    return by_type, edges, model


def main():
    ap = argparse.ArgumentParser(description="LongMemEval belief-revision isolation (ON vs OFF)")
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap #conversations (quick pipeline check)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--extractor", type=str, default="mock", choices=["mock", "ollama"])
    ap.add_argument("--user-only", action="store_true",
                    help="MODE A: store only USER messages as facts (drop assistant turns entirely). "
                         "belief revision should operate on the user's evolving statements, not the "
                         "assistant's verbose responses)")
    ap.add_argument("--role-filtered", action="store_true",
                    help="MODE B: store ALL roles (every memory stays retrievable) but restrict "
                         "belief-revision detection to user facts via the engine's first-class "
                         "belief_source_roles. This is the productized form of --user-only.")
    args = ap.parse_args()
    store_roles = {"user"} if args.user_only else None
    belief_source_roles = ["user"] if args.role_filtered else None

    convs = load_longmemeval(args.data_dir)
    if args.limit:
        convs = convs[:args.limit]
    n_q = sum(len(c.queries) for c in convs)
    logger.info("Loaded %d conversations, %d queries. Model=%s extractor=%s top_k=%d",
                len(convs), n_q, args.model, args.extractor, args.top_k)

    # Quiet the per-conversation adapter + HF download chatter.
    for name in ("cognitive_eval.adapters.tsm", "httpx", "httpcore", "urllib3",
                 "huggingface_hub", "sentence_transformers", "transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)

    logger.info("Running OFF arm (belief revision disabled)... store_roles=%s belief_source_roles=%s",
                store_roles, belief_source_roles)
    off, off_edges, model = run_arm(False, convs, args.model, args.top_k, args.extractor,
                                    store_roles=store_roles, belief_source_roles=belief_source_roles)
    logger.info("Running ON arm (belief revision enabled)...")
    on, on_edges, _ = run_arm(True, convs, args.model, args.top_k, args.extractor,
                              shared_model=model, store_roles=store_roles,
                              belief_source_roles=belief_source_roles)

    logger.info("Belief edges built — OFF: %s   ON: %s", off_edges, on_edges)
    logger.info("=" * 100)
    logger.info("%-26s %4s | %-18s | %-18s | %-18s", "question_type", "n",
                "hit@1  off/on/lift", "hit@3  off/on/lift", "hit@k  off/on/lift")
    logger.info("-" * 100)
    types = sorted(set(off) | set(on), key=lambda t: (t != "knowledge-update", t))
    for t in types:
        o, n = off.get(t, {"n": 0, "h1": 0, "h3": 0, "hk": 0}), on.get(t, {"n": 0, "h1": 0, "h3": 0, "hk": 0})
        nq = max(o["n"], n["n"]) or 1
        def cell(key):
            of, oo = o[key] / (o["n"] or 1), n[key] / (n["n"] or 1)
            return f"{of:.2f}/{oo:.2f}/{oo - of:+.2f}"
        mark = "  <== belief revision" if t == "knowledge-update" else ""
        logger.info("%-26s %4d | %-18s | %-18s | %-18s%s", t, nq, cell("h1"), cell("h3"), cell("hk"), mark)
    logger.info("=" * 100)
    ku_o, ku_n = off.get("knowledge-update"), on.get("knowledge-update")
    if ku_o and ku_n and ku_o["n"]:
        lift1 = ku_n["h1"] / ku_n["n"] - ku_o["h1"] / ku_o["n"]
        liftk = ku_n["hk"] / ku_n["n"] - ku_o["hk"] / ku_o["n"]
        logger.info("KNOWLEDGE-UPDATE (belief revision) lift: hit@1 %+.2f, hit@k %+.2f  (n=%d)",
                    lift1, liftk, ku_o["n"])
        if on_edges["refine"] + on_edges["contra"] == 0:
            logger.info("VERDICT: belief edges NEVER fired on real data — detection heuristic is "
                        "too strict for real corrections (opposition markers / thresholds).")
        elif lift1 > 0.05 or liftk > 0.05:
            logger.info("VERDICT: belief revision MEASURABLY improves knowledge-update retrieval on real data.")
        else:
            logger.info("VERDICT: belief revision fired but adds ~no retrieval lift on real knowledge-update.")
    logger.info("=" * 100)


if __name__ == "__main__":
    main()
