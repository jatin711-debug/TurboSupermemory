#!/usr/bin/env python3
"""LongMemEval — abstraction / concept-expansion isolation on REAL data (W4).

Abstraction is the last cognitive mechanism without a real-data verdict. It is
WIRED (Phase 3 + unit tests: a query reaches a sibling-concept memory through the
learned abstraction parent), but "wired" is not "helps on real conversational
data". This runner isolates it the same disciplined way belief revision was
isolated: two arms with an IDENTICAL corpus + config, differing in ONE switch.

  - OFF arm: `concept_expansion=False` — the augmenter uses only belief
    (Refines/Contradicts) and temporal edges; the concept graph + abstraction
    hierarchy do NOT influence ranking.
  - ON arm:  `concept_expansion=True`  — the 2-hop `mem→concept→mem` and 4-hop
    `mem→concept→parent→sibling-concept→mem` paths are active.

Belief revision is ON and role-filtered in BOTH arms (the W1 production config),
so the per-type lift (ON − OFF) isolates concept + abstraction expansion.

This is a RETRIEVAL proxy (no LLM): does concept/abstraction expansion SURFACE a
relevant memory the query cannot reach by cosine? The ON−OFF lift cancels proxy
noise. Target types where multi-hop concept bridges should help: multi-session
and temporal-reasoning (answers span sessions / related-but-not-nearest facts).

Usage:
    python benchmarks/cognitive_eval/run_abstraction_longmemeval.py --limit 200
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cognitive_eval.adapters.tsm_adapter import TSMAdapter
from cognitive_eval.benchmark_datasets.longmemeval import load_longmemeval
from cognitive_eval.run_belief_longmemeval import hit_at  # shared distinctive-token metric

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("abstraction_longmemeval")


def run_arm(concept_expansion, conversations, model_name, top_k, extractor, shared_model=None):
    """One fresh engine per conversation; belief revision ON + role-filtered in
    both arms, differing only in `concept_expansion`."""
    by_type = defaultdict(lambda: {"n": 0, "h1": 0, "h3": 0, "hk": 0})
    abst = {"abstraction": 0}
    model = shared_model
    for conv in conversations:
        db_path = tempfile.mkdtemp(prefix="tsm_abst_")
        adapter = TSMAdapter(db_path=db_path, embedding_model=model_name, extractor=extractor,
                             cognitive_features=True, belief_revision=True, model=model,
                             belief_source_roles=["user"], concept_expansion=concept_expansion)
        model = adapter.model
        try:
            adapter.add(conv.messages, user_id=conv.conv_id)
            adapter.trigger_consolidation()
            g = adapter.engine.graph_stats()
            abst["abstraction"] += g[6]  # abstraction_count
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
    return by_type, abst, model


def main():
    ap = argparse.ArgumentParser(description="LongMemEval abstraction isolation (concept_expansion ON vs OFF)")
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--limit", type=int, default=200,
                    help="cap #conversations (default 200 — full set OOMs, see PLAN W7)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--extractor", type=str, default="mock", choices=["mock", "ollama"])
    args = ap.parse_args()

    convs = load_longmemeval(args.data_dir)
    if args.limit:
        convs = convs[:args.limit]
    logger.info("Loaded %d conversations. Model=%s extractor=%s top_k=%d",
                len(convs), args.model, args.extractor, args.top_k)
    for name in ("cognitive_eval.adapters.tsm", "httpx", "httpcore", "urllib3",
                 "huggingface_hub", "sentence_transformers", "transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)

    logger.info("Running OFF arm (concept_expansion=False)...")
    off, off_abst, model = run_arm(False, convs, args.model, args.top_k, args.extractor)
    logger.info("Running ON arm (concept_expansion=True)...")
    on, on_abst, _ = run_arm(True, convs, args.model, args.top_k, args.extractor, shared_model=model)

    logger.info("Abstraction edges built — OFF: %s  ON: %s", off_abst, on_abst)
    logger.info("=" * 100)
    logger.info("%-26s %4s | %-18s | %-18s | %-18s", "question_type", "n",
                "hit@1  off/on/lift", "hit@3  off/on/lift", "hit@k  off/on/lift")
    logger.info("-" * 100)
    # concept/abstraction bridges should help multi-hop types most.
    target_types = {"multi-session", "temporal-reasoning", "knowledge-update"}
    types = sorted(set(off) | set(on), key=lambda t: (t not in target_types, t))
    best_lift = -1.0
    worst_lift = 1.0
    summary_types = {}
    for t in types:
        o = off.get(t, {"n": 0, "h1": 0, "h3": 0, "hk": 0})
        n = on.get(t, {"n": 0, "h1": 0, "h3": 0, "hk": 0})
        nq = max(o["n"], n["n"]) or 1

        def cell(key):
            of, oo = o[key] / (o["n"] or 1), n[key] / (n["n"] or 1)
            return f"{of:.2f}/{oo:.2f}/{oo - of:+.2f}"

        def lift(key):
            return n[key] / (n["n"] or 1) - o[key] / (o["n"] or 1)

        mark = "  <== multi-hop target" if t in target_types else ""
        logger.info("%-26s %4d | %-18s | %-18s | %-18s%s", t, nq, cell("h1"), cell("h3"), cell("hk"), mark)
        # track best hit@k lift on a target type; worst on any type (collateral)
        lk = lift("hk")
        summary_types[t] = round(lk, 4)
        if t in target_types:
            best_lift = max(best_lift, lk)
        worst_lift = min(worst_lift, lk)
    logger.info("=" * 100)
    logger.info("Best hit@k lift on a multi-hop target type: %+.2f | worst hit@k lift any type: %+.2f",
                best_lift, worst_lift)
    # A robust win needs a clear gain AND no meaningful loss on any type. A gain
    # on one type paid for by a loss on another is a WASH, not a win — report it
    # honestly (the auto-verdict must not overclaim on a mixed, noisy result).
    if best_lift > 0.05 and worst_lift >= -0.02:
        logger.info("VERDICT: concept/abstraction expansion MEASURABLY helps on real data — keep ON.")
    elif best_lift > 0.05 and worst_lift < -0.02:
        logger.info("VERDICT: concept/abstraction expansion is MIXED — a gain on one type (%.2f) is "
                    "offset by a loss on another (%.2f); net ~neutral, within n-noise.",
                    best_lift, worst_lift)
    elif worst_lift < -0.05:
        logger.info("VERDICT: concept/abstraction expansion HURTS some type — needs tuning or default OFF.")
    else:
        logger.info("VERDICT: concept/abstraction expansion adds ~no real-data lift (honest negative).")
    logger.info("GATE_SUMMARY: %s", json.dumps({
        "best_target_hitk_lift": round(best_lift, 4),
        "worst_hitk_lift": round(worst_lift, 4),
        "abstraction_edges_on": on_abst["abstraction"],
        "type_hitk_lift": summary_types,
    }))


if __name__ == "__main__":
    main()
