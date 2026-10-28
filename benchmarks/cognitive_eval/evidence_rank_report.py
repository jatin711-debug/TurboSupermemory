#!/usr/bin/env python3
"""Why a question's evidence does not reach the context.

For every evidence turn LongMemEval marks (``has_answer``), this reports
whether the extractor stored any fact from it, and how far down the dense
ranking (cosine to the question over the whole history's facts) its best
fact sits. A question is filed under its worst evidence turn:

  not stored   no fact was extracted from the turn: no retrieval change can
               reach it; only turn/session adjacency or the write path can
  rank > 60    stored, but further down than any pool recall() uses
  rank 31-60   inside a wide pool, outside the default one
  rank <= 30   inside the pool: if it is still missing from the context, the
               selection (packer, budget) dropped it

With ``--run RUN.json:ARM`` it also says, per category, how many of those
questions had all their evidence in that arm's context and how many were
judged right. Free: cached extraction and embeddings only (run the same
questions online once if embeddings are missing).

    python benchmarks/cognitive_eval/evidence_rank_report.py --limit 120 \\
        --run ~/.tsm_eval/shipped_stack_eval/runs/ctl_notes600.json:sdk_cognitive
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

CATEGORIES = ("not stored", "rank > 60", "rank 31-60", "rank <= 30")


def category(ranks):
    """The worst of a question's evidence turns (``None`` = nothing stored)."""
    if any(r is None for r in ranks):
        return "not stored"
    worst = max(ranks)
    return "rank > 60" if worst > 60 else "rank 31-60" if worst > 30 else "rank <= 30"


def judged(spec):
    """query_id -> (all evidence in context, judged right) from a run file."""
    path, _, arm = spec.rpartition(":")
    with open(os.path.expanduser(path), encoding="utf-8") as fh:
        tasks = json.load(fh)["tasks"]
    out = {}
    for t in tasks:
        if t["arm"] != arm:
            continue
        turns = set(t.get("turns") or [])
        out[t["query_id"]] = (set(t.get("evidence") or []) <= turns, t.get("correct"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--pool", type=int, default=1)
    ap.add_argument("--variant", choices=["oracle", "s", "m"], default=None)
    ap.add_argument("--embed-model", default="text-embedding-3-small")
    ap.add_argument("--extractor-model", default="gpt-4.1-nano")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--run", default=None, metavar="RUN.json:ARM")
    ap.add_argument("--json", default=None, help="write the per-question rows here")
    args = ap.parse_args()
    os.environ.setdefault("OPENAI_API_KEY", "offline")  # clients are built, never called

    import shipped_stack_eval as harness
    from cognitive_eval.benchmark_datasets.longmemeval import load_longmemeval

    work_dir = args.work_dir or harness.default_work_dir()
    extraction = harness.CachedExtraction(args.extractor_model, offline=True)
    embedder = harness.Embeddings(args.embed_model, os.path.join(work_dir, "emb_overlay.pkl"),
                                  offline=True)
    convs = load_longmemeval(None)[args.skip:args.limit]
    if args.variant:
        convs = load_longmemeval(None, variant=args.variant,
                                 question_ids=[c.queries[0].query_id for c in convs if c.queries])
    size = max(1, args.pool)
    units = [convs[i:i + size] for i in range(0, len(convs), size)]
    outcome = judged(args.run) if args.run else {}

    rows = []
    for unit in units:
        messages, per_message = harness.history_inputs(unit, extraction)
        facts, turn_of, conv_of = [], [], []
        for message, found in zip(messages, per_message):
            for fact in found:
                facts.append(fact)
                turn_of.append(message["turn_index"])
                conv_of.append(message["conversation"])
        evidence = defaultdict(set)
        for m in messages:
            if m["has_answer"]:
                evidence[m["conversation"]].add(m["turn_index"])
        vectors = np.asarray(embedder.encode(facts), dtype=np.float32) if facts else None
        if vectors is not None:
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
        for conv in unit:
            for q in conv.queries:
                if q.is_abstention or not evidence[conv.conv_id]:
                    continue
                ranks = []
                if vectors is not None:
                    qv = np.asarray(embedder.encode(q.query_text), dtype=np.float32)
                    order = np.argsort(-(vectors @ (qv / (np.linalg.norm(qv) + 1e-9))))
                    rank_of = np.empty(len(order), dtype=np.int64)
                    rank_of[order] = np.arange(1, len(order) + 1)
                for turn in sorted(evidence[conv.conv_id]):
                    hits = [i for i, t in enumerate(turn_of)
                            if t == turn and conv_of[i] == conv.conv_id]
                    ranks.append(int(min(rank_of[i] for i in hits)) if hits else None)
                in_context, right = outcome.get(q.query_id, (None, None))
                rows.append({"query_id": q.query_id, "type": q.question_type or "?",
                             "ranks": ranks, "category": category(ranks),
                             "in_context": in_context, "right": right})

    print(f"{len(rows)} questions, {len(units)} histories "
          f"({'pool ' + str(size) if size > 1 else 'one history each'}); "
          f"embedding misses: {embedder.misses} (stand-in vectors when > 0)")
    head =f"  {'evidence turn at worst':16s} {'questions':>9s}"
    if outcome:
        head += f" {'all in context':>15s} {'judged right':>13s}"
    print(head)
    for cat in CATEGORIES:
        sel = [r for r in rows if r["category"] == cat]
        line = f"  {cat:16s} {len(sel):9d}"
        if outcome:
            line += (f" {sum(bool(r['in_context']) for r in sel):15d}"
                     f" {sum(bool(r['right']) for r in sel):13d}")
        print(line)
    print("  by question type:")
    for kind in sorted({r["type"] for r in rows}):
        cells = Counter(r["category"] for r in rows if r["type"] == kind)
        print(f"    {kind:28s} " + "  ".join(f"{c}: {cells.get(c, 0)}" for c in CATEGORIES))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
