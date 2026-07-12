#!/usr/bin/env python3
"""B4 — compress-instead-of-delete (rate-distortion retention).

When memory is over budget, does replacing the evicted tail with a short GIST
beat deleting it outright? Rate-distortion view (2026): a lossy summary within
budget may still answer questions whose exact fact was evicted, where pure
deletion cannot.

Isolation per conversation: keep the most-recent `budget-1` facts as SURVIVORS
in both arms; the older EVICTED tail is either
  - DELETE arm:   dropped entirely (store = survivors), or
  - COMPRESS arm: replaced by ONE gist memory (store = survivors + gist).
Then retrieve under a fixed token budget (MMR, B2) and grade the answer with the
gold-standard judge. We break results out by whether the gold answer lived in an
EVICTED fact — that subset is where compression can possibly help.

Usage:
    python benchmarks/cognitive_eval/compress_eval.py --limit 120 --budget 8 \
        --token-budget 150 --extractor openai --extractor-model gpt-4.1-nano \
        --gist-model gpt-4.1-nano --judge openai --judge-model gpt-4.1-mini --workers 10
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cognitive_eval.adapters.tsm_adapter import TSMAdapter
from cognitive_eval.benchmark_datasets.longmemeval import load_longmemeval
from cognitive_eval.run_belief_longmemeval import key_tokens, _msg_content

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("compress_eval")


def conv_user_facts(extractor, conv):
    """Ordered list of the user's atomic facts (cache-backed extraction)."""
    facts = []
    ctx = []
    for m in conv.messages:
        role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else "user")
        c = _msg_content(m)
        if not c or not c.strip():
            continue
        if role == "user":
            facts.extend(extractor.extract_facts(c, ctx))
        ctx.append(c)
    # de-dup preserving order
    seen, out = set(), []
    for f in facts:
        if f and f not in seen:
            seen.add(f)
            out.append(f)
    return out


def insert_facts(adapter, facts, user_id):
    """Insert raw fact strings directly (bypassing add()'s extraction)."""
    if not facts:
        return
    embs = adapter.model.encode(facts)
    embs = np.asarray(embs, dtype=np.float32)
    for i, f in enumerate(facts):
        mid = f"{user_id}_f{i}"
        adapter._id_to_text[mid] = f
        adapter.engine.insert(id=mid, text=f, embedding=embs[i].astype(np.float32),
                              importance_score=1.0, concepts=adapter._extract_concepts(f),
                              payload=None, scope=user_id, source_role="user")


def main():
    ap = argparse.ArgumentParser(description="B4 compress-instead-of-delete (judged)")
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--budget", type=int, default=8, help="#facts kept (survivors = budget-1)")
    ap.add_argument("--token-budget", type=int, default=150, help="answer-context token budget")
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--extractor", type=str, default="openai")
    ap.add_argument("--extractor-model", type=str, default=None)
    ap.add_argument("--gist-model", type=str, default="gpt-4.1-nano")
    ap.add_argument("--gister", choices=["openai", "ollama"], default="openai")
    ap.add_argument("--judge", choices=["auto", "ollama", "openai"], default="openai")
    ap.add_argument("--judge-model", type=str, default=None)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    from cognitive_eval.judge import create_judge
    jkw = {"ollama_model": args.judge_model, "openai_model": args.judge_model} if args.judge_model else {}
    judge = create_judge(args.judge, **jkw)
    from cognitive_eval.extraction import create_extractor
    ekw = {"openai_model": args.extractor_model} if args.extractor_model else {}
    extractor = create_extractor(args.extractor, **ekw)
    from cognitive_eval.gist import create_gister
    gister = create_gister(args.gister, model=args.gist_model)

    convs = load_longmemeval(args.data_dir)[:args.limit]
    logger.info("Loaded %d conversations. budget=%d token_budget=%d", len(convs), args.budget, args.token_budget)
    for name in ("cognitive_eval.adapters.tsm", "httpx", "httpcore", "urllib3",
                 "huggingface_hub", "sentence_transformers", "transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)

    tasks = []  # (arm, in_evicted, query, texts, gold)
    model = None
    pressured = 0
    for conv in convs:
        facts = conv_user_facts(extractor, conv)
        if len(facts) <= args.budget:
            continue
        pressured += 1
        # FAIR budget: both arms hold exactly `budget` memory slots. They share
        # the same `budget-1` most-recent survivor facts and differ ONLY in the
        # last slot: DELETE fills it with one more individual fact (the newest of
        # the tail); COMPRESS fills it with a gist of the WHOLE older tail. So the
        # question is precisely "is a gist of the tail worth more than one extra
        # individual fact?" — no information-quantity advantage either way.
        survivors = facts[-(args.budget - 1):]           # shared by both arms
        tail = facts[:-(args.budget - 1)]                 # everything older (gist covers this)
        delete_store = facts[-args.budget:]               # survivors + 1 more recent fact
        gist = gister.summarize(tail)
        compress_store = list(survivors) + ([gist] if gist else [])

        # queries whose gold answer is in the older tail the delete arm dropped
        def in_evicted(ans):
            kts = key_tokens(ans)
            if not kts:
                return False
            tok = kts[0]
            in_tail = any(tok in (e or "").lower() for e in tail)
            in_delete = any(tok in (e or "").lower() for e in delete_store)
            return in_tail and not in_delete

        for arm in ("delete", "compress"):
            db = tempfile.mkdtemp(prefix="tsm_b4_")
            adapter = TSMAdapter(db_path=db, embedding_model=args.model, extractor="mock",
                                 cognitive_features=True, belief_revision=False, model=model)
            model = adapter.model
            try:
                store = delete_store if arm == "delete" else compress_store
                insert_facts(adapter, store, conv.conv_id)
                adapter.trigger_consolidation()
                for q in conv.queries:
                    if q.is_abstention:
                        continue
                    texts = adapter.recall_under_budget(q.query_text, token_budget=args.token_budget,
                                                        method="mmr")
                    tasks.append((arm, in_evicted(q.answer_text), q.query_text, texts, q.answer_text))
            finally:
                adapter.close()
                shutil.rmtree(db, ignore_errors=True)

    logger.info("Pressured convs: %d | judging %d answers (%d workers)...", pressured, len(tasks), args.workers)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        verdicts = list(ex.map(lambda t: judge.answer_and_judge(t[2], t[3], t[4]), tasks))

    # agg[subset][arm] = [n, correct]
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for (arm, in_ev, _q, _t, _g), correct in zip(tasks, verdicts):
        for subset in ("all", "answer_in_evicted" if in_ev else "answer_in_survivors"):
            agg[subset][arm][0] += 1
            agg[subset][arm][1] += int(correct)

    logger.info("=" * 90)
    summary = {}
    for subset in ("all", "answer_in_evicted", "answer_in_survivors"):
        d = agg[subset]
        da = d["delete"][1] / (d["delete"][0] or 1)
        ca = d["compress"][1] / (d["compress"][0] or 1)
        n = max(d["delete"][0], d["compress"][0])
        logger.info("%-20s | delete %.2f | compress %.2f | lift %+.2f  (n=%d)", subset, da, ca, ca - da, n)
        summary[subset] = {"delete": round(da, 4), "compress": round(ca, 4), "lift": round(ca - da, 4), "n": n}
    logger.info("=" * 90)
    ev = summary.get("answer_in_evicted", {})
    if ev.get("n", 0) >= 5 and ev["lift"] > 0.05:
        logger.info("B4 VERDICT: compress-instead-of-delete RECOVERS answers whose fact was evicted "
                    "(+%.2f on that subset) — the gist retains queryable detail vs pure deletion.", ev["lift"])
    elif summary["all"]["lift"] > 0.03:
        logger.info("B4 VERDICT: compress gives a small overall judged gain (+%.2f).", summary["all"]["lift"])
    else:
        logger.info("B4 VERDICT: gist does not recover evicted answers over deletion (honest negative — "
                    "the summary is too lossy for specific-fact recall).")
    logger.info("GATE_SUMMARY: %s", json.dumps({"budget": args.budget, "token_budget": args.token_budget,
                "pressured_convs": pressured, "by_subset": summary,
                "gist_calls": gister.calls, "judge_calls": judge.calls}))


if __name__ == "__main__":
    main()
