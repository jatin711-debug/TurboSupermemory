#!/usr/bin/env python3
"""LongMemEval — retention / forgetting under a memory budget (W5).

Reinforcement showed no DIRECT ranking lift (Phase 4). Its real claim is the
retain/forget axis: under a bounded store, a memory that gets USED (retrieved /
rehearsed) should survive eviction over one that doesn't — so the facts a user
keeps coming back to stay available, and dead weight is forgotten. Nobody had
tested that. This eval does, with the same disciplined ON-vs-OFF isolation.

All arms are IDENTICAL in every operation — same corpus, same insertion order,
same budget (`max_records` forces eviction), same rehearsal access signal.
They differ in ONE engine switch — the eviction ranking policy:

  - FIFO   (`access_aware_eviction=False`): naive FIFO — evict oldest-inserted
    first, ignoring access entirely (the "database, not memory" baseline).
  - LEGACY (`access_aware_eviction=True`):  evict by cognitive salience — a
    rehearsed/retrieved memory (high access_count, recently accessed) survives.
  - ACT-R  (`access_aware_eviction=True, actr_activation=True`): same salience
    gate, but candidates are ranked by ACT-R base-level activation
    ln(sum age^-d) over the last K access timestamps instead of the legacy
    access_count x 2^(-age/half_life) score.

Procedure per conversation (only those with > budget user-facts, so eviction
actually bites): insert all user facts → rehearse (bump access on facts, in
ALL arms) → consolidate (evict) → measure. Rehearsal modes:

  - sources (DEFAULT, honest): search the conversation's PAST USER-MESSAGE
    texts — users re-ask about their own facts over time, so the fact SOURCE
    texts are the legitimate, deployable access signal.
  - oracle (contaminated, comparison only): search the EVAL QUERY texts — you
    don't know future queries at eviction time in production. The old +0.41
    lift was mostly this leak (honest lift was +0.08).
  - none: no rehearsal; eviction relies on the engine's intrinsic salience.

Metrics:
  - survival: is the gold-answer fact still ALIVE after eviction?
  - hit@k:    is it still RETRIEVED (end-to-end retain + recall)?
  - judged accuracy: an LLM answers from the retrieved memories and an LLM
    grades vs gold (the GOLD STANDARD).

B3 kill/keep: ACT-R >= LEGACY on judged retention → keep ACT-R (parity still
simplifies the theory); a loss beyond noise (~0.05) is an honest negative.

Usage:
    python benchmarks/cognitive_eval/retention_eval.py --budget 10 --limit 120 \
        --judge openai --rehearse-mode sources
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cognitive_eval.adapters.tsm_adapter import TSMAdapter
from cognitive_eval.benchmark_datasets.longmemeval import load_longmemeval
from cognitive_eval.run_belief_longmemeval import hit_at, key_tokens, prewarm_extraction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("retention_eval")


def gold_memory_ids(answer, id_to_text):
    """Ids of inserted facts whose text carries the gold answer's distinctive
    token — the memories that must survive for the query to be answerable."""
    kts = key_tokens(answer)
    if not kts:
        return []
    tok = kts[0]
    return [mid for mid, text in id_to_text.items() if tok in (text or "").lower()]


def rehearsal_texts(conv, cap=24):
    """The honest (non-oracle) access signal: users re-ask about their own facts
    over time, so rehearse with the conversation's PAST USER-MESSAGE texts (the
    fact SOURCE texts) — never the eval query texts. To bound runtime we search
    an evenly-spaced sample (every Nth user message) of at most `cap` texts per
    conversation (all of them when there are fewer)."""
    texts = [m.content for m in conv.messages
             if m.role == "user" and m.content and m.content.strip()]
    if len(texts) > cap:
        step = len(texts) / cap
        texts = [texts[int(i * step)] for i in range(cap)]
    return texts


def run_arm(access_aware, conversations, budget, model_name, top_k, extractor, shared_model=None,
            judge=None, extractor_instance=None, judge_workers=8, rehearse_mode="oracle",
            actr=False, actr_decay=None, actr_history=None):
    """Identical operations in all arms; only the eviction policy differs.
    `rehearse_mode`: 'oracle' (eval queries — CONTAMINATED, comparison only),
    'sources' (past user-message texts — the honest signal), 'none'.
    `actr=True` switches eviction ranking to ACT-R base-level activation
    (requires access_aware=True). With `judge`, each post-eviction answer is
    additionally scored by the GOLD-STANDARD metric (LLM answers from the
    retrieved memories, LLM grades vs gold)."""
    agg = {"n": 0, "survived": 0, "h1": 0, "h3": 0, "hk": 0, "jn": 0, "jc": 0}
    judge_tasks = []  # (query, texts, gold)
    model = shared_model
    pressured_convs = 0
    for conv in conversations:
        db_path = tempfile.mkdtemp(prefix="tsm_ret_")
        adapter = TSMAdapter(db_path=db_path, embedding_model=model_name, extractor=extractor,
                             extractor_instance=extractor_instance,
                             cognitive_features=True, belief_revision=True, model=model,
                             belief_source_roles=["user"], max_records=budget,
                             access_aware_eviction=access_aware,
                             actr_activation=True if actr else None,
                             actr_decay=actr_decay, actr_history=actr_history)
        model = adapter.model
        try:
            adapter.add(conv.messages, user_id=conv.conv_id)
            n_facts = len(adapter._id_to_text)
            if n_facts <= budget:
                continue  # no budget pressure — eviction wouldn't fire
            pressured_convs += 1
            id_to_text = dict(adapter._id_to_text)

            # Rehearse: the facts a user keeps needing get accessed.
            #   oracle:  search each EVAL QUERY text 3x — an ORACLE leak (you
            #            don't know future queries at eviction time). Kept only
            #            for comparison with the contaminated old numbers.
            #   sources: search an evenly-spaced sample (<=24) of the
            #            conversation's PAST USER-MESSAGE texts 2x each — the
            #            honest, deployable signal (users re-ask about their
            #            own facts over time).
            #   none:    no rehearsal; eviction relies only on the engine's
            #            INTRINSIC salience (importance_auto_scoring /
            #            reinforcement).
            if rehearse_mode == "oracle":
                for q in conv.queries:
                    if q.is_abstention:
                        continue
                    for _ in range(3):
                        adapter.search(q.query_text, user_id=conv.conv_id, top_k=top_k, use_cognitive=False)
            elif rehearse_mode == "sources":
                for text in rehearsal_texts(conv):
                    for _ in range(2):
                        adapter.search(text, user_id=conv.conv_id, top_k=top_k, use_cognitive=False)

            adapter.trigger_consolidation()  # eviction fires under the budget

            for q in conv.queries:
                if q.is_abstention:
                    continue
                agg["n"] += 1
                golds = gold_memory_ids(q.answer_text, id_to_text)
                # survival: did any gold-bearing memory stay alive post-eviction?
                survived = any(adapter.engine.contains_id(mid) for mid in golds) if golds else False
                agg["survived"] += int(survived)
                res = adapter.search(q.query_text, user_id=conv.conv_id, top_k=top_k, use_cognitive=True)
                texts = [r.get("text", "") for r in res]
                agg["h1"] += int(hit_at(q.answer_text, texts, 1))
                agg["h3"] += int(hit_at(q.answer_text, texts, 3))
                agg["hk"] += int(hit_at(q.answer_text, texts, top_k))
                if judge is not None:
                    judge_tasks.append((q.query_text, texts, q.answer_text))
        finally:
            adapter.close()
            shutil.rmtree(db_path, ignore_errors=True)

    if judge is not None and judge_tasks:
        logger.info("Judging %d post-eviction answers concurrently (%d workers)...",
                    len(judge_tasks), judge_workers)
        with ThreadPoolExecutor(max_workers=judge_workers) as ex:
            verdicts = list(ex.map(lambda t: judge.answer_and_judge(t[0], t[1], t[2]), judge_tasks))
        agg["jn"] = len(verdicts)
        agg["jc"] = sum(int(v) for v in verdicts)
    return agg, pressured_convs, model


def main():
    ap = argparse.ArgumentParser(description="LongMemEval retention isolation (B3: FIFO vs legacy "
                                             "access-aware vs ACT-R eviction)")
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--limit", type=int, default=200, help="cap #conversations (full set OOMs, see W7)")
    ap.add_argument("--budget", type=int, default=10, help="max_records cap (facts above this are evicted)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--tsm-embedder", choices=["local", "openai"], default="local",
                    help="embedding backend: local MiniLM (384-d) or OpenAI (--embed-model). "
                         "Use openai to re-validate the retention lift without the weak-embedder confound.")
    ap.add_argument("--embed-model", type=str, default="text-embedding-3-small",
                    help="OpenAI embedding model when --tsm-embedder openai")
    ap.add_argument("--rehearse-mode", choices=["oracle", "sources", "none"], default="sources",
                    help="access-signal rehearsal before eviction: 'sources' (DEFAULT, honest) = "
                         "search the conversation's PAST USER-MESSAGE texts (users re-ask about "
                         "their own facts); 'oracle' = search the EVAL QUERIES themselves "
                         "(CONTAMINATED — kept only for comparison with old numbers); "
                         "'none' = no rehearsal (intrinsic salience only)")
    ap.add_argument("--no-rehearse", action="store_true",
                    help="alias for --rehearse-mode none")
    ap.add_argument("--actr-decay", type=float, default=None,
                    help="ACT-R decay exponent d for the ACT-R arm (default: engine 0.5)")
    ap.add_argument("--actr-history", type=int, default=None,
                    help="ACT-R access-timestamp ring length for the ACT-R arm (default: engine 8)")
    ap.add_argument("--extractor", type=str, default="mock",
                    choices=["mock", "ollama", "openai", "auto"],
                    help="fact extractor; 'auto' = Ollama if reachable else OpenAI")
    ap.add_argument("--judge", choices=["none", "auto", "ollama", "openai"], default="none",
                    help="A2 GOLD STANDARD: LLM answers from the post-eviction retrieval and an "
                         "LLM grades vs gold — does the survival win convert to answers?")
    ap.add_argument("--judge-model", type=str, default=None)
    ap.add_argument("--extractor-model", type=str, default=None,
                    help="override the OpenAI extractor model (RPD limits are per-model; "
                         "e.g. gpt-4.1-nano when gpt-4o-mini's daily quota is spent)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    judge = None
    if args.judge != "none":
        from cognitive_eval.judge import create_judge
        kw = {}
        if args.judge_model:
            kw = {"ollama_model": args.judge_model, "openai_model": args.judge_model}
        judge = create_judge(args.judge, **kw)
        logger.info("LLM-judge ENABLED (%s: %s)", args.judge, type(judge).__name__)
    from cognitive_eval.extraction import create_extractor
    ekw = {"openai_model": args.extractor_model} if args.extractor_model else {}
    shared_extractor = create_extractor(args.extractor, **ekw)

    convs = load_longmemeval(args.data_dir)
    if args.limit:
        convs = convs[:args.limit]
    logger.info("Loaded %d conversations. budget(max_records)=%d model=%s top_k=%d",
                len(convs), args.budget, args.model, args.top_k)
    prewarm_extraction(shared_extractor, convs, workers=max(args.workers, 8))
    for name in ("cognitive_eval.adapters.tsm", "httpx", "httpcore", "urllib3",
                 "huggingface_hub", "sentence_transformers", "transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)

    init_model = None
    if args.tsm_embedder == "openai":
        from cognitive_eval.openai_embedder import OpenAIEmbedder
        init_model = OpenAIEmbedder(model=args.embed_model)
        logger.info("Embeddings: OpenAI %s (dim=%d) — retention re-validation",
                    args.embed_model, init_model.get_sentence_embedding_dimension())

    rehearse_mode = "none" if args.no_rehearse else args.rehearse_mode
    if rehearse_mode == "oracle":
        logger.warning("Rehearsal mode ORACLE: the access signal uses the EVAL QUERIES "
                       "themselves — results are oracle-contaminated and kept only for "
                       "comparison with the old (leaked) numbers.")
    else:
        logger.info("Rehearsal mode: %s", rehearse_mode.upper()
                    + (" (past user-message texts — honest signal)" if rehearse_mode == "sources"
                       else " (intrinsic salience only)"))

    # B3 kill/keep: three arms, identical operations, differing only in the
    # eviction ranking policy.
    arms = [
        ("fifo",   False, False),  # naive FIFO baseline
        ("legacy", True,  False),  # access_count x 2^(-age/half_life)
        ("actr",   True,  True),   # ACT-R base-level activation ln(sum age^-d)
    ]
    results = {}
    model = init_model
    for name, access_aware, actr in arms:
        logger.info("Running %s arm (access_aware_eviction=%s, actr_activation=%s)...",
                    name.upper(), access_aware, actr)
        agg, pressured, model = run_arm(access_aware, convs, args.budget, args.model, args.top_k,
                                        args.extractor, shared_model=model, judge=judge,
                                        extractor_instance=shared_extractor,
                                        judge_workers=args.workers, rehearse_mode=rehearse_mode,
                                        actr=actr, actr_decay=args.actr_decay,
                                        actr_history=args.actr_history)
        results[name] = (agg, pressured)

    n = max(r[0]["n"] for r in results.values()) or 1
    logger.info("=" * 100)
    logger.info("Budget-pressured conversations: %s | scored queries: %d",
                " / ".join(f"{name.upper()} {p}" for name, (_, p) in results.items()), n)
    logger.info("%-16s | %-10s | %-10s | %-10s", "metric", "FIFO", "LEGACY", "ACT-R")
    logger.info("-" * 100)

    def rate(agg, num_key, den_key="n"):
        return agg[num_key] / (agg[den_key] or 1)

    def row(label, key):
        vals = {name: rate(agg, key) for name, (agg, _) in results.items()}
        logger.info("%-16s | %-10.2f | %-10.2f | %-10.2f",
                    label, vals["fifo"], vals["legacy"], vals["actr"])
        return vals

    surv = row("gold survival", "survived")
    row("hit@1", "h1")
    row("hit@3", "h3")
    hk = row("hit@k", "hk")
    judged = None
    if judge is not None and any(agg["jn"] for agg, _ in results.values()):
        judged = {name: rate(agg, "jc", "jn") for name, (agg, _) in results.items()}
        logger.info("%-16s | %-10.2f | %-10.2f | %-10.2f   <== GOLD STANDARD (judge=%s, calls=%d)",
                    "judged accuracy", judged["fifo"], judged["legacy"], judged["actr"],
                    getattr(judge, "model", "?"), judge.calls)
    logger.info("=" * 100)

    # B3 kill/keep verdict: ACT-R >= LEGACY on judged retention -> keep
    # (parity still simplifies the theory); a loss beyond noise (~0.05) kills.
    metric_name = "judged accuracy" if judged is not None else "hit@k"
    metric = judged if judged is not None else hk
    delta_al = metric["actr"] - metric["legacy"]
    delta_lf = metric["legacy"] - metric["fifo"]
    logger.info("LEGACY - FIFO %s: %+.4f | ACT-R - LEGACY %s: %+.4f (metric=%s)",
                metric_name, delta_lf, metric_name, delta_al, metric_name)
    if delta_al >= 0.0:
        verdict = "keep"
        logger.info("B3 VERDICT: KEEP ACT-R — it beats legacy access-aware eviction on %s "
                    "(%+.4f).", metric_name, delta_al)
    elif delta_al >= -0.05:
        verdict = "keep"
        logger.info("B3 VERDICT: KEEP ACT-R — parity with legacy within noise (%+.4f); "
                    "the simpler, principled theory wins the tie.", delta_al)
    else:
        verdict = "kill"
        logger.info("B3 VERDICT: KILL ACT-R — it loses to legacy beyond noise (%+.4f). "
                    "Honest negative.", delta_al)

    summary = {
        "budget": args.budget,
        "embedder": args.embed_model if args.tsm_embedder == "openai" else "minilm-384",
        "rehearse_mode": rehearse_mode,
        "scored_queries": n,
        "metric_for_verdict": metric_name,
        "arms": {
            name: {
                "pressured_convs": p,
                "survival": round(rate(agg, "survived"), 4),
                "hit1": round(rate(agg, "h1"), 4),
                "hit3": round(rate(agg, "h3"), 4),
                "hitk": round(rate(agg, "hk"), 4),
                **({"judged_acc": round(rate(agg, "jc", "jn"), 4)} if judged is not None else {}),
            }
            for name, (agg, p) in results.items()
        },
        "legacy_minus_fifo": round(delta_lf, 4),
        "actr_minus_legacy": round(delta_al, 4),
        "verdict": verdict,
    }
    logger.info("GATE_SUMMARY: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
