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
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cognitive_eval.adapters.tsm_adapter import TSMAdapter
from cognitive_eval.benchmark_datasets.longmemeval import load_longmemeval

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("belief_longmemeval")


def _msg_content(m):
    return m.content if hasattr(m, "content") else (m.get("content", "") if isinstance(m, dict) else "")


def prewarm_extraction(extractor, conversations, workers=12):
    """Extract every unique message CONCURRENTLY up front to fill the extractor's
    cache, so the per-conversation add() (and the second arm) hit the cache
    instead of paying serial LLM latency. No-op for extractors without a cache
    (e.g. mock)."""
    if not hasattr(extractor, "_cache"):
        return
    seen, msgs = set(), []
    for conv in conversations:
        for m in conv.messages:
            c = _msg_content(m)
            if c and c.strip() and c not in seen:
                seen.add(c)
                msgs.append(c)
    if not msgs:
        return
    logger.info("Pre-extracting %d unique messages concurrently (%d workers)...", len(msgs), workers)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda t: extractor.extract_facts(t), msgs))
    if hasattr(extractor, "flush_cache"):
        extractor.flush_cache()
    logger.info("Pre-extraction done (extractor calls=%s).", getattr(extractor, "calls", "?"))

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
            store_roles=None, belief_source_roles=None, verify_demotions=False,
            shared_verifier=None, judge=None, extractor_instance=None, judge_workers=8,
            judge_ks=None):
    """One fresh engine PER conversation (belief detection stays within a user,
    matching a properly-scoped store and isolating the mechanism). The embedding
    model is loaded once and shared across all engines.

    When `judge` is supplied, each query is additionally scored by the
    GOLD-STANDARD metric at every k in `judge_ks` (default [top_k]): the judge
    answers from the TRUNCATED top-k retrieved memories, then grades that answer
    against gold. Retrieval runs ONCE at top_k; truncation gives the low-k
    points (A1) and the per-k context-token counts give accuracy-per-token (A3)
    from the same run. Returned `judged[type][k] = {n, c, tok}`."""
    by_type = defaultdict(lambda: {"n": 0, "h1": 0, "h3": 0, "hk": 0})
    judged = defaultdict(dict)  # type -> k -> {n, c, tok}
    judge_ks = sorted(set(judge_ks or [top_k]))
    edges = {"refine": 0, "contra": 0}
    judge_tasks = []  # (type, query, texts, gold) — judged concurrently after retrieval
    model = shared_model
    verifier = shared_verifier
    if belief_on and verify_demotions and verifier is None:
        from cognitive_eval.verification import NLIVerifier
        verifier = NLIVerifier()
    for conv in conversations:
        db_path = tempfile.mkdtemp(prefix="tsm_lme_")
        adapter = TSMAdapter(db_path=db_path, embedding_model=model_name, extractor=extractor,
                             extractor_instance=extractor_instance,
                             cognitive_features=True, belief_revision=belief_on, model=model,
                             store_roles=store_roles, belief_source_roles=belief_source_roles,
                             verify_demotions=(belief_on and verify_demotions), verifier=verifier)
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
                if judge is not None:
                    # Defer the (I/O-bound) judge calls; run them concurrently
                    # after retrieval so a whole arm isn't gated on serial latency.
                    judge_tasks.append((q.question_type or "?", q.query_text, texts, q.answer_text))
        finally:
            adapter.close()
            shutil.rmtree(db_path, ignore_errors=True)

    if judge is not None and judge_tasks:
        pairs = [(i, k) for i in range(len(judge_tasks)) for k in judge_ks]
        logger.info("Judging %d answers (%d queries x k=%s) concurrently (%d workers)...",
                    len(pairs), len(judge_tasks), judge_ks, judge_workers)

        def _judge_at_k(pair):
            i, k = pair
            _t, q, texts, gold = judge_tasks[i]
            return judge.answer_and_judge(q, texts[:k], gold)

        with ThreadPoolExecutor(max_workers=judge_workers) as ex:
            verdicts = list(ex.map(_judge_at_k, pairs))
        for (i, k), correct in zip(pairs, verdicts):
            tkey, _q, texts, _g = judge_tasks[i]
            cell = judged[tkey].setdefault(k, {"n": 0, "c": 0, "tok": 0})
            cell["n"] += 1
            cell["c"] += int(correct)
            # ~4 chars/token: good enough for a cost curve.
            cell["tok"] += sum(len(t) // 4 for t in texts[:k] if t)
    return by_type, edges, model, verifier, judged


def main():
    ap = argparse.ArgumentParser(description="LongMemEval belief-revision isolation (ON vs OFF)")
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap #conversations (quick pipeline check)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--extractor", type=str, default="mock",
                    choices=["mock", "ollama", "openai", "auto"],
                    help="fact extractor. 'auto' = Ollama if reachable else OpenAI (real LLM "
                         "extraction); 'mock' is the offline gate-only splitter.")
    ap.add_argument("--user-only", action="store_true",
                    help="MODE A: store only USER messages as facts (drop assistant turns entirely). "
                         "belief revision should operate on the user's evolving statements, not the "
                         "assistant's verbose responses)")
    ap.add_argument("--role-filtered", action="store_true",
                    help="MODE B: store ALL roles (every memory stays retrievable) but restrict "
                         "belief-revision detection to user facts via the engine's first-class "
                         "belief_source_roles. This is the productized form of --user-only.")
    ap.add_argument("--verify-demotions", action="store_true",
                    help="W3: gate each proposed supersession through a local NLI cross-encoder "
                         "before the destructive demotion (propose -> verify -> commit).")
    ap.add_argument("--judge", choices=["none", "auto", "ollama", "openai"], default="none",
                    help="W6 GOLD-STANDARD metric: score answers with an LLM judge "
                         "(retrieve -> LLM answers from memories -> LLM grades vs gold), not just "
                         "the retrieval proxy. 'auto' = Ollama if reachable (free) else OpenAI.")
    ap.add_argument("--judge-model", type=str, default=None,
                    help="override judge model (default: qwen2.5:3b for ollama, gpt-4o-mini for openai)")
    ap.add_argument("--judge-ks", type=str, default=None,
                    help="comma-separated ks to judge at by TRUNCATING the top_k retrieval "
                         "(e.g. '1,3,5,10'). One retrieval pass gives the whole accuracy-vs-"
                         "context-budget curve (Phase A1+A3). Default: just top_k.")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrency for LLM extraction/judging (I/O-bound API calls)")
    args = ap.parse_args()
    judge_ks = None
    if args.judge_ks:
        judge_ks = sorted({int(x) for x in args.judge_ks.split(",") if x.strip()})
        if any(k > args.top_k for k in judge_ks):
            ap.error(f"--judge-ks entries must be <= --top-k ({args.top_k})")
    store_roles = {"user"} if args.user_only else None
    belief_source_roles = ["user"] if args.role_filtered else None

    judge = None
    if args.judge != "none":
        from cognitive_eval.judge import create_judge
        kw = {}
        if args.judge_model:
            kw = {"ollama_model": args.judge_model, "openai_model": args.judge_model}
        judge = create_judge(args.judge, **kw)
        logger.info("LLM-judge ENABLED (%s: %s)", args.judge, type(judge).__name__)

    # Build ONE extractor and share it across both arms so its cross-arm cache
    # means the identical corpus is only extracted once (halves LLM cost/time).
    from cognitive_eval.extraction import create_extractor
    shared_extractor = create_extractor(args.extractor)

    convs = load_longmemeval(args.data_dir)
    if args.limit:
        convs = convs[:args.limit]
    n_q = sum(len(c.queries) for c in convs)
    logger.info("Loaded %d conversations, %d queries. Model=%s extractor=%s top_k=%d",
                len(convs), n_q, args.model, args.extractor, args.top_k)

    # Warm the extractor cache concurrently so the arms don't pay serial LLM
    # latency per message (the dominant cost with a real extractor).
    prewarm_extraction(shared_extractor, convs, workers=max(args.workers, 8))

    # Quiet the per-conversation adapter + HF download chatter.
    for name in ("cognitive_eval.adapters.tsm", "httpx", "httpcore", "urllib3",
                 "huggingface_hub", "sentence_transformers", "transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)

    logger.info("Running OFF arm (belief revision disabled)... store_roles=%s belief_source_roles=%s "
                "verify_demotions=%s", store_roles, belief_source_roles, args.verify_demotions)
    off, off_edges, model, _, off_j = run_arm(
        False, convs, args.model, args.top_k, args.extractor,
        store_roles=store_roles, belief_source_roles=belief_source_roles,
        judge=judge, extractor_instance=shared_extractor,
        judge_workers=args.workers, judge_ks=judge_ks)
    logger.info("Running ON arm (belief revision enabled)...")
    on, on_edges, _, _, on_j = run_arm(
        True, convs, args.model, args.top_k, args.extractor,
        shared_model=model, store_roles=store_roles,
        belief_source_roles=belief_source_roles,
        verify_demotions=args.verify_demotions, judge=judge,
        extractor_instance=shared_extractor, judge_workers=args.workers, judge_ks=judge_ks)

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

    # GOLD-STANDARD LLM-judged accuracy at each context budget k (Phase A1+A3):
    # did an LLM answer correctly from the TRUNCATED top-k? Reported per type
    # per k, with the mean context tokens per query at that k (the cost axis).
    judged_summary = None
    if judge is not None:
        ks = judge_ks or [args.top_k]
        logger.info("LLM-JUDGED answer accuracy (gold standard)   [judge=%s model=%s, calls=%d]",
                    type(judge).__name__, getattr(judge, "model", "?"), judge.calls)

        def jcell(jd, t, k):
            c = jd.get(t, {}).get(k)
            return (c["c"] / c["n"], c["tok"] / c["n"], c["n"]) if c and c["n"] else (0.0, 0.0, 0)

        judged_summary = {}
        all_types = sorted(set(off_j) | set(on_j), key=lambda t: (t != "knowledge-update", t))
        for k in ks:
            # mean context tokens/query at this k (ON arm; OFF is ~identical)
            toks = [jcell(on_j, t, k)[1] for t in all_types if jcell(on_j, t, k)[2]]
            avg_tok = sum(toks) / len(toks) if toks else 0.0
            logger.info("-" * 72)
            logger.info("k=%-2d (avg ctx ~%d tok/query)   %-26s %5s | %s",
                        k, avg_tok, "question_type", "n", "acc  off/on/lift")
            ksum = {}
            for t in all_types:
                of, _, no = jcell(off_j, t, k)
                oo, _, nn = jcell(on_j, t, k)
                if not no and not nn:
                    continue
                mark = "  <== belief revision" if t == "knowledge-update" else ""
                logger.info("%36s %-26s %5d | %.2f/%.2f/%+.2f%s", "", t, max(no, nn),
                            of, oo, oo - of, mark)
                ksum[t] = round(oo - of, 4)
            judged_summary[str(k)] = {"type_lift": ksum, "avg_ctx_tokens": round(avg_tok, 1)}
        logger.info("-" * 72)
        # Headline: knowledge-update judged lift per k (the A1 kill/keep signal).
        for k in ks:
            of, _, no = jcell(off_j, "knowledge-update", k)
            oo, _, _ = jcell(on_j, "knowledge-update", k)
            if no:
                logger.info("KNOWLEDGE-UPDATE judged lift @ k=%-2d: %+.2f  (off %.2f -> on %.2f, n=%d)",
                            k, oo - of, of, oo, no)
        low_ks = [k for k in ks if k <= 3]
        if low_ks:
            best_low = max((jcell(on_j, "knowledge-update", k)[0] -
                            jcell(off_j, "knowledge-update", k)[0]) for k in low_ks)
            if best_low > 0.05:
                logger.info("GOLD VERDICT (A1): belief revision improves judged accuracy under a TIGHT "
                            "context budget (k<=3) — the rank win pays off where context is scarce.")
            else:
                logger.info("GOLD VERDICT (A1): no judged-accuracy lift even at k<=3 — the belief-revision "
                            "rank win does not convert to answers at any tested budget.")
        logger.info("=" * 100)

    # Machine-readable one-liner for the regression gate (W2). Contains the
    # knowledge-update lift, total ON edges, and per-type hit@1 lift so the gate
    # can assert non-regression without parsing the human table.
    def _h1_lift(t):
        o, n = off.get(t), on.get(t)
        if not o or not n or not o["n"] or not n["n"]:
            return 0.0
        return round(n["h1"] / n["n"] - o["h1"] / o["n"], 4)
    ku_o2 = off.get("knowledge-update")
    summary = {
        "ku_hit1_lift": _h1_lift("knowledge-update"),
        "ku_n": (ku_o2 or {}).get("n", 0),
        "on_edges": on_edges["refine"] + on_edges["contra"],
        "on_refine": on_edges["refine"],
        "on_contra": on_edges["contra"],
        "type_hit1_lift": {t: _h1_lift(t) for t in (set(off) | set(on))},
    }
    if judged_summary is not None:
        summary["judged_by_k"] = judged_summary
        summary["judge_calls"] = judge.calls
    logger.info("GATE_SUMMARY: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
