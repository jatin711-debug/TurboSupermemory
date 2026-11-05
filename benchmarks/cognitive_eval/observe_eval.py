"""Observation memory on LongMemEval-S: build the contexts, judge with reread.py.

Every session of a question's history is observed (``tsm.observe``), each
observation dated with the session's date (relative times resolved against it
by ``tsm.when``), and the question is asked over the observations. Arms:

    obs_log     every observation of the history, in the order said: the
                consolidated record as a whole (observational memory)
    obs_N       (--select N,...) the observations that best match the
                question within N tokens, in the order said: what
                ``Memory.observations(user, query, token_budget=N)`` returns
                (ranked here by cosine alone, with --embedder)
    mix_N       (--select) the best observations within N/2 tokens and the
                first N/2 tokens of the saved run's own context

The output is in ``shipped_stack_eval.py``'s per-question format, so
``reread.py`` compares it with any saved arm under one reader:

    python benchmarks/cognitive_eval/observe_eval.py --questions-from RUN.json \\
        --out OBS.json
    python benchmarks/cognitive_eval/reread.py --provider anthropic --reads 3 \\
        --arm raw=RUN.json:sdk_raw --arm obs=OBS.json:obs_log

``--questions-from`` takes the question ids (and their question dates) of a
saved run, so both arms answer the same questions. Paid: one observer request
per session not yet in the cache (claude-haiku-5-5 by default, about three
cents a LongMemEval-S history).
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cognitive_eval._secrets import ensure_anthropic_key, ensure_openai_key  # noqa: E402
from tsm.budget import estimate_tokens  # noqa: E402
from tsm.observe import AnthropicObserver, OpenAIObserver  # noqa: E402
from tsm.when import annotate, said_on  # noqa: E402

logger = logging.getLogger("observe_eval")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "longmemeval_official",
                    "longmemeval_s_cleaned.json")


def dated(line: str, session_date: str) -> str:
    """``[YYYY-MM-DD] observation``, a relative time followed by its date."""
    day = said_on(session_date)
    tag = day.isoformat() if day else session_date[:10]
    return f"[{tag}] {annotate(line, day)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--questions-from", required=True, metavar="RUN.json")
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    ap.add_argument("--observer-model", default=None)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--work-dir", default=os.environ.get("TSM_EVAL_DIR") or os.path.join(
        os.path.expanduser("~"), ".tsm_eval", "shipped_stack_eval"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--select", default=None, metavar="N[,N]",
                    help="also arms obs_N and mix_N (token budgets)")
    ap.add_argument("--embedder", default="google/embeddinggemma-2",
                    help="local sentence-transformers model that ranks observations")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    with open(args.questions_from, encoding="utf-8") as fh:
        wanted = {}
        for t in json.load(fh)["tasks"]:
            wanted.setdefault(t["query_id"], t)
    with open(DATA, encoding="utf-8") as fh:
        items = [d for d in json.load(fh) if d["question_id"] in wanted]
    if args.provider == "anthropic":
        if not ensure_anthropic_key():
            sys.exit("no Anthropic key (ANTHROPIC_API_KEY or claude_key.txt)")
        observer = AnthropicObserver(model=args.observer_model or "claude-haiku-5-5",
                                     cache_dir=args.work_dir, workers=args.workers)
    else:
        if not ensure_openai_key():
            sys.exit("OPENAI_API_KEY is not set")
        observer = OpenAIObserver(model=args.observer_model or "gpt-4o-mini",
                                  cache_dir=args.work_dir, workers=args.workers)

    # Every session of every history at once: the observer's pool stays full.
    sessions, owners = [], []
    for k, item in enumerate(items):
        for date, session in zip(item["haystack_dates"], item["haystack_sessions"]):
            sessions.append(session)
            owners.append((k, date))
    logger.info("%d questions, %d sessions to observe (cached ones are free)",
                len(items), len(sessions))
    try:
        found = observer.observe_many(sessions)
    finally:
        observer.flush_cache()
    logs = [[] for _ in items]
    for (k, date), lines in zip(owners, found):
        logs[k].extend(dated(line, date) for line in lines)

    contexts = {"obs_log": logs}
    if args.select:
        budgets = [int(b) for b in args.select.split(",")]
        contexts.update(select(items, logs, wanted, budgets, args.embedder))

    tasks, sizes = [], []
    for arm, per_question in contexts.items():
        for item, log in zip(items, per_question):
            saved = wanted[item["question_id"]]
            if arm == "obs_log":
                sizes.append(estimate_tokens("\n".join(log)))
            tasks.append({"arm": arm, "conversation_id": item["question_id"],
                          "query_id": item["question_id"],
                          "question_type": item["question_type"], "query": item["question"],
                          "gold": saved["gold"],
                          "asked": saved.get("asked") or item["question_date"],
                          "reader": saved.get("reader", "notes"), "retrieved": log,
                          "turns": [None] * len(log), "prediction": None, "correct": None})
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": {"arm": "obs_log", "observer": observer.model,
                               "questions": len(tasks),
                               "mean_log_tokens": sum(sizes) / max(1, len(sizes))},
                   "tasks": tasks}, fh)
    logger.info("wrote %s: %d questions, observation log %.0f tokens on average "
                "(max %d); observer requests %d, tokens in %s out %s", args.out, len(tasks),
                sum(sizes) / max(1, len(sizes)), max(sizes or [0]), observer.calls,
                f"{observer.input_tokens:,}", f"{observer.output_tokens:,}")


def select(items, logs, wanted, budgets, model):
    """``obs_N`` and ``mix_N`` contexts: the observations that best match each
    question (cosine, ``model``) within the budget, in the order said."""
    import numpy as np

    from tsm.embedders import SentenceTransformerEmbedder
    from tsm.interfaces import embed_documents, embed_query

    embedder = SentenceTransformerEmbedder(model)
    out = {f"{kind}_{b}": [] for b in budgets for kind in ("obs", "mix")}
    for n, (item, log) in enumerate(zip(items, logs)):
        if not log:
            for key in out:
                out[key].append([])
            continue
        docs = np.asarray(embed_documents(embedder, log), dtype=np.float32)
        query = np.asarray(embed_query(embedder, item["question"]), dtype=np.float32)
        docs /= np.linalg.norm(docs, axis=1, keepdims=True) + 1e-9
        query /= np.linalg.norm(query) + 1e-9
        order = np.argsort(-(docs @ query))
        cost = [estimate_tokens(line) for line in log]
        saved = wanted[item["question_id"]]["retrieved"]
        for b in budgets:
            for kind, share in (("obs", b), ("mix", b // 2)):
                kept, used = [], 0
                for k in order:
                    if used + cost[k] <= share:
                        kept.append(int(k))
                        used += cost[k]
                lines = [log[k] for k in sorted(kept)]  # the order said
                if kind == "mix":
                    room, raw = b - used, []
                    for text in saved:
                        size = estimate_tokens(text)
                        if size > room:
                            break
                        raw.append(text)
                        room -= size
                    lines = lines + raw
                out[f"{kind}_{b}"].append(lines)
        if (n + 1) % 20 == 0:
            logger.info("selected for %d/%d questions", n + 1, len(items))
    return out


if __name__ == "__main__":
    main()
