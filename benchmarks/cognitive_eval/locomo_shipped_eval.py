#!/usr/bin/env python3
"""tsm.Memory as shipped, on LoCoMo: free evidence score, optionally judged.

LoCoMo (Maharana et al. 2024; snap-research/locomo, ``data/locomo10.json``)
is ten conversations between two people, about 590 turns each over up to 35
dated sessions, with 1,986 questions and the turns that hold each answer
(``evidence``, as dialogue ids like ``D1:3``). It is a different kind of
history from LongMemEval's user-and-assistant chats: two people, either of
whom a question can be about. Nothing in tsm was tuned on it.

Each conversation's store is built once per arm and asked all its questions;
``--jobs`` conversations are worked on at once. Vectors come from OpenAI's
``text-embedding-3-small`` through the harness's embedding cache by default
(``--embedder local`` for a sentence-transformers model, whose import alone
takes minutes on Windows). All ten conversations, three arms: about 4 minutes.
The free score is the share of questions whose evidence turns all reached the
2,000-token context (a turn counts when a memory taken from it is there).
``--judge-sample N`` also reads a fixed sample of N questions (adversarial
ones excluded, as is usual for LoCoMo) with the SDK's reading prompt and
judges the answers, every arm on the same questions in one run. A context is
judged once: verdicts are kept under ``<work-dir>/verdicts`` per judge and
reader, so a later run pays only for the contexts that changed.

Arms:
  facts     the facts extracted from each message (keep_messages=False)
  messages  the facts and the messages, in passages (the default since
            2026-10-29)
  raw       the passages alone (extractor="none")

``--roles ab`` stores the first speaker as "user" and the second as
"assistant" (how LoCoMo is usually fed to a user/assistant memory);
``--roles names`` stores each under their name, where no role prior applies.
Every message's text starts with its speaker's name either way, so the
extractor and the reader know who said it.

Download the data first (about 3 MB), into a directory of its own:
    curl -L -o .cache/datasets/locomo_orig/locomo10.json \\
        https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
"""

import argparse
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop",
              5: "adversarial"}

logger = logging.getLogger("locomo_shipped_eval")


def parse_when(stamp):
    """"1:56 pm on 8 May, 2023" -> "2023-05-08 13:56" (None if unparsable)."""
    try:
        return datetime.strptime(stamp.strip(), "%I:%M %p on %d %B, %Y").strftime(
            "%Y-%m-%d %H:%M")
    except (AttributeError, ValueError):
        return None


def conversation_messages(sample, roles):
    """The messages of one LoCoMo conversation in order, and dialogue id ->
    turn index. Each message's text starts with its speaker's name; a shared
    photo is kept as its caption."""
    conv = sample["conversation"]
    first = conv["speaker_a"]
    numbers = sorted(int(m.group(1)) for k in conv
                     for m in [re.fullmatch(r"session_(\d+)", k)] if m)
    messages, turn_of = [], {}
    for n in numbers:
        when = parse_when(conv.get(f"session_{n}_date_time"))
        for turn in conv[f"session_{n}"]:
            text = (turn.get("text") or "").strip()
            if turn.get("blip_caption"):
                text = f"{text} [shares a photo: {turn['blip_caption']}]".strip()
            if not text:
                continue
            speaker = turn["speaker"]
            if roles == "names":
                role = speaker
            else:
                role = "user" if speaker == first else "assistant"
            messages.append({"role": role, "content": f"{speaker}: {text}",
                             "timestamp": when or "", "turn_index": len(messages) + 1})
            turn_of[turn["dia_id"]] = len(messages)
    return messages, turn_of


def sign_test(wins, losses):
    n = wins + losses
    if not n:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(wins, losses) + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=os.path.join(ROOT, ".cache", "datasets", "locomo_orig",
                                                   "locomo10.json"))
    ap.add_argument("--arms", default="raw", help="comma-separated: facts, messages, raw")
    ap.add_argument("--roles", choices=["ab", "names"], default="ab")
    ap.add_argument("--token-budget", type=int, default=2000)
    ap.add_argument("--conversations", type=int, default=10)
    ap.add_argument("--embedder", choices=["openai", "local"], default="openai",
                    help="openai: --embed-model through the harness's embedding cache (starts in "
                         "seconds); local: --local-embedder, whose import alone takes minutes on "
                         "Windows (sentence-transformers pulls in transformers)")
    ap.add_argument("--embed-model", default="text-embedding-3-small")
    ap.add_argument("--local-embedder", default="google/embeddinggemma-2")
    ap.add_argument("--jobs", type=int, default=4,
                    help="conversations built and asked at once (each in its own store)")
    ap.add_argument("--work-dir", default=os.environ.get(
        "TSM_EVAL_DIR", os.path.join(os.path.expanduser("~"), ".tsm_eval",
                                     "shipped_stack_eval")))
    ap.add_argument("--extractor-model", default="gpt-4o-mini")
    ap.add_argument("--extraction-cache-only", action="store_true",
                    help="a message the extraction cache does not hold gives no facts")
    ap.add_argument("--judge-sample", type=int, default=0,
                    help="judge this many questions (not adversarial), the same for every arm")
    ap.add_argument("--judge-model", default="gpt-4.1-mini")
    ap.add_argument("--reader", choices=["plain", "notes"], default="notes")
    ap.add_argument("--workers", type=int, default=32,
                    help="concurrent requests (extraction, reading and judging); the account "
                         "allows 5,000 a minute")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.dirname(HERE))
    from cognitive_eval._secrets import ensure_openai_key
    from cognitive_eval.shipped_stack_eval import (CachedExtraction, Embeddings, LocalEmbeddings,
                                                   Replay)
    from tsm import Memory

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(arms) - {"facts", "messages", "raw"}
    if unknown:
        ap.error(f"unknown arms: {sorted(unknown)}")
    needs_facts = any(a in ("facts", "messages") for a in arms)
    if needs_facts or args.judge_sample or args.embedder == "openai":
        if not ensure_openai_key():
            if (needs_facts and args.extraction_cache_only and not args.judge_sample
                    and args.embedder == "local"):
                os.environ["OPENAI_API_KEY"] = "offline"  # the client is built, never called
            else:
                sys.exit("no OpenAI key: see benchmarks/cognitive_eval/_secrets.py")

    samples = json.load(open(args.data, encoding="utf-8"))[:args.conversations]
    if args.embedder == "openai":
        # The harness's OpenAI vectors, read-only, plus an overlay of this
        # runner's own (another runner may be writing the shared one).
        embedder = Embeddings(args.embed_model,
                              os.path.join(args.work_dir, "emb_overlay_locomo.pkl"), False)
    else:
        from tsm.embedders import SentenceTransformerEmbedder

        embedder = LocalEmbeddings(SentenceTransformerEmbedder(args.local_embedder),
                                   os.path.join(args.work_dir, "embeddings_local"))
    extraction = (CachedExtraction(args.extractor_model, args.extraction_cache_only)
                  if needs_facts else None)
    settings = {"facts": dict(keep_messages=False),
                "messages": dict(keep_messages=True),
                "raw": dict(keep_messages=True, extractor="none")}

    started = time.time()
    flush_lock = threading.Lock()

    def run_conversation(sample):
        """Every arm's store for one conversation, and its questions' contexts."""
        out = []
        messages, turn_of = conversation_messages(sample, args.roles)
        asked = next((m["timestamp"] for m in reversed(messages) if m["timestamp"]), None)
        # One request for all of the questions instead of one per recall.
        embedder.encode([qa["question"] for qa in sample["qa"]])
        facts = None
        if extraction is not None:
            said = [m["content"] for m in messages]
            # Each message is extracted with what was said before it, as in
            # one pass; the calls are independent, so they run in parallel.
            with ThreadPoolExecutor(args.workers) as pool_:
                facts = list(pool_.map(lambda k: extraction.facts(said[k], said[:k]),
                                       range(len(said))))
            if not args.extraction_cache_only:
                with flush_lock:  # paid for: kept even if a later step stops the run
                    extraction.flush()
        for arm in arms:
            db = tempfile.mkdtemp(prefix="tsm_locomo_")
            kwargs = dict(settings[arm], refinement_cosine_threshold=None,
                          contradiction_cosine_threshold=None)
            if arm != "raw":
                kwargs["extractor"] = Replay(facts)
            mem = Memory(db, embedder=embedder, **kwargs)
            try:
                mem.add(messages, user_id="u")
                for k, qa in enumerate(sample["qa"]):
                    evidence = sorted({turn_of[d] for d in qa.get("evidence") or []
                                       if d in turn_of})
                    got = mem.recall(qa["question"], user_id="u",
                                     token_budget=args.token_budget)
                    out.append({
                        "arm": arm, "conversation": sample["sample_id"],
                        "question_id": f"{sample['sample_id']}_q{k}",
                        "category": qa["category"], "question": qa["question"],
                        "gold": str(qa.get("answer", qa.get("adversarial_answer", ""))),
                        "asked": asked, "evidence": evidence,
                        "turns": [r.get("turn_index") for r in got],
                        "retrieved": [r.get("context") or r["text"] for r in got],
                    })
            finally:
                mem.close()
                shutil.rmtree(db, ignore_errors=True)
        logger.info("%s: %d messages, %d questions (%.0fs)", sample["sample_id"],
                    len(messages), len(sample["qa"]), time.time() - started)
        return out

    # Conversations are independent (a store each); the engine releases the
    # GIL, so a few at once use the cores the embedding requests leave idle.
    with ThreadPoolExecutor(max(1, args.jobs)) as conv_pool:
        tasks = [t for part in conv_pool.map(run_conversation, samples) for t in part]
    embedder.flush()

    def full(t):
        return bool(t["evidence"]) and set(t["evidence"]) <= {x for x in t["turns"] if x}

    print("=" * 96)
    print(f"LoCoMo, {len(samples)} conversations, {args.token_budget}-token context, "
          f"roles {args.roles}: questions with ALL of their evidence in the context")
    for arm in arms:
        rows = [t for t in tasks if t["arm"] == arm and t["evidence"]]
        cells = "  ".join(
            f"{CATEGORIES[c]} {sum(full(t) for t in rows if t['category'] == c)}/"
            f"{sum(t['category'] == c for t in rows)}" for c in sorted(CATEGORIES))
        kept = [t for t in rows if t["category"] != 5]
        print(f"  {arm:9s} {sum(full(t) for t in rows):5d}/{len(rows)}  "
              f"(not adversarial {sum(full(t) for t in kept)}/{len(kept)})  {cells}")
    by = {(t["arm"], t["question_id"]): t for t in tasks}
    for a in arms:
        for b in arms:
            if a < b:
                keys = [q for (arm, q) in by if arm == a and by[(a, q)]["evidence"]]
                w = sum(full(by[(a, q)]) and not full(by[(b, q)]) for q in keys)
                l = sum(full(by[(b, q)]) and not full(by[(a, q)]) for q in keys)
                print(f"  {a} vs {b}: {w} / {l}  p={sign_test(w, l):.3f}")

    if args.judge_sample:
        from cognitive_eval.judge import create_judge

        judge = create_judge("openai", openai_model=args.judge_model)
        pool = sorted({t["question_id"] for t in tasks if t["category"] != 5})
        chosen = set(random.Random(args.seed).sample(pool, min(args.judge_sample, len(pool))))
        todo = [t for t in tasks if t["question_id"] in chosen]

        # A verdict depends only on what the reader and the judge are shown,
        # never on the arm: the same context is judged once, in this run and
        # in every later run with the same judge and reader.
        def key(t):
            return hashlib.sha256(json.dumps(
                [t["question"], t["gold"], t["asked"], t["retrieved"]]).encode("utf-8")
            ).hexdigest()

        cache_path = os.path.join(args.work_dir, "verdicts", "locomo_"
                                  f"{args.judge_model.replace('/', '_')}_{args.reader}.json")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        try:
            with open(cache_path, encoding="utf-8") as fh:
                known = json.load(fh)
        except (OSError, ValueError):
            known = {}
        pending = list({key(t): t for t in todo if key(t) not in known}.values())
        print(f"judging {len(pending)} contexts ({len(todo) - len(pending)} answers reuse "
              f"a verdict), {args.workers} at once")

        def grade(t):
            prediction = judge.answer(t["question"], t["retrieved"], today=t["asked"],
                                      reader=args.reader)
            known[key(t)] = [prediction,
                             bool(judge.judge(t["question"], t["gold"], prediction))]

        try:
            with ThreadPoolExecutor(args.workers) as pool_:
                list(pool_.map(grade, pending))
        finally:
            tmp = cache_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(known, fh)
            os.replace(tmp, cache_path)
        for t in todo:
            t["prediction"], t["correct"] = known[key(t)]
        print("-" * 96)
        print(f"judged by {args.judge_model}, reader {args.reader}, "
              f"{len(chosen)} questions (seed {args.seed})")
        for arm in arms:
            rows = [t for t in todo if t["arm"] == arm]
            cells = "  ".join(
                f"{CATEGORIES[c]} {sum(t['correct'] for t in rows if t['category'] == c)}/"
                f"{sum(t['category'] == c for t in rows)}" for c in sorted(CATEGORIES) if c != 5)
            print(f"  {arm:9s} {sum(t['correct'] for t in rows):4d}/{len(rows)}  {cells}")
        for a in arms:
            for b in arms:
                if a < b:
                    w = sum(by[(a, q)]["correct"] and not by[(b, q)]["correct"] for q in chosen)
                    l = sum(by[(b, q)]["correct"] and not by[(a, q)]["correct"] for q in chosen)
                    print(f"  {a} vs {b}: {w} / {l}  p={sign_test(w, l):.3f}")
        cost = judge.input_tokens * 0.4e-6 + judge.output_tokens * 1.6e-6
        print(f"  {judge.calls} requests, about ${cost:.3f}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"args": vars(args), "tasks": tasks}, fh)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
