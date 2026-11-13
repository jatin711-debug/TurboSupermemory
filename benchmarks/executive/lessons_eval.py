#!/usr/bin/env python3
"""Learning from outcomes, on a stream of tasks checked by code
(``lessons_world.py``, written blind to ``tsm.experience``).

A stream holds tasks of a split's families in a shuffled order. Most families
hide a house rule a default model gets wrong; controls look like a rule
family but the ordinary answer is right (a lesson carried over makes them
wrong); in a drift family the rule changes at the stream's midpoint. After
each task the user reports the outcome, checked by code, with a one-line
correction when it was wrong (``--feedback``):

  correction  the outcome and, on a failure, the user's correction;
  outcome     the outcome only;
  noisy       as ``correction``, with ``--flip`` of the outcomes reversed (a
              success reported as a failure gets a bare "That's not right.").

Arms, each on a fresh store per stream, the same reader (``--reader-model``)
and the same token cap (``--cap``):

  none       the task alone.
  episodes   ``Memory.episodes()``: the k nearest earlier attempts as they
             were (task, answer, outcome, feedback): what ships, and the bar
             any other way of learning from outcomes has to beat.
  oracle     the split's rules stated (the ceiling).

Curated lessons (a model distils each episode into a lesson with helpful /
harmful counters) were the ``lessons`` / ``probation`` arms; on 2026-11-13
they took rule tasks in the second half of a stream to 11 and 20 of 66
against 45 for episodes, and were removed (``LEARNINGS.md``; the arms are at
``git show 6825fd5`` plus that day's commit). A new idea goes in as an arm.

Reported per arm: tasks right overall and in the second half of the stream,
by family kind (rule / control / drift), drift after the change, and the
tokens shown. Tune on ``--split dev``; quote ``--split test`` once.

    python benchmarks/executive/lessons_eval.py --split dev --offline
    python benchmarks/executive/lessons_eval.py --split dev --seeds 1,2 --arms none,episodes,lessons
"""

import argparse
import json
import os
import random
import re
import shutil
import sys
import tempfile
from collections import defaultdict

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "cognitive_eval"))

import lessons_world as world  # noqa: E402
from belief_pairs_eval import EVAL_DIR, OfflineClient  # noqa: E402

USER = "u"
ARMS = ("none", "episodes", "oracle")
READER_SYSTEM = (
    "You are the assistant of the staff at the company described below. A colleague "
    "asks you to do something short. Answer it as well as you can; when notes from "
    "earlier work are given, use what applies to this request and ignore what does "
    "not. Keep any working brief and end with one line: Answer: <the answer alone>.\n\n"
    + world.WORLD_BRIEF)


def final_answer(reply):
    for line in reversed((reply or "").strip().splitlines()):
        match = re.match(r"\s*\**\s*answer\s*\**\s*:\s*(.*)", line, re.I)
        if match:
            return match.group(1).strip().strip("*").strip()
    return ""


def build_memory(db, embedder):
    from tsm import Memory

    return Memory(db, embedder=embedder, extractor="none", experience=True)


def run_stream(split, seed, arm, args, embedder, reader):
    from tsm.experience import episode_block

    tasks = world.stream(split, seed, per_family=args.per_family)
    flips = random.Random(f"flip:{split}:{seed}")
    db = tempfile.mkdtemp(prefix="tsm_lessons_eval_")
    memory = build_memory(db, embedder) if arm == "episodes" else None
    rows = []
    try:
        for position, task in enumerate(tasks):
            block = ""
            if arm == "episodes":
                found = memory.episodes(task["prompt"], user_id=USER, k=args.k)
                block = episode_block(found, token_cap=args.cap)
            elif arm == "oracle":
                block = "House rules:\n" + world.rules_text(split, phase=task["phase"])
            user = (block + "\n\n" if block else "") + "Request: " + task["prompt"]
            answer = final_answer(reader.ask(user))
            right = bool(answer) and world.check(task, answer)
            rows.append({"family": task["family"], "kind": task["kind"],
                         "phase": task["phase"], "late": position >= len(tasks) // 2,
                         "right": right, "tokens": len(block) // 4})
            if memory is None:
                continue
            reported, note = right, (None if right else task["correction"])
            if args.feedback == "outcome":
                note = None
            elif args.feedback == "noisy" and flips.random() < args.flip:
                reported = not right
                note = None if reported else "That's not right."
            memory.feedback(task["prompt"], reported, user_id=USER,
                            source="user", answer=answer or "(no answer)", note=note)
        return rows
    finally:
        if memory is not None:
            memory.close()
        shutil.rmtree(db, ignore_errors=True)


def summarize(rows):
    def rate(sel):
        sel = list(sel)
        return [sum(r["right"] for r in sel), len(sel)]

    out = {"all": rate(rows), "late": rate(r for r in rows if r["late"]),
           "tokens": round(sum(r["tokens"] for r in rows) / max(1, len(rows)), 1)}
    for kind in ("rule", "control", "drift"):
        out[kind] = rate(r for r in rows if r["kind"] == kind)
        out[kind + "_late"] = rate(r for r in rows if r["kind"] == kind and r["late"])
    out["drift_after"] = rate(r for r in rows if r["kind"] == "drift" and r["phase"] == 1)
    fam = defaultdict(list)
    for r in rows:
        fam[r["family"]].append(r)
    out["families"] = {f: rate(v) for f, v in sorted(fam.items())}
    return out


def fmt(pair):
    right, total = pair
    return f"{right}/{total}" if total else "-"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=("dev", "test"), default="dev")
    ap.add_argument("--seeds", default="1,2", help="stream orders, comma-separated")
    ap.add_argument("--per-family", type=int, default=12)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--feedback", choices=("correction", "outcome", "noisy"),
                    default="correction")
    ap.add_argument("--flip", type=float, default=0.15, help="noisy: share of outcomes reversed")
    ap.add_argument("--k", type=int, default=3, help="lessons / episodes shown")
    ap.add_argument("--cap", type=int, default=200, help="tokens of lessons / episodes shown")
    ap.add_argument("--embed-cache", default=os.path.join(EVAL_DIR, "embeddings_lessons"))
    ap.add_argument("--reader-model", default="gpt-4.1-mini")
    ap.add_argument("--offline", action="store_true",
                    help="never call a chat model; count the requests a real run would make")
    ap.add_argument("--json", help="write the summary here")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        if arm not in ARMS:
            sys.exit(f"unknown arm {arm!r}: one of {', '.join(ARMS)}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    from tsm._chat import CachedChat

    if args.offline:  # free: a fake embedder too (the request count does not depend on it)
        from tsm.tests.test_memory import FakeEmbedder

        embedder = FakeEmbedder()
    else:
        from _secrets import ensure_openai_key, key_file_hint
        from tsm.embedders import OpenAIEmbedder

        if not ensure_openai_key():
            sys.exit(key_file_hint())
        embedder = OpenAIEmbedder(cache_dir=args.embed_cache)
    offline = OfflineClient() if args.offline else None
    cache = os.path.join(EVAL_DIR, "executive_chat")
    reader = CachedChat(READER_SYSTEM, model=args.reader_model, client=offline,
                        cache_dir=cache, cache_name="lessons_reader", max_tokens=300,
                        what="lessons reader")

    summary = {}
    print(f"split={args.split} seeds={seeds} feedback={args.feedback} k={args.k} "
          f"cap={args.cap}")
    print(f"{'arm':<10} {'all':>9} {'late':>9} {'rule late':>10} {'ctrl':>8} "
          f"{'ctrl late':>10} {'drift aft':>10} {'tok':>6}")
    for arm in arms:
        rows = []
        for seed in seeds:
            rows += run_stream(args.split, seed, arm, args, embedder, reader)
            reader.flush_cache()
        s = summary[arm] = summarize(rows)
        print(f"{arm:<10} {fmt(s['all']):>9} {fmt(s['late']):>9} {fmt(s['rule_late']):>10} "
              f"{fmt(s['control']):>8} {fmt(s['control_late']):>10} "
              f"{fmt(s['drift_after']):>10} {s['tokens']:>6}")
    print("\nper family (all positions):")
    fams = sorted({f for s in summary.values() for f in s["families"]})
    print(f"{'family':<24}" + "".join(f"{a:>11}" for a in summary))
    for f in fams:
        print(f"{f:<24}" + "".join(f"{fmt(summary[a]['families'].get(f, [0, 0])):>11}"
                                     for a in summary))
    if offline is not None:
        print(f"\n[offline: a real run would make {offline.requests} chat requests]")
    else:
        print(f"\n[reader: {reader.calls} chat requests]")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"args": vars(args), "arms": summary}, fh, indent=2)


if __name__ == "__main__":
    main()
