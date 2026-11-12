#!/usr/bin/env python3
"""Prospective memory against labeled chat histories
(``intentions_scenarios.json``, written blind to ``tsm.intentions``).

Each scenario is one user's messages over a few weeks and a few checks: at
date ``now`` (and optionally in a ``context``), which of the user's own
pending intentions need attention? A check lists ``should_surface`` (pending
and due by then, or brought to mind by the context) and ``must_not_surface``
(closed or cancelled by then, or three or more days early); the rest may be
mentioned or not. Anything surfaced from a message that creates no intention
(a decoy: a past event, someone else's task, a hypothetical, an assistant
suggestion) is wrong.

Arms, each on a fresh store per scenario, messages added in time order and
``consolidate()`` run before every check:

  rules    Memory(intentions=True).due(): formed by rules, closed and
           rescheduled at consolidation, due by date or brought to mind by
           the context (no model call).
  todo     the same intentions, never closed or rescheduled (consolidate()
           is not run): a plain to-do list.
  judged   rules, then each due item judged by LLMReadinessJudge.
  tracked  Memory(intention_tracker=LLMIntentionTracker): a chat model
           (--tracker-model) forms, closes and reschedules intentions at
           consolidation; due() as in rules.
  recall   no intention store: recall() at --budget tokens, and a chat
           model (--reader-model) reads the memories with today's date and
           the context and names the ones that hold something the user has
           to do now.

An item is matched to the intention its message created (or rescheduled);
one from a message that created several is matched by word overlap.

    python benchmarks/executive/intentions_eval.py --split dev --arms rules,todo
    python benchmarks/executive/intentions_eval.py --split dev --arms judged,recall --offline
"""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from datetime import date

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "cognitive_eval"))

from belief_pairs_eval import EVAL_DIR, OfflineClient  # noqa: E402

SCENARIOS = os.path.join(HERE, "intentions_scenarios.json")
USER = "u"

RECALL_SYSTEM = """You are a personal assistant checking what needs the \
user's attention right now. You get today's date, possibly what the user is \
doing at the moment, and numbered memories from the user's past messages \
with their dates.

List the memories that state something the USER themselves still has to do, \
attend or remember, that needs attention today: it is due today or already \
overdue, or what the user is doing now brings it to mind. Leave out anything \
already done or cancelled according to a later memory, anything not due for \
a few more days, other people's tasks, past events, wishes, hypotheticals \
and suggestions the user did not take up. When a task was moved, use its new \
date.

Answer with one line: IDS: followed by the memory numbers, comma-separated, \
or IDS: none."""


def load(split):
    rows = json.load(open(SCENARIOS, encoding="utf-8"))
    return [s for s in rows if split == "all" or s["split"] == split]


def words(text):
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def day(stamp):
    return date.fromisoformat(stamp[:10])


class Matcher:
    """Memory id -> message id, and message -> the intentions it labels."""

    def __init__(self, scenario):
        self.scenario = scenario
        self.memory_to_message = {}
        self.labels = {}
        for m in scenario["messages"]:
            ids = list(m.get("creates") or []) + [r["intention"] for r in m.get("reschedules") or []]
            self.labels[m["id"]] = ids

    def intention_of(self, source_ids, text):
        """The gold intention a surfaced item stands for, or None."""
        messages = [self.memory_to_message.get(mid) for mid in source_ids]
        candidates = []
        for message in messages:
            for iid in self.labels.get(message, []):
                if iid not in candidates:
                    candidates.append(iid)
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        item = words(text)
        what = self.scenario["intentions"]
        return max(candidates, key=lambda iid: len(item & words(what[iid]["what"])))


def build_memory(db, embedder, arm, judge, tracker):
    from tsm import Memory

    return Memory(db, embedder=embedder, extractor="none", keep_messages=True,
                  intentions=arm in ("rules", "todo", "judged", "tracked"),
                  readiness_judge=judge if arm == "judged" else None,
                  intention_tracker=tracker if arm == "tracked" else None)


def recall_check(memory, matcher, check, reader, budget):
    query = "What do I need to do, attend or remember now?"
    if check["context"]:
        query += " " + check["context"]
    results = memory.recall(query, user_id=USER, token_budget=budget, now=check["now"])
    results = sorted(results, key=lambda r: r["id"])
    lines = [f"{k + 1}. {r['context']}" for k, r in enumerate(results)]
    user = (f"Today is {check['now']}.\n"
            + (f"The user right now: {check['context']}\n" if check["context"] else "")
            + "\nMemories:\n" + "\n".join(lines))
    reply = reader.ask(user)
    match = re.search(r"IDS:\s*(.*)", reply or "")
    numbers = [int(n) for n in re.findall(r"\d+", match.group(1))] if match else []
    surfaced = []
    for n in numbers:
        if 1 <= n <= len(results):
            r = results[n - 1]
            surfaced.append(([r["id"]], r["text"]))
    return surfaced


def run_scenario(scenario, arm, embedder, judge, reader, budget, tracker=None, cue=None):
    db = tempfile.mkdtemp(prefix="tsm_intentions_eval_")
    memory = build_memory(db, embedder, arm, judge, tracker)
    matcher = Matcher(scenario)
    try:
        messages = sorted(scenario["messages"], key=lambda m: m["timestamp"])
        added = 0
        out = []
        for check in sorted(scenario["checks"], key=lambda c: c["now"]):
            today = date.fromisoformat(check["now"])
            while added < len(messages) and day(messages[added]["timestamp"]) <= today:
                m = messages[added]
                before = set(memory.engine.scope_ids(USER))
                memory.add([{"role": m["role"], "content": m["content"],
                             "timestamp": m["timestamp"]}], user_id=USER)
                for mid in set(memory.engine.scope_ids(USER)) - before:
                    matcher.memory_to_message[mid] = m["id"]
                added += 1
            if arm != "todo":
                memory.consolidate()
            if arm == "recall":
                surfaced = recall_check(memory, matcher, check, reader, budget)
            else:
                items = memory.due(USER, now=check["now"], context=check["context"],
                                  cue_min_cosine=cue)
                surfaced = [(item.get("source_ids") or [], item["text"]) for item in items
                            if item.get("verdict") in ("act", "observe")]
            out.append((check, [matcher.intention_of(src, text) for src, text in surfaced]))
        return out
    finally:
        memory.close()
        shutil.rmtree(db, ignore_errors=True)


def score(scenarios, results, title, show):
    tp = fn = fp_closed = fp_early = fp_spurious = neutral = 0
    misses = Counter()
    wrong = []
    for scenario, checks in zip(scenarios, results):
        what = scenario["intentions"]
        for check, matched in checks:
            should, must_not = set(check["should_surface"]), set(check["must_not_surface"])
            got = {iid for iid in matched if iid is not None}
            fp_spurious += sum(iid is None for iid in matched)
            tp += len(got & should)
            for iid in should - got:
                fn += 1
                kind = "situation" if what[iid]["situation"] else (
                    "undated" if what[iid]["due_from"] is None else "dated")
                misses[kind + (" +context" if check["context"] else "")] += 1
                wrong.append(f"MISSED  {scenario['id']} {check['now']} {iid}: {what[iid]['what']}"
                             + (f"  [context: {check['context']}]" if check["context"] else ""))
            closed = {c["intention"] for m in scenario["messages"]
                      for c in m.get("closes") or () if day(m["timestamp"]).isoformat()
                      <= check["now"]}
            for iid in got & must_not:
                if iid in closed:
                    fp_closed += 1
                    reason = "closed"
                else:
                    fp_early += 1
                    reason = "early"
                wrong.append(f"WRONG   {scenario['id']} {check['now']} {iid} ({reason}): "
                             f"{what[iid]['what']}")
            neutral += len(got - should - must_not)
    fp = fp_closed + fp_early + fp_spurious
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    print(f"\n{title}")
    print(f"  surfaced {tp} of {tp + fn} that should (recall {recall:.2f}); wrong {fp} "
          f"(closed or cancelled {fp_closed}, too early {fp_early}, from no intention "
          f"{fp_spurious}); precision {precision:.2f}; neither-list {neutral}")
    if misses:
        print("  missed by kind: " + ", ".join(f"{k} {v}" for k, v in sorted(misses.items())))
    if show:
        for line in wrong:
            print("    " + line)
    return {"tp": tp, "fn": fn, "fp_closed": fp_closed, "fp_early": fp_early,
            "fp_spurious": fp_spurious, "neutral": neutral, "recall": recall,
            "precision": precision}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    ap.add_argument("--arms", default="rules,todo")
    ap.add_argument("--embedder", choices=("local", "openai"), default="openai")
    ap.add_argument("--embed-cache", default=os.path.join(EVAL_DIR, "embeddings_intentions"))
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--reader-model", default="gpt-4.1-mini")
    ap.add_argument("--tracker-model", default="gpt-4.1-mini")
    ap.add_argument("--budget", type=int, default=1000, help="recall arm's token budget")
    ap.add_argument("--cue-min-cosine", type=float, default=None,
                    help="due(): how close a context must be to bring an intention to mind "
                         "(default: tsm.intentions.CUE_MIN_COSINE)")
    ap.add_argument("--offline", action="store_true",
                    help="never call a chat model; count the requests a real run would make")
    ap.add_argument("--show", action="store_true", help="print every miss and wrong item")
    ap.add_argument("--json", help="write the summary here")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    from _secrets import ensure_openai_key, key_file_hint

    paid = args.embedder == "openai" or any(a in ("judged", "recall", "tracked") for a in arms)
    if paid and not args.offline and not ensure_openai_key():
        sys.exit(key_file_hint())
    if args.embedder == "openai":
        if args.offline:
            ensure_openai_key()
        from tsm.embedders import OpenAIEmbedder

        embedder = OpenAIEmbedder(cache_dir=args.embed_cache)
    else:
        from tsm.embedders import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()
    offline = OfflineClient() if args.offline else None
    cache = os.path.join(EVAL_DIR, "executive_chat")
    judge = reader = None
    if "judged" in arms:
        from tsm.intentions import LLMReadinessJudge

        judge = LLMReadinessJudge(model=args.judge_model, client=offline, cache_dir=cache)
    if "recall" in arms:
        from tsm._chat import CachedChat

        reader = CachedChat(RECALL_SYSTEM, model=args.reader_model, client=offline,
                            cache_dir=cache, cache_name="intentions_reader", max_tokens=200,
                            what="intentions reader")
    tracker = None
    if "tracked" in arms:
        from tsm.intentions import LLMIntentionTracker

        tracker = LLMIntentionTracker(model=args.tracker_model, client=offline, cache_dir=cache)
    scenarios = load(args.split)
    summary = {}
    for arm in arms:
        results = [run_scenario(s, arm, embedder, judge, reader, args.budget, tracker,
                                args.cue_min_cosine)
                   for s in scenarios]
        checks = sum(len(s["checks"]) for s in scenarios)
        summary[arm] = score(scenarios, results,
                             f"split={args.split} arm={arm} embedder={args.embedder} "
                             f"({len(scenarios)} scenarios, {checks} checks)", args.show)
    for backend in (judge, reader, tracker):
        flush = getattr(backend, "flush_cache", None)
        if callable(flush):
            flush()
    if offline is not None:
        print(f"\n[offline: a real run would make {offline.requests} chat requests]")
    else:
        for name, backend in (("judge", judge), ("reader", reader), ("tracker", tracker)):
            if backend is not None:
                calls = getattr(backend, "calls", None)
                if calls is None:
                    calls = getattr(getattr(backend, "_chat", None), "calls", "?")
                print(f"[{name}: {calls} chat requests]")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"split": args.split, "embedder": args.embedder, "arms": summary}, fh,
                      indent=2)


if __name__ == "__main__":
    main()
