#!/usr/bin/env python3
"""Belief revision against labeled sentence pairs (``belief_pairs.jsonl``).

Each pair is an older and a newer first-person statement with a label:

  update   the newer statement replaces the older one (a moved city, a
           corrected date, a habit that stopped): the older one should stop
           being served;
  coexist  both stay true (another relative, another trip, a second allergy,
           an extra detail): hiding the older one loses a fact;
  same     the same fact reworded: either outcome is fine (not scored).

The pairs are stored through the shipped ``tsm.Memory`` (local MiniLM
embeddings, no extraction, no API key), ``consolidate()`` runs, and the
script reports which older statements ended up superseded:

  recall      share of ``update`` pairs whose older statement was superseded
  wrong       share of ``coexist`` pairs whose older statement was superseded:
              a fact that is still true, marked stale
  precision   of everything superseded, the share that should have been

What being superseded costs depends on the setup: with a verifier the
statement is removed from recall, without one it is ranked lower and flagged.

``--layout isolated`` gives every pair its own user, so only the pair itself
can interact; ``--layout mixed`` stores all pairs under one user, where
statements of other pairs compete as neighbours, as in a real store. The
pairs were written independently, so in the mixed layout a statement of one
pair can truly replace a statement of another ("I live in Lisbon" and, from
another pair, "I moved to Austin"); the labels do not cover those, and the
report counts them apart ("by another pair's statement").

The pairs are split in half by position within each category. Tune on
``--split dev``; quote ``--split test``.

    python benchmarks/cognitive_eval/belief_pairs_eval.py --verifier none
    python benchmarks/cognitive_eval/belief_pairs_eval.py --verifier nli --split test
    # a chat model: OpenAI (the key is read from the environment or the
    # gitignored key file, see _secrets.py) ...
    python benchmarks/cognitive_eval/belief_pairs_eval.py --verifier llm
    # ... or any OpenAI-compatible server, here a local Ollama one
    python benchmarks/cognitive_eval/belief_pairs_eval.py --verifier llm \
        --llm-base-url http://localhost:11434/v1 --llm-model qwen3.5:4b
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

PAIRS = os.path.join(HERE, "belief_pairs.jsonl")


def load_pairs(split):
    rows = [json.loads(line) for line in open(PAIRS, encoding="utf-8") if line.strip()]
    position = Counter()
    out = []
    for row in rows:
        index = position[row["category"]]
        position[row["category"]] += 1
        half = "dev" if index % 2 == 0 else "test"
        if split in ("all", half):
            out.append(row)
    return out


def parse_overrides(items):
    out = {}
    for item in items or []:
        key, _, raw = item.partition("=")
        try:
            out[key] = json.loads(raw)
        except ValueError:
            out[key] = raw
    return out


def run(pairs, layout, verifier, overrides, embedder, rounds=1):
    """Store the pairs, consolidate, and return per-pair outcomes."""
    from tsm import Memory

    db = tempfile.mkdtemp(prefix="tsm_belief_pairs_")
    memory = Memory(db, embedder=embedder, extractor="passthrough", verifier=verifier,
                    **overrides)
    try:
        def user(i):
            return "u" if layout == "mixed" else f"u{i}"

        # Every older statement first, then every newer one: the order facts
        # arrive in when the update comes in a later session.
        for field in ("older", "newer"):
            for i, pair in enumerate(pairs):
                stored = memory.add([{"role": "user", "content": pair[field]}], user_id=user(i))
                assert stored == 1, pair
        n = len(pairs)
        old_ids = [f"{user(i)}_{i + 1}" for i in range(n)]
        new_ids = [f"{user(i)}_{n + i + 1}" for i in range(n)]
        texts = {r["id"]: r["text"] for r in memory.engine.get_records(old_ids + new_ids) if r}
        for i, pair in enumerate(pairs):
            assert texts[old_ids[i]] == pair["older"] and texts[new_ids[i]] == pair["newer"]

        committed = sum(memory.consolidate() for _ in range(rounds))
        resolved = {r["id"]: r["current_id"] for r in memory.engine.resolve_beliefs(old_ids + new_ids)}
        outcomes = []
        for i, pair in enumerate(pairs):
            current = resolved[old_ids[i]]
            outcomes.append({
                "pair": pair,
                "old_hidden": current != old_ids[i],
                "by_own_newer": current == new_ids[i],
                # The statement that replaced it, when that is not its own newer one.
                "by_other": texts.get(current) if current not in (old_ids[i], new_ids[i]) else None,
                "new_hidden": resolved[new_ids[i]] != new_ids[i],
                "new_by": texts.get(resolved[new_ids[i]]),
            })
        return outcomes, committed
    finally:
        memory.close()
        shutil.rmtree(db, ignore_errors=True)


def report(outcomes, title):
    by_category = defaultdict(list)
    for o in outcomes:
        by_category[(o["pair"]["label"], o["pair"]["category"])].append(o)
    print(f"\n{title}")
    print(f"  {'label':<8} {'category':<28} {'pairs':>5} {'superseded':>11} {'newer too':>10}")
    for (label, category), rows in sorted(by_category.items(), key=lambda kv: kv[0]):
        hidden = sum(o["old_hidden"] for o in rows)
        collateral = sum(o["new_hidden"] for o in rows)
        print(f"  {label:<8} {category:<28} {len(rows):>5} {hidden:>6} ({hidden / len(rows):>4.0%})"
              f" {collateral:>7}")
    updates = [o for o in outcomes if o["pair"]["label"] == "update"]
    coexist = [o for o in outcomes if o["pair"]["label"] == "coexist"]
    true_hides = sum(o["old_hidden"] for o in updates)
    false_hides = sum(o["old_hidden"] for o in coexist) + sum(
        o["new_hidden"] for o in outcomes if o["pair"]["label"] != "same")
    recall = true_hides / max(1, len(updates))
    precision = true_hides / max(1, true_hides + false_hides)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    wrong = sum(o["old_hidden"] for o in coexist)
    print(f"  recall {recall:.3f} ({true_hides}/{len(updates)} updates)   "
          f"wrong {wrong}/{len(coexist)} coexisting"
          f"   precision {precision:.3f}   F1 {f1:.3f}")
    cross = sum(o["by_other"] is not None for o in coexist)
    if cross:
        print(f"  of the {wrong} wrong: {wrong - cross} by the pair's own newer statement, "
              f"{cross} by another pair's statement")
    return {"recall": recall, "precision": precision, "f1": f1,
            "true_hides": true_hides, "false_hides": false_hides,
            "updates": len(updates), "coexist": len(coexist),
            "wrong": wrong, "wrong_by_other_pair": cross}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    ap.add_argument("--layout", choices=("isolated", "mixed", "both"), default="both")
    ap.add_argument("--verifier", choices=("none", "nli", "llm"), default="none")
    ap.add_argument("--llm-model", default="gpt-4o-mini")
    ap.add_argument("--llm-base-url", help="OpenAI-compatible server (default: OpenAI)")
    ap.add_argument("--llm-batch", type=int, default=8, help="candidate pairs per request")
    ap.add_argument("--llm-cache", help="directory for the verdict cache")
    ap.add_argument("--llm-arg", action="append", metavar="KEY=VALUE",
                    help="extra chat-completion argument (JSON value), repeatable; "
                         "e.g. reasoning_effort=none for a local thinking model")
    ap.add_argument("--min-cosine", type=float, default=None,
                    help="candidate floor for --verifier llm (default: the verifier's)")
    ap.add_argument("--per-record", type=int, default=None,
                    help="older candidates per new fact for --verifier llm")
    ap.add_argument("--margin", type=float, default=None,
                    help="judge only neighbours this close to the closest one "
                         "(default: the verifier's; a negative value turns the rule off)")
    ap.add_argument("--rounds", type=int, default=1,
                    help="consolidate this many times; later rounds must not add "
                         "supersessions the first one did not make")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="MemoryEngine option override (JSON value), repeatable")
    ap.add_argument("--show", choices=("none", "errors"), default="none",
                    help="print the pairs that were decided wrongly")
    ap.add_argument("--json", help="write the summary here")
    args = ap.parse_args()

    from tsm.embedders import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder()
    pairs = load_pairs(args.split)
    overrides = parse_overrides(args.set)
    summary = {}
    for layout in (("isolated", "mixed") if args.layout == "both" else (args.layout,)):
        verifier = None
        if args.verifier == "nli":
            from tsm.verification import NLIVerifier

            verifier = NLIVerifier(allow_download=False)
        elif args.verifier == "llm":
            from tsm.verification import LLMVerifier

            if not args.llm_base_url:
                # OpenAI itself: the key comes from the environment or the
                # gitignored key file, never from the command line.
                sys.path.insert(0, HERE)
                from _secrets import ensure_openai_key, key_file_hint

                if not ensure_openai_key():
                    sys.exit(key_file_hint())
            options = {}
            if args.min_cosine is not None:
                options["candidate_min_cosine"] = args.min_cosine
            if args.per_record is not None:
                options["candidates_per_record"] = args.per_record
            if args.margin is not None:
                options["candidate_margin"] = None if args.margin < 0 else args.margin
            verifier = LLMVerifier(model=args.llm_model, base_url=args.llm_base_url,
                                   batch_size=args.llm_batch, cache_dir=args.llm_cache,
                                   request_timeout=600.0,
                                   request_kwargs=parse_overrides(args.llm_arg), **options)
        outcomes, committed = run(pairs, layout, verifier, overrides, embedder, args.rounds)
        if args.verifier == "llm":
            print(f"\n  [{verifier.calls} requests to {args.llm_model} so far]")
        summary[layout] = report(
            outcomes,
            f"split={args.split} layout={layout} verifier={args.verifier} "
            f"overrides={overrides or '{}'} ({len(pairs)} pairs, {committed} committed by verifier)")
        if args.show == "errors":
            for o in outcomes:
                label = o["pair"]["label"]
                wrong = (label == "update" and not o["old_hidden"]) or (
                    label == "coexist" and o["old_hidden"])
                if wrong:
                    kind = "MISSED " if label == "update" else "HIDDEN "
                    print(f"    {kind}[{o['pair']['category']}] {o['pair']['older']!r} -> "
                          f"{o['pair']['newer']!r}")
                    if o["by_other"]:
                        print(f"            replaced by another pair's {o['by_other']!r}")
                if o["new_hidden"] and label != "same":
                    print(f"    NEWER  [{o['pair']['category']}] {o['pair']['newer']!r}\n"
                          f"            replaced by another pair's {o['new_by']!r}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"split": args.split, "verifier": args.verifier,
                       "overrides": overrides, "layouts": summary}, fh, indent=2)


if __name__ == "__main__":
    main()
