"""Read saved contexts several times and compare arms on the mean.

The reader is not deterministic at temperature 0: on 121 identical
LongMemEval-S contexts, two judged runs differed on 5 or 6 verdicts
(2026-11-05). A difference of that size between two arms is noise when
each context is read once. This reads every context of the given runs
``--reads`` times (reader + judge, as ``shipped_stack_eval.py`` asks them,
no verdict cache) and reports, per arm, the mean number answered right and
the questions where one arm is right more often than the other.

    python benchmarks/cognitive_eval/reread.py --reads 5 \\
        --arm raw=RUN.json:sdk_raw --arm dated=RUN.json:sdk_raw_dated

Each ``--arm NAME=PATH:ARM`` takes the contexts of arm ``ARM`` from the
per-question results ``PATH`` (``--out`` of the harness). Arms are compared
on the questions they share; identical contexts are read once per read and
shared between arms. Paid: one reader and one judge request per distinct
context per read.

``--system NAME=FILE`` (repeatable; ``NAME=-`` is the shipped
``tsm.reading.ANSWER_SYSTEM``) reads every arm with each of these reader
instructions instead, as arms ``ARM/NAME``: a prompt change compared on the
same contexts.
"""

import argparse
import json
import logging
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cognitive_eval._secrets import ensure_openai_key  # noqa: E402
from cognitive_eval.judge import create_judge  # noqa: E402

logger = logging.getLogger("reread")


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH:ARM")
    ap.add_argument("--reads", type=int, default=5)
    ap.add_argument("--reader", default="notes", choices=["plain", "notes", "tools"])
    ap.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    ap.add_argument("--judge-model", default=None,
                    help="default gpt-4.1-mini (openai) or claude-haiku-5-5 (anthropic)")
    ap.add_argument("--reader-effort", default=None,
                    help="anthropic: the reader's effort (low ... max)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=None, help="write every read here (JSON)")
    ap.add_argument("--system", action="append", default=[], metavar="NAME=FILE",
                    help="--reader notes: read with these instructions too ('-': shipped)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.provider == "openai" and not ensure_openai_key():
        sys.exit("OPENAI_API_KEY is not set")
    if args.judge_model is None:
        args.judge_model = "claude-haiku-5-5" if args.provider == "anthropic" else "gpt-4.1-mini"

    arms = {}
    for spec in args.arm:
        name, rest = spec.split("=", 1)
        path, arm = rest.rsplit(":", 1)
        with open(path, encoding="utf-8") as fh:
            tasks = [t for t in json.load(fh)["tasks"] if t["arm"] == arm]
        if not tasks:
            sys.exit(f"no tasks of arm {arm} in {path}")
        arms[name] = {t["query_id"]: t for t in tasks}
    systems = {None: None}
    if args.system:
        systems = {}
        for spec in args.system:
            name, file = spec.split("=", 1)
            if file == "-":
                systems[name] = None
            else:
                with open(file, encoding="utf-8") as fh:
                    systems[name] = fh.read().strip()
        arms = {f"{arm}/{name}": dict(tasks, **{"": name})
                for arm, tasks in arms.items() for name in systems}
    shared = set.intersection(*(set(a) - {""} for a in arms.values()))
    names = list(arms)

    def key(t, system=None):
        return json.dumps([t["query"], t["gold"], t.get("asked"), t["retrieved"], system])

    distinct = {}  # key -> (task, system name)
    for a in arms.values():
        for q in shared:
            distinct.setdefault(key(a[q], a.get("")), (a[q], a.get("")))
    judge = create_judge(args.provider, openai_model=args.judge_model,
                         anthropic_model=args.judge_model)
    judge.reader_effort = args.reader_effort
    logger.info("%d arms, %d shared questions, %d distinct contexts, %d reads each",
                len(arms), len(shared), len(distinct), args.reads)

    def read(job):
        k, _ = job
        t, system = distinct[k]
        how = {"reader": args.reader}
        if systems.get(system):
            how["system"] = systems[system]
        if t.get("asked"):
            how["today"] = t["asked"]
        if t.get("turns"):
            how["turns"] = t["turns"]
        prediction = judge.answer(t["query"], t["retrieved"], **how)
        return k, prediction, bool(judge.judge(t["query"], t["gold"], prediction))

    jobs = [(k, n) for n in range(args.reads) for k in distinct]
    right = {k: 0 for k in distinct}
    log = {k: [] for k in distinct}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for k, prediction, ok in pool.map(read, jobs):
            right[k] += ok
            log[k].append([prediction, ok])

    rate = {name: {q: right[key(a[q], a.get(""))] / args.reads for q in shared}
            for name, a in arms.items()}
    print(f"\nmean answered right over {args.reads} reads, {len(shared)} questions")
    for name in names:
        per_read = sum(rate[name].values())
        always = sum(r == 1 for r in rate[name].values())
        print(f"  {name:12s} {per_read:6.1f}   (right every read: {always})")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            wins = sum(rate[a][q] > rate[b][q] for q in shared)
            losses = sum(rate[a][q] < rate[b][q] for q in shared)
            print(f"  {a} vs {b}: more often right {wins} / less often {losses} "
                  f"(sign test p = {sign_test(wins, losses):.3f})")
    print(f"reader+judge requests: {judge.calls}, tokens in {judge.input_tokens:,} "
          f"out {judge.output_tokens:,}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"reads": args.reads, "arms": {n: dict(r) for n, r in rate.items()},
                       "log": {k: log[k] for k in distinct}}, fh)


if __name__ == "__main__":
    main()
