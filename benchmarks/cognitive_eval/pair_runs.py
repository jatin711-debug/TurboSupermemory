#!/usr/bin/env python3
"""Compare two arms question by question, from the same run file or two.

    python benchmarks/cognitive_eval/pair_runs.py RUN_A.json:ARM_A RUN_B.json:ARM_B [--by-type]

Prints how many shared questions each got right, how many only one of them
got right, and the two-sided sign-test p of that split. Questions are matched
by question id, so a LongMemEval-S run pairs with an oracle run of the same
questions.
"""

import json
import math
import sys
from collections import defaultdict


def load(spec):
    path, _, arm = spec.rpartition(":")
    with open(path, encoding="utf-8") as fh:
        tasks = json.load(fh)["tasks"]
    return {t["query_id"]: t for t in tasks if t["arm"] == arm and t.get("correct") is not None}


def sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    a, b = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(a.keys() & b.keys())
    wins = sum(bool(a[k]["correct"]) and not b[k]["correct"] for k in keys)
    losses = sum(bool(b[k]["correct"]) and not a[k]["correct"] for k in keys)
    print(f"{len(keys)} shared questions: A {sum(bool(a[k]['correct']) for k in keys)} right, "
          f"B {sum(bool(b[k]['correct']) for k in keys)} right; A only {wins}, B only {losses}, "
          f"p={sign_test(wins, losses):.4f}")
    if "--by-type" in sys.argv:
        rows = defaultdict(lambda: [0, 0, 0])
        for k in keys:
            row = rows[a[k]["question_type"]]
            row[0] += 1
            row[1] += bool(a[k]["correct"])
            row[2] += bool(b[k]["correct"])
        for kind, (n, right_a, right_b) in sorted(rows.items()):
            print(f"  {kind:28s} n={n:3d}  A {right_a:3d}  B {right_b:3d}  "
                  f"net {right_a - right_b:+d}")


if __name__ == "__main__":
    main()
