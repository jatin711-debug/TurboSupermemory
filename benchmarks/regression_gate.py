#!/usr/bin/env python3
"""W2 — TurboSuperMemory regression gate.

One command that FAILS if any proven property regresses. Run it before every
commit that touches the engine or cognitive layer. It exists because a refactor
silently destroyed cognition once before (the 2026-06-29 fusion no-op), and
because belief-revision detection over-fired catastrophically on real data
(Stage A) before the mutual-nearest-neighbour fix. The gate locks in both wins.

Checks (each PASS/FAIL; the gate exits nonzero if any FAIL):

  1. cargo fmt --check                     — formatting is clean
  2. cargo clippy -D warnings (workspace)  — no lint regressions
  3. cargo test (workspace, ex-python)     — Rust unit/integration suite green
  4. synthetic belief (refinement+contra)  — clean-data lift held, no false demotion
  5. LongMemEval smoke (role-filtered)     — KU lift non-negative, NO edge explosion,
                                             NO single-session collateral (Stage-A guard)
  6. recall audit                          — ANN recall floor intact

The two eval scripts print a machine-readable `GATE_SUMMARY: {json}` line that
this gate parses, so it never depends on the human-readable tables.

Usage (MUST run under Python 3.12 so the subprocess evals load turbomemory.pyd):
    python benchmarks/regression_gate.py            # full gate
    python benchmarks/regression_gate.py --quick    # smaller LongMemEval limit
    python benchmarks/regression_gate.py --no-rust  # skip cargo checks (fast iter)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks"
COG = BENCH / "cognitive_eval"
PY = sys.executable

# --- environment (mirrors the documented build/run setup) -------------------
os.environ.setdefault(
    "PYO3_PYTHON", r"C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe"
)
os.environ.setdefault("CARGO_PROFILE_DEV_DEBUG", "0")
os.environ.setdefault("CARGO_PROFILE_TEST_DEBUG", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# audit_recall.py imports `turbomemory` from the repo root.
os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

# --- thresholds (tuned to catch REGRESSIONS, not fine noise) ----------------
SYNTH_MIN_LIFT = 0.9          # synthetic belief lift is ~+1.00 when healthy
SYNTH_MAX_FALSE_DEMOTION = 0.05
LME_MIN_KU_LIFT = -0.10       # smoke n is small; assert non-regression, not exact
LME_MIN_TYPE_LIFT = -0.15     # single-session collateral must stay bounded
LME_EDGE_MIN = 10             # detection must actually fire (role-filtered)
LME_EDGE_MAX = 800            # Stage-A over-firing / reverse-MNN removal blows past this
RECALL_FLOOR_PCT = 95.0

results = []  # (name, ok, detail)


def _run(cmd, **kw):
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, **kw)


def _record(name, ok, detail):
    results.append((name, bool(ok), detail))
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name} - {detail}", flush=True)
    return ok


def check_cmd(name, cmd):
    t = time.time()
    p = _run(cmd)
    ok = p.returncode == 0
    if not ok:
        tail = "\n".join((p.stderr or p.stdout).splitlines()[-20:])
        sys.stderr.write(f"\n----- {name} output (tail) -----\n{tail}\n")
    return _record(name, ok, f"exit={p.returncode} ({time.time() - t:.0f}s)")


def _gate_summary(text):
    for line in reversed(text.splitlines()):
        m = re.search(r"GATE_SUMMARY:\s*(\{.*\})", line)
        if m:
            return json.loads(m.group(1))
    return None


def check_synthetic_belief():
    for mode in ("refinement", "contradiction"):
        t = time.time()
        p = _run([PY, str(COG / "belief_revision.py"), "--mode", mode, "--quick"])
        s = _gate_summary(p.stdout + "\n" + p.stderr)
        if s is None:
            _record(f"synthetic belief [{mode}]", False,
                     f"no GATE_SUMMARY (exit={p.returncode})")
            continue
        ok = (s["mean_lift"] >= SYNTH_MIN_LIFT
              and s["mean_false_demotion"] <= SYNTH_MAX_FALSE_DEMOTION)
        _record(f"synthetic belief [{mode}]", ok,
                f"lift={s['mean_lift']:+.2f} (>={SYNTH_MIN_LIFT}), "
                f"false_dem={s['mean_false_demotion']:.2f} "
                f"(<={SYNTH_MAX_FALSE_DEMOTION}) ({time.time() - t:.0f}s)")


def check_longmemeval_smoke(limit):
    t = time.time()
    p = _run([PY, str(COG / "run_belief_longmemeval.py"),
              "--limit", str(limit), "--role-filtered"])
    s = _gate_summary(p.stdout + "\n" + p.stderr)
    if s is None:
        _record("LongMemEval smoke (role-filtered)", False,
                f"no GATE_SUMMARY (exit={p.returncode})")
        return
    ku = s["ku_hit1_lift"]
    edges = s["on_edges"]
    worst_ss = min([v for k, v in s["type_hit1_lift"].items()
                    if k.startswith("single-session")] or [0.0])
    ok = (ku >= LME_MIN_KU_LIFT
          and LME_EDGE_MIN <= edges <= LME_EDGE_MAX
          and worst_ss >= LME_MIN_TYPE_LIFT)
    _record("LongMemEval smoke (role-filtered)", ok,
            f"KU_hit1_lift={ku:+.2f} (>={LME_MIN_KU_LIFT}), "
            f"edges={edges} (in [{LME_EDGE_MIN},{LME_EDGE_MAX}]), "
            f"worst_single_session={worst_ss:+.2f} (>={LME_MIN_TYPE_LIFT}) "
            f"({time.time() - t:.0f}s)")


def check_recall_audit():
    t = time.time()
    p = _run([PY, str(BENCH / "audit_recall.py"),
              "--num-items", "2000", "--dimension", "64", "--num-queries", "50"])
    text = p.stdout + "\n" + p.stderr
    m = re.search(r"ANN reranked Recall@\d+:\s*([\d.]+)%", text)
    if m is None:
        _record("recall audit", False, f"no recall line (exit={p.returncode})")
        return
    pct = float(m.group(1))
    _record("recall audit", pct >= RECALL_FLOOR_PCT,
            f"ANN recall={pct:.1f}% (>={RECALL_FLOOR_PCT}%) ({time.time() - t:.0f}s)")


def main():
    ap = argparse.ArgumentParser(description="TurboSuperMemory regression gate (W2)")
    ap.add_argument("--quick", action="store_true",
                    help="smaller LongMemEval limit (faster, noisier)")
    ap.add_argument("--no-rust", action="store_true",
                    help="skip cargo fmt/clippy/test (fast iteration on evals)")
    ap.add_argument("--no-evals", action="store_true",
                    help="skip the python evals (Rust checks only)")
    args = ap.parse_args()

    print("=" * 78)
    print("TurboSuperMemory regression gate (W2)")
    print("=" * 78)

    if not args.no_rust:
        check_cmd("cargo fmt --check", ["cargo", "fmt", "--all", "--", "--check"])
        check_cmd("cargo clippy -D warnings",
                  ["cargo", "clippy", "--workspace", "--all-targets", "--", "-D", "warnings"])
        check_cmd("cargo test (workspace ex-python)",
                  ["cargo", "test", "--workspace", "--exclude", "turbomemory_python"])

    if not args.no_evals:
        check_synthetic_belief()
        check_longmemeval_smoke(limit=20 if args.quick else 40)
        check_recall_audit()

    print("=" * 78)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("-" * 78)
    all_ok = passed == total
    print(f"{'GATE PASS' if all_ok else 'GATE FAIL'}: {passed}/{total} checks passed")
    print("=" * 78)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
