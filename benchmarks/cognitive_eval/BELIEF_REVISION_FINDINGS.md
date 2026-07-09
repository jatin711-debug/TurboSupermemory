# Belief-Revision Eval — Phase 1 Findings (2026-06-29)

Harness: `benchmarks/cognitive_eval/belief_revision.py`. Captured on the post-fix
release build (pure graph-delta fusion). This is the eval the frozen baseline
(`benchmarks/BASELINE.md`) called for: one that can **fail** when cognition adds no value.

## Headline metric

```
belief_accuracy   = P( the CURRENT belief outranks its SUPERSEDED version )
stale_suppression = P( the stale fact is NOT rank 1 )
feature_lift      = belief_accuracy(CogON) - belief_accuracy(CogOFF)   # isolates the mechanism
```

Three arms share identical config; the ONLY difference between CogON and CogOFF is whether
the belief-revision edge (Contradicts / Refines) is built. So `feature_lift` attributes
value to the *mechanism*, not to generic lexical/association recall.

## The eval is calibrated (control passes)

Easy regime (`--dimension 64 --distractors 0 --subtlety 0.10 --gap 0`):

| arm | belief_acc | corr_recall | stale_suppr |
|---|--:|--:|--:|
| ANN | 0.00 | 1.00 | 0.00 |
| CogOFF | 0.62 | 1.00 | 0.62 |
| CogON | **1.00** | 1.00 | 1.00 |

`feature_lift = +0.38`, `vs ANN = +1.00`. **The metric demonstrably registers a real
belief-revision win when the mechanism fires.** So zeros below are a true negative, not a
broken harness.

## Realistic sweep — contradiction (`--probes 24 --gap 20 --subtlety 0.25`)

| distractors | arm | belief_acc | corr_recall | stale_suppr |
|--:|---|--:|--:|--:|
| 100 | ANN | 0.00 | 0.67 | 0.00 |
| 100 | CogOFF | 0.00 | 0.92 | 0.00 |
| 100 | CogON | 0.00 | 0.92 | 0.00 |
| 1000 | ANN | 0.00 | 0.42 | 0.00 |
| 1000 | CogOFF | 0.00 | 0.79 | 0.00 |
| 1000 | CogON | 0.00 | 0.79 | 0.00 |
| 10000 | ANN | 0.00 | 0.29 | 0.00 |
| 10000 | CogOFF | 0.00 | 0.79 | 0.00 |
| 10000 | CogON | 0.00 | 0.79 | 0.00 |

**Mean feature lift: +0.00.** Refinement mode (100/1000) is identical: all arms 0.00.

## What this proves

1. **Cognition lifts correction *recall*** (0.29 → 0.79 at 10k distractors) — it reliably
   surfaces the correction as a candidate. That part works and is worth keeping.

2. **Cognition does NOT revise the belief.** `belief_accuracy` and `stale_suppression` are
   **0.00 everywhere**: the outdated fact is *always* rank 1, the correction *always* ranks
   below it. The agent keeps answering with the stale belief.

3. **The old benchmark's "win" was a recall illusion.** `cognitive_benchmark.py` scored
   contradiction as a win because the correction moved rank 99 → 2. By the stricter
   "does the current belief actually supersede the old one?" metric, that is still a loss.

## Root cause (informs Phase 2 + Phase 3)

The Contradicts/Refines mechanism is **purely additive on the correction** — it boosts the
*target* of the edge. It never **demotes the source** (the superseded fact) in the ranking
the agent actually sees:

- `contradiction_weaken_factor` weakens the stale fact's *outgoing Association edge
  weights*. But when the agent queries the topic directly, the stale fact is the
  cosine-nearest seed, and cosine dominates the fused score. Weakened outgoing association
  edges don't lower that cosine, so the stale fact stays rank 1.
- The additive graph delta on the correction is bounded and cannot overcome the cosine gap
  to the stale fact (same structural ceiling as abstraction-at-scale).

**Implication for the plan:**
- **Phase 2 (confidence-scaled fusion)** can help the correction climb, but additive
  boosting alone cannot reliably put it *above* a cosine-nearest stale fact.
- **Phase 3 is the real fix:** belief revision must act at **consolidation time on the
  stored/ranked representation of the superseded fact** — e.g. an explicit
  supersession/demotion signal that lowers the stale fact's effective retrieval score (not
  just its outgoing association weights), so that even pure cosine (`alpha=1.0`) returns the
  current belief first. Target: `belief_accuracy(CogON) > 0` at `alpha=1.0`.

## Reproduce

```bash
python benchmarks/cognitive_eval/belief_revision.py --mode contradiction --probes 24 --distractors 100 1000 10000
python benchmarks/cognitive_eval/belief_revision.py --mode refinement   --probes 24 --distractors 100 1000
# calibration control (should show CogON belief_acc ~1.0):
python benchmarks/cognitive_eval/belief_revision.py --dimension 64 --distractors 0 --subtlety 0.10 --gap 0 --probes 8
```
