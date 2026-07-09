# Cognitive MVP — Phase Progress

Living record of the four-mechanism cognitive MVP effort (belief revision + reinforcement +
abstraction + forgetting/importance). Baseline: [PHASE0_GROUND_TRUTH.md](PHASE0_GROUND_TRUTH.md).
Ship rule: a change ships only if it improves (or holds) cognition's marginal lift over plain
ANN **without** regressing recall@10.

---

## Phase 1 — Fix belief-eval geometry ✅ (2026-07-09)

**Problem (from Phase 0).** `belief_revision.py` modeled a "correction" as `randn`-jittered
vector; in 768-dim that jitter's norm scales with √dim, so `subtlety=0.25` produced
cos(stale, correction) ≈ 0.11 — near-orthogonal. Detection gates on cosine ≥ 0.5, so **zero
belief edges formed** and `belief_accuracy` was pinned at 0.00 regardless of demotion quality.

**Change.** `benchmarks/cognitive_eval/belief_revision.py`:
- Added `vec_at_cosine(base, target_cos, rng)` — returns a unit vector at an *exact* cosine to
  `base` (`c*base + sqrt(1-c^2)*base_perp`), dimension-independent.
- The correction now sits at a controlled cosine to the **stale fact** (not a randn jitter off
  the topic center).
- Renamed the `--subtlety` knob to `--correction-cos` (default 0.7) with honest semantics.

**Result (1000 distractors, probes=24, gap=20).** CogON vs CogOFF (belief edge is the only diff):

| mode | correction_cos | ANN | CogOFF | **CogON** | feature_lift |
|---|--:|--:|--:|--:|--:|
| contradiction | 0.85 / 0.70 / 0.55 | 0.00 | 0.00 | **1.00** | **+1.00** |
| refinement | 0.85 / 0.70 / 0.55 | 0.00 | 0.00 | **1.00** | **+1.00** |
| contradiction | 0.45 / 0.35 (below gate) | 0.00 | 0.00 | 0.00 | +0.00 |

**Conclusion.** Belief revision (Contradicts/Refines edge + supersession demotion) is **validated
at scale and fully feature-attributable** for corrections at cosine ≥ 0.5. The demotion mechanism
was correct all along — Phase 0's "no value" was an eval-geometry false negative. The clean
drop to 0.00 below cos 0.5 is the **detection gate**, and is exactly Phase 2's target.

**Caveat.** This proves belief revision *fires* and *flips* the ranking; it does NOT yet prove
*safety* (no false demotion of coexisting facts) — that's the Phase 2 precision arm.

---

## Phase 2 — Broaden detection + precision arm (in progress)

Targets: (a) link belief pairs **below** the cos-0.5 gate using shared-concept +
temporal-adjacency + text signals; (b) add a precision arm — inject coexisting facts (same
concept, both valid, no supersession) and assert they are NOT demoted. Goal: extend the working
regime below cos 0.5 **while** keeping false-demotion low.
