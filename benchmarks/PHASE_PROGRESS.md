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

## Phase 2 — Detection precision + precision arm ✅ (2026-07-09)

**Problem (measured).** A coexisting-facts diagnostic (same concept, both valid, non-opposing,
same cosine as a real correction) showed the detector at **precision 0.50** for BOTH mechanisms:
it linked all 8 genuine belief pairs (recall 1.0) AND all 8 coexisting pairs (false-demoting
half of everything it touched). Phase 1's +1.00 belief_accuracy had been bought with destructive
false positives — the root risk from the original review.

**Change.**
- **Refinement text-overlap floor** (`refinement_text_threshold`, default 0.25). A refinement is
  a *re-statement* of the same claim → high text overlap. Coexisting facts (independent content,
  overlap ~0.1) are rejected. `check_refinements` now fetches record text and gates on Jaccard.
- **Contradiction opposition-marker gate** (`contradiction_require_opposition`, default true).
  `has_opposition_marker()` (new, `extract.rs`) fires on explicit negation/contrast cues ("not",
  "actually", "instead", "no longer", "n't", …). A genuine contradiction *opposes* the old claim;
  coexisting facts do not. Lightweight bag-of-cues (documented as heuristic, not NLI).
- Both default-on (safe by default), configurable, exposed as Python kwargs. The one marker-less
  contradiction unit test and `cognitive_benchmark.py`'s contradiction scenario were updated to be
  *faithfully* opposing ("…is not compiled; it actually runs through interpretation").
- **Precision arm** added to `belief_revision.py`: injects coexisting-fact pairs and reports
  `false_demotion` (fraction wrongly linked with a belief edge).

**Result.**

| check | before | after |
|---|--:|--:|
| Detection precision (coexisting diagnostic), contradiction & refinement | 0.50 | **1.00** |
| Belief eval (cc=0.7, 100/1000 distractors) — belief_acc / false_demotion | 1.00 / (unmeasured) | **1.00 / 0.00** |
| Cognitive benchmark scale / toy | 3/4 · 2/4 | **3/4 · 2/4** (contradiction feature-help YES preserved) |

Rust suite **173 passed / 0 failed**, clippy clean.

**Conclusion.** Belief revision is now proven (feature_lift +1.00) **and safe** (false_demotion
0.00) — a memory that revises beliefs without burying coexisting facts. Recall floor held.

**Deferred (safe follow-up).** The detection cosine gate is caller-configurable; the new precision
gates make *lowering* it below 0.5 safe (the text/opposition signals prevent false positives), so
extending the working regime below cos 0.5 is now a tuning exercise rather than a safety risk.

---

## Phase 3 — Re-wire abstraction into retrieval (next)

Abstraction is architecturally dead: the augmenter expands 1 hop from *memory* seeds and drops
non-memory targets, and `Association` edges are memory↔concept only — so concept/abstraction paths
are never traversed. Re-wire retrieval (concept-seeded or 2-hop) so a query hitting one concept can
reach memories of a sibling concept through the abstraction parent, and prove isolated lift on the
abstraction scenario.
