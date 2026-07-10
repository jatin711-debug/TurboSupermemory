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

## Phase 3 — Re-wire abstraction into retrieval ✅ wiring (2026-07-10)

**Problem (from Phase 0).** The augmenter expanded 1 hop from *memory* seeds and dropped every
non-memory target; since `Association` edges are memory↔concept only, the concept graph and the
whole abstraction hierarchy were **never traversed** — `build_abstractions()` produced nodes/edges
that had zero effect on retrieval.

**Change (`activation.rs`).** Replaced the memory-only 1-hop expansion with concept-mediated
traversal from each memory seed:
- `mem → concept → mem` (2-hop association spread, boost 0.3) — reach memories sharing a concept.
- `mem → concept → parent → sibling-concept → mem` (abstraction bridge, boost 0.6) — reach a
  *sibling*-concept memory via the learned abstraction parent. The bridge pays a single `decay`
  (not per-hop) because it is a deliberate learned link, not incidental multi-hop noise.
- `mem → mem` Refines/Contradicts (strong ×1.0) and Temporal (×0.5) preserved.
- Suppressed hub concepts are skipped; expansion stays bounded by `expansion_max_candidates`.
New unit test `augmenter_reaches_sibling_concept_via_abstraction` proves the target is reached via
`a → parent → b → target` with a positive delta.

**Validation.** Rust **174 passed / 0 failed**, clippy clean. No regression on the critical
guarantees: ANN Recall@5 **100%**; belief revision held (contradiction 1.00/1.00, refinement
0.96/1.00, false_demotion 0.00); cognitive('memory') Recall@5 99.8%.

**Honest limitation — surfacing is blocked by fusion, deferred to Phase 5.** The abstraction bridge
now *reaches* the sibling-concept target, but at scale the target does not *surface* in the top-k.
Root cause (diagnosed): `hydrate_and_fuse` normalizes the graph delta by the result-set `max_delta`,
and the cosine-near ANN seeds accumulate large deltas (lexical + rich concept associations) that
shrink the far-cosine target's normalized delta below the distractor cosine floor. The same
result-set-relative normalization drove cognitive('') empty-query recall from 1.2% → 0.0%. Both are
Phase 5 (fusion) — the abstraction mechanism is wired and reachable; it needs an **absolute**
(non-result-set-relative) graph-boost to surface. Also observed: naive concept extraction
(`max_concepts=10`) builds hundreds of spurious abstraction parents from junk tokens, diluting the
bridge — hub suppression (concept evolution, C3) is the robustness follow-up.

---

## Phase 4 — Isolate reinforcement lift (next)

Reinforcement shows no feature-attributable lift (CogON==CogOFF everywhere). Build an isolated eval
where a rehearsed memory must outrank a cosine-nearer non-rehearsed one; keep/tune only if it shows
lift.

## Phase 5 — Unify + calibrate fusion (next; unblocks Phase 3 surfacing)

Replace the result-set-relative `delta / max_delta` normalization with a bounded **absolute**
graph boost so cosine-far, graph-reached memories (abstraction targets) surface on their own delta
magnitude, and empty-query temporal displacement stops burying true neighbors. Re-validate all
evals + recall.
