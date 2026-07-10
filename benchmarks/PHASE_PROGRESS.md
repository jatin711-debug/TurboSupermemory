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

## Phase 4 — Isolate reinforcement lift ✅ measured (2026-07-10)

**Eval (`reinforcement_eval.py`, new).** Controlled-cosine geometry: per probe, an `anchor` ANN
seed plus **three sibling memories** at equal low cosine to the query (all non-ANN, reachable only
via `anchor → concept → sibling`). The siblings are symmetric, so cold retrieval orders them purely
by insertion tie-break; we rehearse the **last-inserted** sibling (the cold tie-break *loser*) so
any lift is attributable to reinforcement, not geometry. Metric: `P(rehearsed sibling outranks its
2 cold siblings)`; `lift = CogON(rehearsed) − CogOFF(not rehearsed)`.

**Result: no isolated lift.**

| distractors | CogOFF | CogON | lift |
|--:|--:|--:|--:|
| 100 | 0.25 | 0.29 | +0.04 |
| 1000 | 0.46 | 0.38 | −0.08 |

Mean lift **−0.02** — noise around zero. (Two earlier eval designs were discarded: equal-cosine
symmetric pairs gave a deterministic *tie-break* confound, CogOFF=CogON=1.00; a single low-cosine
memory was surfaced into the top-8 by base concept-expansion regardless of rehearsal.)

**Finding.** The edge-reinforcement mechanism works at the graph level (unit tests confirm
`reinforce` strengthens edges), but it is **decoupled from retrieval ranking**: (a) the augmenter's
graph boost only surfaces *non-ANN* candidates — reinforcing an ANN candidate's concept edge does
not re-rank it (the normal pool skips ANN hits); (b) base concept expansion already surfaces a
reachable memory whether or not it is rehearsed, subsuming reinforcement's contribution. This
rigorously confirms the BASELINE observation (CogON==CogOFF) with an isolated eval.

**Interpretation & next step.** Reinforcement's real role is the **retain/forget loop** (edge decay
+ importance feedback via `recompute_importance`), an axis distinct from direct ranking — the right
way to demonstrate it is a retention eval (rehearsed memory persists while a never-accessed one
decays/evicts), not a ranking eval. Making reinforcement *directly* re-rank would need an explicit
access-salience term in fusion (a positive analogue of the supersession-demotion factor); that is a
new, separately-validated feature, deliberately **not** bolted on here to avoid destabilizing the
validated belief/recall results.

---

# Stage A — Real-data validation (LongMemEval) — THE GO/NO-GO

Everything in Phases 0–5 was validated on synthetic evals. Stage A tests belief revision on
**real conversational data** (LongMemEval), the industry benchmark, to answer the existential
question: does the cognitive layer's marquee feature actually help in the real world?

**Harness (`run_belief_longmemeval.py`, new).** LongMemEval test set (147 questions; 22 are
`knowledge-update` = the belief-revision subset: "what camera do I have NOW", "PREVIOUS status
before current"). Real `all-MiniLM-L6-v2` embeddings, mock sentence extractor, **one isolated
engine per conversation** (belief detection stays within a user), belief revision **ON vs OFF**
(only the Contradicts/Refines edges + supersession demotion differ; everything else identical).
Metric: distinctive-token answer-containment at rank 1/3/k — a retrieval proxy, so absolute numbers
are noisy but the **ON−OFF lift** is trustworthy (proxy noise cancels). The loader was fixed to
preserve `question_type`/`is_abstention`; the adapter gained `belief_revision` + shared-`model`
flags.

**Result: belief revision is NET-NEGATIVE on real data.**

| question_type | n | hit@1 off/on/lift | hit@3 off/on/lift | hit@k off/on/lift |
|---|--:|---|---|---|
| **knowledge-update** | 22 | 0.27/0.27/**+0.00** | 0.73/0.55/**−0.18** | 0.91/0.82/**−0.09** |
| single-session-user | 20 | 0.75/0.65/−0.10 | 0.90/0.70/−0.20 | 1.00/0.85/**−0.15** |
| single-session-preference | 9 | 0.11/0.00/−0.11 | 0.33/0.11/−0.22 | 0.33/0.22/−0.11 |
| single-session-assistant | 16 | 0.25/0.06/−0.19 | 0.31/0.44/+0.12 | 0.62/0.62/+0.00 |
| temporal-reasoning | 38 | 0.37/0.32/−0.05 | 0.45/0.45/+0.00 | 0.58/0.61/+0.03 |
| multi-session | 36 | 0.11/0.14/+0.03 | 0.22/0.22/+0.00 | 0.39/0.42/+0.03 |

Belief edges built across 147 conversations: **refine 23,255 + contra 5,056 = 28,311** — refinement
fires on ~20% of ALL fact pairs, essentially unconditional. It **never helps** knowledge-update (the
subset it targets) and **hurts** the single-session types (single-session-user 1.00 → 0.85).

**Root cause — the detector over-fires catastrophically on real text.** The thresholds (refinement
cosine ≥ 0.5 + text-overlap ≥ 0.25 + shared concept) were calibrated on synthetic near-orthogonal
vectors with a wide cosine spread. Real embeddings (MiniLM) have a **compressed** cosine
distribution where most same-topic sentence pairs clear 0.5, and real conversations are full of
legitimately-similar-but-not-superseding facts. Each spurious refinement demotes the "older"
memory, so within a conversation later sentences bury earlier ones en masse → correct answers fall
out of top-k. **The Phase-2 precision (1.00 on clean synthetic coexisting facts) does not transfer
to real text.**

**Honest caveats.** (1) Retrieval-containment proxy, not LLM-judged — but the OFF arm has high
recall and ON demonstrably drops it, so the harm is real, not a proxy artifact. (2) Lightweight
MiniLM — a different embedding distribution could shift the thresholds, but 28k edges shows the
problem is structural, not a tuning nudge. (3) Mock sentence-splitting inflates near-duplicate
facts — real LLM extraction (ollama, currently down) would dedupe and likely reduce over-firing,
but the demotion-hurts result stands. The gold-standard metric (retrieve → LLM answer → LLM judge)
is the follow-up once an LLM is available.

**Verdict & route implication.** For the current implementation this is a **NO**: belief revision
must not be scaled or shipped as-is — it degrades real-world retrieval. This is the single most
important finding of the effort and it confirms the no-compromises route: the production blocker is
**detection trustworthiness on real text (Stage B)** — thresholds calibrated to the embedding
distribution (or adaptive/percentile-based), mutual-nearest-neighbour + LLM/NLI verification before
the *destructive* demotion, recoverable/bounded demotion, and scope-respecting detection (it
currently ignores scope). Only once belief revision is net-positive on real data does scaling the
cognitive layer make sense. The vector store itself (OFF arm) retrieves well — that part is solid.

---

# Stage B — Trustworthy detection (mutual-nearest-neighbour) — FLIPPED TO POSITIVE

Stage A showed belief revision over-fires on real text (28k edges, ~20% of all fact pairs) and is
net-negative. Stage B fixes the *detection*, not the demotion.

**Change (`engine.rs` `check_refinements` / `check_contradictions`).**
- **Mutual-nearest-neighbour gate** (`nearest_superseding_neighbor`): a supersession fires only if
  the two memories are *each other's* nearest same-concept neighbour. This is **scale-free** — it
  does not depend on the absolute cosine threshold (which means different things across embedding
  models and caused the over-firing), so it is the structural cure.
- **One edge per new memory** — a new fact supersedes its single best prior fact, not up to 10
  qualifying neighbours (the main over-firing source).
- **Scope-respecting** — detection no longer pairs memories across users (fixes a multi-tenancy
  leak: it previously could demote user A's memory based on user B's).
- **Bounded demotion** (`SUPERSESSION_DEMOTION_FLOOR = 0.3`) — a memory is never buried below 30%
  of its score no matter how many times it is flagged, capping the blast radius of any residual
  false positive.

**Result (LongMemEval test set, ON vs OFF, per-conversation).**

| metric | Stage A | **Stage B** |
|---|--:|--:|
| belief edges (147 convs) | refine 23,255 / contra 5,056 | **refine 6,701 / contra 429** (−75%) |
| knowledge-update hit@1 lift | +0.00 | **+0.23** (0.27 → 0.50) |
| knowledge-update hit@3 lift | −0.18 | **+0.00** |
| single-session-user hit@3 lift | −0.20 | −0.05 |
| single-session-assistant hit@3 lift | (n/a) | +0.12 |
| verdict | "NO value" | **"MEASURABLY improves knowledge-update"** |

Belief revision is now **net-positive where it matters** (current fact surfaced at rank 1 far more
often on the belief-revision subset), and the Stage-A collateral damage is roughly halved. Synthetic
`belief_revision` held +1.00 / false_demotion 0.00 (MNN doesn't break clean pairs); storage suite
green (74+3); clippy clean.

**Residual (next iteration).** 6,701 refinements is still over-firing — within a single conversation,
sequential same-topic sentences can be mutually-nearest, so refinement still demotes some legitimate
facts, leaving minor harm on single-session-user (−0.10 hit@k) and the KU recall tail (−0.09). The
next tightening is a stronger refinement signal (higher text-overlap floor — a genuine re-statement,
not just topical similarity — and/or LLM/NLI verification before the destructive demotion), plus the
gold-standard LLM-judge metric once ollama is available. But the direction is proven: **MNN turned
the marquee feature from a liability into a measurable win on real data.**

## Phase 5 — Unify + calibrate fusion ✅ (2026-07-10)

**Problem.** `hydrate_and_fuse` computed the graph boost as `act / max_act` — normalized by the
result-set maximum. A weak incidental signal (a temporal successor on an empty query) got inflated
to full strength when every delta was small (empty-query cognitive recall 1.2% → 0.0%), while a
genuine far-cosine signal got crushed when some other candidate had a large delta.

**Change (`engine.rs::hydrate_and_fuse`).** Replaced the result-set-relative normalization with an
**absolute, saturating** transform:

```
final = cos + (1 - alpha) * act / (1 + act)      # was: cos + (1 - alpha) * act/max_act
```

The boost now depends only on the candidate's OWN graph signal, bounded in `[0, 1)`, so it neither
inflates weak signals nor crushes genuine ones. Demotion (belief revision) is unchanged.

**Result.**

| metric | before | after |
|---|--:|--:|
| ANN Recall@5 (floor) | 100% | **100%** |
| Cognitive('memory') Recall@5 | 99.8% | **100%** |
| Cognitive('') empty-query Recall@5 | 0.0% | **100%** |
| Belief revision (contra / refine, 100&1000) | 1.00 / 0.96–1.00 | **1.00 / 0.96–1.00** (false_dem 0.00) |
| Cognitive benchmark scale / toy | 3/4 · 2/4 | 3/4 · **1/4** |

Storage suite **74 passed**, clippy clean. The empty-query displacement is fully fixed and the
fusion is now interpretable. The toy 2/4→1/4 is the loss of the abstraction *non-attributable* win
(the old max-normalization was artificially inflating a far-cosine candidate's boost); the one
*feature-attributable* win (contradiction, toy) is preserved, and scale is unchanged.

**Crystallized abstraction root cause.** With correct (absolute) fusion, the abstraction target
still doesn't surface because the `cognitive_benchmark` scenario builds vectors with `make_close_vec`
— which, like the pre-Phase-1 belief geometry, yields **near-orthogonal** vectors in 768-dim (seeds
at cosine ~0.1 to the query). Graph propagation is scaled by that weak seed activation, so every
propagated delta (including the abstraction bridge) is negligible. The fix is the **same
controlled-cosine geometry** Phase 1 applied to belief — the abstraction mechanism is wired and
proven in isolation (unit test); demonstrating its lift needs the scenario geometry fixed and seed
propagation to carry lexical/seed strength (Phase 4/abstraction follow-up).
