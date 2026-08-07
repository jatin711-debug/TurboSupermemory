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

**Stage B.2 — the residual was assistant boilerplate.** A dump of the actual refinement pairs
showed the remaining 6,701 refinements were overwhelmingly the mock extractor sentence-splitting the
*assistant's* verbose, bulleted, repetitive responses (e.g. `"* After 1-2 years... 70-80%"` →
`"* After 2-3 years... 50-60%"` — different list items, not a supersession; `"What is your budget?"`
→ `"My budget is $800"` — question→answer). A conversational memory should store the **user's** facts,
not replay the assistant's chatter, so the adapter gained a `store_roles` filter (eval flag
`--user-only`). With user-only facts:

| metric | all roles | **user-only** |
|---|--:|--:|
| belief edges (147 convs) | refine 6,701 / contra 429 | **refine 683 / contra 80** (−90%) |
| knowledge-update hit@1 lift | +0.23 | **+0.23** (held) |
| single-session-user lift | −0.10 hit@k | **+0.00** |
| single-session-preference lift | −0.11 | **+0.00** |
| single-session-assistant lift | (n/a) | **+0.00** |

Belief revision keeps its full +0.23 knowledge-update rank-1 win and the collateral damage is
**eliminated** — every single-session type is exactly 0.00 (belief revision no longer touches
memories it shouldn't). Base recall even improved from the reduced noise (KU hit@3 OFF 0.73 → 0.86).

**Net Stage B verdict: belief revision is now genuinely net-positive on real data with no downside** —
+0.23 rank-1 on the belief-revision subset, zero collateral, 763 (not 28,311) edges. The path from
here is the gold-standard LLM-judge metric (needs ollama) and LLM/NLI verification for the highest-
stakes demotions, but the mechanism is validated: **mutual-nearest-neighbour detection + user-scoped
facts make belief revision work on real conversational memory.**

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

---

# W1 — Role-aware memory as a first-class engine feature ✅ (2026-07-10)

Stage B.2 proved user-scoping eliminates belief-revision collateral, but the fix lived **only in
the eval adapter** (`store_roles` dropped assistant messages before insert). Any real TSM user still
got Stage-A over-firing. W1 moves role-awareness into the engine.

**Change.**
- `Record`/`MetaRecord` gain a durable `source_role: Option<String>` (`serde(default)`, survives WAL
  replay). Unlike `scope` it **never filters retrieval** — every role stays searchable.
- New `TierConfig.belief_source_roles` (default `None` = legacy role-blind). When set,
  `check_refinements` / `check_contradictions` / `nearest_superseding_neighbor` only let in-list
  roles create or receive a supersession edge (`role_allowed` helper).
- Write path: `insert_with_payload_role` / `insert_batch_with_payload_role` /
  `update_with_payload_role` (old signatures delegate with `None`; REST/gRPC/existing callers
  unchanged). Exposed via Python (`source_role` kwarg + `belief_source_roles` ctor kwarg), REST, and
  gRPC (proto field 8). Rust tests: role-filtered exclusion (+ asserts the excluded memory is still
  retrievable) and `source_role` WAL survival. Storage suite **74 → 76** green, clippy clean.

**Two-way productization decision, on the FULL LongMemEval set (500 conversations, n=72
knowledge-update — larger/harder than the 147-subset used for Stage A/B, hence smaller absolute
lifts; all arms show ~+0.08 KU, so the smaller number is a dataset property, not a regression).**

| metric | baseline (store-all, role-blind) | mode a (`--user-only`, adapter drops assistant) | **mode b (`--role-filtered`, engine)** |
|---|--:|--:|--:|
| knowledge-update hit@1 lift | +0.08 | +0.08 | **+0.07** |
| single-session-user collateral (hit@1) | **−0.12** | −0.03 | **+0.00** |
| single-session-assistant recall (hit@k, absolute) | 0.80 | **0.27** | **0.77** |
| worst single-session lift | −0.12 | −0.03 | **−0.03** |
| belief edges (500 convs) | 22,708 | 2,506 | **2,345** |

**Verdict: mode b strictly dominates and is the recommended production config.** It keeps the belief
win (+0.07 ≈ baseline's +0.08), **eliminates** the single-session-user collateral (+0.00 vs baseline
−0.12), *and* preserves assistant-answerable recall (0.77 ≈ baseline 0.80) — whereas mode a throws
assistant messages away and craters that recall to 0.27. The engine role-filter is therefore
strictly better than the eval-only `store_roles` hack it replaces: same clean belief behavior,
without discarding retrievable content. Recommended production config:
`belief_source_roles=["user"]` + tag every insert with its `source_role`. Engine default stays
`None` (backward-compatible; a caller who opts in without tagging roles simply gets no
supersessions, never silent role-blind behavior). Committed `865224d` (code) + this section.

# W2 — Regression gate ✅ (2026-07-10)

Automated guard so a refactor can't silently destroy the cognitive-layer wins (as the 2026-06-29
fusion no-op did, and as Stage-A over-firing did). `benchmarks/regression_gate.py` + `make gate`,
documented in `AGENTS.md` ("run before every commit that touches the engine/cognitive layer").

**Checks (PASS/FAIL table, nonzero exit on any fail):** `cargo fmt --check`; `clippy -D warnings`
(workspace); `cargo test` (workspace ex-python); synthetic belief refinement+contradiction (assert
lift ≥ +0.9, false_demotion ≤ 0.05); LongMemEval smoke `--limit 40 --role-filtered` (assert KU
hit@1 lift ≥ −0.10, ON edges ∈ [10, 800] — the over-firing ceiling, worst single-session lift ≥
−0.15); recall audit (ANN ≥ 95%). The two evals emit a machine-readable `GATE_SUMMARY: {json}` line
the gate parses, so it never depends on the human tables.

**Full clean run: GATE PASS 7/7** (synthetic +1.00/false_dem 0.00 both modes; LME smoke edges 248,
no collateral; recall 100%).

**Sabotage test (proves the gate has teeth).** Two findings:
1. *Disabling the reverse-MNN mutual check* did **not** explode the edge count (248 → 335) — because
   "one edge per new memory" independently bounds volume; reverse-MNN removal is a **precision**
   regression (collateral +0.00 → −0.11, KU lift +0.00 → +0.20), not a volume one. At smoke scale
   (`--limit 40`, per-type single-session n ≈ 5) that collateral signal is too noisy for a tight
   threshold — so the gate's *reliable* teeth are the deterministic checks, not smoke per-type rates.
   (Catching a reverse-MNN-class precision regression needs the **full** 500-conv run, which is the
   per-workstream manual step, not the fast pre-commit gate.)
2. *Breaking the demotion* (making supersession a no-op — the "cognition silently no-ops" class, same
   shape as the 2026-06-29 fusion bug) collapsed synthetic belief lift **+1.00 → +0.00** for both
   refinement and contradiction, tripping the `lift ≥ 0.9` check. **Gate FAILS as designed.** Code
   reverted; clean build restored to +1.00/+1.00.

# W3 — Verified demotion (propose → verify → commit) ✅ (2026-07-10)

Demotion is the only *destructive* cognitive action. MNN (W1/Stage B) is a
*geometric* gate; W3 adds a *semantic* gate before a memory is ever buried — with
no LLM server (local NLI cross-encoder on GPU).

**Engine (behavior-preserving refactor).** Detection split from commitment:
`propose_refinements`/`propose_contradictions`/`propose_supersessions` are pure
(no graph mutation, no demotion) and return `ProposedSupersession`
{old,new,offsets,kind,cosine}; `commit_supersessions` (and
`commit_supersessions_by_id` for the Python round-trip) apply the edge + bounded
demotion + concept transfer. `check_refinements`/`check_contradictions` are now
thin propose+commit wrappers, so the auto-commit path is identical (storage
76→77 tests green; unverified full-500 reproduced W1 exactly: 2,349 edges, KU
+0.069). New `defer_supersession_commit` config lets consolidation hand the
decision to the caller. Exposed via PyO3 (`propose_supersessions`,
`commit_supersessions`, `defer_supersession_commit`).

**Verifier (`verification/nli.py`).** Local NLI cross-encoder
(`cross-encoder/nli-deberta-v3-xsmall`) loaded via `transformers` directly
(sentence_transformers is unusable here — torchcodec/FFmpeg). premise=new,
hypothesis=old; **accept contradiction+entailment, REJECT neutral** — neutral =
coexisting facts, the exact false positive that caused single-session collateral.
Proven on examples: Camry-vs-F150 → contradiction (accept), hiking-vs-color →
neutral (reject), moved-to-Boston → entailment (accept).

**Validation (LongMemEval, 200 conversations, same convs, mode b).**

| metric | unverified | **verified (NLI)** |
|---|--:|--:|
| belief edges | 962 (879 R / 83 C) | **393 (382 R / 11 C)** −59% |
| knowledge-update hit@1 lift | +0.11 | **+0.07** (held; Δ≈1 q at n=28) |
| temporal-reasoning hit@1 | −0.02 | **+0.00** |
| single-session-assistant hit@k | −0.05 | **+0.00** |
| every non-KU type (h1/h3/hk) | some noise | **exactly +0.00** |

NLI audit on 288 real proposed pairs (`verify_supersessions.py`): **58% rejected
as neutral** (coexisting facts), 42% accepted (97 entailment + 23 contradiction)
— the 42% accept rate matches the edge reduction. **Verdict: verification holds
the knowledge-update win, halves the destructive demotions, and drives all
collateral to exactly zero** — the NLI gate removes the coexisting-fact false
positives that geometry alone cannot. Opt-in (`verify_demotions` /
`--verify-demotions`); the default no-verifier path is unchanged (gate still
7/7). Small honest caveat: KU +0.11→+0.07 is within n=28 noise but the verifier
may reject an occasional genuine refinement — a margin/threshold tune (exposed:
`min_margin`, `accept_labels`) could recover it; the default rule favors
precision (never demote a still-true memory).

**NOTE (out of scope, flagged separately):** the per-conversation eval harness
(one MemoryEngine per conversation) OOMs at the full 500-set — a native-memory
leak across engine create/close cycles (close() likely not releasing
mmap/redb/tantivy/usearch). Hence the 200-conv validation. Relevant to W7 (scale).

# W4 — Abstraction / concept-expansion real-data verdict ✅ (2026-07-11)

Abstraction was the last cognitive mechanism without a real-data verdict — WIRED
(Phase 3 + unit test: a query reaches a sibling-concept memory through the learned
abstraction parent) but unproven on real conversational data. W4 isolates it.

**Mechanism (`SpreadingConfig.concept_expansion`, default true).** When false, the
augmenter skips the concept-mediated branch entirely (2-hop `mem→concept→mem` and
4-hop `mem→concept→parent→sibling-concept→mem`), leaving only belief + temporal
traversal. Exposed via PyO3 + adapter. Unit test
`concept_expansion_toggle_isolates_abstraction_reachability` is a clean A/B — and
revealed the old Phase-3 test was partly reachable via the auto-created **temporal**
chain, not abstraction; the new test inserts a filler to break temporal adjacency so
the target is reachable ONLY through the abstraction bridge (graph suite 68→69).

**Isolation eval (`run_abstraction_longmemeval.py`, 200 convs, belief ON +
role-filtered in BOTH arms; only `concept_expansion` differs). Abstraction edges
build identically in both arms (61,399 vs 61,435) — only retrieval USE differs.**

| question_type (n) | hit@k lift (ON − OFF) |
|---|--:|
| temporal-reasoning (52) | **+0.06** (0.52→0.58) |
| knowledge-update (28) | −0.04 |
| multi-session (48) | −0.04 |
| all single-session types | +0.00 |

**Verdict: MIXED / marginal — honest near-neutral, NOT a broad win.** Concept +
abstraction expansion gives a real, mechanistically-sensible lift on
**temporal-reasoning** (+0.06 hit@k — the type where multi-hop concept bridges connect
facts across time), but pays for it with small losses on knowledge-update and
multi-session (−0.04 each). Net across types ≈ neutral, and every delta is 1–3
questions (within noise at n=28–52). Single-session types are untouched (+0.00), as
expected — they need no multi-hop. This does NOT clear the four-mechanism MVP bar of a
robust *isolated* lift; it clears it only narrowly and only for temporal-reasoning.

**Product call:** keep `concept_expansion` default ON (it is the established behavior,
belief evals W1/W3 ran with it on, and the temporal gain is the largest single
effect), but it is a candidate for **per-query-type gating** (on for
temporal-reasoning, off for multi-session/KU) in future — the current global setting
trades a temporal gain for multi-session/KU hit@k losses. The auto-verdict threshold
was tightened (worst-type ≥ −0.02 for a "win") so it reports MIXED honestly rather
than overclaiming.

**Four-mechanism MVP status after W4:** belief revision = **strong real-data positive**
(W1/W3: KU rank-1 win, zero collateral, NLI-verified); reinforcement (ranking) =
**honest negative** (Phase 4, no isolated lift); abstraction = **marginal/mixed**
(narrow temporal-reasoning gain, ~neutral overall); forgetting/retention = W5 (next).
The synthetic `cognitive_benchmark.py` geometry rework (planned W4 step to show
abstraction lift in an ideal low-distractor regime) is **deferred/subsumed** — the
mechanism is already proven wired by the unit test, and the real-data verdict (the one
that decides) is in; re-demonstrating a toy-scale lift would not change it.

# W5 — Retention / forgetting under a memory budget ✅ (2026-07-11)

Reinforcement showed no direct RANKING lift (Phase 4). Its real claim is the
retain/forget axis: under a bounded store, a memory that gets USED should survive
eviction over one that doesn't. W5 tests that — and it is the "forgetting" half of
the four-mechanism MVP.

**Mechanism (`TierConfig.access_aware_eviction`, default true).** `evict()` ranks
victims by cognitive salience — `access_score = access_count × 2^(-age/half_life)`
plus a grace window for recently-accessed records — so a rehearsed/retrieved memory
survives. Set false for the naive **FIFO** baseline (evict oldest-inserted first,
access ignored). Exposed via PyO3 + adapter; new `engine.contains_id` (+PyO3) for
gold-fact survival checks. Rust test `fifo_eviction_ignores_rehearsal`: FIFO drops a
heavily-rehearsed oldest record that access-aware eviction keeps (storage 77→78).

**Isolation eval (`retention_eval.py`, 200 convs, budget max_records=10).** Both arms
are IDENTICAL in every operation — same inserts, same rehearsals (3× per query text,
bumping access on the facts that query needs), same budget forcing eviction — and
differ only in `access_aware_eviction`. Scored on the 199 budget-pressured
conversations (n=188 queries).

| metric | OFF (FIFO) | ON (access-aware) | lift |
|---|--:|--:|--:|
| gold survival | 0.18 | **0.60** | **+0.41** |
| hit@1 | 0.06 | 0.34 | +0.27 |
| hit@3 | 0.11 | 0.47 | +0.36 |
| hit@k | 0.18 | 0.60 | +0.41 |

**Verdict: STRONG POSITIVE — the retain/forget mechanism works.** A used memory
survives budget-pressure eviction 3.3× more often under the cognitive policy than
under FIFO; end-to-end retrieval (hit@k) tracks survival exactly (+0.41). This is the
strongest mechanism result after belief revision, and it **reconciles the Phase-4
reinforcement negative**: reinforcement (access_count) doesn't re-rank, but it *does*
drive retention — reinforcement's value is real, on its proper axis. Honest caveat:
the rehearsal signal is oracle (we access exactly the query-relevant facts), modelling
"important facts get used"; the eval proves the engine *translates access into
retention* (+0.41 vs FIFO's ~0), which is the mechanism claim. In production the
rehearsal comes from real query traffic.

## Four-mechanism MVP scorecard — COMPLETE (2026-07-11)

The MVP bar (user, 2026-07-09): all four cognitive mechanisms proven with **isolated,
feature-attributable** lift on real data. Final verdicts:

| mechanism | real-data verdict |
|---|---|
| **Belief revision** | ✅ **strong positive** — KU rank-1 win, zero collateral, NLI-verified, role-scoped (W1/W3) |
| **Reinforcement** | ✅ **positive on its true axis** — no direct ranking lift (Phase 4) BUT drives retention (W5) |
| **Forgetting / retention** | ✅ **strong positive** — +0.41 gold survival under budget pressure (W5) |
| **Abstraction** | 🟡 **marginal/mixed** — narrow temporal-reasoning gain, ~neutral overall (W4) |

Three of four mechanisms are clear real-data wins; abstraction is honestly marginal.
The whole cognitive layer is now isolation-validated on the industry benchmark, first-
class in the engine (not eval hacks), and guarded by `make gate`. Remaining plan:
W6 gold-standard LLM-judge metric (needs ollama), W7 scale/ops (incl. the flagged
per-conversation eval-harness memory leak).

# W7 — Scale: measure first, then fix the dominant cost (2026-07-11)

Per PLAN W7, profile cognitive consolidation BEFORE optimizing, so the fix targets
the real bottleneck. `profile_consolidation.py` uses one long-lived engine (avoids
the flagged per-conversation eval-harness leak).

**Profile (dim=768, cognitive features on, single consolidation over a fresh store):**

| phase | N=2k | N=10k | scaling |
|---|--:|--:|---|
| ingest (insert_batch) | 0.13s | 0.64s | ~linear |
| **propose_supersessions (MNN)** | **13.2s** | **68.9s** | ~linear in N, but per-cycle |
| recompute_importance | 0.26s | 1.46s | O(N), minor |
| deduplicate (disabled) | 0 | 0 | — |
| evict (O(N) scan) | ~0 | ~0 | negligible |
| **trigger_consolidation (whole)** | **16.9s** | **277s** | **superlinear** |
| flush (+ graph JSON) | 0.46s | 0.55s | — |
| graph + store on disk | 47 MB | 246 MB | ~24 KB/record |

**Finding: supersession detection (mutual-nearest-neighbour) is the dominant cost** —
it does ~O(N) ANN searches (forward + reverse per candidate, ×refinement+contradiction)
**every consolidation cycle**, re-scanning ALL live records even though only newly-
inserted ones can be the "newer" side of a supersession. 69s at 10k → hopeless at 1M.
The full cycle is 277s at 10k because it runs that scan ~4× plus commit/concept-transfer,
concept-evolution, abstraction-building, and segment HNSW builds — several of which are
also superlinear (future work). The graph+store footprint (~24 KB/record → ~2.4 GB at
100k, ~24 GB at 1M) is a second, separate scale wall (the graph persists as JSON).

**Fix (matches PLAN's "seq-cursor"): incremental supersession detection.** New
`TierConfig.incremental_supersession_detection` (default false). A per-engine
`supersession_watermark` (AtomicU64) records the max `insert_seq` checked; `propose_*`
process only records at/after it (cands are newest-first, so they break at the
watermark), and `trigger_consolidation` advances it past the current max seq after
detection. So a steady-state cycle costs **O(new records), not O(total)** — a new memory
is checked once, when it's new (an older memory it supersedes is still found via the ANN
index). One-time bulk-load cost is unchanged. Correct across cycles (Rust test
`incremental_detection_finds_cross_cycle_refinement`: a supersession first detectable in
cycle 2 is still found). On a reloaded store the watermark starts past all loaded
history (no re-scan of the past). Opt-in so no default-behavior change; storage 78→79.

**Steady-state speedup (base=20k + delta=200, time the 2nd cycle):**

| 2nd consolidation | time | |
|---|--:|---|
| incremental OFF (full re-scan of 20,200) | 487 s | |
| incremental ON (delta 200 only) | 111 s | **4.4× faster** |

**Honest read:** incremental detection cut the propose cost from ~376 s (≈77% of the
cycle) to ~1 s — a huge reduction of the single dominant pass. But the cycle only sped
up 4.4× because **~111 s of OTHER O(N) consolidation work remains** and is NOT yet
incremental: importance recompute, segment sealing / HNSW builds, abstraction +
concept-vocabulary evolution, and the full graph-JSON snapshot. So the W7 fix removes
the biggest head of a multi-headed O(N) beast; reaching 1M needs the remaining passes
made incremental too (seq-cursor for importance; only-new-records for abstraction/vocab;
delta-encoded or binary graph snapshot instead of full JSON). The profiler
(`profile_consolidation.py`) is the tool to drive that follow-on, one measured head at a
time. **W7 status:** dominant cost measured + fixed (4.4× steady-state, propose now
O(delta)); remaining consolidation O(N) passes + graph-snapshot format + API/ops
hardening (tracing, auth, Docker) + the flagged per-conversation native-memory leak are
scoped follow-ons.

# W6 — Gold-standard LLM-judge metric (the moment of truth) ⚠️ (2026-07-11)

Everything W1–W5 measured was a RETRIEVAL proxy (does a top-k memory contain the gold
token?). W6 adds the real LongMemEval metric with REAL extraction: OpenAI (gpt-4o-mini)
extracts facts, retrieves, answers using ONLY the retrieved memories, then grades the
answer vs gold. Role-filtered belief ON vs OFF, 120 conversations, top_k=10.

**Retrieval proxy (real OpenAI extraction — confirms the prior mock-based result holds):**
knowledge-update hit@1 OFF 0.39 → ON 0.50 (**+0.11**); the rank win is real and survives
real extraction.

**Gold-standard LLM-judged ANSWER ACCURACY (OFF / ON / lift):**

| question_type (n) | judged acc lift |
|---|--:|
| knowledge-update (18) | 0.39 / 0.39 / **+0.00** |
| single-session-preference (9) | +0.11 |
| single-session-assistant (11) | +0.09 |
| temporal-reasoning (28) | −0.04 |
| single-session-user (18) | −0.11 |
| multi-session (31) | −0.13 |

**GOLD VERDICT: the belief-revision RETRIEVAL win does NOT translate into an
answer-accuracy win at top_k=10.** knowledge-update judged accuracy is identical
(+0.00); other types are small-n noise that roughly cancels. This is the single most
important honest finding of the whole effort, and it is exactly why a gold-standard
metric exists.

**Why (the mechanism, not an excuse).** Belief revision improves the RANK of the current
fact and demotes the stale one. But the judge feeds the LLM the WHOLE top-10 set, so if
the current fact is anywhere in those 10 — which it usually is at k=10 — the LLM answers
correctly regardless of rank. Rank-within-context is invisible to an answer metric when
context is generous. So the proxy (rank-sensitive) and the gold standard (rank-insensitive
once the fact is in-context) measure different things, and here they disagree.

**What this does and does not say.**
- It does NOT invalidate belief revision as a mechanism — the current fact genuinely
  ranks higher and stale facts are demoted (proven W1–W3, NLI-verified).
- It DOES say: at a generous context budget (top_k=10), that rank improvement buys no
  extra answer accuracy on this benchmark. The value, if any, lives in **low-k / tight
  token-budget** regimes (top_k = 1–3), where excluding the stale fact and surfacing the
  current one changes what the LLM even sees. That is the open, testable hypothesis
  (`--top-k 3` judged rerun) — untested here, so no claim is made.
- Retention/forgetting (W5) is a DIFFERENT axis this metric did not probe.

**Honest bottom line for the product thesis:** the cognitive layer's headline claim
("belief revision surfaces the current fact") is TRUE at the retrieval level but, at
top_k=10, does NOT yet convert to better answers. Whether it converts at small k is the
next experiment. This is the number that should drive the "is this a product?" decision —
and it says "not proven yet at the answer level," which is worth knowing honestly rather
than shipping on a proxy. Run: gpt-4o-mini, 367 judge calls, ~$0.25.

---

# Phase A — kill-gate experiments (Roadmap v2)

## A1 + A3 — judged accuracy vs context budget ❌ KILL SIGNAL (2026-07-11)

One retrieval pass at top_k=10, answers judged at TRUNCATED k ∈ {1,3,5,10}
(gpt-4o-mini extraction + judge, 120 convs, role-filtered belief ON vs OFF,
1,363 judge calls, ~$0.50). The hypothesis: the belief-revision rank win pays
off where context is tight (k≤3), because excluding the stale fact changes
what the LLM sees.

**Result — the hypothesis is FALSE:**

| k | avg ctx tok/query | knowledge-update judged acc (off→on) | lift |
|--:|--:|---|--:|
| 1 | ~17 | 0.44 → 0.44 | **+0.00** |
| 3 | ~55 | 0.44 → 0.39 | **−0.06** |
| 5 | ~91 | 0.50 → 0.39 | **−0.11** |
| 10 | ~187 | 0.33 → 0.28 | **−0.06** |

No judged-accuracy lift at ANY context budget — including k=1, the most favorable
regime. Worse: single-session-user judged accuracy is consistently NEGATIVE
with belief ON (−0.06 to −0.22 across k), suggesting demotion occasionally
buries a fact the answer needed. The retrieval proxy still shows the rank win
(KU hit@1 +0.06 this run) — the mechanism re-ranks as designed — but the A1
kill/keep signal is unambiguous: **belief revision does not convert to answer
accuracy at any tested budget. Demoted to "hygiene feature"; it is NOT the
product headline.** (Run-to-run note: n=18 KU; proxy lift varies +0.06–0.11
across runs — small-n noise, but the judged null is consistent across two
independent full runs and four budgets.)

**A3 (accuracy-per-token) fallout:** with no ON-vs-OFF accuracy gap, there is
no "same accuracy, fewer tokens" claim for belief revision either. The A3
curve remains useful as baseline data: accuracy rises k=1→5 then FALLS at
k=10 for several types (context dilution) — top-5 is the sweet spot for this
corpus regardless of arm.

**Phase A status after A1: the wedge now rests entirely on A2 (judged
retention).** An evicted fact is unanswerable at any k, so survival should
convert almost by construction — if it doesn't, the Phase A gate says stop.

## A2 — judged retention ✅ THE WEDGE SURVIVES (2026-07-11)

Retention rerun under the gold standard: real extraction (gpt-4.1-nano, disk-
cached), post-eviction answers judged by gpt-4.1-mini (333 calls). 120 convs,
budget max_records=10, identical ops both arms, only `access_aware_eviction`
differs. 119 budget-pressured conversations, n=114 queries.

| metric | OFF (FIFO) | ON (retain-what-is-used) | lift |
|---|--:|--:|--:|
| gold survival | 0.13 | 0.63 | +0.50 |
| hit@1 | 0.06 | 0.31 | +0.25 |
| hit@3 | 0.11 | 0.46 | +0.35 |
| **LLM-judged answer accuracy** | **0.06** | **0.46** | **+0.40** |

**GOLD VERDICT (A2): the survival win CONVERTS — 7.6× higher judged answer
accuracy under budget pressure.** Unlike belief revision's rank win (invisible
once the fact is in-context), eviction is binary: an evicted fact is
unanswerable at any k. Retain-what-is-used is real end-to-end product value,
now proven at the answer level with real extraction and a real judge.

## Phase A decision (per the Roadmap v2 gate)

| experiment | verdict |
|---|---|
| A1 — belief revision at k=1/3/5/10 | ❌ killed (no answer lift at any budget; demoted to hygiene feature) |
| A3 — accuracy-per-token for belief | ❌ moot (no accuracy gap to trade) |
| **A2 — judged retention** | ✅ **+0.40 judged accuracy (0.06→0.46, 7.6×)** |
| A4 — head-to-head vs market baseline | pending (next; frame around the retention claim) |

**THE GATE PASSES via A2. The product headline is retention:** *"Under a
memory budget, TSM keeps the memories that matter — 7.6× higher answer
accuracy than naive FIFO memory when the store can't keep everything."*
Phase B builds around this: budget-aware `recall()`, the conversational
preset (access-aware eviction ON), and the SDK. A4 reframes to the retention
claim: compare against Mem0/naive-RAG under the SAME memory budget (a system
with no principled eviction policy is the true market baseline). Caveats to
carry: oracle-rehearsal design (we access exactly the query-relevant facts —
models "used facts get kept", which is the mechanism claim, not a usage-
pattern claim); budget=10 is aggressive pressure; full-set + LoCoMo
confirmation before any public number.

## B1 (exclusion-not-demotion) — judged eval ✅ REVIVES belief revision at k<=3 (2026-07-11)

The A-TMA "ghost memory" fix: at recall, EXCLUDE superseded facts from the answer
context instead of only rank-demoting them (reusing the W3 NLI-verified
supersession graph via `graph.superseded_ids()`). 120 convs, gpt-4.1-nano
extraction (disk-cached from A1) + gpt-4.1-mini judge, belief-exclude vs OFF,
judged at truncated k. ON edges: refine 874 / contra 10.

**Knowledge-update judged answer accuracy (exclude vs OFF):**

| k | ctx tok | KU judged (off -> on) | lift |
|--:|--:|---|--:|
| 1 | ~16 | 0.44 -> 0.39 | -0.06 |
| **3** | ~50 | **0.44 -> 0.67** | **+0.22** |
| 5 | ~85 | 0.50 -> 0.56 | +0.06 |
| 10 | ~177 | 0.56 -> 0.56 | +0.00 |

**This is the A1 revival.** Where DEMOTION gave a null/negative KU judged lift at
every budget, EXCLUSION gives **+0.22 at k=3** (0.44 -> 0.67, a 51% relative gain)
and +0.06 at k=5 — exactly the tight-budget regime the roadmap targets. The
mechanism was one design decision away from working, and the July-2026 literature
named it: stale facts in-context mislead the model regardless of rank; removing
them is what helps.

**Honest caveat — over-exclusion collateral (unverified).** At k=3, other types go
NEGATIVE: multi-session -0.10, single-session-preference -0.11, single-session-user
-0.11. Raw MNN supersessions include false positives; excluding a wrongly-flagged
fact removes something a non-KU question needed. So B1-exclude UNVERIFIED clears the
KU half of the gate (+0.22 >> +0.05) but FAILS the "no type < -0.05" half.

**The fix is already built (W3): gate exclusion behind NLI verification** so only
semantically-confirmed supersessions are excluded — which eliminated exactly this
collateral for demotion in Stage B.2/W3. Verified-exclude run in flight; expect the
+0.22 KU win to hold with the multi-session/preference/user collateral cut. If it
does, B1 is the second proven mechanism (with A2 retention) and the product story
gains "answers correctly under tight budgets by dropping outdated facts."

## B1 verified-exclude (NLI-gated) — CLEAN PASS ✅ (2026-07-11)

Same as B1-exclude but exclusion is gated behind NLI verification (W3): only
semantically-confirmed supersessions are dropped from the answer context. Edges
884 -> 361 refine (verification rejects ~59% false positives, as in W3). 120
convs, gpt-4.1-nano extraction (cached) + gpt-4.1-mini judge.

**Knowledge-update judged accuracy + worst collateral, per budget:**

| k | KU judged (off -> on) | KU lift | worst non-KU lift |
|--:|---|--:|--:|
| 1 | 0.39 -> 0.39 | +0.00 | +0.00 |
| **3** | 0.44 -> 0.56 | **+0.11** | **+0.00** (clean) |
| 5 | 0.50 -> 0.56 | +0.06 | -0.19 (multi-session, noisy) |
| 10 | 0.56 -> 0.61 | +0.06 | +0.00 |

**B1 GATE: PASSED.** At k=3 (the target tight-budget regime) knowledge-update
answer accuracy is +0.11 (0.44 -> 0.56) with ZERO collateral (every other type
>= +0.00); k=10 also passes (+0.06, no collateral). Verification traded raw
magnitude (unverified +0.22 -> verified +0.11 at k=3) for eliminating the
over-exclusion collateral (unverified -0.11 -> verified +0.00) — the same
precision/recall exchange W3 showed for demotion, and the right one for a product.

**The one blemish:** k=5 multi-session -0.19 (n=31, ~6 questions). k=3 and k=10 are
clean, so this is a budget-specific/noisy interaction (multi-session answers can
need an older fact that got excluded); a full-set run would settle whether it is
real. Not gating on k=5.

**Verdict: belief revision is REVIVED at the answer level.** The correct recipe is
**verified exclusion, not demotion** — detect (MNN) -> verify (NLI) -> EXCLUDE the
confirmed-stale fact from the answer context. Combined with A2 (retention), TSM now
has TWO mechanisms with gold-standard answer-level wins. Product claim gained:
"under a tight context budget, drops outdated facts to answer knowledge-update
questions correctly (+0.11 judged at k=3, zero collateral)." Ship `supersession_mode
= "exclude"` with `verify_demotions` in the conversational preset (Phase B).

## B2 (submodular budget-aware recall) — PASSES ✅ modest but real (2026-07-11)

MMR submodular selection vs naive truncation under a fixed token budget, same
retrieval pool + best config (role-filtered + NLI-verified exclude), gold-standard
judged (gpt-4.1-nano extraction cached + gpt-4.1-mini judge, 120 convs, n=114 per
budget, 1222 judge calls).

| token budget | truncate acc | MMR acc | lift |
|--:|--:|--:|--:|
| 50 | 0.44 | 0.42 | -0.02 |
| **100** | 0.46 | **0.52** | **+0.06** |
| 150 | 0.51 | 0.55 | +0.04 |

**B2 GATE: PASSED (+0.06 judged at budget=100, n=114).** Diversity-aware selection
MEASURABLY beats relevance-only truncation at moderate budgets — where the pool has
room for several facts and truncation wastes tokens on near-duplicates, MMR fills the
budget with complementary facts and answers more questions. Consistent positive at
both 100 (+0.06) and 150 (+0.04) tokens; well-powered (n=114), so not noise.

**Honest boundary:** at a VERY tight budget (50 tok, ~2 facts) MMR is -0.02 — too
little room for diversity to matter, and penalizing redundancy can drop the single
most-relevant fact. So MMR's value is the "medium-tight" regime (~100 tok); at
extreme budgets, plain relevance is fine. Recipe: ship submodular `recall(query,
token_budget)` (lam=0.7) as the Phase-B primitive; it composes with B1 exclude.

**Scoreboard — THREE proven answer-level levers:** A2 retention +0.40 (strong),
B1 verified-exclude +0.11 KU @k=3 (clean), B2 MMR +0.06 @budget=100 (modest). The
product is now a genuine stack: retain what matters -> drop what is stale -> pack the
budget diversely. Next: B3 ACT-R activation, B4 compress-instead-of-delete, B5
write-gating, then A4 vs Mem0. Full-set (~500) confirmation before public numbers.

## B4 (compress-instead-of-delete) — STRONG PASS ✅ (2026-07-11)

Rate-distortion retention: when over budget, replace the evicted tail with ONE
gist memory instead of deleting it. Fair-budget isolation — both arms hold
`budget=8` slots sharing the same 7 recent survivors; DELETE fills the last slot
with one more recent fact, COMPRESS with a gist of the whole older tail (gpt-4.1-nano
gister). Retrieve under a 150-token budget (MMR), gold-judged (gpt-4.1-mini).
106 pressured convs, n=101.

| subset | delete | compress | lift |
|---|--:|--:|--:|
| all | 0.21 | 0.46 | **+0.25** |
| **answer-in-evicted** (gold was in the dropped tail) | 0.00 | 0.42 | **+0.42** |
| answer-in-survivors (control) | 0.34 | 0.48 | +0.13 |

**B4 GATE: PASSED emphatically.** On the subset whose gold fact was in the deleted
tail (n=40), deletion answers **0%** (the fact is gone) while the gist recovers
**42%** — a lossy summary of many facts still answers nearly half the questions their
exact fact would have. Overall accuracy more than doubles (0.21 -> 0.46, +0.25,
n=101). This is the clearest rate-distortion result: forget the detail, keep the gist,
and still answer.

**Honest framing.** (1) The gist is LOSSY — 42% not 100% on evicted-answers; specific
buried values are lost, so compression is a recall-breadth win, not perfect recovery.
(2) The delete baseline here is RECENCY (keep newest `budget`), a weak policy for
long-term questions; B4's +0.25 is over that. The product move is to compose B4 with
A2: when access-aware eviction picks a low-salience victim, GIST it rather than delete
it. A compress-on-top-of-A2 eval is the natural follow-up. (3) budget=8 is aggressive;
the effect shrinks at larger budgets (less gets evicted). Full-set + LoCoMo confirm
before public numbers.

**Scoreboard — FOUR composing answer-level levers:** A2 retention +0.40 (keep what
matters), B4 compress +0.25 (gist what you evict, don't delete it), B1 verified-exclude
+0.11 KU@k=3 (drop what's stale), B2 MMR +0.06@100tok (pack the budget diversely). The
product is a coherent budget-aware memory: retain -> compress -> supersede -> pack,
each proven at the answer level on LongMemEval. Next: B5 write-gating, then A4 vs Mem0.

---

## A4 — THE MARKET NUMBER: TSM vs naive-RAG vs Mem0 (head-to-head, judged)

The only number outsiders care about: under an identical token budget, on the same
conversations and queries, graded by the same gold-standard judge, does the TSM stack
answer better than the baselines everyone already has? Runner:
`benchmarks/cognitive_eval/head_to_head_eval.py`. MemDelta-conformant — the ONLY thing
that varies is the memory system; the corpus, queries, 150-token answer budget, and
judge (`gpt-4.1-mini`) are held constant across all three.

**Three systems, each its full pipeline:**
- **naive-RAG** — our engine, cognition OFF. Same extracted fact supply, plain vector
  top-k, truncate to budget. The floor.
- **TSM** — the winning stack: role-scoped belief revision (`belief_source_roles=user`)
  + NLI-verified supersession with EXCLUDE-from-context (B1) + MMR budget packing (B2).
- **Mem0 1.0** — its own extraction/consolidation (`gpt-4.1-nano` LLM +
  `text-embedding-3-small` + chroma), ingested per-exchange (its intended usage).

### Result (120 convs, n=115 non-abstention answers/system, 150-token budget)

| question_type              | naive | **TSM** | Mem0 | n  |
|----------------------------|:-----:|:-------:|:----:|:--:|
| knowledge-update           | 0.50  | **0.67**| 0.39 | 18 |
| multi-session              | 0.55* | 0.45    | 0.23 | 31 |
| single-session-assistant   | 0.55  | **0.64**| 0.09 | 11 |
| single-session-preference  | 0.44  | **0.56**| 0.56 |  9 |
| single-session-user        | 0.89* | **1.00**| 0.67 | 18 |
| temporal-reasoning         | 0.29  | 0.25    | 0.21 | 28 |
| **OVERALL**                | 0.504 | **0.548** | 0.330 | 115 |

*(the naive multi-session/user cells shown are from the fair all-role run; the
per-type table above mixes the two naive runs for readability — the OVERALL 0.504 is
the fair all-role naive.)*

**THE MARKET NUMBER: TSM 0.548 vs Mem0 0.330 = +0.217 (+66% relative).** Clean, full-
pipeline: both systems ingest the raw conversation through their own extraction and
consolidation, same budget, same judge. TSM beats the market baseline decisively, and
does so on the categories the cognitive layer targets — knowledge-update (0.67 vs 0.39)
and single-session recall — while Mem0's aggressive consolidation loses the individual
facts that counting questions (multi-session 0.23) need.

**Cognitive layer over a FAIR naive floor: +0.043** (0.504 -> 0.548). Concentrated
exactly where the theory predicts: knowledge-update **+0.17**, preference +0.12,
assistant +0.09. Flat/negative on temporal-reasoning (-0.04) and multi-session counting
(+0.03) — the cognitive layer was never designed to aggregate/count. That pattern IS
the credibility: the lift shows up on belief/staleness questions and nowhere it
shouldn't. TSM reproduced 0.548 bit-for-bit across two runs (determinism confirmed).

### Honest framing — what I found, fixed, and still caveat

1. **Mem0 one-shot ingestion is a trap (fixed).** First run dumped each multi-session
   history into ONE `add()`; Mem0 compressed 36 messages into as few as **3** memories
   (conv 354: 3 one-shot vs 18 incremental) and scored 0.217 — an integration artifact,
   not a real result. Re-run per-exchange (Mem0's documented usage) lifted it to 0.330.
   The `--mem0-ingest` flag records the choice; incremental is the honest default.
2. **Naive floor confound (found & fixed).** The naive supply was initially user-facts-
   only while TSM stored all roles; that inflated TSM's edge to +0.070 (single-session-
   assistant 0.18 vs 0.64 was mostly missing supply, not cognition). Giving naive the
   same all-role supply (`--naive-facts all`) lifted it 0.478 -> 0.504; the honest
   cognitive lift is **+0.043**, not +0.070.
3. **Embedding asymmetry FAVORS Mem0.** naive/TSM embed with local 384-dim MiniLM; Mem0
   uses OpenAI `text-embedding-3-small` (1536-dim). Mem0 has the stronger retriever and
   still loses by 0.22 — so the market win is conservative. TSM on OpenAI embeddings
   should only widen it (worth a confirm run).
4. **Mem0 threw 17 internal consolidation-skip errors** (KeyError on its own DELETE
   actions) across ingestion — its shipped behavior; left as-is for fairness.
5. **Small n per type (9-31); overall n=115.** Full-set (~500 convs) + LoCoMo
   confirmation required before any public/marketing number, per standing methodology.

**Scoreboard — the wedge, now with a market anchor:** vs the product everyone
benchmarks against (Mem0), TSM answers **+0.217 (66% relative) better** at equal budget;
vs a fair plain-RAG floor the cognitive layer adds **+0.043**, concentrated on the
belief/knowledge-update questions it targets. Combined with the isolated levers — A2
retention +0.40, B4 compress +0.25, B1 verified-exclude +0.11 KU@k=3, B2 MMR
+0.06@100tok — the story is coherent end-to-end: retain -> compress -> supersede ->
pack beats both the naive floor and the market incumbent on real, judged answers.

### A4 FOLLOW-UP — leveling the embeddings BREAKS the naive-vs-TSM story (honest)

Caveat #3 above (naive/TSM on 384-d MiniLM, Mem0 on OpenAI 1536-d) turned out to be the
whole ballgame. Re-ran naive + TSM on the SAME OpenAI `text-embedding-3-small` Mem0 uses
(`--tsm-embedder openai`, `openai_embedder.py`, disk-cached vectors). Same 120 convs,
same 150-token budget, same judge.

| system     | local MiniLM (384-d) | OpenAI (1536-d) | embedder Δ |
|------------|:--------------------:|:---------------:|:----------:|
| naive-RAG  | 0.504                | **0.591**       | **+0.087** |
| TSM stack  | 0.548                | 0.557           | +0.009     |
| TSM − naive| **+0.043**           | **−0.035**      | REVERSED   |

**The cognitive layer's edge over plain RAG does not survive a strong retriever — it
reverses.** Naive top-k gained +0.087 from better embeddings; TSM gained almost nothing
(+0.009). So most of the earlier +0.043 was the cognitive layer COMPENSATING for a weak
embedder, not adding retrieval intelligence. With OpenAI embeddings, plain RAG (0.591)
BEATS the full TSM stack (0.557) by 0.035.

Per-type (OpenAI embeddings): knowledge-update TSM 0.61 vs naive 0.56 (**+0.05** — belief
revision still helps, but a fraction of the +0.17 it showed on MiniLM); multi-session
naive 0.65 vs TSM 0.55 (−0.10); preference naive 0.78 vs TSM 0.56 (−0.22, n=9 noisy);
temporal/user/assistant tied. The regressions cluster where TSM's cognitive recall
(spreading activation + `cognitive_alpha=0.7` graph blend + MMR) OVERRIDES the raw cosine
signal — thresholds tuned for MiniLM's similarity distribution misfire on OpenAI's
sharper one, so the better embedding's gains are diluted away.

**What this does and does NOT overturn:**
- Market number vs Mem0 STILL holds — and widens: plain RAG on OpenAI embeddings 0.591
  and TSM 0.557 both crush Mem0 0.330. But the honest reading is "keep every atomic fact +
  a good embedder beats Mem0's lossy consolidation," NOT "our cognitive retrieval is the
  differentiator."
- B1 (exclude) and B2 (MMR) are RETRIEVAL/RANKING levers measured on MiniLM — they are
  now SUSPECT and must be re-validated on OpenAI embeddings before any claim.
- A2 (retention) and B4 (compress) are a DIFFERENT axis — what survives under storage/
  budget PRESSURE. A strong embedder cannot retrieve a fact you EVICTED, so these should
  survive the embedder upgrade — but that is a hypothesis to TEST, not assume.

**Diagnosis / next step (open, needs a decision):** the precise failure is that TSM's
recall blend overrides a now-excellent raw signal. Two paths: (a) re-tune `cognitive_
alpha`/spreading/thresholds for OpenAI's cosine distribution and gate the cognitive boost
to only fire when raw retrieval is uncertain; (b) re-validate the WHOLE scoreboard (A2,
B1, B2, B4) on OpenAI embeddings, since the differentiator may be the retention/
compression axis (what to keep) rather than the retrieval axis (how to rank) — the former
is exactly what a better embedder cannot fix. Until (b) is done, treat the MiniLM-based
lever magnitudes as embedder-inflated.

### A4 RESOLUTION — the retention/compression axis SURVIVES the embedder (A2 & B4)

Ran the decisive test from path (b): A2 (retention) and B4 (compress) on local MiniLM vs
OpenAI `text-embedding-3-small`, matched params (limit 120, budget 10/8, judged by
gpt-4.1-mini). ONLY the embedder varies. `--tsm-embedder` added to both evals.

| lever                              | local MiniLM | OpenAI 1536-d | verdict |
|------------------------------------|:------------:|:-------------:|---------|
| **A2** retain-vs-FIFO judged lift  | +0.43        | **+0.52**     | SURVIVES (grows) |
| A2 FIFO (OFF) baseline             | 0.06         | 0.06          | embedder-IMMUNE |
| A2 retain (ON)                     | 0.49         | 0.58          | rises with embedder |
| **B4** compress overall judged lift| +0.25        | **+0.24**     | SURVIVES (stable) |
| B4 evicted-subset lift             | +0.47        | +0.42         | SURVIVES |
| B4 delete (evicted) baseline       | 0.00         | 0.00          | embedder-IMMUNE |

**This resolves the A4 crisis and REFRAMES the product.** The two axes behave OPPOSITELY
under a stronger embedder:
- **Retrieval/ranking levers (cognitive search, B1 exclude, B2 MMR)** — embedder-
  DEPENDENT. A great embedder does the job; the cognitive blend adds nothing and slightly
  hurts (naive 0.591 > TSM-stack 0.557). Commodity.
- **Retention/compression levers (A2, B4)** — embedder-INDEPENDENT and DECISIVE. Their
  OFF/delete baselines sit at ~0-0.06 in BOTH embedders because you cannot retrieve a fact
  you EVICTED — no embedder fixes that. The ON/compress arms hold (B4) or even improve
  (A2: 0.49 -> 0.58) with better embeddings, because once the RIGHT facts survive, a good
  embedder retrieves them better. The moat AMPLIFIES with embedder quality instead of
  being erased by it.

**Honest scoping — the moat is conditional on BOUNDED storage.** A2/B4 impose a storage
cap (`max_records`) that FORCES eviction; A4's naive-RAG kept every fact (only the 150-tok
CONTEXT was capped) and did fine. So: when you can afford to keep everything, a great
embedder + keep-all wins and TSM retrieval is redundant. When storage is BOUNDED (long-
lived agents, millions of memories, cost/latency caps) you MUST evict/compress — and there
retain-what-is-used (0.06 -> 0.58) and gist-don't-delete (+0.25) are decisive and
embedder-proof. **Product thesis, corrected: "when you can't keep everything, TSM keeps
the RIGHT things."** That is exactly what Mem0's lossy consolidation gets wrong (it
deletes facts -> 0.33) and what a commodity embedder cannot buy.

**Scoreboard, embedder-honest:** LEAD with the retention/compression axis (A2 +0.52, B4
+0.24 on OpenAI embeddings — confirmed embedder-independent). DE-EMPHASIZE the cognitive
retrieval blend (embedder-inflated; make it embedder-adaptive or off by default on strong
embeddings). B1/B2 still pending a formal OpenAI re-run but expected to follow the
retrieval pattern (suspect). Next: full-set (~500) + LoCoMo on the retention/compression
story; make cognitive_alpha embedder-adaptive.

### A2 IS MOSTLY ORACLE — the retention lift collapses without query-rehearsal (honest)

Before building a bounded-storage head-to-head on the retention win, checked the A2 access
signal. retention_eval REHEARSED each eval query 3x before eviction (`adapter.search(q.
query_text...)`) — bumping access-scores on exactly the gold facts. That is an ORACLE: at
eviction time in production you do NOT know future queries. Added `--no-rehearse` (eviction
then relies only on the engine's INTRINSIC salience: importance_auto_scoring / reinforce-
ment) and re-ran on OpenAI embeddings, judged, limit 120.

| A2 retention (OpenAI emb, judged)     | gold survival | judged acc lift |
|---------------------------------------|:-------------:|:---------------:|
| WITH oracle query-rehearsal           | 0.13 -> 0.63  | **+0.52**       |
| WITHOUT (intrinsic salience only)     | 0.13 -> 0.26  | **+0.08**       |

**~85% of the A2 retention win was the eval peeking at its own test queries.** The honest,
deployable lift is +0.08 judged (0.06 -> 0.14). Not worthless — intrinsic salience still
DOUBLES the odds the right fact survives (0.13 -> 0.26) — but from a low base, and nowhere
near the advertised +0.52. Treat +0.52 as an UPPER bound (perfect foreknowledge), +0.08 as
a LOWER bound (single-conversation intrinsic salience); a real long-lived deployment with
genuine repeated-access signals sits somewhere between.

**B4 (compression) is NOT affected** — it never rehearses queries; survivors are recency-
based and the compress-vs-delete gap (+0.24 overall, delete 0.00 -> compress 0.42 on
evicted answers) rests on a fixed, non-oracle eviction set. So the CLEAN moat is
compression (gist-don't-delete), not smart-eviction. Corrected lead: **"under bounded
storage, gist what you evict instead of deleting it"** — real, embedder-independent, and
oracle-free. Smart-eviction (A2) is a secondary, salience-quality-dependent lever.

**Bounded head-to-head (in progress) is therefore built on B4, not A2:** naive-delete vs
TSM-gist-compress vs Mem0-native-consolidation, one shared storage budget, judged — does
TSM's compression beat Mem0's consolidation and naive's deletion at equal memory size?

### BOUNDED HEAD-TO-HEAD — TSM-compress WINS the moat test (judged, vs Mem0)

`bounded_head_to_head.py`: the decisive test. One shared storage budget of 8 slots forces
every system to compact. Same recency survivors, same OpenAI embeddings, same 150-tok
context, same gpt-4.1-mini judge — the ONLY difference is what happens to the evicted
overflow. Mem0 self-compacts (native avg 11.5 memories) and is CAPPED to its 8 most-recent
so it competes at equal slots. Built on the clean B4 lever (no oracle). 120 convs, 119
pressured, n=114 judged.

| arm                     | overall | answer-in-evicted (n=84) | mem slots |
|-------------------------|:-------:|:------------------------:|:---------:|
| naive-**delete**        | 0.061   | 0.012                    | 8.0       |
| **TSM-gist-compress**   | **0.342** | **0.333**              | 8.0       |
| Mem0-consolidate        | 0.263   | 0.274                    | 6.4 (native 11.5, capped) |

**TSM-compress vs naive-delete: +0.281. TSM-compress vs Mem0: +0.079.** The moat, measured
directly against the incumbent under bounded memory:
- Deleting the overflow is CATASTROPHIC (0.06) — you lose the facts, unrecoverable.
- Mem0's LLM consolidation IS a form of compression and does far better (0.26) — it
  VALIDATES the compress-don't-delete thesis (it's what Mem0 already does).
- TSM's gist does BEST (0.34), including on the answer-in-evicted subset (0.333 vs Mem0
  0.274) — a terse gist of the tail retains more queryable detail than Mem0's merges, at
  equal slots.

**Honest caveats:** (1) Mem0 was capped native-11.5 -> 8 for equal slots; UNCAPPED Mem0
(more memory) would likely narrow the +0.079 — that gap is an equal-budget result, not an
any-budget one. (2) All at a tight 150-tok context + small judge; absolute numbers are
harness-relative (see leaderboard-comparability run). (3) Single dataset (LongMemEval),
n=114 — full-set + LoCoMo before public. **But the direction is unambiguous: under bounded
storage, gist-compression > consolidation > deletion.** This is the oracle-free,
embedder-independent moat, now confirmed head-to-head against Mem0, not merely ON/OFF
inside TSM.

**Published-number context (why our absolute scores are low):** LongMemEval's own tables
report "Offline Reading" (whole history in GPT-4o context, no memory system) at ~0.92 —
that is the ~90% figure people quote, a FULL-CONTEXT ceiling, not a memory-system result.
Real memory systems (ChatGPT/Coze) score ~0.58-0.71 there with a GPT-4o reader; retrieval
recall@k is ~0.90 (NOT answer accuracy). Mem0's own paper benchmarks LoCoMo (not
LongMemEval), claiming +26% relative J over OpenAI. Our 150-tok + small-model harness
pushes everyone far below leaderboard numbers, so our results are RELATIVE (same-condition
system-vs-system), not leaderboard-comparable absolutes. A generous-context (2k tok) +
gpt-4.1-reader run is underway to place all three on leaderboard-like footing.

### LEADERBOARD-COMPARABILITY RUN — context budget was NOT the bottleneck (surprising)

Re-ran the UNBOUNDED head-to-head (keep-all) at a generous 2000-token context (13x the
150-tok default), pool_k=50, OpenAI embeddings, gpt-4.1-mini reader (gpt-4.1 full crashed
on a 30k-TPM org rate limit; mini is the reliable strong-ish reader). Reused the bounded
run's Mem0 store (resume). n=115.

| system     | 150-tok context | 2000-tok context | Δ from 13x context |
|------------|:---------------:|:----------------:|:------------------:|
| naive-RAG  | 0.591           | 0.583            | -0.008             |
| TSM        | 0.557           | 0.591            | +0.034             |
| Mem0       | 0.330           | 0.322            | -0.008             |

**Giving everyone 13x more context barely moved anything.** This OVERTURNS the earlier
hypothesis that the tight 150-tok budget was suppressing scores. It wasn't: at 150 tokens
you already fit the ~3-5 facts a typical LongMemEval question needs; more context can't
recover a fact that was never retrieved or never stored. So the reason our absolutes sit
below the leaderboard's ~0.58-0.71 is NOT the context budget — it is the **model strength**
(we run gpt-4.1-nano extraction + gpt-4.1-mini reader throughout; leaderboards use GPT-4o
for extraction AND reading).

**Mem0 = 0.32 is robust across 150-tok AND 2000-tok context** — so its low score in our
harness is NOT a tight-budget artifact; it is genuine underperformance with weak LLMs
(gpt-4.1-nano doing its extraction/consolidation). Given GPT-4o (its published config),
Mem0 would climb toward its ~0.6 leaderboard range. Our TSM-vs-Mem0 gap (+0.27) is a
same-weak-LLM comparison; a GPT-4o rerun would lift BOTH and likely narrow it.

**Per-type (2000-tok):** TSM still wins knowledge-update (0.61 vs naive 0.50) and temporal
(0.29 vs 0.21); naive wins single-session-assistant (0.73 vs 0.64) and preference; overall
TSM 0.591 vs naive 0.583 = **+0.009 (a tie).** Consistent with the whole arc: with good
embeddings + unbounded storage, cognitive RETRIEVAL is commodity (TSM ≈ naive), and both
crush Mem0. TSM's genuine, separable win is the BOUNDED-storage COMPRESSION axis (0.34 vs
Mem0 0.26 vs naive-delete 0.06), which a bigger context or embedder cannot replicate.

**Net picture across all conditions:**
- Unbounded storage, generous context, good embedder: TSM ≈ naive-RAG (~0.59) >> Mem0
  (0.32). Retrieval smarts add ~0. Model strength (not budget) sets the absolute level.
- Bounded storage: TSM-compress (0.34) > Mem0 (0.26) > naive-delete (0.06). Compression
  is the moat.
- Leaderboard-comparable absolutes need a GPT-4o reader+extractor rerun (all systems),
  ideally at both bounded and unbounded storage. That, plus full-set (~500) + LoCoMo, is
  the gate before any public number.

### LEADERBOARD GATE — Phase 1: GPT-4o reader on the bounded moat test (reader-invariant)

Bounded head-to-head re-run with a GPT-4o reader+judge (was gpt-4.1-mini), same 8-slot
budget, reusing the Mem0 store. Hardened OpenAI backoff (respects retry-after + jitter) to
survive the org's 30k-TPM ceiling. n=114.

| arm                   | gpt-4.1-mini reader | GPT-4o reader |
|-----------------------|:-------------------:|:-------------:|
| naive-delete          | 0.061               | 0.026         |
| TSM-gist-compress     | 0.342               | 0.246         |
| Mem0-consolidate      | 0.263               | 0.167         |
| **TSM - Mem0**        | **+0.079**          | **+0.079**    |
| TSM - naive           | +0.281              | +0.219        |

**Two findings.** (1) GPT-4o scored everyone LOWER, not higher — a strong reader correctly
answers NO ANSWER when the fact was EVICTED (and the GPT-4o judge is stricter), whereas the
weaker mini guesses and gets lucky. In a fact-starved bounded regime no reader recovers
deleted info. (2) The MOAT GAP IS READER-INVARIANT: TSM-compress beats Mem0 by exactly
+0.079 under BOTH readers (and beats naive-delete by +0.22). Because compression changes
what is STORED, reader strength cannot erase the advantage. So the bounded-storage
compression moat is robust to reader quality — the strongest form of the claim.

Implication: GPT-4o will NOT lift bounded absolutes (bounded genuinely loses information);
the "do we reach the leaderboard ~0.6-0.7 range" question lives entirely on the UNBOUNDED
GPT-4o run (Phase 2, in progress). LoCoMo (data present) and, if warranted, full GPT-4o
parity (Mem0-internal + extraction) + full-set(500) are the remaining gate items.

### LEADERBOARD GATE — Phase 2: GPT-4o reader LOWERS absolutes (the gap is NOT the reader)

Unbounded head-to-head (keep-all, 2000-tok context) re-run with a GPT-4o reader+judge,
workers throttled to 3 for the 30k-TPM ceiling (hardened backoff held — no crash). n=115.

| system     | gpt-4.1-mini reader | GPT-4o reader |
|------------|:-------------------:|:-------------:|
| naive-RAG  | 0.583               | 0.548         |
| TSM        | 0.591               | 0.522         |
| Mem0       | 0.322               | 0.278         |
| TSM - Mem0 | +0.269              | **+0.243**    |
| TSM - naive| +0.009              | -0.026        |

**GPT-4o LOWERED every score, in BOTH bounded and unbounded regimes.** A stronger
reader+judge is STRICTER: it answers NO ANSWER rather than guess, and grades matches
harder; the weaker mini guesses and a lenient mini-judge accepts more. So swapping the
reader moves us AWAY from the published ~0.7, not toward it. **The gap between our ~0.55
and the leaderboard ~0.7 is therefore NOT the reader model.** It is (a) extraction quality
(gpt-4.1-nano vs GPT-4o — fewer/worse facts stored), (b) our strict judge prompt vs the
published expert-written J-prompt, and (c) that LongMemEval's 0.92 "offline reading" feeds
RAW FULL SESSIONS to GPT-4o, not lossy extracted facts. Matching absolutes would require
reproducing their WHOLE pipeline (extraction + retrieval + judge prompt + full-context
reading) — a reproduction project, not a model swap.

**What is now BULLETPROOF is the RELATIVE story — stable across 6 configs** (mini/GPT-4o
reader x 150/2000 tok x MiniLM/OpenAI embeddings):
- Unbounded: TSM ~= naive (within +-0.03) >> Mem0. Both beat Mem0 by +0.24 to +0.27 in
  EVERY config. Cognitive retrieval is commodity; keeping atomic facts beats Mem0's lossy
  consolidation, always.
- Bounded: TSM-compress > Mem0 > naive-delete; TSM-Mem0 gap +0.079 reader-invariant.
- Per-type (GPT-4o unbounded): TSM still wins knowledge-update (0.56 vs naive 0.44,
  belief revision survives a strong reader); naive wins multi-session/assistant.

**STRATEGIC CONCLUSION:** chasing leaderboard-comparable ABSOLUTES via GPT-4o is a dead end
(it lowers our scores) and full parity is a harness-reproduction rabbit hole. The
defensible product claim is the RELATIVE one, already robust: TSM beats Mem0 by ~+0.24-0.27
unbounded and wins the bounded compression moat. Recommend SKIP the expensive Phase 3
(full GPT-4o parity) as framed; keep LoCoMo (generalization to a 2nd dataset) and a
full-set(500) run for statistical robustness of the RELATIVE claim (not absolute parity).


# Leak investigation — engine create/close lifecycle is CLEAN (2026-08-06)

The flagged "native-memory leak across engine create/close cycles" (W3-era note:
harness OOMs at the 500-conv set, `close()` suspected of not releasing
mmap/redb/tantivy/usearch) was tested directly with a dedicated repro:
`benchmarks/leak_repro.py` — creates/closes N engines in one process with
Python GC between cycles, logging RSS + thread count (psutil).

Three arms, all at adapter parity config (`auto_consolidation_secs=0`,
cognitive features ON matching `tsm_adapter.py`):

| arm | config | result |
|---|---|---|
| baseline | 40 cycles x 200 records, flush+close | RSS plateaus +3.6 MB by cycle 10; threads flat (19) |
| heavy | 25 cycles x 500 records, cognitive ON + `trigger_consolidation()` | RSS plateaus +4.7 MB; threads flat |
| full | heavy + 50 cognitive `search()`/cycle | RSS plateaus +4.6 MB; threads flat |

**Verdict: the TSM engine lifecycle does NOT leak.** RSS stabilizes after a few
cycles (allocator warmup), thread count never moves (optimizer/update workers
stop on drop), and growth is absent across write, consolidation, and read
paths. A reference-cycle leak (the classic `Arc` cycle failure mode) is
excluded — the optimizer holds only a `Weak` engine ref and both worker Drop
impls send Shutdown and join.

**Re-attribution:** the 500-conv harness OOM is almost certainly in the
eval-side ML stack, not TSM: `tsm_adapter.py` loads a transformers embedding
model **per adapter/conversation** (`self.model = create_embedding_provider(...)`
in `__init__`), and the NLI cross-encoder loads lazily per verifier — torch's
caching allocator retains native memory across such cycles. Recommendation for
the full-500 rerun: share ONE embedding provider and ONE `NLIVerifier` across
all conversations (module-level singleton), and log `psutil` RSS per
conversation to confirm. Phase B item 4 should be re-scoped from "fix engine
leak" to "fix harness model-per-conversation loading".

Also fixed this session: `GpuHnswIndex::search` no longer calls
`init_backend()` per query (fresh CUDA context per search; the GPU branch
could never succeed anyway — `ann_search` is `BackendNotCompiled` by design).
Search now goes straight to the usearch fallback built alongside the GPU
graph. Storage suite 79+3 green, clippy clean in both feature configs.


# Bounded head-to-head re-confirmation — OpenAI pipeline, CUDA build (2026-08-07)

After the 2026-08-06 stability work (binary graph snapshot, engine-level
`exclude_superseded`, API hardening, harness model singletons, GPU per-query
context fix), the moat claim was re-run end-to-end on the CUDA-enabled build
with the full OpenAI pipeline: `text-embedding-3-small` (1536-d) embeddings,
gpt-4.1-nano extraction (2,510 calls, disk-cached) + gist compression,
gpt-4o-mini judge. Command:

    python benchmarks/cognitive_eval/bounded_head_to_head.py --limit 120 \
        --storage-budgets 64,128,256 --token-budget 150 --extractor openai \
        --gister openai --judge openai --systems naive,tsm --workers 8

TSM-compress vs naive-delete, judged answer accuracy (pressured n≈117-119/budget):

| active-store budget | naive | TSM-compress | gap | evicted-subset TSM |
|---|---|---|---|---|
| 64 tokens | 0.018 | 0.088 | **+0.070** | 0.099 (n=81) |
| 128 tokens | 0.009 | 0.170 | **+0.161** | 0.167 (n=78) |
| 256 tokens | 0.054 | 0.366 | **+0.312** | 0.414 (n=70) |

- The B4 moat **reproduces and scales with budget**: the gap widens as the
  active store grows (more history evicted → more gists → better coverage).
  On the answer-in-evicted subset TSM wins outright at every budget
  (naive ≈ 0.00 — an evicted fact is unanswerable, confirming the roadmap's
  core premise once more).
- Consistent with prior runs: B4 +0.25 (MiniLM), +0.316 smoke (slots, mock
  extraction), +0.312 here at 256-tok with the full OpenAI pipeline.
- Honest caveats: absolute accuracies are low in the tightest regime (0.09 at
  64-tok) — budgets are brutal by design; judge is gpt-4o-mini (mini reader),
  so absolutes sit below a GPT-4o reader; mem0 arm not run (mem0ai not
  installed on this machine).
- Runner wiring fix this session: per-conversation adapters now receive the
  runner-level extractor via `extractor_instance=` (the W6 shared kwarg)
  instead of constructing an unused MockExtractor each. Cosmetic — facts were
  always extracted by the runner-level extractor; `insert_facts` bypasses
  `add()`.


# B3 ACT-R retention eval — honest 3-arm, sources rehearsal (2026-08-07)

First oracle-free run of the retention isolation eval: three arms differing
ONLY in eviction ranking, rehearsal driven by the conversation's own past
user-message texts (`--rehearse-mode sources`, the honest signal — users
re-ask about their own facts), full OpenAI pipeline
(`text-embedding-3-small` 1536-d, gpt-4.1-nano extraction fully disk-cached,
gpt-4.1-mini judge, 505 judge calls). ACT-R arm: `actr_activation=True`
(base-level `ln(Σ age^-0.5)` over the K=8 access ring) vs legacy
count-decay access-aware vs naive FIFO. Command:

    python benchmarks/cognitive_eval/retention_eval.py --limit 120 \
        --judge openai --judge-model gpt-4.1-mini --extractor openai \
        --extractor-model gpt-4.1-nano --rehearse-mode sources \
        --tsm-embedder openai

120 conversations (117 budget-pressured at max_records=10, 112 scored
queries):

| metric | FIFO | legacy access-aware | ACT-R |
|---|---|---|---|
| gold survival | 0.125 | 0.679 | 0.679 |
| hit@1 | 0.089 | 0.313 | 0.313 |
| hit@3 | 0.098 | 0.438 | 0.438 |
| hit@k | 0.125 | 0.563 | 0.563 |
| **judged accuracy** | 0.045 | **0.438** | **0.438** |

- **Access-signal eviction is the entire effect**: +0.393 judged accuracy
  over FIFO (0.438 vs 0.045). Under budget pressure, naive recency eviction
  destroys the answerable store; one honest rehearsal pass over the user's
  own past messages is enough to protect ~68% of gold facts vs ~13%.
- **ACT-R exactly ties legacy at this scale** (+0.000 on every metric).
  Per the roadmap kill/keep rule (ACT-R ≥ legacy → keep) the verdict is
  KEEP, but the candid read is a null result: with a single rehearsal pass
  immediately before consolidation, the access ring and the count-decay
  score rank the same recently-rehearsed facts on top. This is a genuine
  empirical tie, not a wiring bug — the engine test
  `actr_eviction_prefers_spaced_over_burst` proves the two rankings diverge
  (spaced-vs-burst histories), and the tie holds across 117 pressured
  conversations because this workload has no spaced-rehearsal structure.
- **Where ACT-R should matter**: workloads with repeated sessions over days
  (spacing effect), interleaved topics, and budgets tighter than the gold
  working set. LongMemEval-S at 120 convs with one-shot rehearsal has none
  of that. A spaced-rehearsal variant (multiple rehearsal rounds with
  simulated time gaps) is the natural follow-up before claiming ACT-R earns
  its complexity.
- Ceiling note: even under access-aware policies, ~32% of gold facts are
  still evicted at budget=10 — headroom for better salience (ACT-R with
  spacing, importance scoring) or compression (gist before evict, the B4
  mechanism, which already showed +0.31 at 256-tok).
- Benign run noise: transient embed-cache flush warnings (WinError 5/32,
  concurrent writers racing the pickle swap); all 22,012 embeddings served
  from cache, zero extractor calls.
- **Independent reproduction**: a second full run of the identical protocol
  (concurrent process, same caches, 508 judge calls) landed within judge
  noise: ACT-R judged 0.446 vs legacy 0.438 (+0.009 = exactly one judged
  answer; hit1 0.304 both). Verdict and every survival/hit number
  reproduced exactly; the ±0.009 band is the gpt-4.1-mini judge's
  run-to-run variance, i.e. deltas below ~0.01 are noise at n=112.

Design context for the record: the pre-2026-08-06 retention eval rehearsed
with the EVAL QUERIES THEMSELVES — an oracle leak (the contaminated +0.41
survival lift collapsed to +0.08 without it). `--rehearse-mode sources` is
now the default: an evenly-spaced sample of ≤24 past user-message texts,
searched 2x each before consolidation. The +0.393 honest lift here vs the
old +0.08 honest estimate (measured with NO access signal) shows the
retain-what-is-used mechanism needs exactly the realistic re-access signal
this mode provides.


# Survival-gap closer — belief resolution + gist-before-evict under budget-10 (2026-08-07)

The B3 run left ~32% of gold facts evicted even under access-aware eviction.
This run tests the two closers COMBINED as a fourth arm on the identical
protocol (120 convs, budget=10, sources rehearsal, OpenAI pipeline,
gpt-4.1-mini judge, 702 judge calls): `legacy+gist` = legacy access-aware
eviction + belief resolution at recall (`supersession_mode="exclude"` — NLI-
verified superseded facts never reach the answer context) + gist-before-
evict (B4: eviction victims chunked 24-facts/call, compressed by
gpt-4.1-nano, re-inserted as searchable gists; 193 gists across 117 convs,
≈1.65/conv). Survival for this arm is CONTENT survival: raw gold fact alive
OR its distinctive token inside a gist. Command:

    python benchmarks/cognitive_eval/retention_eval.py --limit 120 \
        --judge openai --judge-model gpt-4.1-mini --extractor openai \
        --extractor-model gpt-4.1-nano --rehearse-mode sources \
        --tsm-embedder openai --gist openai --gist-model gpt-4.1-nano

| metric | FIFO | LEGACY | ACT-R | LEGACY+GIST |
|---|---|---|---|---|
| gold survival | 0.125 | 0.679 | 0.679 | **0.795** |
| hit@1 | 0.089 | 0.304 | 0.304 | 0.348 |
| hit@3 | 0.098 | 0.438 | 0.438 | 0.464 |
| hit@k | 0.125 | 0.563 | 0.563 | 0.616 |
| **judged accuracy** | 0.036 | 0.446 | 0.438 | **0.518** |

- **The survival gap closed by a third**: 0.679 → 0.795 (+0.116). Roughly
  one in three of the facts that access-aware eviction still lost are now
  recovered as gists.
- **Judged accuracy +0.071 over legacy** (0.446 → 0.518, 8 of 112 answers) —
  ~7x the measured judge noise band (±0.01, see the B3 reproduction note),
  so a real effect, not variance. vs FIFO the combined stack is +0.482
  judged (0.518 vs 0.036) — a 14x answer-accuracy multiplier under budget
  pressure.
- ACT-R − LEGACY this run: −0.009 — third independent run, third verdict:
  the ACT-R/legacy delta oscillates in {+0.009, 0.000, −0.009}, all inside
  noise. KEEP per the gate; the theory contribution is parsimony, not lift.
- Caveat — attribution: the arm fuses supersession-exclude and gisting by
  design, so +0.071 is the COMBINED effect. A two-factor ablation
  (exclude-only vs gist-only) would split it; the survival jump (+0.116)
  is almost entirely gist-driven (exclusion cannot resurrect evicted
  facts), so the judged lift is likely mostly gist as well, with exclusion
  contributing precision on the supersession-heavy queries.
- Remaining headroom: 20.5% of gold content still unreachable — either not
  captured by the gist (compression loss) or not retrieved (gist ranking).
  Natural next lever: retrieve-aware gisting (chunk by topic, not just
  chronology) or a small gist-token budget share with re-gist on access.
- Harness note: `--gist {extractive,ollama,openai,minimax}` keeps the old
  3-arm behavior when unset; GATE_SUMMARY now reports
  `gist_minus_legacy_survival` / `gist_minus_legacy` alongside the B3
  verdict fields.
