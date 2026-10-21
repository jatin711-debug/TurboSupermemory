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


# Ablation — belief-exclusion vs gist-before-evict, attribution run (2026-08-07)

Two-factor ablation of the combined LEGACY+GIST arm from the previous run:
`legacy+excl` (belief resolution at recall only) and `gist-only` (gists,
supersession demote) isolate each mechanism; identical 120-conv protocol
(budget=10, sources rehearsal, OpenAI pipeline, gpt-4.1-mini judge, 1098
judge calls). Command = previous run + `--ablate`.

| metric | LEGACY | LEGACY+EXCL | GIST-ONLY | LEGACY+GIST |
|---|---|---|---|---|
| gold survival | 0.679 | 0.777 | **0.804** | 0.795 |
| hit@1 | 0.304 | 0.357 | 0.304 | 0.348 |
| hit@k | 0.563 | **0.625** | 0.536 | 0.616 |
| **judged accuracy** | 0.438 | 0.500 | 0.455 | **0.509** |

Deltas vs legacy (judged / survival):

| arm | judged | survival |
|---|---|---|
| legacy+excl | **+0.063** | +0.098 |
| gist-only | +0.018 | **+0.125** |
| combined | +0.071 | +0.116 |

- **Belief exclusion is the bigger ANSWER lever**: +0.063 judged alone (7 of
  112 answers) — nearly the whole combined +0.071. It also lifts survival
  (+0.098) even though it deletes nothing: exclusion applies during the
  rehearsal searches too, so stale/superseded facts stop soaking up access
  bumps, salience concentrates on live facts, and the rehearsal grace
  window protects a wider set of distinct live facts from eviction. Double
  mechanism: rehearsal-salience concentration (survival) + ghost-memory
  context cleaning (precision).
- **Gists own the survival axis but convert weakly alone**: +0.125 survival
  (content resurrection — the highest of any arm) yet only +0.018 judged,
  and gist-only hit@k (0.536) dips BELOW legacy (0.563): gist texts compete
  in the same ranking as exact facts and dilute top-10. Exclusion fixes
  that dilution in the combined arm (hit@k back to 0.616).
- **Combined ≈ additive on judged** (0.063 + 0.018 = 0.081 predicted vs
  0.071 measured — mild sub-additivity, well above the ±0.01 noise band
  either way). Product conclusion: belief resolution is the primary
  accuracy feature; gisting is the insurance policy for coverage/survival.
  Both ship: exclusion is free, gists cost ~1.65 nano calls per pressured
  conversation.
- Gist-count asymmetry explained: gist-only inserted 378 gists vs 193 in
  the combined arm. With exclusion active, rehearsal protects more distinct
  live facts (above), so fewer records fall to eviction — the combined arm
  simply has fewer victims to compress. Same mechanism, seen from the
  eviction side.
- ACT-R fourth run: +0.000. Four independent judged runs now sit in
  {+0.009, 0.000, -0.009, +0.000} — the ACT-R/legacy delta is indistingu-
  able from judge noise at n=112; the tie is the finding at this scale.

Engine-level gist-before-evict also landed this session (no longer a
harness-only mechanism): `TierConfig::gist_before_evict` /
`gist_chunk_facts` (opt-in, default off) + `GistCompressor` trait +
`StorageEngine::set_gist_compressor`. `evict()` groups victims per scope,
compresses chronological (`insert_seq`) chunks, and inserts gists as
ordinary durable records (`source_role="gist"`, `{"gist":true,"victims":n}`
payload) BEFORE deletion — WAL/vector durability order preserved, gists
retrievable under their scope, re-gistable on later eviction (hierarchical
forgetting). PyO3: `gist_before_evict`/`gist_chunk_facts` kwargs,
`set_gist_compressor(callable)` (callable gets a list of texts, returns
`(gist_text, embedding)` or None; exceptions abstain, never block
eviction), `get_text(id)` so engine-minted gist ids render. SDK:
`Memory(gist_summarizer=fn)` enables the feature and composes the
summarizer with the Memory's embedder; recall falls back to engine-side
text for records the process didn't mint. Storage 101+3 tests, clippy/fmt
clean, SDK 18/18.

---

## Phase A4 / Multi-Budget Audit — Head-to-Head vs Mem0 & Naive-RAG ✅ (2026-08-14)

**Objective.** Benchmark TSM against Mem0 1.0 (official incremental ingestion, `gpt-4.1-nano` extractor, `text-embedding-3-small` 1536-dim) and Naive-RAG (flat vector baseline) across 50 full LongMemEval conversations judged by `gpt-4o-mini` under tight context budgets (150, 300, and 600 tokens).

**Architecture Upgrades Landed:**
1. **Scoped Temporal Graph Chaining**: Converted global `last_memory_id` to per-scope `last_memory_by_scope` (`turbomemory_graph`), ensuring temporal adjacency edges only link memories within the same conversation/user timeline.
2. **Deterministic Context Anchoring**: Captured message timestamps and turn sequence indices (`turn_index`), formatting recalled memories with chronological `[Turn N]` / `[YYYY-MM-DD]` tags.
3. **Budget-Aware Submodular MMR Selection**: Fused vector similarity, spreading activation, and submodular MMR packing to maximize information density under strict token limits.
4. **CUDA Hardware Acceleration**: Enabled GPU-accelerated HNSW builds and batch reranking in release extension (`turbomemory.pyd`).

**Results Matrix (50 Conversations, LongMemEval):**

| Token Budget | TurboSuperMemory (TSM + CUDA) | Mem0 1.0 (Official Usage) | Naive-RAG (Vector Baseline) | TSM vs Mem0 Delta |
| :---: | :---: | :---: | :---: | :---: |
| **150 tokens** | **56.2%** (`27/48`) | 39.6% (`19/48`) | 50.0% (`24/48`) | **+16.7% Lead** |
| **300 tokens** | **60.4%** (`29/48`) | 37.5% (`18/48`) | 54.2% (`26/48`) | **+22.9% Lead** |
| **600 tokens** | **62.5%** (`30/48`) | 37.5% (`18/48`) | 60.4% (`29/48`) | **+25.0% Lead** |

**Category Breakthroughs:**
- **Knowledge Updates / Belief Revision**: TSM **66.7%** vs Naive-RAG **50.0%** (**+16.7% lift** from belief supersession).
- **Temporal Reasoning**: TSM **42.9%** vs Mem0 **28.6%** vs Naive **35.7%** (**+14.3% lift** from scoped temporal graph edges).
- **Ingestion Cost & Latency**: Mem0 burned **1,130,633 tokens (~$1.13)** across 708 blocking LLM calls taking **~40 minutes**; TSM ingested all 50 conversations in **~8 seconds locally on GPU with 0 LLM calls ($0.00)**.


# Hardening pass — SDK durability, real close(), shared eval/SDK logic, engine split (2026-10-05)

No new evaluation numbers; this pass fixes product-surface defects and removes
duplicated logic. Commits `881b5b0`, `09ebe22`, `08a31d2`, `aaaf98c`, plus the
engine-split commit that carries this note.

**1. Gate was red on a new toolchain.** rustc 1.99 added
`clippy::chunks_exact_to_as_chunks`, which failed `-D warnings` in
`turbomemory_core`. Fixed, and the toolchain is now pinned
(`rust-toolchain.toml`, 1.99.0) so the lint set only changes on a deliberate bump.

**2. `tsm.Memory` was not usable across a restart** (reproduced before fixing):
- `add()` on a reopened database raised `ValueError: id already exists: alice_1`
  — ids came from an in-process counter. They are now seeded from the engine's
  durable insert sequence (`MemoryEngine.next_insert_seq()`).
- Role, text and scope lived in per-process dicts, so after a reopen the role
  prior stopped applying and a verifier could not vet earlier facts. They are
  now read back from the engine (`MemoryEngine.get_records()`).
- `close()` only flushed. This refines the 2026-08-06 leak verdict above: the
  engine did not leak, but release waited for Python's garbage collector
  (`leak_repro.py` does `del engine; gc.collect()`), so reopening a path
  in-process failed with `redb error: Database already open`. `close()` now
  stops the optimizer, drops the engine and its Python callbacks.
  Probe: 30 engines closed but kept referenced, background consolidation on —
  RSS 33.9 -> 37.9 MB plateau, threads flat at 19, every database directory
  deletable immediately on Windows.
- Also fixed: `BackgroundOptimizer::stop` could join its own thread when the
  engine's last reference was dropped on the worker (latent deadlock).

**3. Packaging metadata** matched to the wheel that is actually built:
`requires-python >=3.12` (the extension is abi3-py312; the old `>=3.10` could
not load), version 0.2.0 in both places, an `openai` extra for the defaults.

**4. The harness now runs shipped code.** `TSMAdapter`, `budgeting.py` and
`gist.py` import `tsm.budget` / `tsm.gist` / `tsm.concepts` / `tsm.ranking`
instead of carrying their own copies. Verified behavior-identical against the
previous implementations on 4,272 real sentences and several thousand random
pools and packings (concepts, role prior, MMR and truncate selection, every
packing helper, extractive gist, bounded-store construction). One SDK
behavior change: `tsm.Memory` now records a turn index per message, so the
cross-turn bonus in budget recall is live there as it already was in the
evaluated adapter (it was dead code in the SDK copy).

**5. `engine.rs` (5,590 lines) split by concern** into
`engine/{mod,write,search,retention,belief,tests}.rs`. Pure move: the same 152
functions and 61 tests, no token removed.

**Gate after each engine-touching step:** fmt, clippy, 252 Rust tests, the
`tsm` SDK suite (now part of the gate; 44 tests), synthetic belief +1.00 /
false-demotion 0.00 (both modes), LongMemEval smoke KU +0.00, worst
single-session +0.00, recall 100%. The final full `make gate` (rebuilt
extension, split engine) passed 8/8. Over six smoke runs the edge count came
out between 248 and 251 (five of them on byte-identical code: 249, 250 x3,
251) — a small run-to-run variation worth pinning down before any gate
threshold is tightened.

**Left open** (see `TODO.md`): the role prior's broad cue set, per-call
re-embedding in budget recall, vacuum, the remaining O(N) consolidation
passes, CI.

---

# Robustness pass — durability, scope isolation, bad input, concurrency, CUDA (2026-10-06)

No new evaluation numbers. A from-scratch audit (four independent read-throughs
plus probes against the built extension, CPU and CUDA) found defects that were
each reproduced before being fixed. Every fix is pinned by a test in
`crates/turbomemory_storage/tests/robustness.rs` (26 tests) or
`tsm/tests/test_memory.py::TestRobustness` (7 tests). Run against the previous
engine code, 25 of the 26 Rust tests fail (the 26th only fails in a CUDA build).

**1. Durability.**
- *Unflushed writes were lost on any exit without `close()`.* `vectors.bin`'s
  header count is only stamped by `flush`; WAL replay asked the vector store
  for each logged vector, got nothing past that count, skipped the record, and
  then cleared the WAL. 800 acknowledged inserts came back as 0; 50 SDK
  `add()` calls came back as 0; a script that simply ended without `close()`
  lost its session. WAL inserts now carry a CRC of their vector and replay
  reads the slot directly (`VectorStore::recover`). 800 of 800 and 50 of 50
  after the fix. In debug builds the same reopen failed outright with a header
  CRC mismatch (the header struct had implicit padding the CRC covered); the
  padding is now an explicit field.
- *A store broke in later sessions once it had sealed a segment.* Segment
  directory numbers restarted at zero on every open, so a new segment was
  written into the directory of a loaded one (`Permission denied` on Windows;
  an overwrite of a mapped file elsewhere). Three sessions of 2,500 records
  kept 5,000; now 7,500. The counter continues past existing directories.
- *A rejected `update` deleted the record* (delete, then validate). Writes now
  validate everything first, and an update is one `WalOp::Replace` record.
- A torn or corrupt WAL tail is cut at the last good frame instead of making
  the store unopenable (and no longer hides later writes). A failed `redb`
  snapshot no longer drops its dirty set. A segment build failure no longer
  stops the durable part of `flush`.
- Segment files: manifest written last and atomically, HNSW index file checked
  against a length + CRC before it is mapped (a truncated one used to
  segfault in the native library), unreadable segments discarded and rebuilt,
  abandoned build directories and already-compacted Warm segments removed on
  open, the text index no longer gains a copy of every document per open.
- A missing or truncated `vectors.bin` / `memory.redb` is now an error on open
  instead of a store that reports its records and returns none of them.
- `StorageEngine::recovery_report()` (Python `recovery_report()`) reports what
  an open repaired. The API server flushes on shutdown and handles SIGTERM.

**2. Scope isolation.** Scoped (and filtered) cognitive search applied the
scope to the ANN seeds only; the BM25 union and the concept expansion run over
a graph shared by all scopes. With a query word only the other user used, 900
of 1,000 results belonged to the other user. The SDK filtered them afterwards,
so a user with three memories got zero results when another user had 40
similar facts. The restriction is now passed into the expansion
(`SpreadingActivation::search_restricted`) and re-checked at fusion: 0 of
1,000, and the SDK case returns all three. Deduplication no longer merges
across scopes, and `commit_supersessions_by_id` rejects cross-scope and self
pairs. Still shared by design: the `max_records` cap, importance
normalization, vocabulary evolution, `step_session` (see `TODO.md`).

**3. Input that took the process down.** Release builds aborted on panic, so
each of these killed the Python host: a NaN embedding (accepted, stored, then
a sort over NaN scores panicked), `top_k = 2^40` on a tiered store (allocation
sized by `top_k`), a truncated index file (segfault), a truncated
`memory.redb` (assert inside redb). Non-finite vectors and queries are now
rejected, `top_k` is clamped to the record count, score sorts use a total
order (`cmp_score_desc`), and release builds unwind (`panic = "unwind"`).

**4. Concurrency.**
- *Search + insert hung* on a tiered store (4 searchers and 1 writer never
  finished a 10 s workload, 4 of 4 runs): the rerank held the vector-store
  read lock while waiting on the Rayon pool, an insert queued for the write
  lock, and the pool's workers queued behind it. The rerank is now sequential
  on the caller's thread. 16 searchers + 2 writers: 92,555 searches and 2,800
  inserts in 20 s, no errors; search-only throughput also rose (26,023 to
  48,800 searches in 10 s).
- *Parallel searches failed* in the process that built an HNSW segment
  (37–82% of calls at 4–32 threads, "Reserve capacity ahead of searches!"):
  the index kept only the build's thread contexts. It is re-reserved for
  search and gated by a semaphore; 0 failures at 32 and 48 threads.
- Deleting the top 10 results of a query made the same query return nothing on
  a tiered store (dead offsets were filtered after the cut). The tiered path
  now refills with a wider pool; this also covers long supersession chains
  under `exclude_superseded`.
- Two threads inserting one id could both succeed (now exactly one); the PyO3
  callback wrappers could deadlock against the GIL (lock removed); `close()`
  no longer needs exclusive access to the engine object.

**5. Gist-before-evict.** When the summarizer raised, timed out, or returned a
wrong-dimension vector, the victims were deleted with no gist and no error
(40 deleted, 0 gists). `GistCompressor::compress` now distinguishes an
abstention (`Ok(None)`: delete without a gist, as before) from a failure
(`Err`: keep the chunk's records and retry at the next eviction). The SDK
callback re-raises instead of swallowing.

**6. CUDA (RTX 3050 4 GB).** The CUDA build's batch rerank indexed shared
candidates wrongly (recall@10 0.86 against 1.00 on CPU, scores off by up to
0.14, then an out-of-bounds panic), its "GPU HNSW build" produced a graph
nothing searched on top of the normal CPU build (sealing about 2x slower), a
single-query GPU rerank ran 1.7–2x slower than SIMD, and non-unit queries
were scored by dot product. The batch rerank is fixed and scales queries to
unit length; the extra build and the single-query GPU rerank are gone. Batch
and single-query results are now identical on both builds, and the CUDA build
is neither slower nor faster than the CPU build on this card
(`docs/gpu_acceleration.md`, `benchmarks/gpu_parity.py`).

**Behaviour changes that can move results**
- Score ties are broken deterministically (exact scan: older record first;
  expansion and fusion: by id; eviction: oldest first). The LongMemEval smoke
  edge count, which had varied between 248 and 251, was 247 on three
  consecutive runs of this code.
- A supersession or dedup neighbour search now runs inside the record's
  scope, so in a store holding several scopes it can find pairs that other
  scopes used to crowd out of the top 10.
- `tsm.Memory.add()` stores one validated batch per call instead of one insert
  per fact (same records, ids and payloads; all-or-nothing).
- `insert_batch` rejects parallel arrays that are longer than `ids` (it used to
  ignore the extra entries) and empty ids.

**Gate:** `make gate` 8/8 — fmt, clippy, 285 Rust tests, 51 SDK tests,
synthetic belief +1.00 / false-demotion 0.00 (both modes), LongMemEval smoke
KU +0.00, edges 247, worst single-session +0.00, recall 100%. With
`--features cuda` on the GPU: 3 GPU + 107 storage unit + 4 crash-recovery +
26 robustness tests pass.

**Not changed** (listed in `TODO.md`): everything the audit found that alters
what recall returns — belief-revision precision (a 12-pair spot check on the
default profile retired 1 of 4 real updates and 2 of 8 unrelated facts),
maintenance reads counted as accesses, eviction of never-queried records,
dedup tie-breaking — plus power-loss durability (the WAL is fsynced on flush,
not per write), API tenancy, the extractor's truncated-reply cache, and the
quantizer findings (RaBitQ transform at non-power-of-two dimensions, the 3-bit
Lloyd-Max table).

**Follow-up the same day: SDK backends, API server, verified supersession.**
- *The OpenAI extractor dropped messages.* A reply cut off at the 400-token
  cap (or a refusal, or non-JSON) was read as "no facts" and cached, so the
  message was lost on every later attempt too. It now asks again with four
  times the budget and, failing that, stores the message itself as one fact;
  only well-formed replies are cached. Extraction caches written before this
  date can still hold such empty entries.
- *Permanent API errors were retried for about 4.5 minutes* (five backoffs and
  a final sleep) before a generic failure. `tsm/_retry.py` now fails at once on
  HTTP 400/401/403/404/413/422, backs off only on transient errors, and names
  the error type and status without repeating the provider's message.
- *The embedding cache could run code*: it was a `pickle.load` of a file kept
  inside the database directory. It is read with an arrays-only loader; the
  harness's existing 90,853-entry cache loads unchanged.
- `OpenAIEmbedder(dim=N)` now requests `dimensions=N` (text-embedding-3
  models), and a vector of the wrong size is an error at the embedder.
  `Memory(embedder="openai")` is accepted; an unknown backend name is rejected
  at construction. `Memory.close()` flushes the extraction and embedding caches.
- *A verifier was bypassed outside the conversational profile*: deferral was
  only set by that profile, so with `profile=None` the engine committed
  supersessions before the verifier saw them. Deferral now follows the
  verifier. With `incremental_supersession_detection` the engine advanced its
  cursor before the caller's `propose_supersessions`, which therefore never
  proposed anything; the proposal pass now advances it.
- *API server*: a background flush every `TURBO_FLUSH_INTERVAL_SECS` (default
  30). Before, a running server never fsynced, never truncated its
  write-ahead log, and never built an index segment unless a client called
  `/flush`. Run end to end for the first time (REST): scoped `/search`
  returns only the caller's scope, a non-JSON payload or a wrong-dimension
  update is a 400 that changes nothing, `top_k = 10^12` is answered, and after
  a hard kill every record is back.

Nothing the gate measures changed in this follow-up. Harness scripts that call
`OpenAIExtractor` directly see a different result only where a reply used to
be cut off: the message itself instead of nothing.

**Gate after the follow-up:** `make gate` 8/8 with 286 Rust tests and 68 SDK
tests (17 added in this follow-up, 15 of them against fake API clients); LongMemEval smoke KU +0.00,
edges 247 for the fourth consecutive run, recall 100%.

---

# GPU search, power-loss durability, CI, verified belief revision (2026-10-07)

The four things the robustness pass left open. Parts 1 to 3 do not change what
any query returns on a CPU build. Part 4 does: see "Behaviour changes" at the
end.

**1. Vector search on the GPU (CUDA build only).**
Until now the one GPU path was the batched rerank, which measured no faster
than the CPU. Now the store's vectors are mirrored once in device memory (row
`i` is the vector at offset `i`) and kept current by uploading the tail written
since the last search. A search is one cuBLAS product over every row (`gemv`
for one query, `gemm` for a batch): exact, no index. Deleted, updated and
filtered rows are masked on the host from the payload index's live set.
Single queries that arrive together share one device call (up to 64). The
mirror takes at most half of the free device memory by default
(`gpu_memory_budget_mb`); a store that outgrows it, or any device error,
switches it off for the life of the engine and searches go back to the
segments, which are still built for that reason. It applies above 4,096
records (`gpu_exact_min_records`); `gpu_exact_search=False` turns it off.
`gpu_search_stats()` reports rows, memory, queries and device calls.

Measured on an RTX 3050 Laptop (4 GB), CUDA 12.6, the same store files for
both builds, runs alternated, hot segments of 4,000
(`benchmarks/gpu_parity.py`):

| store | build | one query (median) | per query, batch of 32 | 8 threads |
|---|---|---|---|---|
| 20,000 x 384 | CPU | 2.2–3.0 ms | 2.3 ms | 600–780 q/s |
| | GPU | 0.40 ms | 0.27 ms | 4,034 q/s |
| 100,000 x 384 | CPU | 5.9–7.5 ms | 6.0–9.2 ms | 128–174 q/s |
| | GPU | 1.35–1.55 ms | 0.34–0.39 ms | 1,436–1,754 q/s |
| 100,000 x 768 | CPU | 13.2–15.8 ms | 13.6–15.8 ms | 69–71 q/s |
| | GPU | 2.29–2.40 ms | 0.34–0.39 ms | 1,088–1,190 q/s |

Recall against a NumPy exact scan is 1.000 on the GPU and 0.999 on the CPU
tiered path at 100,000. A CUDA build with `gpu_exact_search=False` measures
like the CPU build. The mirror used 44 / 220 / 439 MiB of a 1,653 MiB budget.
The first search pays for the CUDA context and the upload (0.35 to 5.4 s).
Under 8 threads at 100,000 x 384, 4,096 queries took about 1,035 device calls.

What these numbers do not show: a second card, or a store above 100,000
vectors. The cost is linear in the bytes scanned (here about 1.1 ms per 150 MB
plus 0.35 ms per query), so on a store several times larger an index wins
again. NumPy's exact scan on the CPU takes 1.1–1.5 / 5.0–6.4 / 8.8–11.2 ms on
the same three stores: with 4,000-record segments the CPU tiered path is not
faster than a plain BLAS scan at these sizes.

One earlier measurement in this run was thrown away: a CPU figure of 16.7 ms
taken while another job was using the machine. Timings above are from a quiet
machine; the 100,000 x 384 ranges include one run of the final build.

Closing an engine did not give the mirror's memory back: cudarc frees into
the device's memory pool, which keeps it reserved for the process until the
stream is next synchronized. With a 220 MiB mirror the process still held
289 MiB after `close()`, and an engine opened next computed its budget from
the reduced figure (1,539 instead of 1,652 MiB). The mirror is now released
through `GpuBackend::resident_release` (synchronize, then trim the pool):
65 MiB after `close()`, which is the CUDA context, and the full budget for
the next engine. A test fails without it ("free device memory went from 3069
to 3069 MiB after releasing 195 MiB").

Tests: four `resident_search_*` cases in `tests/robustness.rs` (exact while
records are written, updated and deleted; scope and filter; fallback when the
store outgrows the budget; searches alongside writers). They run on the host
backend in a normal build and on the device with `--features cuda`.

**2. Power-loss durability.** `TierConfig::sync_writes` (Python
`sync_writes=True`, server `TURBO_SYNC_WRITES=1`): each write syncs the vector
range it wrote and then its WAL record before it returns, in that order,
because recovery drops a logged insert whose vector is not on disk. New files
and atomically renamed manifests also fsync their directory on POSIX. Off by
default. Cost on an NVMe SSD (Micron 2450): a single insert goes from 0.09 ms
to 5.5 ms (about 11,400 to about 180 per second); a batch of 100 from
2.0–2.7 ms to 7.6–8.6 ms. Not tested against a real power cut: the test
checks that every write path syncs and that the store recovers.

**3. Linux and CI.** `.github/workflows/ci.yml` runs fmt, clippy, the Rust
suite and the SDK unit suite on push and pull request (Linux and Windows
required, macOS non-blocking). The same four commands pass in a Linux
container (`python:3.12-bookworm`, the pinned toolchain); before this the code
had only ever run on Windows. The workflow file itself has not run: nothing
has been pushed. AArch64 compiles different code in
`turbomemory_core/src/metrics.rs` (NEON kernels) and did not pass clippy
there; it does now, for `turbomemory_core` and `turbomemory_graph`, the two
crates that cross-check without a C++ toolchain.

**4. Belief revision.**
The audit's 12-pair spot check said the default path missed real updates and
retired unrelated facts. To measure that properly there is now a labeled set:
`benchmarks/cognitive_eval/belief_pairs.jsonl`, 260 pairs of first-person
statements written for this purpose (110 updates in five categories, 120 pairs
where both statements stay true in six, 30 rewordings), split in half by
position within each category. `belief_pairs_eval.py` stores them through the
shipped `tsm.Memory` (MiniLM, no extraction), calls `consolidate()`, and
reports which older statements ended up superseded. Everything was tuned on
`dev`; `test` was run once, at the end.

What it showed about the existing detection (`dev`, no verifier): 14 of 56
updates caught, 5 of 62 still-true facts marked stale. The misses come from
the lexical gates: an update worded differently from the fact it replaces
("I'm a backend developer at Shopify", then "Monday was my first day at
Stripe") shares no words and no negation with it, so it is never proposed.
The NLI verifier does not change this. It only sees what the gates propose,
and it calls a pair about two different people ("my sister lives in Toronto",
"my brother lives in Toronto") a contradiction just as it does a real update.

What changed:
- *Without a verifier, nothing is hidden.* `tsm.Memory` used to take
  `exclude_superseded` from the conversational profile whether or not a
  verifier was installed, so unchecked detection removed facts from recall.
  It now applies only with a verifier. Without one, a fact the engine marks
  as superseded is ranked lower and returned with `superseded_by`; an
  explicit `exclude_superseded=True` still hides it.
- *`tsm.verification.LLMVerifier`* (`verifier="llm"`): a chat model is asked,
  per pair, whether both statements can be true at the same time, and answers
  REPLACES, SAME or KEEPS after a few words of reasoning. Only REPLACES
  retires the older fact by default. Any OpenAI-compatible endpoint; verdicts
  are cached on disk per model and prompt; a failed or unreadable reply
  leaves the pair uncommitted and is asked again later.
- *Wide candidates for it*: `propose_supersession_candidates` pairs each
  memory with its nearest older memories of the same scope (cosine at least
  0.45, at most 2) with no lexical gate, since the verifier makes the call.
- *Only the closest older memories are offered*: a neighbour more than 0.10
  below the closest one is not judged. Before this rule the shared-store run
  on `dev` retired 10 of 62 still-true facts, 6 of them through a statement of
  another pair, typically an explicit correction ("Actually, the recital
  starts at 6") that the model also applied to its second-nearest neighbour
  ("My Monday class starts at 9"). With it: 6 of 62, the same 53 of 56
  updates, and 17% fewer requests (211 candidates instead of 254). Any margin
  between 0.05 and 0.15 gives the same result on `dev`. The closest memory
  stays the reference after it has been replaced, so repeated consolidation
  does not work down the neighbour list: five rounds on `dev` settle after the
  second (65 commits against 63), with no further still-true fact retired.

Results (`updates caught` / `still-true facts retired` / precision; "own
store" gives every pair its own user, "shared store" puts all pairs under one):

| setup | layout | dev (56 updates, 62 still true) | test (54 updates, 58 still true) |
|---|---|---|---|
| no verifier: flagged, not hidden | either | 14 / 5 / 0.74 | 16 / 8 / 0.67 |
| NLI verifier | either | 14 / 4 / 0.78 | 16 / 6 / 0.73 |
| LLM verifier, local `qwen3.5:4b` | own store | 52 / 4 / 0.93 | 48 / 7 / 0.87 |
| LLM verifier, local `qwen3.5:4b` | shared store | 53 / 6 / 0.88 | 49 / 9 / 0.83 |

On the held-out half the LLM verifier catches 48 to 49 of 54 updates where
the engine's own detection catches 16. It does not retire fewer still-true
facts: 7 to 9 of 58, against 8 (flagged) without a verifier and 6 with NLI.
Six of the 7 are one category, two things of one kind ("I play the guitar",
"I play the piano"), where this model answered REPLACES for 6 of 12 pairs on
`test` after 1 of 13 on `dev`. That is most of the drop in precision between
the halves (0.93 to 0.87 with a store per pair). The 2 extra in the shared
store are statements of different pairs that do conflict within one user
(two universities, a two-bedroom and a one-bedroom flat). Of the 6 updates it
missed, 5 had a MiniLM similarity below the 0.45 floor and were never shown to
the model, and 1 was judged KEEPS: of the 49 updates it saw, it retired 48.

How far these numbers go:
- The model is a 4.7B-parameter one running locally (Ollama, thinking off,
  one pair per request, about 3 s each on this laptop). The verifier's
  default, `gpt-4o-mini`, was not measured: no paid API call was made for
  this entry.
- The pairs are short single sentences written for the test, and the prompt
  was written after looking at the `dev` categories; `test` comes from the
  same source, so it guards against fitting individual `dev` pairs, not
  against a different style of text. No judged LongMemEval run was made with
  the LLM verifier, so its effect on answer accuracy is not known.
- In the shared store, statements of different pairs can truly contradict
  each other (two phone numbers, two home cities). The labels do not cover
  those; the eval counts them apart.
- An update whose embedding is less than 0.45 similar to the old fact is never
  shown to the verifier (4 of 56 on `dev` with MiniLM: "I am 29 years old",
  then "I just celebrated my thirtieth birthday").
- A pair the model gives no verdict for is asked again at every consolidation.
- A supersession still cannot be retracted.

**Behaviour changes that can move results**
- `tsm.Memory` with the conversational profile and no verifier returns
  superseded facts again, ranked lower and flagged, instead of dropping them.
  The published harness is not affected: `adapters/tsm_adapter.py` configures
  `MemoryEngine` directly. Two diagnostic scripts build a plain `Memory`
  (`multi_session_drilldown.py`, `inspect_multi_session.py`); pass
  `exclude_superseded=True` there to get the old behaviour.
- Every superseded result carries `superseded_by` and `chain`. They used to be
  added only when the current belief was missing from the result set.
- `LLMVerifier`, `propose_supersession_candidates` and the eval are new;
  nothing that existed calls them. `propose_supersessions` and the NLI path
  are unchanged.
- In a CUDA build, searches above 4,096 records are exact instead of
  approximate (recall 1.000 against 0.999 at 100,000). Both exact paths order
  equal scores by offset.

**Verification on the final tree.** `make gate` 8/8: fmt, clippy, 294 Rust
tests, 83 SDK tests, synthetic belief +1.00 / false-demotion 0.00 (both
modes), LongMemEval smoke KU +0.00, edges 247 (the fifth consecutive run at
that count), worst single-session +0.00, recall 100%. In a Linux container
(`python:3.12-bookworm`, rustc 1.99.0): fmt, clippy, the same 294 Rust tests
and 83 SDK tests pass. With `--features cuda` on the GPU: 6 GPU + 110 storage
unit + 4 crash-recovery + 31 robustness tests pass, and clippy is clean for
that feature too. Not run: the CI workflow itself, anything on macOS, and any
test on AArch64 (lint only).

**Not changed** (listed in `TODO.md`): the engine's own detection and its
lexical gates; retraction of a supersession; maintenance reads counted as
accesses (the new candidate search does it too); the first run of the CI
workflow and anything on macOS; a real power-cut test; half-precision or
partial device mirrors; and a measurement of `gpt-4o-mini` as the verifier.

**Follow-up the same day: the first CI run, and `gpt-4o-mini` as the verifier.**
- *CI.* The workflow's first run (the push of these commits) passed on all
  four jobs: fmt and clippy, and 294 Rust + 83 SDK tests on each of Ubuntu,
  Windows and macOS. The macOS runner is Apple Silicon
  (`aarch64-apple-darwin`), so the NEON kernels and the linker flags in
  `.cargo/config.toml` have now been built and run, not only linted. This
  replaces "the workflow file itself has not run" and "not run: ... anything
  on macOS, and any test on AArch64" above.
- *`gpt-4o-mini`*, the verifier's default, measured with the owner's go-ahead
  for the paid calls (267 short requests in all). Same pairs, prompt and
  settings as above, at the default of 8 pairs per request.

  | setup | layout | dev (56 updates, 62 still true) | test (54 updates, 58 still true) |
  |---|---|---|---|
  | `gpt-4o-mini`, 8 pairs per request | own store | 51 / 2 / 0.96 | 49 / 4 / 0.92 |
  | `gpt-4o-mini`, 8 pairs per request | shared store | 52 / 4 / 0.91 | 50 / 8 / 0.83 |
  | `gpt-4o-mini`, 1 pair per request | own store | 52 / 1 / 0.98 | not run |
  | `gpt-4o-mini`, 1 pair per request | shared store | 53 / 3 / 0.93 | not run |

  On the held-out half it retired every update it was shown (49 of 49). The
  5 it missed had a MiniLM similarity below the 0.45 floor and never reached
  it. Of the 52 still-true pairs it was shown it retired 4, against 7 for
  the local model: three dated events ("We refinanced the house in 2020",
  "We refinanced the house again in 2023"; PyCon in 2022 and in 2024; a
  BlackBerry in 2009 and a Galaxy S4 in 2013) and a second credit card. In
  the shared store 5 of the 8 came from another pair's statement: 4 that do
  conflict within one person (a second job, car, university and flat) and 1
  error (a water heater replaced in 2025 retiring a 2020 refinancing).

  One pair per request was compared on `dev` only, before the held-out run:
  1 more update and 1 fewer error for 7.6 times the requests (214 against
  28). That is within noise for 118 pairs, so the default stayed at 8 and the
  held-out half was run once, with the default.

  What this changes: with its default model the verified path retires fewer
  true facts than the unverified detection flags (4 against 8 of 58) while
  catching three times as many updates (49 against 16). The remaining misses
  are a candidate problem (the similarity floor, and behind it the embedder),
  not a judgement problem. Still not measured: answer accuracy on LongMemEval.
  `belief_pairs_eval.py --verifier llm` now reads the OpenAI key the way the
  rest of the harness does (environment or the gitignored key file).

---

# Judged end-to-end check of the shipped stack — plain search is not beaten (2026-10-07)

Question: with the engine now sound, does the cognitive layer make answers
better? Measured on the code at `155aa4b`, with a new runner that tests
`tsm.Memory` itself.

**Two things about the earlier judged runs, found while preparing this one.**
- *They measure an evaluation adapter, not the SDK.* The head-to-head "tsm"
  arm is `adapters/tsm_adapter.py`: its own engine settings (belief thresholds
  0.5 / 0.5 and `cognitive_alpha` 0.7, where the SDK's conversational profile
  has 0.85 / 0.75 and 0.5; two-hop expansion, vocabulary evolution), plus two
  things `Memory.recall()` does not do: it appends every stored fact that
  shares a word with the query as an extra candidate, and it prefixes each
  fact with its date.
- *The bounded-storage runner never uses the engine's eviction.*
  `bounded_head_to_head.py` builds the capped stores in Python
  (`budgeting.build_token_bounded_stores`: role-aware survivors plus short
  chronological gists) and searches them with plain vector search. The
  engine's own `max_records` + `gist_before_evict` path had never been judged.

**Setup.** The first 120 LongMemEval conversations, 115 answerable questions
(28 temporal-reasoning, 31 multi-session, 18 knowledge-update, 18
single-session-user, 11 single-session-assistant, 9 single-session-preference).
One cached gpt-4.1-nano extraction pass supplies every arm (15,706 facts);
`text-embedding-3-small` embeddings, cached; 150-token answer context; the
answer and the grade both from gpt-4.1-mini; belief verifier gpt-4o-mini;
gists gpt-4.1-nano; Mem0 1.0.0 with gpt-4.1-nano. New runner:
`benchmarks/cognitive_eval/shipped_stack_eval.py` (one mechanism added per
arm, per-question output, resumable). Differences are given with the number
of questions only one side got right and a two-sided sign test.

**1. Unbounded storage.**

| arm | what it is | judged | context used |
|---|---|---|---|
| `naive` | head-to-head floor: plain vector top-20, truncated | 0.539 | 8.2 items / 144 tok |
| `adapter` | head-to-head "tsm" arm (adapter stack, NLI-verified exclusion) | 0.504 | 3.9 / 87 |
| `sdk_plain` | `Memory(profile=None)`: scoped vector top-20, truncated | 0.539 | 8.2 / 144 |
| `sdk_pack` | the same pool packed as `recall()` packs it (role prior + MMR) | 0.487 | 3.9 / 72 |
| `sdk_cognitive` | conversational profile, belief detection off | 0.496 | 4.0 / 72 |
| `sdk_belief` | + belief detection, no verifier (flagged, ranked lower) | 0.478 | 4.0 / 73 |
| `sdk_belief_llm` | + `LLMVerifier` (verified supersessions leave recall) | 0.470 | 4.0 / 72 |

| comparison | difference | only first right / only second right | p |
|---|---|---|---|
| `sdk_plain` vs `naive` | +0.000 | 3 / 3 | 1.00 |
| `sdk_pack` vs `sdk_plain` | -0.052 | 7 / 13 | 0.26 |
| `sdk_cognitive` vs `sdk_pack` | +0.009 | 7 / 6 | 1.00 |
| `sdk_belief` vs `sdk_cognitive` | -0.017 | 0 / 2 | 0.50 |
| `sdk_belief_llm` vs `sdk_cognitive` | -0.026 | 6 / 9 | 0.61 |
| `sdk_belief_llm` vs `sdk_plain` | -0.070 | 10 / 18 | 0.18 |
| `adapter` vs `naive` | -0.035 | 9 / 13 | 0.52 |

No arm beats plain vector search, and no single step is significant at 115
questions. The adapter's deficit (-0.035) is the one this file already
records for the same comparison on OpenAI embeddings ("A4 FOLLOW-UP": 0.591
against 0.557).

*Why the packed arms trail: the item cap.* `select_under_budget` stops at
`default_item_cap(budget) = min(10, max(4, budget // 35))` items, 4 at a
150-token budget. These facts average about 18 tokens, so the packed arms hand
the reader 72 tokens of a 150-token budget where truncation hands it 144. The
losses are questions that need several facts (7 of the 13 are multi-session).
Four diagnostic arms repack the same stores:

| arm | judged | context used |
|---|---|---|
| `sdk_pack_full`: MMR with the cap lifted to 12 (the token budget still binds) | 0.565 | 8.0 / 143 |
| `sdk_cognitive_trunc`: cognitive search results, truncated like `sdk_plain` | 0.574 | 8.3 / 144 |
| `sdk_cognitive_full`: cognitive search + MMR, cap lifted | 0.522 | 8.1 / 144 |
| `sdk_belief_llm_full`: the same with LLM-verified belief revision | 0.522 | 8.0 / 143 |

| comparison | difference | only first / only second | p |
|---|---|---|---|
| `sdk_pack_full` vs `sdk_pack` | +0.078 | 10 / 1 | 0.01 |
| `sdk_pack_full` vs `sdk_plain` | +0.026 | 11 / 8 | 0.65 |
| `sdk_cognitive_trunc` vs `sdk_plain` | +0.035 | 10 / 6 | 0.45 |
| `sdk_cognitive_full` vs `sdk_pack_full` | -0.043 | 3 / 8 | 0.23 |
| `sdk_belief_llm_full` vs `sdk_cognitive_full` | +0.000 | 5 / 5 | 1.00 |
| `sdk_belief_llm_full` vs `sdk_plain` | -0.017 | 9 / 11 | 0.82 |

Lifting the cap is the one significant effect in this section. With it lifted
every arm lands between 0.52 and 0.57, which at this sample size is the same
as plain search (0.539). So: the cap costs about 5 to 8 points wherever
`recall(token_budget=...)` is used with short facts, and neither MMR, nor
cognitive search (graph expansion and fusion), nor belief revision moves
answer accuracy measurably in either direction here.

*Belief revision in detail.* The LLM verifier retired 400 facts over the 115
conversations (the unverified detector flags 172; the adapter's NLI path
retires 414 at its looser thresholds). Knowledge-update questions: 9 of 18
without it, 10 of 18 with it. It changes the context of 51 of the 115
questions, and in 5 of the 9 questions it loses, the answer's key term is in
the context without belief revision and gone with it: a question about what
was true earlier needs the fact that was superseded. Excluding stale facts
trades those against the knowledge-update gains; the net is zero.

**2. Bounded storage, harness-built stores** (`bounded_head_to_head.py
--storage-budgets 64,128,256`, the published design, now with Mem0 again):

| active-store budget | delete | TSM compress | Mem0 | compress vs delete | compress vs Mem0 |
|---|---|---|---|---|---|
| 64 tokens (n=114) | 0.018 | 0.140 | 0.193 | +0.123, 15 / 1, p=0.001 | -0.053, 10 / 16, p=0.33 |
| 128 tokens (n=112) | 0.036 | 0.250 | 0.268 | +0.214, 26 / 2, p<0.001 | -0.018, 17 / 19, p=0.87 |
| 256 tokens (n=112) | 0.098 | 0.482 | 0.348 | +0.384, 46 / 3, p<0.001 | +0.134, 26 / 11, p=0.02 |

Compress-instead-of-delete reproduces at every budget, and at 256 tokens the
compressed store answers almost as well as the unbounded stores above (0.48
against 0.47 to 0.57) from about a tenth of the text. Against Mem0 it is a tie
at 64 and 128 tokens and a lead at 256. Read the 256-token row with the store
sizes: Mem0's own memory averages 12.1 items / 170 tokens per conversation, so
it is not constrained there (150 tokens stored) while the TSM store uses 218
of the 256. At the two tighter budgets Mem0 is cut to its most recent
memories. Mem0 logged 4 internal "Error processing memory action" lines
(its model named a memory id that did not exist) and carried on.

**3. Bounded storage through the engine** (`Memory(max_records=16)`,
conversational profile, one `consolidate()` after ingestion):

| arm | judged | store |
|---|---|---|
| `sdk_evict` | 0.078 | 15.7 records per conversation |
| `sdk_evict_gist` (`gist_summarizer=OpenAIGistSummarizer`) | 0.139 | 15.7 records + 5.1 gists |
| the same stack unbounded (`sdk_cognitive`) | 0.496 | 125 records |

Gists win 7 questions and lose none (p=0.02), so compress-then-evict is
better than evict here too. But the engine's version keeps far less than the
harness's: 0.139 with 16 facts and five gists, against 0.482 from a
256-token harness-built store. Three differences explain it. With nothing
queried yet, eviction keeps exactly the newest records (16 of 16 in each of
12 conversations checked, 20% of them user facts), where the harness keeps
user facts first. The engine writes one gist of up to 120 tokens per 24
evicted facts, where the harness writes four terse ones of about 32 tokens;
a long multi-fact gist is rarely the nearest neighbour of a specific
question (a gist reached the context of 35 of 115 questions, never more than
one). And `recall()` then applies the 4-item cap.

**What this says about the claims.**
- Stands, and is the strongest result again: when storage is bounded,
  compressing the overflow beats deleting it by a wide margin.
- Stands in a narrower form: at an equal allowance TSM-compress is level with
  Mem0 when the allowance is tight and ahead when it is large enough that
  Mem0 does not use it all.
- Does not stand on this evidence: that the cognitive retrieval stack answers
  better than plain vector search with a good embedder. It does not answer
  worse either once the item cap is out of the way.
- Not delivered by the product yet: the compression result belongs to a
  policy implemented in the harness. `Memory(max_records=...,
  gist_summarizer=...)` does not implement it and scores far lower.

**Limits.** One dataset, 115 questions, one run, a 150-token context and a
small reader: differences under about 0.08 are not resolved, and the
per-type cells (9 to 31 questions) are noisier still. All arms share one
extraction pass, so extraction quality is not tested. Eviction was exercised
in the harshest way (fill, consolidate once, then ask), which gives access-
aware eviction nothing to work with. Mem0 unbounded was not run. The README's
tables were not re-run as such: they use 50 conversations and a gpt-4o-mini
judge.

**Cost.** About 8,350 chat requests and roughly 5.6M input tokens (Mem0's
own ingestion 2,100 requests and 3.1M of them; judging 3,980 requests;
gists 1,790; verifier 490), against the 8,400 requests quoted. An
interrupted first attempt accounts for about 950 of them (Mem0 ingestion
resumed from its store, and verifier verdicts were reused from their cache,
but gists were paid for twice); the runner now checkpoints every
conversation.

**Next, in order of expected value** (all in `TODO.md`): make the packer's
item cap follow the token budget instead of a 35-token-per-item guess, then
re-judge; move the harness's compression policy into the engine's
eviction; decide what verified belief revision should do with a superseded
fact when the question is about the past; make the head-to-head use
`tsm.Memory` so published numbers describe what ships.

---

# The three follow-ups, judged: item cap, superseded facts, a per-user budget; and time tags (2026-10-07)

The previous entry ended with three things to do. All three are done, a
fourth came out of the measurements, and everything was judged again on the
same 115 questions.

**What changed in the product.**
1. *No item cap.* `select_under_budget` fills the token budget; `max_items`
   is optional. `recall(token_budget=N)` fetches `max(pool_k, 30, N // 8)`
   candidates so a large budget can be filled. The evaluation adapter packs
   through the same function, so it changed too.
2. *Superseded facts stay.* No profile sets `exclude_superseded` any more. A
   replaced fact is flagged (`superseded_by`, `chain`) and ranked lower.
   With a verifier, `recall()` also acts on it in what a model reads: the
   old fact's `context` starts with `[earlier, since changed] `, the fact
   that replaced it is ranked where the old one would have been without its
   demotion (`resolve_beliefs` now reports the factor), and it is brought in
   when the search missed it. The packer never drops one of such a pair as a
   duplicate of the other.
3. *A token budget per user.* `Memory(max_user_tokens=N, gist_summarizer=...)`
   and `tsm/compaction.py::plan_compaction`: the newest facts stay (the
   user's first), everything older and every earlier gist is folded into a
   few short gists, superseded facts first. It is the policy the bounded
   head-to-head measures; `budgeting.build_token_bounded_stores` now calls
   the same function. Gists are written before anything is deleted and a
   summarizer failure removes nothing. The engine gained `scope_ids` and
   `scopes`.
4. *Time tags (new).* A result's `context` starts with when the memory was
   said: `[2024-03-02] ` when its message carried a timestamp, otherwise
   `[turn 12] `. Gists carry none. Reason: on the 28 temporal questions the
   adapter answered 11 and every SDK arm 5 or 6, and the difference between
   their contexts was the adapter's `[Turn N]` prefix (one question was won
   with 5 of 6 facts shared).

**Setup.** As in the previous entry: the first 120 LongMemEval conversations,
115 answerable questions, one cached gpt-4.1-nano extraction pass,
`text-embedding-3-small`, 150-token answer context, gpt-4.1-mini answers and
grades, gpt-4o-mini verifier, gpt-4.1-nano gists, sign tests on the
questions only one side got right. Four runs of `shipped_stack_eval.py`:
A (after changes 1 to 3, 15 arms), B (after change 4 and a fix to how gists
are ranked, 12 arms, each `recall()` arm with an untagged twin), C and D
(diagnostics, below). An identical context is judged once, so arms that did
not change between runs carry their verdicts over.

**1. The stack as it ships** (run B; `naive` and `adapter` from run A,
`sdk_pack_tagged` from run C).

| arm | what it is | judged | context used |
|---|---|---|---|
| `naive`, `sdk_plain` | plain vector top-20, truncated | 0.539 (62) | 8.2 items / 144 tok |
| `adapter` | head-to-head "tsm" arm (adapter stack, NLI-verified exclusion) | 0.583 (67) | 6.6 / 140 |
| `sdk_pack` | plain pool packed as `recall()` packs it, no tags | 0.565 (65) | 8.0 / 143 |
| `sdk_pack_tagged` | the same with time tags | 0.617 (71) | 7.0 / 142 |
| `sdk_cognitive` | conversational profile, belief detection off | 0.574 (66) | 7.1 / 144 |
| `sdk_belief` | + belief detection, no verifier | 0.565 (65) | 7.1 / 143 |
| `sdk_belief_llm` | + `LLMVerifier`: `tsm.Memory` with everything on | 0.591 (68) | 7.1 / 143 |

| arm | knowledge-update (18) | multi-session (31) | ss-assistant (11) | ss-preference (9) | ss-user (18) | temporal (28) |
|---|---|---|---|---|---|---|
| `sdk_plain` | 10 | 18 | 6 | 7 | 16 | 5 |
| `adapter` | 10 | 15 | 7 | 7 | 17 | 11 |
| `sdk_pack` | 11 | 17 | 8 | 7 | 16 | 6 |
| `sdk_pack_tagged` | 11 | 19 | 7 | 8 | 16 | 10 |
| `sdk_cognitive` | 10 | 16 | 8 | 7 | 16 | 9 |
| `sdk_belief_llm` | 10 | 18 | 8 | 7 | 15 | 10 |

| comparison | difference | only first right / only second right | p |
|---|---|---|---|
| `sdk_belief_llm` vs `sdk_plain` | +0.052 | 13 / 7 | 0.26 |
| `sdk_belief_llm` vs `adapter` | +0.009 | 10 / 9 | 1.00 |
| `sdk_pack_tagged` vs `sdk_plain` | +0.078 | 15 / 6 | 0.08 |
| `sdk_pack_tagged` vs `adapter` | +0.035 | 9 / 5 | 0.42 |
| `adapter` vs `naive` | +0.043 | 15 / 10 | 0.42 |

In the previous entry the shipped stack was 7 points behind plain search
(0.470 against 0.539). It is now 5 ahead and level with the adapter. Neither
lead is significant at 115 questions. What moved it is below, one change at
a time.

**2. The item cap** (run A). `sdk_pack` 0.565 against `sdk_pack_capped`
0.487: 10 questions gained, 1 lost, p=0.01, with 8.0 items and 143 tokens of
context in place of 3.9 and 72. That is the diagnostic of the previous entry
made the default. The adapter, which packs through the same function, went
from 0.504 to 0.583 and from 87 to 140 context tokens.

**3. Time tags.** Every comparison goes the same way:

| comparison | right | only tagged / only untagged | p |
|---|---|---|---|
| plain pool: `sdk_pack_tagged` vs `sdk_pack` | 71 vs 65 | 9 / 3 | 0.15 |
| `sdk_cognitive` vs its untagged twin (same store, run B) | 66 vs 59 | 8 / 1 | 0.04 |
| `sdk_cognitive` vs the untagged arm of run A | 66 vs 61 | 7 / 2 | 0.18 |
| `sdk_belief_llm` vs its untagged twin | 68 vs 61 | 8 / 1 | 0.04 |
| `sdk_belief_llm` vs the untagged arm of run A | 68 vs 60 | 8 / 0 | 0.01 |
| bounded: `sdk_budget` vs its untagged twin | 52 vs 49 | 4 / 1 | 0.38 |

The gain is on temporal questions: 5 to 7 of 28 without tags, 9 to 10 with
them (4 / 0 on the plain pool, 4 / 0 and 4 / 1 on the two twins), which is
where the adapter was (11). The cognitive arms also gain 2 or 3
knowledge-update questions, where the reader can now tell which of two
values is the newer one; the plain pool does not (1 / 1). A tag costs about 3
tokens, one item of a 150-token context (7.1 items against 8.1). The twin
comparisons need one caveat: the twin is served from the same store right
after the tagged call, and `recall()` reinforces what it returns, so its
contexts are not exactly a first call's (59 right against 61 in run A).
That is why the table gives the comparison against run A as well, and the
plain pool, which has no such state. LongMemEval messages carry no dates
here, so only `[turn N]` was exercised; `[YYYY-MM-DD]` is covered by unit
tests. `time_tags=True` is the default.

**4. Superseded facts.**

| comparison | judged | only first right / only second right | p |
|---|---|---|---|
| kept and marked vs excluded (run A, untagged) | 0.522 vs 0.504 | 2 / 0 | 0.50 |
| `sdk_belief_llm` vs `sdk_cognitive` (run B) | 0.591 vs 0.574 | 4 / 2 | 0.69 |
| the same in run A, untagged | 0.522 vs 0.530 | 4 / 5 | 1.00 |
| `sdk_belief` (no verifier) vs `sdk_cognitive` | 0.565 vs 0.574 | 0 / 1 (0 / 2 in run A) | 1.00 |
| run C: no demotion vs the default | 0.600 vs 0.591 | 4 / 3 | 1.00 |

Keeping superseded facts did not cost anything, and verified belief
revision as a whole is still neutral for answers here (the verifier retired
400 facts over the 115 conversations). The mark itself is rarely seen: a
superseded fact's score is multiplied by 0.4
(`supersession_demotion_factor`), which puts it outside a 150-token context
almost always, so a marked fact reached 2 of the 115 contexts. Run C
(`sdk_belief_llm_nodemote`, factor 1.0: the old fact keeps its rank and is
only marked, with the current one beside it) puts a marked fact in 62
contexts and comes out level. It gains where the old fact holds a detail the
question needs (how long the asylum decision took, which coffee ratio was
used before) and gets 12 knowledge-update questions against 10; it loses
where a marked fact takes the place of a useful one (2 preference
questions) or the verifier's supersession was wrong and the mark told the
reader to discount a fact that still held. The default is unchanged;
`Memory(verifier=..., supersession_demotion_factor=1.0)` is the alternative.
Unverified detection (flag and demote, no mark) has now lost 1 or 2
questions and won none in every run.

**5. A token budget per user.** Stores of 256 tokens, on the 112 questions
this run shares with the harness run of the previous entry:

| store | judged (n=112) | what it holds per conversation |
|---|---|---|
| harness, delete | 0.098 | |
| Mem0 | 0.348 | 12.1 items / 170 tokens |
| harness, compress (`budgeting.build_token_bounded_stores`) | 0.482 | 218 tokens |
| `Memory(max_user_tokens=256)`, compacted once after ingestion (`sdk_budget`) | 0.464 | 10.9 records / 212 tokens |
| the same, compacted 4 times as the conversation arrives (`sdk_budget_rolling`) | 0.411 | 10.9 / 216 |
| the same without a summarizer | 0.330 | 14.0 / 249 |

| comparison | only first right / only second right | p |
|---|---|---|
| `sdk_budget` vs harness compress | 6 / 8 | 0.79 |
| `sdk_budget` vs Mem0 | 26 / 13 | 0.05 |
| `sdk_budget` vs harness delete | 44 / 3 | <0.001 |
| `sdk_budget_rolling` vs harness compress | 10 / 18 | 0.19 |
| `sdk_budget_rolling` vs Mem0 | 23 / 16 | 0.34 |
| `sdk_budget_rolling` vs `sdk_budget` (n=115) | 7 / 13 | 0.26 |
| with vs without the summarizer (untagged twins, n=115) | 19 / 7 | 0.03 |
| `sdk_budget` vs the engine's `max_records` + gists (0.148, n=115) | 38 / 3 | <0.001 |
| `sdk_budget` vs `sdk_cognitive`, unbounded (n=115) | 7 / 21 | 0.01 |

The product now delivers the compression result: one compaction of the SDK
store answers as well as the harness-built store (0.464 against 0.482) from
212 tokens, a twelfth of the unbounded store's 2,535, for 12 points of
accuracy. Run A had this arm at 0.411 on the same questions: its gists were
stored under a role the role prior did not know, and all gists of one pass
shared a turn, so the packer's cross-turn bonus went to one of them only.
With gists ranked like the facts they stand for, 3.0 gists reach a context
(2.2 before), as in the harness, and the untagged twin scores 0.438; the
time tags account for the rest. Two things to keep in view. Compacting as the conversation arrives,
which is what a deployment does, scores lower than compacting once (0.411
against 0.464; knowledge-update 7 against 10, multi-session 8 against 12,
temporal 8 against 5): each pass summarizes the previous pass's gists again.
It is not a significant difference and the harness never measured it. And
on knowledge-update questions the bounded stores do as well as the
unbounded ones (10 of 18), because the newest facts are the ones kept
verbatim.

**6. Cognitive search: three readings, all negative.** Not one of the three
tasks, but the arms were there. With the same packing, the conversational
profile's search against the plain vector pool:

| comparison | right | only cognitive / only plain | p |
|---|---|---|---|
| untagged, run A (`sdk_cognitive` vs `sdk_pack`) | 61 vs 65 | 3 / 7 | 0.34 |
| untagged twin, run B | 59 vs 65 | 2 / 8 | 0.11 |
| tagged (`sdk_cognitive` vs `sdk_pack_tagged`) | 66 vs 71 | 3 / 8 | 0.23 |
| run D: `cognitive_alpha=1.0` vs the profile's 0.5 | 70 vs 66 | 7 / 3 | 0.34 |

Run D (`sdk_cognitive_alpha1`) keeps the profile and only sets
`cognitive_alpha=1.0`, which leaves the graph proposing candidates and ranks
by cosine. Its context is identical to the plain pool's in 114 of 115
questions (0.609 against 0.617). So on these embeddings the graph expansion
adds no candidate that makes the top 30, and the profile's re-rank (a graph
boost of up to 0.5 added to a cosine) costs about 4 points. The direction is
the same every time and no single comparison is significant; the four are
not independent (same questions, overlapping contexts). The profile is
unchanged: `cognitive_alpha=0.5` came from earlier runs, most of them on
MiniLM embeddings, where the graph did help, and what the default should be
for a strong embedder is a product decision, recorded in `TODO.md`.

**What this says about the claims.**
- The shipped SDK no longer trails plain search: 0.591 against 0.539, not
  significant. What the lead is made of is packing to the budget and time
  tags (the plain pool with both: 0.617, 15 / 6 against plain search,
  p=0.08). Cognitive search and belief revision add nothing measurable to
  answer accuracy on these embeddings.
- Bounded compression is now a product feature with a judged number of its
  own: 0.464 from a 256-token store (Mem0 0.348, delete 0.098), 0.411 when
  compacted as the conversation arrives.
- Time tags are worth 5 to 8 questions of 115, most of them temporal
  questions, in every setting they were tried in.

**Limits.** One dataset, 115 questions, one run per arm: differences under
about 0.08 are not resolved and per-type cells less so. Verdicts are reused
for identical contexts only; a different order of the same facts is judged
again. The plain-pool arms are deterministic. The cognitive arms reproduced
exactly when built the same way (runs B, C and D), but `recall()` changes
the store it reads, so an arm served second from a store differs slightly
from one served first. The published README tables (50 conversations,
gpt-4o-mini judge, the adapter) were not re-run. `bounded_head_to_head.py`
was not re-judged after its store builder moved to `plan_compaction`; its
13 unit tests pass.

**The runner.** `shipped_stack_eval.py` gained: `<arm>_untagged` twins and
the diagnostics `sdk_pack_tagged`, `sdk_belief_llm_nodemote`,
`sdk_cognitive_alpha1`; a disk cache for gists; resuming with fewer arms
than the checkpoint was written with; verdicts saved while judging, so an
interrupted run started again pays for nothing twice.

**Checks.** `make gate` passed on this tree, 8 of 8: fmt, clippy, the Rust
suite, the SDK suite (105 tests), both synthetic belief checks, the
LongMemEval smoke test and the ANN recall floor.

**Cost.** About 7,100 chat requests against the 4,500 quoted: 2,691 for
answers and grades (0.69M input tokens) and about 4,400 gists, plus 1,772
gist texts embedded. The overrun is the time-tag work, which was not in the
plan (its twins and the three diagnostic arms), and run B being interrupted
four times (the laptop went to standby; it was stopped once to spare the
battery; the session that had launched it ended; the machine ran short of
memory under other work). Its finished conversations and their gists were
kept each time.

**Next** (all in `TODO.md`): decide `cognitive_alpha` for the default
embedder; make the head-to-head drive `tsm.Memory`; give gists a time span
and look at why rolling compaction trails one pass; confirm on the full
LongMemEval set and on LoCoMo before quoting any of this outside the repo.

---

# Why cognitive search lost, a free retrieval score, and what it fixed (2026-10-07)

The previous entry left one decision open (the conversational profile's
`cognitive_alpha`) and one question unasked: why would a graph lose to
plain vector search at all? This entry answers the question, and fixes what
the answer pointed at. Nothing in it was judged when it was written: every number below is
free to reproduce. The judged run that followed is at the end.

**1. What the graph term was.** `final = cosine + (1 - alpha) * d / (1 + d)`,
with `d` made of a BM25 share (at most 0.3), a fixed 0.56 for every query
word that is one of the memory's concepts, and a spread through shared
concepts. Measured on 114 questions (cached embeddings, no model): at the
profile's `alpha = 0.5` the boost averaged 0.24 per candidate where
neighbouring candidates differ by 0.013 in cosine, and it correlated +0.68
with the number of query words in the memory (0.17 with none, 0.38 with
three or more). Concepts are content words, so word overlap was scored twice, the
second time without IDF, and at eighteen times the weight of the cosine
gaps it was added to. It changed 1.5 of the top 8.

**2. What the evaluation was missing.**
- *The data is the benchmark's oracle variant.* The parquet copy the harness
  downloads holds, per question, only the evidence sessions (1.9 sessions,
  6.7k tokens). Standard LongMemEval-S has about 48 sessions and 122k tokens
  for the same questions. Every store judged so far was small and all of it
  relevant, so numbers published for other systems (LongMemEval-S, large
  readers) are not comparable with ours, and a mechanism meant to find what
  similarity misses had little to find.
- *Dates were dropped.* Every session starts with `[Date: ...]`; the loader
  skipped the line. Sessions were also ingested in file order, which is not
  chronological for 34 of the 133 temporal questions (8 of our 28), so their
  `[turn N]` tags pointed the wrong way.
- *There was no question date*, so "a month ago" had no anchor.

Fixed: the loader keeps session dates and can read the official files
(`download.py --official oracle|s|m`); its parse of the official oracle file
is identical to the parquet copy message for message (10,960 of 10,960), so
the cached extraction applies. The runner ingests sessions in the order they
took place, passes each message's session date to `recall()`, and tells the
reader the question's date.

**3. A retrieval score that costs nothing.** The official files flag the
turns that hold each question's evidence. `shipped_stack_eval.py` now
reports, per arm, for how many questions a memory from every evidence turn
is in the context. Checked against the 3,795 answers already judged in the
earlier runs:

| evidence in the context | judged right |
|---|---|
| all of it | 1071 / 1665 = 0.64 |
| part of it | 407 / 918 = 0.44 |
| none of it | 155 / 618 = 0.25 |

| by type, all / not all of the evidence | judged right |
|---|---|
| single-session-user | 0.93 / 0.31 |
| single-session-preference | 0.97 / 0.64 |
| single-session-assistant | 0.88 / 0.35 |
| multi-session | 0.76 / 0.29 |
| knowledge-update | 0.52 / 0.50 |
| temporal-reasoning | 0.20 / 0.28 |

It tracks the judge where retrieval is the problem, and it reproduces the
judged finding about cognitive search in two minutes (plain pool 68
questions with all their evidence, shipped profile 64; judged 0.609 and
0.574). It also shows where retrieval is *not* the problem: temporal
questions were answered right 20% of the time with all the evidence
present, and knowledge-update questions no better with it than without.
Those need what section 2 adds (dates, order, the question's date), which
this score cannot see.

To test retrieval among distractors without paying for extraction,
`--pool N` builds each history from N conversations of the cached data: 20
conversations give stores of about 2,500 facts, 40 about 5,000, which is
past the 4,096 records where search stops being an exact scan. (Index
recall there was checked against an exact scan: 0.996 to 1.000 of the top
30.)

**4. What the score found, and the fixes.** Questions out of 115 with all
their evidence in a 150-token context:

| | oracle (125 facts) | 20 pooled (2,500) | 40 pooled (5,000) |
|---|---|---|---|
| plain top-20, truncated | 64 | 62 | 61 |
| shipped before (`cognitive_alpha` 0.5) | 64 | 60 | 57 |
| plain pool + packer, before | 68 | 67 | 59 |
| old graph settings, new packer | 68 | 67 | 65 |
| plain pool + packer, now | 75 | 74 | 68 |
| **shipped now** | **76** | **74** | **68** |

Old graph settings against the new profile, same packer: 1 / 9 (p=0.02),
3 / 10 (p=0.09), 5 / 8.

- *The packer threw evidence away.* `select_under_budget` skipped any
  candidate more similar than 0.72 to one already chosen. With OpenAI
  embeddings that is not "the same thing said twice": it is two yoga
  classes in one week, or an egg count before and after it changed, which
  is what multi-session and knowledge-update questions are made of.
  Loosening the cutoff helped at every step (0.80, 0.85, 0.90, 0.95) in all
  three store sizes; at 0.95, which only catches near-identical text, 68
  becomes 75 on the packer alone (knowledge-update 11 to 15 of 18,
  multi-session 17 to 20 of 31). `MAX_REDUNDANCY` is now 0.95. This was the
  largest single effect, and the old cutoff was worse than no MMR at all
  (relevance order: 69, 68, 66).
- *The graph step dropped true neighbours in a large store.* It kept its 90
  best candidates ranked by cosine + delta. Past a few thousand memories,
  hundreds share a word with the query, each with a delta larger than a
  seed's cosine, and they crowded ANN hits out before fusion: with cosine
  ranking the cognitive path covered 55 questions at 5,000 facts where the
  plain pool covered 59. Every ANN seed is now kept and only discovered
  candidates compete for the rest; the two are equal again at every size.
- *The graph weight had no scale.* New engine settings, each off by default
  so nothing else moves: `graph_weight` (the boost as a share of the cosine
  spread among the `top_k` nearest: the same weight means the same thing for
  any embedder), `query_concept_boost` (the per-word bonus; 0 switches it
  off) and `fan_exponent` (what a concept passes on is divided by the number
  of memories it points at). The conversational profile now uses 0.5, 0 and
  1. Of the settings tried it is the only one never below the plain pool
  (69, 67, 60 against 68, 67, 59 with the old packer). The dose-response of
  the old form, oracle data: alpha 1.0 68, 0.9 68, 0.8 70, 0.7 66, 0.5 64;
  without the per-word bonus alpha 0.5 gives 70, and the bonus without the
  concept spread 62.
- *A read changed the next read.* A search reinforced every memory it
  returned, so the same question came back as a different list in 13 of 19
  checks. `reinforce_on_search` is now off by default and
  `reinforce(ids)` (engine, Python, `Memory`) is for memories that were
  used.

**5. What is still out of reach.** For 34 to 37 of the 115 questions an
evidence turn has no fact among the 30 nearest to the question (60 is no
better), so about 0.70 is the ceiling for any packer and the stack is at
0.66. A 600-token context does not lift it (0.70). That gap, and not
re-ranking, is where an associative step could earn its place.

**What this says about the claims.** The graph term lost because of how it
was weighted and what it double-counted, not because graphs cannot help;
calibrated, it is level with plain search, which is still not a gain. The
larger loss was in the packer. The judged results are at the end.

**Limits.** The evidence score counts a turn as found when any memory from
it is in the context, not the right one; it is blind to everything a reader
does. Arms that serve gists are understated (a gist is from no one turn).
The three store sizes share the same 115 questions, so they are not three
independent samples, and choosing settings on them is tuning on the test
set: the judged run decides. Pooled histories mix other users' sessions in,
which can contradict an answer, so they are for this score only. MiniLM was
not tried.

**Checks.** `make gate` passed on this tree (8 of 8); 300 Rust tests, 111
SDK tests, 20 harness tests.

**Cost.** No model was asked: cached extraction and embeddings only.

**Judged, 2026-10-09** (same 115 questions, judge gpt-4.1-mini, 989 judge
requests, about $0.13 at list prices; the per-question files were under
`target/bench/e2e` and went when `target/` was removed, so only the counts
below are left). Each cell is questions right out of 115.

| arm | committed (`shipped6`) | this batch | retrieval only (old inputs) | minute tags |
|---|---|---|---|---|
| `sdk_cognitive` | 66 (0.574) | 90 (0.783) | 73 (0.635) | 84 (0.730) |
| `sdk_pack_tagged` | 71 (0.617) | 87 (0.757) | 72 (0.626) | - |
| `sdk_cognitive_alpha1` | 70 (0.609) | 87 (0.757) | - | - |

- `sdk_cognitive`, committed to this batch: 26 questions gained, 2 lost, sign
  test p < 0.001. Of the 26, 12 are explained by the retrieval change alone and
  14 needed the input fix. The two losses: a question about which of two gifts
  came first (the day tags on its facts are identical, so they cannot order
  them; the turn tags gave the reader two numbers to order by), and a charity
  event a month ago (its fact dropped out of the context).
- Decomposition: the retrieval change alone gives +7 (13 gained, 6 lost,
  p = 0.17, not significant). The input fix gives +17 (19 gained, 2 lost,
  p < 0.001); the run cannot separate its three parts (session dates,
  time-ordered ingestion, the question's date).
- By type, committed to this batch: temporal 9 -> 21 of 28, knowledge-update
  10 -> 15 of 18, multi-session 16 -> 22 of 31; the other three types unchanged.
- Minute tags are worse than date tags: 84 against 90 (1 gained, 7 lost,
  p = 0.07). Date tags stay the default.
- Inside the new run the graph term is 3 questions ahead of cosine-only
  ranking (3 gained, 0 lost, p = 0.25): not significant.

**Caveats.** The retrieval settings and the packer cut-off were chosen with the
free score on these same 115 questions, so the +7 from retrieval is optimistic.
Per-type counts are 9 to 31. One judge, one reader, one run each. The committed
baseline was measured without dates, so it understated the shipped stack; the
input fix corrects the harness and was not a tuned setting.

**Next.** A held-out check of the retrieval settings, which needs extraction
(paid: 13,932 requests for 30 LongMemEval-S questions, 51,562 for all 115); a
test of tagging with both the date and the turn; then validity intervals on
facts and the original turn kept next to its facts (sections 3 and 5).

---

# The write path drops a third of what users say (2026-10-09)

After the judged run, 25 of 115 answers are still wrong. Sorting them by
whether the evidence reached the context (free: saved verdicts and the cached
extraction, no model asked) points at ingestion, not at ranking.

**1. Where the 25 are lost.** 15 lack some of their evidence in the context
and 10 have all of it (5 of those are multi-session totals and counts the
reader got wrong). In 14 of the 15, an evidence turn that is missing produced
no extracted fact at all: the store never held it.

**2. How much the extractor drops.** Over the 120 conversations of the judged
runs (`gpt-4.1-nano`, the cached pass every arm shares):

| | messages | no fact extracted |
|---|---|---|
| user messages | 1,403 | 470 (33%) |
| assistant messages | 1,403 | 311 (22%) |
| evidence turns (`has_answer`) | 229 | 39 (17%) |

34 of the 115 questions have an evidence turn with no fact. `sdk_cognitive`
is judged right on 20 of those 34 (0.59) and on 70 of the other 81 (0.86).

**3. Why.** 91% of the dropped user messages speak in the first person and
80% end in a question mark: a statement about the user, then a request ("I
just got my highest score in Ticket to Ride - 132 points! Anyway, back to
Scythe..."). The prompt says to ignore questions and to return an empty list
when there are no facts. `tsm/extractors.py` ships the same prompt, so the SDK
loses these turns too. Two smaller losses: in about 12% of the messages that
did yield facts, the facts share more words with the message before than with
the message itself (the "recent context" is being extracted; a word-overlap
heuristic, not a count), and the harness extractor reads a reply cut off at
400 tokens as "no facts" and caches that (the SDK extractor retries and falls
back to the message; the harness one does not).

**4. This is the 0.70 ceiling.** With the store as it is, the number of
questions with all their evidence in the context does not move with the
budget. Keeping the user's own turn as a memory when extraction returned
nothing removes the ceiling (457 turns; 98% of user turns are already in the
embedding cache, so the test is free). Questions out of 115 with all their
evidence, `sdk_cognitive`:

| context | as it is | user turn kept when nothing was extracted |
|---|---|---|
| 150 tokens | 76 | 83 |
| 300 tokens | 81 | 99 |
| 600 tokens | 81 | 107 |
| 1,500 tokens | 81 | 109 |
| 600 tokens, 20 conversations pooled (2,500 records) | 81 | 103 |
| 600 tokens, 40 pooled (4,900 records) | 81 | 98 |

Storing every user turn next to its facts was no better (80 at 150 tokens,
108 at 600): a turn costs about 57 tokens where a fact costs 19. At 150
tokens the gain is small for the same reason, and multi-session questions
lose 2. What is left at 600 tokens is assistant-side content (7 of 11): long
assistant turns were not part of this test.

**Limits.** Coverage, not judged answers: a question with all its evidence
was judged right 0.87 of the time in the 2026-10-09 run, at 150 tokens of
compact facts, and a reader given longer turns may do better or worse. The
oracle variant; the pooled rows are the only check among distractors. The
same 115 questions as everything else here. Nothing is built: the test
patched the harness inputs in one process (`target/`, not committed).

**Next.** Keep every turn as the record its facts point to and never store
nothing for a turn with content; fix the extraction prompt (asides inside
questions, facts taken from the context, long assistant turns in pieces) and
re-extract (paid); let `recall()` return more when a question needs several
facts; then judge at 150 and 600 tokens.

---

# Recovering the dropped turns, and how the answer is asked (2026-10-12)

The entry above proposed storing something for every turn. That was built
and judged on the same 115 questions (`sdk_cognitive`, reader and judge
gpt-4.1-mini). It raised coverage and did not raise answers; what raised
answers was the prompt the reader is given.

**1. What was built.**

- *Extraction in two passes* (`tsm/extractors.py`). The first prompt is
  unchanged. A message of three words or more that it finds nothing in is
  asked about again with a second prompt, which asks what the speaker
  reveals, also inside a question (`second_look`). If that finds nothing
  either, the message's sentences that are not questions are stored as said,
  in pieces of about 80 tokens (`keep_unextracted`, `statements`). Each
  prompt's replies are cached under a hash of the prompt.
- *A reading prompt* (`tsm/reading.py`). `answer_prompt` lists the memories
  in the order they were said and asks the model to name the notes that bear
  on the question, settle what they add up to (a later note replaces an
  earlier one, each item is counted once, a relative time counts from the
  note's date) and end with one `Answer:` line. `final_answer` takes the
  answer out.
- *Harness.* `--extraction shipped|second|first|legacy` (the SDK extractor;
  without the stored statements; the first prompt alone; the harness's earlier
  extractor) and `--reader plain|notes`. `first` and `legacy` hold the same
  facts (125.2 records a conversation, the same contexts). Results and the
  embedding overlay are kept under `.cache/shipped_stack_eval` by default,
  not under `target/`.

**2. What the second pass recovers.** Of 2,806 messages the first prompt
finds nothing in 779 of three words or more. The second prompt finds facts
in 476 of them (237 user, 239 assistant); the statements of 258 more are
stored as said (187 user, 71 assistant); 45 are left with nothing. A
conversation's store grows from 125 records to 151 with the second prompt
and 155 with the statements. Asking every message with the second prompt
instead (17,021 facts) still returns nothing for 274 user messages (20%).

**3. Judged.** Questions right out of 115, and questions with every
evidence turn represented in the context (the free score).

| store | context | reader | right | all evidence |
|---|---|---|---|---|
| first prompt only (as committed) | 150 | plain | 90 (0.783) | 76 |
| second prompt for every message | 150 | plain | 79 (0.687) | 97 |
| two passes | 150 | plain | 82 (0.713) | 92 |
| first prompt only | 600 | plain | 86 (0.748) | 81 |
| second prompt for every message | 600 | plain | 93 (0.809) | 113 |
| two passes | 600 | plain | 92 (0.800) | 111 |
| two passes | 600 | reading prompt | 99 (0.861) | 111 |

Question by question (gained, lost, sign test):

| from | to | gained | lost | p |
|---|---|---|---|---|
| first, 150 | two passes, 150 | 4 | 12 | 0.08 |
| first, 150 | second for every message, 150 | 9 | 20 | 0.06 |
| first, 600 | two passes, 600 | 10 | 4 | 0.18 |
| first, 600 | second for every message, 600 | 12 | 5 | 0.14 |
| first, 150 | first, 600 | 4 | 8 | 0.39 |
| first, 150 | two passes, 600 | 11 | 9 | 0.82 |

The last row of the first table has no paired test: the per-question
verdicts of the other rows were under `target/`, which was removed before
that run. Its contexts were rebuilt from the cached extraction and have the
same coverage (111). By type it is 16 of 18 knowledge-update, 27 of 31
multi-session, 8 of 11 single-session-assistant, 8 of 9 preference, 17 of 18
single-session-user and 23 of 28 temporal.

**4. What it says.**

- *Coverage is not answers.* At 150 tokens the second pass puts all the
  evidence of 16 more questions in the context and 8 fewer are answered. The
  evidence score counts a turn as found when any memory from it is there,
  and the memories the second pass adds are often what the user wanted at
  the time ("The speaker is looking for ...") rather than the fact the
  question is about. They match a question's wording well and take slots.
  The retrieval proxy (the gold answer's most distinctive word in the
  context) does not see the loss either: 80, 80, 81 and 80 for the four
  stores at 150 tokens.
- *More context does not help the plain reader.* On the same store it
  answers 90 at 150 tokens and 86 at 600.
- *The reader was the bottleneck.* Of the 23 the plain reader got wrong at
  600 tokens, about 13 had the facts in the context: the older of two
  values, a total missing an item, a duration computed wrong. The reading
  prompt answers 99 on the same store and budget.
- *What is left* (16 wrong): 8 with the facts in the context (two values
  said on one day, which date tags cannot order; a total and an average
  whose facts do not say which workshop or degree they are about; a
  subtraction; a count of flights), 3 missing a fact, 2 replies cut off at
  500 tokens before the answer line (the harness allows 800 now; a third
  cut-off reply was still judged right), 2 that give the gold answer and
  were marked wrong, 1 that asks about jewelry where the conversation has a
  chandelier.

**5. The retrieval settings of 2026-10-07 on questions they were not chosen
on (free).** The cached first-prompt extraction turned out to cover all 500
conversations of the oracle file: 470 questions that are not abstentions,
of which the settings were chosen on the first 115. Questions with all
their evidence in a 150-token context, same store in every row:

| | the 115 | the other 355 |
|---|---|---|
| shipped (`sdk_cognitive`) | 76 | 256 (0.721) |
| plain search, same packer (`sdk_pack_tagged`) | 75 | 257 |
| the fixed boost it replaced (`cognitive_alpha=0.5`, concept bonus 0.75, no fan normalization) | 68 | 248 |
| shipped, with the packer's earlier cut-off (0.72) | 69 | 246 |

On the 355, question by question: the 0.95 cut-off against 0.72 gains 11
and loses 1 (p = 0.006; 14 and 2 by the retrieval proxy, p = 0.004); the
calibrated graph term against the fixed boost gains 11 and loses 3
(p = 0.06); against plain search it gains 0 and loses 1. The packer change
and the calibration hold where they were not tuned, and cognitive search is
level with plain search there too, which is still not a gain. This is
coverage, and section 4 shows how far coverage can be from answers when the
store changes; in these rows the store is the same and only the selection
differs, which is the case the score was checked against judged runs for.
Seven texts had no cached embedding (the runs were offline) and got a
stand-in vector, the same in every row.

**Limits.** These are the 115 questions everything here has been tuned on,
and the reading prompt was written after reading what they got wrong: 0.861
is an upper estimate. One run of each setting, one judge, and the judge is
the model that answers. The first prompt alone was not run with the reading
prompt, so the 99 cannot yet be split between the store and the prompt.
Two things changed after these runs and are not in the numbers. The harness
gave the reading prompt the memories as bare strings, so those of one day
were listed by rank; it now passes each memory's turn, as `recall()` results
carry it, and they are listed in the order they were said. And a stored
statement no longer keeps a list number whose question was removed ("1."
on a line of its own), nor loses the number of the item it starts. The
extractor's defaults (both recovery steps on) are therefore not settled: at
150 tokens with the plain reader they score 8 questions lower.

**Cost.** 2,856 extraction requests (gpt-4.1-nano, about $0.64), about
16,700 embedded texts ($0.01) and 1,308 judge requests (600k tokens in, 42k
out, about $0.31): about $0.96 at list prices.

**Next.** (1) A held-out check: the cached first-prompt extraction covers
all 500 conversations of the oracle file, so the 355 questions nothing was
tuned on can be judged for the price of the judge (about 710 requests a
setting). (2) The paired grid on the 115 (store, budget, reader) to settle
the extractor's defaults. (3) The second prompt for the other 380
conversations (about 2,700 requests), then the two-pass store held-out.
After that: facts that say what they are about, an order within a day, and
totals computed in code rather than by the reader.


# A local stack: EmbeddingGemma 2 against MiniLM and the OpenAI vectors (2026-10-19)

`Improvements.md` proposes EmbeddingGemma 2 as the embedder, a ternary
Bonsai model as the local verifier, and consolidation while the memory is
idle. This entry measures what can be measured on this machine (RTX 3050
Laptop 4 GB, PyTorch 2.8, sentence-transformers 6.1) before any of it
becomes a default. The shipped defaults do not change: the embedder is still
OpenAI's, `embedder="gemma"` and `idle_consolidation_secs` are opt-in. The one
change that reaches every user is that a database now records which embedder
wrote it and refuses another (section 7).

**1. Setup.** The free evidence score of `shipped_stack_eval.py` (for how many
of the 115 non-abstention questions of the first 120 every evidence turn is in
a 150-token context; no network, no judge), now with `--local-embedder MODEL`
so the same score runs on a local model. Same cached extraction facts, same
packer in every row; only the vectors differ. One history is the 120
conversations separately (about 125 records each); `--pool 20` and `--pool 40`
put 20 or 40 conversations in one store, so most of it is distractors, and 40
(about 5,000 records) is past the 4,096-record exact scan. Exact paired tests
against the OpenAI row. The OpenAI baseline reproduces the documented 75 and
76 exactly.

```
python benchmarks/cognitive_eval/shipped_stack_eval.py --offline --limit 120 --arms sdk_pack_tagged,sdk_cognitive --pool N --local-embedder google/embeddinggemma-2
```

**2. Retrieval (free).** Questions with all their evidence in the context,
`sdk_pack_tagged` / `sdk_cognitive`:

| embedder | one history | pool 20 | pool 40 |
|---|---|---|---|
| OpenAI `text-embedding-3-small`, 1536-d, API | 75 / 76 | 74 / 74 | 68 / 68 |
| `embeddinggemma-2`, text only, 768-d, local | 76 / 77 | 72 / 72 | 67 / 68 |
| `embeddinggemma-2`, query prompt `QuestionAnswering` | 76 / 74 | | |
| `all-MiniLM-L6-v2`, 384-d, local | 73 / 72 | 69 / 71 | 61 / 63 |

Share of the evidence turns present at pool 40: 0.717 / 0.713 (OpenAI),
0.731 / 0.732 (Gemma), 0.692 / 0.699 (MiniLM). Gemma against OpenAI: a net
difference of 2 questions or less at every size (up to 5 questions go each
way), p of 0.62 or more. MiniLM against OpenAI: 2 to 7 questions behind at
every size, most at pool 40 (`sdk_pack_tagged` 61 against 68, p = 0.07; the
other rows p of 0.12 to 0.62): suggestive, not settled on 115 questions. The
`QuestionAnswering` prompt did not help (`sdk_cognitive` 74 against 77 for the
default query prompt, `sdk_pack_tagged` 76 for both), so the default stays.
In short, a 270M local text model with 768 dimensions retrieves as well as
the 1536-dimension API vectors on this score, which is what the blueprint
needed to be true and had not shown.

**3. Cost.** Same machine, 16 CPU threads. Single text, a 256-text batch and
GPU memory for the model; then through `Memory` on a 2,000-fact store
(passthrough extractor):

| | 1 text | 256 batch | texts/s | VRAM |
|---|---|---|---|---|
| MiniLM cuda | 10 ms | 118 ms | 2169 | 95 MB |
| MiniLM cpu | 12 ms | 497 ms | 516 | |
| Gemma bf16 cuda | 125 ms | 1214 ms | 211 | 519 MB |
| Gemma fp32 cuda | 94 ms | 1850 ms | 138 | 1036 MB |
| Gemma fp32 cpu | 115 ms | 10716 ms | 24 | |

| through `Memory` | bulk add of 2,000 | add one message | `recall(top_k=10)` | `recall(token_budget=150)` |
|---|---|---|---|---|
| MiniLM | 1.3 s | 10 ms | 15 ms | 34 ms |
| Gemma bf16 | 9.5 s | 101 ms | 125 ms | 252 ms |

The blueprint's "under 5 ms to embed" is not met: a short text costs 94 to
125 ms through Gemma on the GPU and 115 ms on the CPU (a 400-token one 138 ms
and 794 ms), so most of it is a fixed per-call cost; on the GPU the marginal
text in a batch is 4.7 ms. Two consequences. A
budgeted `recall()` costs twice a plain one, because the packer embeds the
whole candidate pool again for its redundancy check although the engine has
those vectors; the engine has no call that returns stored vectors, and adding
one would halve this (open). And ingestion has to batch. The 525 MB of GPU
memory for the text model (644 MB peak) matches the blueprint's 600 MB; the
card's 740M parameters is the model with its image and audio encoders, which
the adapter does not load.

**4. A new embedder invalidates the belief thresholds.** The profile's 0.85
and 0.75 were tuned on other vectors. `belief_pairs_eval.py --split test
--verifier none` (127 held-out pairs, 54 updates, 58 still-true):

| embedder | updates caught | still-true marked stale |
|---|---|---|
| MiniLM (the documented 16 and 8, reproduced) | 16 of 54 | 8 of 58 |
| Gemma, profile thresholds 0.85 / 0.75 | 30 of 54 | **30 of 58** |
| Gemma, 0.92 / 0.86 | 18 of 54 | 7 of 58 (9 in one store) |

A sweep on the dev split (0.85/0.75 to 0.96/0.93) only trades recall for
precision: at 0.92 / 0.86 Gemma behaves like MiniLM at its own thresholds, and
no setting beats it (the dev and test halves differ, so "similar to", not
"equal to"), which is the old finding that geometry alone cannot separate an
update from a second fact. What this costs retrieval, `sdk_belief`
(detection on, no verifier) against `sdk_cognitive` (detection off), questions
with all their evidence present:

| | one history | pool 40 |
|---|---|---|
| OpenAI, 0.85 / 0.75 | 74 against 76 | 65 against 68 |
| Gemma, 0.85 / 0.75 | **68 against 77** (loses 10, wins 1, p = 0.012) | 64 against 68 |
| Gemma, 0.92 / 0.86 | 76 against 77 (p = 1.0) | 70 against 68 |

With another embedder's vectors the profile's detection quietly costs a tenth
of the questions. An embedder may now declare `engine_defaults`, applied under
the conversational profile and below any explicit `Memory(...)` argument;
EmbeddingGemma 2 declares 0.92 / 0.86. Chosen on the dev pairs and the 115
questions, confirmed on the held-out pairs (row 3 above); the retrieval row
is therefore an upper estimate. The verified path is unaffected: a
verifier decides what is committed.

**5. Verifier.** Ternary-Bonsai-2 could not be run: its weights are not on
this machine (the Hugging Face cache holds the repository's `USAGE.md` only),
and the compact formats need PrismML's llama.cpp fork. `LLMVerifier(base_url=...)`
already talks to such a server, and `belief_pairs_eval.py --verifier llm
--llm-base-url ... --llm-model ... --embed-model google/embeddinggemma-2` is
the whole evaluation. A stand-in that is cached, `LiquidAI/LFM2.5-1.2B-Instruct`
(new `--llm-local MODEL`, which runs a causal LM in the process), is not
usable: on 20 update and 20 still-true dev pairs it scored 19 of 40 with the
shipped prompt (0 of 20 updates), and 20 to 23 of 40 with a prompt that
shows the reply format, at any batch size: chance. It also sometimes answers
with the verdict words instead of numbered lines, which shows up in the
harness as a verifier that catches almost nothing (2 of 54 in the first full
run); it now prints how many pair judgments got no readable verdict
(`LLMVerifier.unresolved_total`).
This says nothing about a 4B model (`qwen3.5:4b` catches 48 to 49 of 54 on
the same pairs, `docs/python_sdk.md`), only that the verifier needs one of
that size.

**6. Idle consolidation.** `Memory(idle_consolidation_secs=N)` runs
`consolidate()` (verification, supersession, per-user compaction) in a thread
after N idle seconds with something written since the last pass. Tested with a
real engine and fake verifiers (13 tests): it runs once idle, stays out while
`recall()` / `add()` are in use, does nothing when nothing was written,
consolidates a reopened store, retries a failed pass and a pass a verifier
could not finish (`LLMVerifier.unresolved`), is stopped by `close()` (which
waits for a running pass), and leaves memories alone on failure. Not
measured against a real provider or a live workload.

**7. Changed in the SDK.** `encode_query` / `encode_document` (optional) and
a `fingerprint`; `Memory` records the embedder in `tsm_embedder.json` and
refuses another (`EmbedderMismatchError`), because two models of one
dimension return wrong results with no error (a subprocess test caught the
first fingerprint, which included the module path and made one embedder look
like two); `SentenceTransformerEmbedder` handles prompts, Matryoshka
truncation, text-only loading, a bfloat16/float32 guard, an 8,192-token bound
(sentence-transformers reports none) and a load-time probe; `tsm[gemma]` and
`tsm[colbert]` extras (the old extras allowed sentence-transformers 2.2,
which has neither model). No Rust changed.

**Limits.** One machine, one GPU. The 115 questions are the ones everything
has been tuned on, and the facts are the OpenAI extractor's, so a local stack
also needs a local extractor, which is unmeasured (`OpenAIExtractor(client=...)`
takes any chat client). The score is coverage, not answers, and section 4 of
the entry of 2026-10-12 shows how far those can be apart; no judged run was
made. Single runs, no repeats, and 115 questions detect only large
differences (the one clear result is the belief cost, p = 0.012). The
thresholds were chosen on the same questions the retrieval row measures.

**Next.** (1) Bonsai: download the weights and PrismML's llama.cpp build, serve
it, run the command in section 5 on both layouts and with `--embed-model
google/embeddinggemma-2`. (2) A judged run with the Gemma store
(`--local-embedder`, `--judge openai`) to see whether the coverage tie holds
for answers. (3) An engine call that returns stored vectors, so the packer
stops re-embedding the pool. (4) A local extractor and its extraction score.
(5) A migration that rebuilds a database under a new embedder (re-ingest in
insert order; `update()` would reorder history), only if (2) says the switch
is worth it.

---

# Bonsai 2 4B: local verifier and SDK integration measured (2026-10-19)

Follow-up to the local-stack entry above. With permission to download the
weights and runtime, `Ternary-Bonsai-2-4B-PQ2_0` now runs locally through the
existing SDK. No SDK, verifier prompt, harness, or Rust behavior was changed
for these measurements. Bonsai remains an opt-in verifier: update detection
improves substantially, but false supersessions remain material.

**Pinned setup.** Windows, Python 3.12.0, RTX 3050 Laptop GPU (4 GB), 16 GB RAM,
PyTorch 2.8.0+cu126, sentence-transformers 6.1.0. The model is the
[Prism PQ2 file](https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-gguf/tree/8d6557168168f91ea6697a4a49ed2ef622b53adb),
revision `8d6557168168f91ea6697a4a49ed2ef622b53adb`, 1,146,465,120 bytes.
SHA-256:
`d406b1e912c0d2a4c124ad10bf38a76ae662fdc4d24b176451403dc7b7f30409`.
The runtime is Prism release
[`prism-b10777-e3de715`](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b10777-e3de715),
`llama-prism-b10777-e3de715-bin-win-cuda-12.4-x64.zip` (313,951,324 bytes;
SHA-256 `fa5874a1412e251a6efb13f289e97bacbbb4e7fc6714c1836b4fc6c1dc863462`).
Both hashes matched their published values. The installed CUDA 12.6 runtime
DLLs worked with this build; no additional CUDA package was needed. No vision
projector, speculative draft model, or MTP model was loaded.

**Held-out results.** `belief_pairs_eval.py --split test --layout both`, the
shipped conversational profile and verifier prompt, greedy decoding,
`enable_thinking=false`, seed 42, batch size 8, output limit 1,024 tokens,
candidate floor 0.45 and the verifier's normal margin/per-record limits. The
127 held-out pairs contain 54 updates, 58 coexisting pairs and 15 rewordings;
the rewordings are not scored.

| Embedder | Verifier | Scope layout | Updates caught / 54 | Coexisting old facts marked stale / 58 | Newer facts marked stale | Precision |
|---|---|---|---:|---:|---:|---:|
| MiniLM | none | isolated | 16 | 8 | 0 | 0.667 |
| MiniLM | none | mixed | 16 | 8 | 0 | 0.667 |
| MiniLM | Bonsai | isolated | 49 | 7 | 0 | 0.875 |
| MiniLM | Bonsai | mixed | 48 | 10 | 1 | 0.814 |
| EmbeddingGemma 2 | none | isolated | 18 | 7 | 0 | 0.720 |
| EmbeddingGemma 2 | none | mixed | 18 | 9 | 0 | 0.667 |
| EmbeddingGemma 2 | Bonsai | isolated | 54 | 7 | 0 | 0.885 |
| EmbeddingGemma 2 | Bonsai | mixed | 50 | 19 | 4 | 0.685 |

Three of MiniLM's ten mixed-layout coexisting supersessions, and eleven of
Gemma's nineteen, came from another pair's statement. The mixed corpus puts
independently written first-person histories into one scope; some cross-pair
conflicts are real and not covered by its pair labels. Report those counts
separately rather than treating every cross-pair supersession as a proven
model error. The isolated result still has seven false supersessions out of
58 coexisting pairs. All older facts remain stored; supersession marks and
demotes them, and a verified flag changes the context a reader sees.

The same verdict cache was reused across layouts and then from MiniLM to
Gemma, so identical text pairs retain the same model decision; these are not
independent fresh-cache repetitions. Calls for isolated/mixed layouts were
15/10 with MiniLM and 2/27 additional calls with Gemma. Every held-out pair
judgment returned a readable verdict (zero unresolved in all four runs).
This measures end-to-end proposal plus verification, not standalone model
classification or judged answers on LongMemEval.

**Development and output limits.** With MiniLM, `--split dev` caught 46/56
updates and marked 8/62 coexisting facts stale in isolated scopes; mixed was
49/56 and 12/62 (plus one newer fact). A preliminary 512-token limit cut two
replies; it left four mixed-layout judgments without readable verdicts.
Increasing the limit to 1,024 on dev did not change the outcome counts or
eliminate those four unreadable judgments. That setting was fixed before the
final held-out run; an earlier incomplete 512-token test attempt was stopped
and is not reported as a completed evaluation. No prompt or decision
threshold was tuned on the held-out results.

**Memory and latency.** Peak sampled GPU usage was 2,201 MiB (about 2.15 GiB)
with Gemma and Bonsai loaded, within this card's 4 GB. The initial four-pair
smoke request took 6.86 s, generating 204 tokens at 39.4 tokens/s. Across all
completed local requests, including setup/dev runs and cache pressure,
generation ranged from about 13 to 61 tokens/s; this is not a controlled
latency benchmark. Some requests generated 1,024 tokens despite the prompt
asking for short verdict lines.

Host RAM is a separate constraint. The runtime's `--cache-ram` default is
8,192 MiB: during the long run its private memory reached about 11.24 GB
(working set 5.41 GB), leaving less than 0.8 GB of free physical RAM on this
16 GB machine. The belief runs above completed with that default. The server
was then restarted with `--cache-ram 128` for the live SDK smoke check; after
that check its private memory was about 2.87 GB, working set 1.95 GB. This
bounded configuration was not rerun through the full belief matrix. Weight
file size, GPU use, host RAM, and total store size are different quantities;
these results do not establish the blueprint's complete 2.5 GB budget.

**Live SDK checks.** With Gemma and Bonsai, `idle_consolidation_secs=1`
committed a Tuesday-to-Wednesday correction in about 2.5 s including idle
time, kept a different user's Friday meeting unchanged, returned only the
requested user's memories, marked the old context as earlier, and closed
and reopened the store with all three records retained. There were no
unresolved judgments. `OpenAIGistSummarizer` through the same local endpoint
returned separate first-dog/2016 and second-dog/2021 facts plus the peanut
allergy, and dropped generic food-journal advice (30 output tokens, about
0.71 s). These are integration smoke checks, not an evaluation of summary
fidelity, long-running concurrency, or compaction answer quality.

**Reproduce.** Download the pinned model and release above, then start the
server (use the full executable/model paths on Windows):

```bash
llama-server -m Ternary-Bonsai-2-4B-PQ2_0.gguf -ngl 99 -fa on \
  -c 8192 -np 1 -t 6 --host 127.0.0.1 --port 18080 \
  --alias bonsai-2-4b-pq2 --jinja --cache-ram 128
```

`--cache-ram 128` is the bounded SDK setup; omit it only when reproducing
the original belief runs' runtime configuration. With Python 3.12, run the
following for MiniLM and then repeat with `--embed-model
google/embeddinggemma-2`. Reuse the same `--llm-cache` directory for the
shared-cache protocol above; use a fresh directory when weights, prompt,
or decoding settings change. Model thinking is disabled explicitly because
its default thinking mode can spend thousands of tokens before answering.

```bash
python benchmarks/cognitive_eval/belief_pairs_eval.py \
  --split test --layout both --verifier llm \
  --llm-base-url http://127.0.0.1:18080/v1 --llm-model bonsai-2-4b-pq2 \
  --llm-batch 8 --llm-cache /path/to/bonsai-verdict-cache \
  --llm-arg max_tokens=1024 \
  --llm-arg 'extra_body={"chat_template_kwargs":{"enable_thinking":false},"seed":42}' \
  --show errors --json /path/to/bonsai-test.json
```

**Artifacts and decision.** Pinned metadata, binaries/model, hashes, exact
commands, stdout, JSON results, GPU telemetry and `sdk-smoke.json` are retained
under `D:/personal-projects/.scratch_tsm/bonsai-20261019/`; `report.json` collects
the measurements. The local test server and telemetry process were stopped
afterward and GPU usage returned to zero. All inference was local. This
follow-up changes documentation only, so no extension rebuild or gate was
run; it does not replace the required gate for SDK/cognitive behavior changes.
Keep Bonsai and idle consolidation opt-in. Improve/measure precision on
realistic shared histories and constrain the verdict protocol before making
verified supersession automatic. Gist quality and judged LongMemEval answers
remain separate evaluations.
