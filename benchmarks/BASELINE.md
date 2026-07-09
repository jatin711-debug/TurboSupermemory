# Cognitive Layer — Frozen Baseline (2026-06-29)

This is the **reference point** for the production-grade hardening effort. Every later
change (confidence-scaled fusion, consolidation-time belief revision, etc.) is measured
against the numbers below. A change ships only if it improves — or holds — cognition's
**marginal lift over plain ANN** without regressing recall@10.

Captured immediately after the "pure graph delta" fusion fix (the no-op bug fix), on a
fresh `cargo build --release --package turbomemory_python`.

---

## How to reproduce

```bash
# Build + copy extension (Windows)
cargo build --release --package turbomemory_python
copy /Y target\release\turbomemory.dll turbomemory.pyd

# Toy regime (near-orthogonal vectors, no distractors)
python benchmarks/cognitive_benchmark.py --dimension 64 --distractors 0

# Realistic scale (clustered 768-dim embeddings, 1000 distractors/scenario)
python benchmarks/cognitive_benchmark.py
```

Config defaults at capture time: `cognitive_alpha = 0.7`, `semantic_alpha = 1.0`,
`lexical_alpha = 0.3`, bounded augmenter `iterations = 1`, fusion
`final = cosine + (1 - cognitive_alpha) * normalized_graph_delta`.

---

## `cognitive_benchmark.py` results

The benchmark reports two columns that matter:
- **CogON won** — did `search()` (cognitive) beat `search_ann()` (plain ANN)?
- **Feature helps** — did CogON beat **CogOFF** (same path, the *specific* feature disabled)?
  This is the column that isolates whether the *mechanism under test* (Refines edge,
  reinforcement, Contradicts edge, abstraction edge) actually did the work.

### Toy regime (`--dimension 64 --distractors 0`)

| Scenario | CogON won | CogOFF won | **Feature helps** |
|---|:--:|:--:|:--:|
| Abstraction traversal | YES | YES | **no** |
| Refinement surfacing | no | no | **no** |
| Reinforcement boosting | no | no | **no** |
| Contradiction surfacing | YES | no | **YES** |

**Won: 2/4. Feature-attributable: 1/4 (contradiction only).**

### Realistic scale (default: 768-dim, 1000 distractors)

| Scenario | CogON won | CogOFF won | **Feature helps** | Notes |
|---|:--:|:--:|:--:|---|
| Abstraction traversal | no | no | **no** | target never enters top-k; cosine-near distractors dominate |
| Refinement surfacing | YES | YES | **no** | new_fact: ANN rank 99 → Cog rank 2 |
| Reinforcement boosting | YES | YES | **no** | mem_a: ANN rank 99 → Cog rank 2 |
| Contradiction surfacing | YES | YES | **no** | new_correction: ANN rank 99 → Cog rank 2 |

**Won: 3/4. Feature-attributable: 0/4.**

---

## Honest readout (the reason Phase 1 exists)

1. **Cognition (the bounded augmenter as a whole) genuinely beats plain ANN at scale** —
   3/4 scenarios go from rank 99 (missed entirely) to rank 2. That part is real and valuable.

2. **But almost none of those wins are attributable to the specific cognitive mechanism
   being tested.** In every realistic win, **CogOFF wins too** — the target is surfaced by
   the *generic* BM25 lexical boost + Association/concept expansion (the target shares
   concepts/keywords with the query), not by the Refines/reinforcement/Contradicts edge.
   The only isolated, feature-attributable win in the entire matrix is **contradiction in
   the toy regime**.

3. **Abstraction-at-scale loses outright** — the target never enters the top-k; a flat
   additive delta cannot overcome the growing cosine gap to near-by distractors. This is a
   structural ceiling of additive fusion, not a tuning miss (motivates Phase 2:
   confidence-scaled fusion).

4. **The 4-scenario benchmark is a wiring test, not a validation strategy.** It cannot
   distinguish "the graph helped" from "good lexical recall would have helped anyway."
   Phase 1 builds an eval where plain ANN (and generic lexical expansion) *demonstrably
   fail*, so the marginal value of each cognitive mechanism is measurable and falsifiable.

### Known staleness to clean up (Phase 2)
- `cognitive_benchmark.py` still passes `spreading_iterations=6` / `spreading_decay` in
  the contradiction config. These are **dead kwargs** under the bounded augmenter
  (`iterations` defaults to 1, multi-iteration spreading was removed from the hot path).

---

## Industry benchmarks (NOT re-run this session)

`LongMemEval` and `LoCoMo` require downloaded datasets and long runtimes; they were not
re-executed for this baseline. Last recorded numbers (from README, pre-hardening):

| Benchmark | Last recorded |
|---|---|
| LongMemEval (quick) | 100% recall@10 |
| LoCoMo-MC10 | infrastructure validated only |

> Caveat already noted in the design log: LongMemEval/LoCoMo mainly reward recall@k and
> under-exercise reinforcement / contradiction / abstraction, so they push tuning toward
> "just good ANN." They are necessary regression guards, not proof of cognitive value.

---

## Decision gate (applies to every later phase)

Re-run: **Phase 1 belief-revision eval** + `cognitive_benchmark.py` + LongMemEval/LoCoMo
quick. Ship a change only if it improves (or holds) cognition's marginal lift **without**
regressing recall@10.
