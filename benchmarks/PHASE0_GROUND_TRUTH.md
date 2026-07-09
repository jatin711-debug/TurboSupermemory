# Phase 0 — Ground Truth (2026-07-09)

Measured state of the **uncommitted cognitive rework** on the `evaluation` branch
(bounded augmenter rewrite + supersession-demotion belief-revision fix), captured on a
fresh release build. This is the reference point for the four-mechanism MVP effort
(belief revision + reinforcement + abstraction + forgetting/importance), superseding the
pre-demotion numbers in [BASELINE.md](BASELINE.md) and
[cognitive_eval/BELIEF_REVISION_FINDINGS.md](cognitive_eval/BELIEF_REVISION_FINDINGS.md).

## Environment

- Rust 1.96.0, `cargo test` debug stripped (`CARGO_PROFILE_{DEV,TEST}_DEBUG=0`, LNK1102 workaround).
- Python **3.12.0** (`C:\Users\User\AppData\Local\Programs\Python\Python312`), numpy 1.26.4.
  The PyO3 extension must be built against **and run with** 3.12 — the machine's default
  `python` is 3.14, which will ABI-fail on import.
- `audit_recall.py` has no `setup_extension()`; run it with `PYTHONPATH=<repo root>` or it
  raises `ModuleNotFoundError: turbomemory`.

## Build / test / lint — GREEN

| Check | Result |
|---|---|
| `cargo test --workspace --exclude turbomemory_python` | **172 passed / 0 failed** (core 29 · graph 66 · storage 74 · crash-recovery 3) |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean (exit 0) |
| `cargo build --release -p turbomemory_python` | builds; `import turbomemory` OK (24 methods) |

The 1,600-line rework compiles, is clippy-clean, and passes the Rust suite — including the
new `supersession_demotion_*` unit tests. Code stability is fine; **result** stability is not.

## Headline: belief revision does NOT fire at scale

`belief_revision.py`, `supersession_demotion_factor = 0.4` (default), CogON vs CogOFF vs ANN.

**Calibration control** (`--dimension 64 --distractors 0 --subtlety 0.10 --gap 0 --probes 8`) — the mechanism works in the easy regime:

| arm | belief_acc | corr_recall | stale_suppr |
|---|--:|--:|--:|
| ANN | 0.00 | 1.00 | 0.00 |
| CogOFF | 0.62 | 1.00 | 0.62 |
| CogON | **1.00** | 1.00 | 1.00 |

`feature_lift = +0.38`, `vs ANN = +1.00`.

**Realistic sweep** (`--probes 24 --gap 20 --subtlety 0.25`), contradiction AND refinement, distractors 100 / 1000:

| distractors | arm | belief_acc | corr_recall | stale_suppr |
|--:|---|--:|--:|--:|
| 100 | ANN / CogOFF / CogON | 0.00 / 0.00 / 0.00 | 0.67 / 0.92 / 0.92 | 0.00 |
| 1000 | ANN / CogOFF / CogON | 0.00 / 0.00 / 0.00 | 0.42 / 0.79 / 0.79 | 0.00 |

**Mean feature lift +0.00. Identical to the pre-demotion FINDINGS.md numbers.** CogON == CogOFF
exactly; `stale_suppression = 0.00` (the stale fact is *always* rank 1).

## Root cause: detection never creates the edge (not demotion strength)

Diagnostic at 1000 distractors, contradiction mode:

```
graph_stats -> refinement_count: 0   contradiction_count: 0
[python]  edge(stale->?): []  stale_rank=1(1.211)  corr_rank=2(0.589)
[database]edge(stale->?): []  stale_rank=1(1.204)  corr_rank=2(0.612)
...
```

**Zero belief-revision edges are created at scale**, so demotion never runs. Detection
requires `cosine(stale, correction) >= contradiction_cosine_threshold` (0.5), but the
eval's `jitter` multiplies a `randn` vector whose norm grows as √dim, so `subtlety` does
NOT mean "cosine distance":

| eval `subtlety` | mean cos(stale, correction) | cos(query, stale) | cos(query, correction) |
|--:|--:|--:|--:|
| 0.10 | 0.259 | 0.768 | 0.197 |
| **0.25** (default) | **0.112** | 0.769 | 0.086 |
| 0.50 | 0.054 | 0.769 | 0.040 |

At the default `subtlety=0.25`, the "correction" sits at cosine **0.11** to the stale fact —
essentially orthogonal, far below the 0.5 gate. **No cosine-threshold detector can link the
pair**, and the correction only reaches rank 2 via lexical/concept boost (never cosine).
The fused scores confirm it: stale ≈ 1.2 (cos 0.77 + boost), correction ≈ 0.59 (cos 0.09 + boost);
even a 0.4× demotion on the stale (→0.48) would flip it IF the edge existed.

**Implication.** `FINDINGS.md`'s "cognition adds no value" verdict was measured in a regime
where the mechanism physically cannot engage — a false negative baked into the eval geometry,
not proof the idea fails. This splits Phase 1 in two:
- **(A) Fix the eval geometry.** Generate a correction with a *controlled target cosine* to
  the stale fact (same cluster, cos ~0.6–0.85, like real text embeddings), not a
  dim-exploding `randn` jitter. Only then can the eval distinguish "mechanism works" from
  "mechanism can't see the pair."
- **(B) Broaden detection** beyond a bare cosine gate — link belief pairs via shared-concept
  + temporal-adjacency + text signals, since genuine corrections can sit at moderate cosine.

## Cognitive benchmark — reproduces BASELINE.md exactly (no regression)

`cognitive_benchmark.py`.

**Scale (768-dim, 1000 distractors):** won **3/4**, feature-attributable **0/4**.
Refinement/Reinforcement/Contradiction each go ANN rank 99 → Cog rank 2 (recall win), but
CogOFF wins too (generic lexical/association), and in Contradiction the stale `old_false_claim`
is still **rank 1** (correction rank 2) — same non-revision as the belief eval. Abstraction
never enters top-k.

**Toy (64-dim, 0 distractors):** won **2/4**, feature-attributable **1/4** (contradiction only:
CogON new rank 1 / old rank 2; CogOFF old rank 1).

## Recall floor — intact

`audit_recall.py --num-items 20000 --dimension 768`:

- **ANN candidate Recall@5: 100.0%**, ANN reranked Recall@5: 100.0% (floor unaffected by the rework).
- `cognitive('memory')` Recall@5: 100.0%.
- `cognitive('')` (empty query text) Recall@5: **1.2%**, results systematically off-by-one — with
  no lexical signal the **Temporal-edge boost displaces the true nearest neighbor with its
  insertion-successor** (`mem_N` → `mem_{N+1}`). Cognitive-robustness bug for Phase 2 fusion/temporal
  calibration; does not affect the ANN floor.

## What Phase 0 establishes for the four-mechanism MVP

1. The tree is buildable, green, clippy-clean, and now committed — code stability is a solved problem.
2. **Belief revision (Contradicts/Refines + demotion): unproven at scale, root cause is detection + eval geometry, NOT demotion.** First MVP work: fix eval geometry (A) + broaden detection (B), then re-measure `feature_lift`.
3. **Abstraction: architecturally dead in retrieval.** The augmenter expands 1 hop from *memory* seeds and drops non-memory targets ([activation.rs](../crates/turbomemory_graph/src/activation.rs) target-kind filter) and explicitly skips `Abstraction` edges — so `build_abstractions()` output is never traversed. Must be re-wired (concept-seeded / multi-hop) before it can earn lift.
4. **Reinforcement: no feature-attributable lift** in any regime (CogOFF matches CogON). Needs an isolated eval.
5. **Fusion is fragile** — result-set-relative normalization + additive boost + multiplicative demotion, plus the empty-query temporal displacement above. Needs a single documented, calibrated score.

## Reproduce

```powershell
$env:PYO3_PYTHON="C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe"
cargo build --release -p turbomemory_python
Copy-Item target\release\turbomemory.dll turbomemory.pyd -Force

# belief revision (the decisive eval)
& $env:PYO3_PYTHON benchmarks\cognitive_eval\belief_revision.py --dimension 64 --distractors 0 --subtlety 0.10 --gap 0 --probes 8   # control -> CogON 1.00
& $env:PYO3_PYTHON benchmarks\cognitive_eval\belief_revision.py --mode contradiction --probes 24 --gap 20 --subtlety 0.25 --distractors 100 1000
& $env:PYO3_PYTHON benchmarks\cognitive_eval\belief_revision.py --mode refinement   --probes 24 --gap 20 --subtlety 0.25 --distractors 100 1000

# regression guards
& $env:PYO3_PYTHON benchmarks\cognitive_benchmark.py                       # scale 3/4, attributable 0/4
& $env:PYO3_PYTHON benchmarks\cognitive_benchmark.py --dimension 64 --distractors 0   # toy 2/4, attributable 1/4
$env:PYTHONPATH="."; & $env:PYO3_PYTHON benchmarks\audit_recall.py --num-items 20000 --dimension 768   # ANN recall@5 100%
```
