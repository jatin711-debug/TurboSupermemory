# TurboSuperMemory — Execution Plan: Robust Cognitive Layer

> **Audience:** a fresh Claude (Opus) session picking this up cold. Everything you need is in
> this file + the pointers it names. Read this fully before touching code.
> **Branch:** `evaluation`. **Written:** 2026-07-10, at commit `81edded`.

---

## 1. Where the project stands (do not re-derive this)

TSM is a Rust memory engine (6 crates, PyO3 bindings) whose differentiator is the **cognitive
layer** — belief revision, reinforcement/forgetting, abstraction, importance — on top of a solid
tiered vector store. The vector store is proven (100% recall@10 at 100k×1536); the cognitive layer
is the product and was, until recently, unvalidated on real data.

**What is now PROVEN (don't redo):**

- **Belief revision is net-positive on real data.** On LongMemEval (147 conversations, real
  conversational QA), belief ON vs OFF: knowledge-update hit@1 **0.27 → 0.50 (+0.23)**, **zero**
  collateral damage on every other question type, 763 edges (down from 28,311). This took three
  stages: Stage A found the original detector catastrophically over-fires on real text
  (net-negative, `30b10c9`); Stage B fixed it with **mutual-nearest-neighbour detection**
  (`b88130d`); Stage B.2 eliminated the residual with **user-scoped facts** (`81edded`).
- **Reinforcement has NO direct ranking lift** (honest negative, Phase 4 `a07c768`). Its real role
  is retention under a memory budget — untested, see W4.
- **Abstraction is wired but unproven** (Phase 3 `17b741e` made the augmenter traverse
  mem→concept→parent→sibling; unit test proves reachability; no benchmark lift shown yet). See W3.
- Fusion is absolute-saturating `cos + (1-α)·act/(1+act)` (Phase 5 `0b2d709`); empty-query
  recall fixed. Synthetic belief eval: lift +1.00, false_demotion 0.00.

**The two hard-won lessons (they cost the most and must guide everything):**

1. **Synthetic proof does NOT transfer to real data.** Phase 2 had precision 1.00 on synthetic
   coexisting facts; the same detector built 28,311 spurious edges on real conversations. Every
   mechanism claim needs a real-data (LongMemEval) ON-vs-OFF verdict before it counts.
2. **A memory stores the USER's facts, not the assistant's chatter.** ~90% of residual false
   supersessions were the mock extractor sentence-splitting the assistant's bulleted boilerplate.
   Role-scoping eliminated all collateral damage while keeping the full win.

**Living record:** `benchmarks/PHASE_PROGRESS.md` — read the Stage A / B / B.2 sections before
changing anything in detection. Update it (and commit) after every workstream below.

---

## 2. Ground rules (non-negotiable methodology)

1. **ON/OFF isolation.** Every cognitive claim is measured as feature-ON vs feature-OFF on an
   identical corpus + config. The *lift* is the only trusted number (proxy noise cancels).
2. **Real data decides.** LongMemEval is the go/no-go arbiter. Synthetic evals are for geometry
   and regression only.
3. **Honest negatives are deliverables.** If a mechanism shows no lift, document it in
   PHASE_PROGRESS.md and default it OFF. Do not tune until a checkmark appears (that is how
   Stage A happened).
4. **Destructive actions need a precision argument + bounded blast radius.** Demotion already has
   MNN + one-edge-per-memory + floor 0.3 + scope-respect. Keep that bar for anything new
   (eviction, merging).
5. **Synthetic-eval geometry trap:** at 768-dim, `base + randn*jitter` produces near-orthogonal
   vectors (cos ~0.11) — synthetic scenarios MUST use controlled cosine
   (`vec_at_cosine(base, target_cos, rng)` in `benchmarks/cognitive_eval/belief_revision.py`).
   This trap caused a months-long false negative once already.
6. **Every workstream ends with:** `cargo fmt --all` applied, `cargo clippy --workspace
   --all-targets -- -D warnings` clean, `cargo test --workspace --exclude turbomemory_python`
   green, evals re-run, PHASE_PROGRESS.md updated, ONE commit.

---

## 3. Environment & build (Windows — follow exactly, this wastes hours otherwise)

```powershell
# Python MUST be 3.12 (default python is 3.14 → ABI import failure)
$py = "C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe"
$env:PYO3_PYTHON = $py

# Debug info off or link.exe dies with LNK1102 on the storage test binary
$env:CARGO_PROFILE_DEV_DEBUG = "0"; $env:CARGO_PROFILE_TEST_DEBUG = "0"

# Build + refresh the extension (repo-root turbomemory.pyd is what evals import)
cargo build --release -p turbomemory_python
Copy-Item target\release\turbomemory.dll turbomemory.pyd -Force

# Embedding model is cached — run offline (unset only for a one-time new-model download)
$env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"
```

- **Only ONE python process may hold `turbomemory.pyd`** — run eval scripts sequentially, never
  in parallel.
- `audit_recall.py` needs `$env:PYTHONPATH = "D:\personal-projects\TurboSuperMemory"`.
- LongMemEval data: `benchmarks/cognitive_eval/data/longmemeval/*.parquet` (already downloaded).
- Embeddings run on CUDA (RTX 3050); MiniLM is the cached model.
- **ollama server is DOWN.** The `ollama` python lib is installed but there is no server. Anything
  needing an LLM is gated to W6 — do not block other work on it.

**Key commands:**

```powershell
# Real-data arbiter (full test set ~25 min; --limit 25 for a smoke)
& $py benchmarks\cognitive_eval\run_belief_longmemeval.py --user-only

# Synthetic belief regression (expect lift +1.00, false_demotion 0.00)
& $py benchmarks\cognitive_eval\belief_revision.py --mode refinement --quick
& $py benchmarks\cognitive_eval\belief_revision.py --mode contradiction --quick

# Reinforcement isolation (expect ~0 lift — that's the recorded honest negative)
& $py benchmarks\cognitive_eval\reinforcement_eval.py

# E2E + recall floor
& $py verify.py
& $py audit_recall.py
```

**Key files:**

| What | Where |
|---|---|
| Detection (MNN, demotion) | `crates/turbomemory_storage/src/engine.rs` — `nearest_superseding_neighbor`, `check_refinements`, `check_contradictions`, `SUPERSESSION_DEMOTION_FLOOR` |
| Fusion | `engine.rs::hydrate_and_fuse` |
| Augmenter / spreading activation | `crates/turbomemory_graph/src/activation.rs` |
| Concept extraction, opposition markers, vocab evolution | `crates/turbomemory_graph/src/extract.rs`, `graph.rs::evolve_vocabulary` |
| Config (all cognitive knobs) | `crates/turbomemory_storage/src/config.rs` (`TierConfig`) |
| Python bindings | `crates/turbomemory_python/src/lib.rs` |
| Eval adapter (where `store_roles` lives TODAY — eval-only!) | `benchmarks/cognitive_eval/adapters/tsm_adapter.py` |
| LongMemEval runner | `benchmarks/cognitive_eval/run_belief_longmemeval.py` |
| Results record | `benchmarks/PHASE_PROGRESS.md` |

---

## 4. Baseline numbers (regression reference — reproduce before changing anything)

Full LongMemEval test set, MiniLM, mock extractor, `--user-only`, at `81edded`:

| metric | expected |
|---|--:|
| ON-arm edges (147 convs) | refine 683 / contra 80 |
| knowledge-update hit@1 OFF→ON | 0.27 → 0.50 (**lift +0.23**) |
| knowledge-update hit@3 OFF | ~0.86 |
| every single-session type lift (h1/h3/hk) | **+0.00** |
| synthetic belief lift / false_demotion | +1.00 / 0.00 |
| Rust tests | storage 74 + crash 3 + graph + core, all green |
| cognitive_benchmark (768-dim/1000 distractors · toy) | 3/4 · 1/4 (toy regression at Phase 5 is known) |

Small-n caveat: knowledge-update n=22, so hit@1 granularity is ~0.05. Treat |lift| < 0.05 as noise.

---

## 5. Workstreams (in order)

### W1 — Productize the B.2 win: first-class role-aware memory  ✅ DONE (2026-07-10, `865224d` + PHASE_PROGRESS "W1")

**Outcome:** shipped `source_role` on records + `belief_source_roles` config + role-gated detection,
exposed through Python/REST/gRPC. On the full 500-conv LongMemEval, **mode b (`--role-filtered`)
strictly dominates**: KU hit@1 +0.07 (≈ baseline +0.08), single-session-user collateral eliminated
(+0.00 vs baseline −0.12), assistant recall preserved (0.77 vs mode-a's crater to 0.27), 2,345 edges.
It is strictly better than the eval-only `store_roles` hack. Recommended prod config
`belief_source_roles=["user"]` + tag inserts; engine default stays `None` (backward-compatible).
Note: the +0.23 in the old B.2 note was a 147-subset small-n (n=22) artifact; the representative
full-set lift is ~+0.07–0.08 across all arms. Storage tests 74→76, gate green. Original spec below.

---

**Why first:** the entire +0.23 win currently depends on `store_roles` — a filter inside the
**eval adapter**. The engine, Python API, and REST/gRPC have no concept of message role. Any real
user of TSM today gets Stage-A behavior (28k spurious edges). The validated fix must live in the
product.

**Design:**
- Add `source_role: Option<String>` to `Record`/`MetaRecord` with `#[serde(default)]` — mirror
  exactly how `scope` was added (C4, 2026-06-21: insert/update/delete paths, WAL replay, Python
  kwargs on `insert`/`insert_batch`/`update`, REST + gRPC proto). Grep `scope` through
  `engine.rs`, `wal`, `lib.rs`, `rest.rs`, `grpc.rs` and do the same.
- New `TierConfig` field `belief_source_roles: Option<Vec<String>>` (default `None` = legacy
  behavior). When set, `check_refinements`/`check_contradictions` AND
  `nearest_superseding_neighbor` only consider records whose `source_role` is in the list —
  assistant/tool memories stay retrievable but can never create or receive supersession edges.
- Update `TSMAdapter` to pass roles through instead of (or in addition to) dropping messages.

**Decide empirically, don't assume:** B.2 tested mode (a) "don't store assistant turns at all."
The engine feature enables mode (b) "store everything, role-filter belief detection." Run the
LongMemEval eval in mode (b) (new flag `--role-filtered` alongside `--user-only`) and compare all
four: {store-all, user-only} × {belief on/off}. Ship whichever mode wins as the documented
default; record the loser's numbers too.

**Acceptance gate:**
- Engine-level config reproduces B.2-class numbers: KU hit@1 lift ≥ +0.20, every single-session
  lift ≥ −0.05, ON edges ≤ ~1000.
- Rust unit tests: role-filtered detection (assistant fact between two user facts never gets an
  edge), WAL-replay survival of `source_role` (mirror the scope tests).
- fmt/clippy/tests green; PHASE_PROGRESS.md "W1" section; one commit.

### W2 — Regression gate ✅ DONE (2026-07-10, PHASE_PROGRESS "W2")

**Outcome:** `benchmarks/regression_gate.py` + `make gate` + AGENTS.md docs. 7 checks
(fmt/clippy/tests + synthetic belief + role-filtered LongMemEval smoke + recall), evals emit
`GATE_SUMMARY: {json}`. Full clean run **PASS 7/7**. Sabotage findings: reverse-MNN removal is a
*precision* not *volume* regression (edge ceiling doesn't catch it; too noisy at smoke n≈5 — needs
the full run); breaking demotion collapses synthetic lift +1.00→+0.00 and the gate **FAILS as
designed**. Reverted to clean +1.00. Original spec below.

---

**Why now:** every subsequent workstream churns `engine.rs`/`activation.rs`. Without an automated
gate, a refactor can silently destroy the +0.23 (it happened once: the fusion no-op of 2026-06-29
made cognition a silent no-op for weeks).

**Design:** `benchmarks/regression_gate.py` + Makefile target `make gate`:
1. `cargo fmt --all -- --check`, `clippy -D warnings`, workspace tests (with the debug=0 env).
2. Synthetic belief (both modes, `--quick`): assert lift ≥ +0.9, false_demotion = 0.
3. LongMemEval smoke (`--limit 25`, role-aware mode from W1): assert KU lift ≥ 0, every
   single-session lift ≥ −0.08 (smoke-n is small — assert non-regression, not exact values),
   ON edges within [50, 2000].
4. `audit_recall.py` quick pass (recall floor 100%).
Print a PASS/FAIL table; exit nonzero on failure. Full 147-conv run stays manual per workstream.
Document in `AGENTS.md`: "run `make gate` before every commit."

**Acceptance:** gate passes at HEAD; sabotage test — commenting out the reverse-MNN check makes
the gate FAIL (prove it can catch the Stage-A class of bug).

### W3 — Verified demotion ✅ DONE (2026-07-10, PHASE_PROGRESS "W3")

**Outcome:** engine split into pure `propose_supersessions` + `commit_supersessions`
(auto-commit path behavior-identical, 77 tests); `defer_supersession_commit` config;
PyO3 exposed. Local NLI cross-encoder verifier (`verification/nli.py`, transformers
not sentence_transformers) — accept contradiction+entailment, reject neutral. On
200-conv LongMemEval: verification holds KU +0.07, **cuts edges 962→393 (−59%)**, and
drives **all collateral to exactly 0.00**; NLI audit shows 58% of geometric proposals
are neutral coexisting facts (correctly rejected). Opt-in; default path unchanged
(gate 7/7). Discovered + flagged a pre-existing native-memory leak in the per-conv
eval harness (OOMs at full-500; relevant to W7). Original spec below.

---

**Why:** demotion is the only destructive cognitive action. MNN is a *geometric* gate; the
no-compromises bar is a *semantic* check before burying a memory. An NLI cross-encoder runs
locally on the GPU — this is NOT blocked on ollama.

**Design:**
- Engine: split detection from commitment. `propose_supersessions() -> Vec<(old_id, new_id,
  kind, cosine)>` (pure, no graph mutation); `commit_supersessions(pairs)` (creates edges +
  demotes — exactly today's logic). Default path (no verifier installed) = propose + auto-commit
  inside consolidation, bit-identical to today. Expose both via PyO3.
- Python verifier: `sentence-transformers` CrossEncoder `cross-encoder/nli-deberta-v3-xsmall`
  (~70MB, one-time download — temporarily unset the HF offline vars). Batch-score candidate
  pairs; commit rule by kind: Contradicts requires NLI(new, old) = contradiction; Refines
  requires NOT mutual-entailment (mutual entailment = same fact restated, still fine to link)
  — but **calibrate empirically**: dump the ~763 LongMemEval candidate pairs (adapt
  the diagnostic pattern in PHASE_PROGRESS Stage B.2 — print old/new text per pair), hand-label
  ~50, pick the rule maximizing precision at ≥0.9 recall of true supersessions.
- Wire into `TSMAdapter` as opt-in (`verify_demotions=True`).

**Acceptance:** with verification ON: KU hit@1 lift ≥ +0.20 (held), collateral stays 0.00, edges
≤ B.2, labelled-sample precision ≥ 0.9. Graceful no-verifier fallback proven by the gate still
passing with verification OFF.

### W4 — Abstraction real-data verdict ✅ DONE (2026-07-11, PHASE_PROGRESS "W4")

**Outcome:** added `concept_expansion` isolation toggle (SpreadingConfig + PyO3 +
adapter) + `run_abstraction_longmemeval.py`. 200-conv isolation (belief on + role-
filtered both arms): **MIXED / marginal** — temporal-reasoning hit@k **+0.06** but
knowledge-update and multi-session **−0.04** each; net ~neutral, within n-noise; single-
session untouched. Not a robust isolated win; clears the MVP bar only narrowly for
temporal-reasoning. Kept default ON (established behavior + temporal gain); flagged as a
future per-query-type gating candidate. A/B unit test also revealed the old Phase-3
abstraction test was partly reachable via the temporal chain (fixed with a filler).
Synthetic geometry rework deferred/subsumed (mechanism already proven by unit test; real
data decides). Four-mechanism MVP: belief=strong+, reinforcement=honest−, abstraction=
mixed, retention=W5. Original spec below.

---

**Why:** MVP bar = all four mechanisms proven with isolated lift. Belief ✅, reinforcement =
recorded honest negative on ranking, forgetting → W5. Abstraction is the one still in limbo.

**Steps:**
1. Fix the synthetic scenario first: `cognitive_benchmark.py` abstraction scenario uses
   jitter-based vectors → near-orthogonal → seed activation starves (ground rule 5). Port
   `vec_at_cosine`. Also diagnose the toy-regime regression (2/4 → 1/4 at Phase 5) while there.
2. Hubs: naive extraction builds hundreds of concepts; a hub concept floods activation. The C3
   machinery already exists (`evolve_vocabulary`, `suppressed_concepts`, hub_fraction) but is
   opt-in and untuned — tune it on the LongMemEval corpus (concept degree distribution) before
   measuring.
3. Real-data verdict: LongMemEval ON/OFF where OFF = concept/abstraction expansion disabled in
   the augmenter, ON = enabled (belief revision identical in both arms — isolate ONE mechanism).
   Add a toggle (config or adapter) if none exists. Target types: multi-session,
   temporal-reasoning, knowledge-update tail.

**Acceptance:** lift > +0.05 on ≥1 question type with all other types ≥ −0.05 → keep ON and
record; otherwise honest negative → default OFF, record why. Either outcome closes the MVP
question for abstraction.

### W5 — Retention / forgetting ✅ DONE (2026-07-11, PHASE_PROGRESS "W5")

**Outcome:** added `access_aware_eviction` flag (cognitive salience vs naive FIFO) +
`engine.contains_id` + `retention_eval.py`. 200-conv isolation (identical ops both arms,
only eviction policy differs, budget=10): **gold survival OFF 0.18 → ON 0.60 (+0.41)**,
hit@k +0.41 — a used memory survives budget eviction 3.3× more often. STRONG POSITIVE;
reconciles the Phase-4 reinforcement negative (reinforcement drives retention, not
ranking). **Four-mechanism MVP COMPLETE:** belief=strong+, reinforcement=positive-on-
retention, forgetting=strong+, abstraction=marginal. Rust test
`fifo_eviction_ignores_rehearsal` (storage 77→78). Original spec below.

---

**Why:** Phase 4 proved rehearsal doesn't re-rank. The honest claim left for
reinforcement/decay/importance is: **under a memory budget, TSM retains what matters**. Nobody
has ever tested that — and it's the "forgetting" half of the MVP.

**Design:** new `benchmarks/cognitive_eval/retention_eval.py`:
- Corpus: LongMemEval conversations (real facts), inserted under a tight `max_records` budget so
  eviction MUST fire during consolidation.
- Rehearsal signal: search a subset of facts mid-stream (the ones queries will later need).
- Arms: (1) cognitive ON (`importance_auto_scoring` + reinforcement + decay + eviction),
  (2) baseline: same budget, cognitive OFF (eviction by recency/insertion only).
- Metrics: answer-survival rate (is the gold fact still alive post-eviction) and hit@k on the
  standard queries.
**Acceptance:** survival/hit lift ON vs OFF > +0.05 → mechanism proven (record config guidance);
else honest negative + analysis of what eviction used instead. Watch out: grace-period logic
(`recency_half_life_secs / 8`) can mask everything — size the budget so eviction genuinely bites.

### W6 — Gold-standard metric + real extractor (GATED: needs ollama server or an API key)

Everything above uses the retrieval-containment proxy. The publishable number is
retrieve → LLM answers → LLM judges (the official LongMemEval protocol).

1. Stand up `ollama serve` with a small instruct model (or wire an API-key client).
2. Add answer+judge stages to `run_belief_longmemeval.py` (keep the proxy — it's the free
   regression signal; the LLM metric is the headline).
3. Re-run the W1 mode comparison + W3 verification with `extractor="ollama"` (real fact
   extraction instead of mock sentence-splitting) — confirm the conclusions hold when facts are
   clean. This closes the biggest honest caveat in the Stage A/B record.
4. Generalization: full LongMemEval (train+test ≈ 500 convs) and/or LoCoMo; one run with a
   768-dim embedding model to confirm MNN's scale-free claim across embedding distributions.

**Acceptance:** LLM-judged per-type accuracy, belief ON ≥ OFF on knowledge-update, no type
regresses; results recorded with the proxy numbers side-by-side.

### W7 — Scale + ops 🟡 CORE FIX DONE (2026-07-11, PHASE_PROGRESS "W7")

**Outcome:** `profile_consolidation.py` measured the dominant cost — MNN supersession
detection re-scans all N every cycle (69s@10k, 277s whole cycle). Fixed with an
incremental seq-cursor (`incremental_supersession_detection`, opt-in): steady-state cycle
now O(new) not O(total) — **4.4× faster** 2nd consolidation (base=20k+200: 487s→111s), with
propose cut ~376s→~1s. Honest: the remaining 111s is OTHER O(N) passes (importance, segment
HNSW builds, abstraction/vocab evolution, graph-JSON snapshot at ~24KB/record) — still to be
made incremental for 1M. Rust test `incremental_detection_finds_cross_cycle_refinement`
(storage 78→79). Remaining W7 (follow-ons): other consolidation O(N) passes, graph snapshot
format, API/ops (tracing/auth/Docker), flagged native-memory leak. Original spec below.

---

- Profile consolidation with cognitive features at 100k/1M inserts: MNN costs ~2 ANN queries per
  new memory per cycle; `recompute_importance`, dedup, and graph rebuild are O(live records)
  scans (`engine.rs` ~1240–1700). Measure first, then make consolidation incremental
  (seq-cursor) where the numbers demand it.
- Graph JSON snapshot size + load time at scale (it persists the whole graph as JSON today).
- Then: API-server hardening, tracing/metrics, auth, Docker. (`CLOUD_DEPLOY.md` exists — audit
  it against reality.)

---

## 6. What NOT to do

- Do **not** re-tune detection thresholds (`refinement_cosine_threshold` etc.) chasing LongMemEval
  numbers — MNN made detection scale-free precisely so thresholds stop mattering; re-tuning
  reintroduces the Stage-A failure mode.
- Do **not** trust any synthetic scenario that builds vectors with `randn` jitter (ground rule 5).
- Do **not** run two TSM python processes at once (`.pyd` lock).
- Do **not** delete or rewrite the Stage A negative results in PHASE_PROGRESS.md — the negative
  record is the evidence base for the current design.
- Do **not** start W7 scale work before the W2 gate exists.

## 7. Definition of done (this plan)

1. W1–W5 complete: all four cognitive mechanisms have a real-data verdict (win or recorded
   honest negative), the winning behaviors are first-class engine features (not eval-adapter
   hacks), demotion is semantically verified, and `make gate` protects all of it.
2. W6 done when LLM infra exists: headline LLM-judged accuracy published in README.
3. PHASE_PROGRESS.md tells the full story; every workstream is one reviewable commit on
   `evaluation`.
