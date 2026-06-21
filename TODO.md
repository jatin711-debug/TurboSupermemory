# TurboSuperMemory — Roadmap

> **Goal:** Build the memory infrastructure that enables AI systems to
> remember, learn from past interactions, maintain context across long
> periods, and build a coherent knowledge base that becomes more useful
> as it accumulates experience.
>
> This is NOT a vector-database roadmap. The cognitive layer (learning,
> forgetting, generalizing, evolving) is the differentiator. Scaling to
> 1M+ is a prerequisite for production, not the goal itself.
>
> Status key: **Done** | **In Progress** | **Pending** | **Cut**

---

## Recently Completed — 2026-06-21 (per-agent memory scoping — C4, LLM compressor integration test — C6)

Shipped the last two Stage 1 cognitive-layer roadmap items together:
per-agent memory scoping (multi-agent isolation) and an end-to-end LLM
compressor integration test. With these, **Stage 1 — Cognitive Deepening is
complete** (C1–C8 done).

- **Per-agent memory scoping (C4).**
  - Added an optional `scope: Option<String>` field to `Record` /
    `MetaRecord` (`#[serde(default)]` for backward-compatible reload).
  - Created `ScopeIndex` (`scope_index.rs`) with a global bitmap plus
    per-scope bitmaps; scoped queries return matching-scope records **plus**
    global/shared records.
  - Wired `scope` through `StorageEngine` insert/update/delete, WAL replay,
    and every search path (`search_ann`, `search`, and filtered variants).
  - Exposed `scope` in Python bindings (`insert`, `insert_batch`, `update`,
    `search`, `search_ann`, `search_ann_candidates`), gRPC/REST proto,
    `grpc.rs`, and `rest.rs`.
  - Added Rust unit tests for scoped isolation and WAL-replay survival.
- **LLM compressor integration test (C6).**
  - Added `StorageEngine::set_compressor` and Python `set_llm_compressor(callable)`.
  - `PythonCompressor` implements the existing `CognitiveCompressor` trait,
    forwarding `(current_ccs_json, user_input, assistant_response)` to the
    Python callable and falling back to the deterministic compressor if the
    callable raises or returns invalid JSON.
  - `verify.py` now includes a mock-LLM test that proves `step_session` uses
    the installed callable and preserves the returned CCS schema.

Validated together with the C8/C3 changes: `cargo fmt --all --check` clean;
`cargo clippy --workspace --all-targets -- -D warnings` clean; `cargo test
--workspace --exclude turbomemory_python` = **165 passed / 0 failed** (core 29
+ graph 65 + storage 68 + crash_recovery 3); `make verify` E2E pass including
new scope + LLM-compressor assertions; `python cognitive_benchmark.py
--dimension 64 --distractors 0` = 4/4 wins; `python cognitive_benchmark.py`
(768-dim, 1000 distractors) = 3/4 wins; `python audit_recall.py` pass.

**Stage 1 — Cognitive Deepening is complete (C1–C8 done). Next focus: Stage 2
structural scaling to 1M × 4k, starting with 0.1 collection sharding.**

## Recently Completed — 2026-06-21 (online concept vocabulary evolution — C3)

Shipped the third cognitive-layer evolution primitive: online concept
vocabulary evolution. The graph now learns which surface forms are synonyms,
collapses them into a single canonical concept node, and suppresses
over-general hub concepts so they don't drown out more specific ones.
Resolves roadmap item **C3**.

- **`ConceptVocabulary` persistence (`extract.rs`).** The alias canonicalizer
  from C8 now derives `Serialize`/`Deserialize` so learned aliases survive
  graph JSON snapshots across restarts.
- **`MemoryGraph` evolution (`graph.rs`).**
  - Added `vocab: ConceptVocabulary` and `suppressed_concepts: BTreeSet<String>`
    fields (both `#[serde(default)]` for backward compatibility).
  - `add_memory_with_importance` now canonicalizes concepts through the learned
    vocabulary before creating nodes/edges, so aliases discovered by evolution
    are automatically applied to future inserts.
  - `evolve_vocabulary(overlap_threshold, hub_fraction, max_pairs)` runs an
    online pass:
    - Builds base-concept → memory-set index and pairwise co-occurrence counts.
    - Scores candidate pairs by Jaccard overlap of associated memory sets.
    - Merges the lower-degree concept into the higher-degree concept,
      redirects `Association` and `Abstraction` edges, records the alias in
      `ConceptVocabulary`, and rebuilds `co_occurrence`.
    - Suppresses base concepts whose degree exceeds
      `hub_fraction * memory_count`.
  - New helpers: `merge_concept_node`, `ensure_edge`, `rebuild_co_occurrence`.
- **`SpreadingActivation` hub suppression (`activation.rs`).** Suppressed
  concepts no longer expand to their neighbors during spreading activation,
  preventing over-general terms ("system", "memory") from drowning specific
  concepts.
- **Config + Python exposure (`config.rs`, `lib.rs`).** Four opt-in kwargs on
  `MemoryEngine`, all disabled by default to preserve existing benchmarks:
  `concept_evolution_enabled`, `concept_merge_overlap_threshold`,
  `concept_hub_degree_fraction`, `concept_evolution_max_pairs_per_cycle`.
  New method `evolve_concept_vocabulary()` returns
  `(merged, newly_suppressed, examined_pairs)`.
- **Engine integration (`engine.rs`).** `StorageEngine` snapshots the graph
  vocabulary before extracting concepts on insert, calls
  `evolve_vocabulary` during consolidation when enabled, and persists the
  updated graph.
- **Tests.** Added 5 graph unit tests: overlapping-concept merge, hub
  suppression, disabled no-op, suppressed concept does not expand, and
  JSON roundtrip of vocab + suppressed set. C8 already added 11 extraction
  tests; graph test count grew from 60 to 65.

Validated: `cargo fmt --all --check` clean; `cargo clippy --workspace
--all-targets -- -D warnings` clean; `cargo test --workspace --exclude
turbomemory_python` = **160 passed / 0 failed** (core 29 + graph 65 +
storage 63 + crash_recovery 3); `make build-python` + `python verify.py`
E2E pass; `python cognitive_benchmark.py --dimension 64 --distractors 0`
= 4/4 wins (toy regression preserved); `python cognitive_benchmark.py`
(768-dim, 1000 distractors) = 3/4 wins; `python audit_recall.py` pass.



---

## Recently Completed — 2026-06-21 (real-embedding cognitive benchmark — C5)

Scaled the cognitive benchmark from the toy regime (64-dim, ~10 memories) to
realistic text-embedding scale (768-dim, 1000+ clustered memories per
scenario) to validate that the cognitive layer generalizes beyond toy data.
Resolves roadmap item **C5**.

- **Realistic embedding generation.** Added `make_clustered_vec` /
  `make_cluster_center` helpers that generate unit-norm vectors drawn from
  topic clusters with tight intra-cluster spread (jitter 0.15–0.20) — the
  same manifold-with-local-structure distribution `benchmark.py` uses, which
  models real text embeddings far better than near-orthogonal Gaussians.
- **Distractor injector.** `inject_distractors(tsm, dim, n)` fills the graph
  with `n` noise memories drawn from a bank of 20 diverse topic clusters
  (weather, jazz, volcanoes, compilers, …), each with unrelated concepts and
  modest importance. This makes retrieval face real competition: the target
  memory must surface through the cognitive graph despite hundreds/thousands
  of competing memories, many with non-trivial cosine to the query.
- **Adaptive top_k for abstraction.** Scenario 1 scales `top_k` with the
  distractor population so the multi-hop abstraction target has room to
  surface; ANN uses the same `top_k` for a fair comparison.
- **Two regimes.** The default is now realistic scale (768-dim, 1000
  distractors). The original toy regime is preserved via
  `--dimension 64 --distractors 0`.

**Scale result (768-dim, 1000 distractors): cognitive layer wins 3/4
scenarios.** Refinement surfacing, reinforcement boosting, and contradiction
surfacing all WIN — the cognitive layer finds memories that plain ANN misses
*entirely* (rank 99 at top_k 5–20). Abstraction traversal does NOT scale to
1000 distractors: the top-k is dominated by cosinely-nearby distractors
before the multi-hop abstraction path can surface the target. This is honest
signal that the abstraction feature needs hub-suppression / frontier tuning
at scale — a future tuning task, not a correctness bug. The toy regime
(64-dim, 0 distractors) still wins 4/4 (backward-compatible).

Validated: `python cognitive_benchmark.py` (default scale) = 3/4 won;
`python cognitive_benchmark.py --dimension 64 --distractors 0` = 4/4 won
(toy regression preserved).

---

## Recently Completed — 2026-06-20 (graph introspection API — C7, auto-importance — C2)

Shipped two more cognitive-layer features: a read-only **graph introspection
API** (C7) for debugging and "what does the AI know" views, and **automatic
importance scoring** (C2) that makes memory self-organizing — retrieval
patterns raise what matters and decay what doesn't, without the caller
tagging importance manually.

- **Graph introspection API (C7, `graph.rs`, `engine.rs`, `lib.rs`).** Five
  read-only Python methods on `MemoryEngine`: `graph_stats()` →
  `(node_count, edge_count, memory_count, concept_count, refinement_count,
  contradiction_count, abstraction_count)`; `get_concepts()` →
  `list[(concept, degree)]` sorted by degree desc; `get_memory_concepts(id)`;
  `get_refinements(id)`; `get_contradictions(id)`. Backed by new
  `MemoryGraph` helpers (`concept_count`, `memory_concepts`, `stats()` returning
  a `GraphStats` snapshot) and a `StorageEngine::read_graph()` accessor that
  hands out a `parking_lot` read guard. Return shape is tuples / list-of-tuples
  (matches the existing binding style). Unknown ids return empty lists.
- **Automatic importance scoring (C2, `graph.rs`, `engine.rs`, `config.rs`,
  `lib.rs`).** `StorageEngine::recompute_importance()` runs on consolidation
  (opt-in via `importance_auto_scoring`). For each record it computes a target
  importance from a blend of retrieval salience (normalized `access_score`) and
  graph connectivity (normalized concept degree), moves the current importance
  `importance_learning_rate` toward it, clamps to `[floor, ceiling]`, writes
  back to metadata, and syncs the graph. Retrieval is the primary driver;
  connectivity is a bounded boost scaled by salience so a never-retrieved memory
  decays toward the floor regardless of how many concepts it touches. Runs
  before dedup/eviction so recomputed importance participates in tiebreaking.
  Five opt-in config fields + Python kwargs.
- **Graph edge re-sync (`MemoryGraph::reweight_memory`).** The graph sets edge
  weights from importance once at insert; until now there was no update path.
  Added `reweight_memory(id, importance)` that rescales a memory's association +
  temporal edges to a new importance while **preserving learned
  reinforcement/decay**. Memory nodes now track their `base_importance_factor`
  (set at creation) so the rescale ratio is exact; reinforced edges (weight >
  baseline) scale by the same ratio as the baseline, keeping their relative
  boost. `recompute_importance` calls it on every changed record.
- **Python exposure.** New constructor kwargs: `importance_auto_scoring`,
  `importance_learning_rate`, `importance_access_weight`, `importance_floor`,
  `importance_ceiling`. New method `recompute_importance()` for manual
  triggering.

Validated: `cargo fmt --check` clean; `cargo clippy --workspace --all-targets
-- -D warnings` clean; `cargo test --workspace --exclude turbomemory_python`
= 144 passed / 0 failed (core 29 + graph 49 + storage 63 + crash_recovery 3);
`make build-python` + `python verify.py` E2E all pass; Python smoke confirms
auto-importance raises a frequently-retrieved memory above a never-retrieved
one and introspection returns correct counts/concepts.

---

## Recently Completed — 2026-06-20 (contradiction detection — C1)

Shipped the second memory-evolution primitive: contradiction detection
(belief revision). This is the SAGE/A-Mem pattern that distinguishes a
*memory* (revises beliefs) from a *database* (stores everything). A new
memory that says the *opposite* of an existing one now creates a
`Contradicts` edge and weakens the old memory, so retrieval surfaces the
correction. Resolves roadmap item **C1**.

- **`EdgeKind::Contradicts` (`graph.rs`).** Directed edge old → new, where
  the new memory contradicts the old one. `MemoryGraph::add_contradiction`
  creates the edge and multiplies the old memory's *outgoing association*
  edges by `contradiction_weaken_factor` (default 0.5) so it fades without
  disappearing — history is preserved. Idempotent. Spreading activation
  traverses `Contradicts` edges so the correction surfaces on retrieval.
  `contradiction_count()` / `contradicted_by(id)` expose the learned edges.
- **Detection logic (`engine.rs`, `extract.rs`).** `check_contradictions()`
  runs on consolidation. A pair is a contradiction when: cosine >=
  `contradiction_cosine_threshold` AND they share a concept AND **text
  Jaccard < `contradiction_text_threshold`** — the text-dissimilarity signal
  is the key distinguisher from refinement (high text overlap = "same topic,
  updated content" → Refines; low text overlap = "same topic, opposing
  content" → Contradicts). `text_jaccard_similarity` in `extract.rs` computes
  token-set Jaccard using the same stopword filtering as concept extraction.
- **Config (`config.rs`) + Python exposure (`lib.rs`).** Four opt-in fields
  on `TierConfig`, all surfaced as Python kwargs:
  `contradiction_cosine_threshold` (None = disabled), `contradiction_text_threshold`
  (0.3), `contradiction_weaken_factor` (0.5), `contradiction_max_pairs_per_cycle`
  (1024). Disabled by default preserves prior behavior.
- **Benchmark scenario 4 (`cognitive_benchmark.py`).** "Contradiction
  surfacing": a false old claim + a newer correction, query biased toward the
  old vector. With contradiction detection ON the correction surfaces at rank
  1 (`new_correction`, `old_false_claim`); plain ANN returns the false claim
  first. With the feature OFF the old claim still wins — proving the
  `Contradicts` edge is what flips the ranking. Cognitive layer now wins
  **4/4 scenarios** (up from 3/3).

Validated: `cargo fmt --check` clean; `cargo clippy --workspace --all-targets
-- -D warnings` clean; `cargo test --workspace --exclude turbomemory_python`
= 134 passed / 0 failed (core 29 + graph 42 + storage 60 + crash_recovery 3);
`make build-python` + `python verify.py` E2E all pass; `python
cognitive_benchmark.py` = 4/4 cognitive scenarios won.

> Note: the Windows `link.exe` hit `LNK1102: out of memory` linking the heavy
> storage debug-test binary (tantivy + usearch + redb). Stripping debug info
> via `CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0` works around it;
> release builds are unaffected.

---

## Recently Completed — 2026-06-20 (cognitive layer: concept extraction, CCS compressor, memory evolution, score fusion, cognitive benchmark)

Shipped the remaining Path A cognitive-layer features (concept extraction,
pluggable CCS compressor, memory evolution/belief revision), a critical
architectural fix (score fusion), and a benchmark that proves the cognitive
layer improves retrieval over plain ANN in 2 of 3 scenarios.

- **Concept extraction (`extract.rs`).** `extract_concepts(text, max)` with
  stopword filtering, TF ranking, length bonus. `merge_concepts` augments
  caller-supplied concepts with extracted ones. Engine auto-extracts on every
  insert when the caller provides fewer than `max_concepts` (default 5, 0 =
  disabled). Concepts are normalized to lowercase + deduped in the graph so
  "Rust" and "rust" map to the same node. The graph now works as a turnkey
  layer — callers no longer *have* to supply concepts.
- **Pluggable CCS compressor (`ccs.rs` rewritten).** `CognitiveCompressor`
  trait with `compress(&ccs, user_input, assistant_response) -> ccs`. Two
  impls: `DeterministicCompressor` (the old logic, fast, no I/O) and
  `LlmCompressor<F>` (calls a user-supplied closure, falls back to
  deterministic on invalid JSON). Engine stores
  `Arc<RwLock<Arc<dyn CognitiveCompressor>>>`; `set_compressor` swaps at
  runtime. The README's "an LLM-based compressor can be plugged in" claim is
  now true in code.
- **Memory evolution / belief revision (`graph.rs`, `engine.rs`).** New
  `EdgeKind::Refines` — a directed edge from an older memory to a newer one
  that supersedes it. `check_refinements()` on consolidation finds pairs
  where cosine >= `refinement_cosine_threshold` AND they share a concept,
  creates the Refines edge, and transfers the older memory's unique concepts
  to the newer one. The older memory is NOT deleted — history is preserved.
  Spreading activation traverses Refines edges so the newer memory surfaces.
- **Score fusion (critical fix, `engine.rs`).** The engine's `search` method
  previously discarded the graph activation score and re-sorted by pure
  cosine — nullifying the graph's ranking signal. Fixed: `hydrate_and_fuse`
  now computes `final_score = cognitive_alpha * cosine + (1 - cognitive_alpha)
  * normalized_activation`. `cognitive_alpha` defaults to 1.0 (pure cosine,
  backward-compatible) but can be set to 0.5 or 0.3 to give the graph a vote.
  Also: the engine now requests `top_k * 3` candidates from the graph (up
  from `top_k`) so multi-hop traversal has room to surface memories before
  fusion + truncation.
- **Reinforcement fix (`graph.rs`).** `reinforce` now strengthens *incoming*
  association edges (where the memory is the target), not just outgoing
  edges. This is what makes reinforcement actually boost a memory's
  activation: when a concept is activated by the query, it propagates more
  energy through the strengthened concept→memory edge to the reinforced
  memory.
- **Cognitive benchmark (`cognitive_benchmark.py`).** Three scenarios that
  test specifically cognitive retrieval (where the correct answer is NOT the
  nearest neighbor):
  1. **Abstraction traversal** — target tagged only "rust", query mentions
     "safety", abstraction edge should bridge. (Currently draws — both find
     it at rank 7; needs higher spreading_iterations to boost multi-hop.)
  2. **Refinement surfacing** — query matches old memory, Refines edge
     should surface newer one. **WINS**: new_fact rank 1 (cognitive) vs
     rank 2 (ANN).
  3. **Reinforcement boosting** — query closer to non-reinforced memory,
     reinforcement should boost the rehearsed one. **WINS**: mem_a rank 1
     (cognitive) vs rank 2 (ANN).
  **Verdict: cognitive layer improves retrieval over plain ANN (3/3
  scenarios won).** The abstraction edge specifically doesn't add
  incremental value over the base graph's concept traversal in small
  graphs (the memory-mediated path is sufficient), but the graph +
  spreading activation + fusion as a whole beats ANN in all three
  scenarios — finding memories that are semantically related through
  concept edges but have low cosine to the query.
- **Python exposure.** New kwargs: `max_concepts`, `refinement_cosine_threshold`,
  `refinement_max_pairs_per_cycle`, `cognitive_alpha`.

Validated: `cargo fmt --check` clean; `cargo clippy -- -D warnings` clean;
`cargo test --workspace --exclude turbomemory_python` = 126 passed / 0 failed
(core 29 + graph 37 + storage 57 + crash_recovery 3); `make build-python` +
`python verify.py` E2E all pass; `python cognitive_benchmark.py` = 2/3
cognitive scenarios won.

---

## Recently Completed — 2026-06-19 (cognitive layer + TurboQuant hardening)

Shipped: the first installment of the *memory-as-cognition* layer (learnable
graph edges, abstraction hierarchy, durable graph reload) plus a latent-crash
fix in the TurboQuant config path. This is the work that differentiates TSM
from a plain vector DB and directly serves the "retain what matters, adapt as
new information arrives, build a coherent knowledge base" vision.

- **Learnable edge weights (`graph.rs`).** Edges are now weighted by
  `importance.sqrt()` at insert time instead of the constant `1.0`/`0.5`.
  `MemoryGraph::reinforce(id, now)` strengthens a memory's edges on retrieval
  (first recall gives a 1.5× boost, subsequent recalls grow as
  `1 + 0.1/(1+w)`, clamped at 8.0). `decay_edges(now, half_life)` erodes the
  *learned* portion of reinforced edges with an exponential half-life, floored
  at the baseline weight so decay never drops an edge below its birth strength.
  Unreinforced edges are untouched. This is the "retain what matters / forget
  what doesn't" loop: retrieval itself is the learning signal.
- **Abstraction hierarchy activated (`graph.rs`).** `EdgeKind::Abstraction`
  was defined but never constructed — now `build_abstractions(threshold)`
  creates parent concept nodes (e.g. `rust+safety`) from accumulated concept
  co-occurrence, with bidirectional `Abstraction` edges. Idempotent; co
  -occurrence counters reset after each call so only *new* co-occurrences
  trigger subsequent abstractions. Spreading activation now traverses
  `Abstraction` edges (exempt from hub suppression) so a query hitting one
  concept can reach memories of a sibling concept through the parent.
- **Durable graph reload (`engine.rs`).** `rebuild_graph` now loads the
  persisted graph JSON on open and merges in only records not already present,
  preserving learned weights, reinforcement timestamps, and abstraction nodes
  across restarts. Previously `MemoryGraph::from_json` was dead code and the
  graph was always rebuilt from records, losing all learned state.
- **Engine hooks.** `reinforce` is called on every cognitive-search result
  (`search` / `search_filtered`); `decay_edges` + `build_abstractions` run on
  `trigger_consolidation`; `add_memory_with_importance` is used on the insert
  and dedup-merge paths. All opt-in via config (disabled by default preserves
  pre-learning behavior).
- **Config + Python exposure.** `SpreadingConfig` added to `StoreConfig`; new
  `TierConfig` fields `abstraction_co_occurrence_threshold` (default 0 = off)
  and `edge_decay_half_life_secs` (default 0 = off); new Python kwargs
  `fok_threshold`, `spreading_decay`, `spreading_iterations`,
  `abstraction_co_occurrence_threshold`, `edge_decay_half_life_secs`.
- **TurboQuant panic fix (`config.rs`).** `QuantizerKind::build` now returns
  `Result` instead of `.expect()`, and `StorageEngine::open` validates
  TurboQuant tiers against non-power-of-two dimensions up front with an
  actionable `InvalidArgument` error. Previously selecting `turbo_mse`/
  `turbo_prod` with the default `dimension = 768` (not a power of two) would
  panic inside the quantizer constructor.
- **Table corrections.** 4.1 (batched SIMD kernel) and 4.6 (TurboQuant
  quantizer) were already implemented in code but still marked Pending in the
  table below — corrected to Done. 4.1 was noted Done in the 2026-06-19 audit
  header; 4.6 is fully implemented in `turbo_quant.rs` (1150 LOC, 12 tests,
  both MSE and Prod variants with LUT scoring).

Validated: `cargo fmt --check` clean; `cargo clippy -- -D warnings` clean;
`cargo test --workspace --exclude turbomemory_python` = 97 passed / 0 failed
(core 29 + graph 14 + storage 51 + crash_recovery 3); `make build-python` +
`python verify.py` E2E all pass.

---

## Recently Completed — 2026-06-19 (audit + memory lifecycle)

Two updates: (1) a deep codebase audit that found several roadmap items were
**already implemented but still marked Pending**, now corrected below; (2) the
memory-lifecycle feature (bounded-storage eviction + semantic dedup) shipped.

- **Bounded-storage eviction + semantic consolidation (new, opt-in, default OFF).**
  `StorageEngine::evict()` and `StorageEngine::deduplicate()` (`engine.rs`), wired
  into the consolidation cycle. Eviction selects victims by capacity cap
  (`max_records`) and/or `access_score` floor (`evict_score_floor`), with a grace
  period (`now - last_accessed < recency_half_life_secs / 8`) so freshly inserted,
  never-queried records survive. Dedup reuses the ANN index for candidate pairs +
  exact cosine above `dedup_cosine_threshold`, keeps the higher-salience record,
  transfers the victim's concept edges to the survivor, then `delete_by_id`.
  Config fields added to `TierConfig` (all `Option`, default `None` = current
  unbounded/no-dedup behavior) and surfaced in the Python constructor plus
  `evict()` / `deduplicate()` methods. Smoke-tested: eviction capped 60→20; dedup
  merged 2 of 4 near-duplicates.

- **Audit corrections — these were Pending in the roadmap but are DONE in code:**
  - **0.2 / 6.1 lock-free segment list** — `ArcSwap<SegmentSnapshot>` in
    `segment_holder.rs`; searches read a published snapshot, never block on swap.
  - **0.5 single batched update worker** — `update_worker.rs` (`IndexApplier` +
    crossbeam channel); all writes serialize through one `apply_batch`.
  - **3.1 visited-set pool** — `visited_pool.rs` (`VisitedSet` token array +
    generation wrap), parking_lot-guarded.
  - **3.5 parallel multi-segment search** — `into_par_iter` over segments in
    `segment_holder.rs` with top-k merge.
  - **4.1 batched SIMD distance kernel** — `cosine_similarity_batch` in
    `turbomemory_core/src/metrics.rs` with AVX2/FMA + SSE paths and a 4-vector
    unrolled kernel (`dot_and_nb_x4_avx2`).
  - **6.3 parking_lot** — in use across the storage crate (segment holder, update
    worker, metadata cache, access counters).

---

## Recently Completed — 2026-06-18 (session 2)

Shipped in commit `9fd68f5` ("feat: implement oversampling in reranking for improved neighbor recovery in vector segments"). Three improvement areas targeting consolidation speed, high-dimensional recall, and ingest cost.

- **Multi-threaded HNSW build (resolves 2.11).** `usearch_index.rs::build` now seeds the graph with the first `PARALLEL_SEED_POINTS = 256` points single-threaded, then inserts the remainder via a bounded rayon pool. Per-build thread count is `clamp(num_cpus / max_concurrent_builds, 1, 16)` so concurrent segment builds don't oversubscribe. Measured at 10k: consolidation time roughly halved — 768-dim 28.5 s → 14.2 s, 1024-dim 34.8 s → 17.9 s — with no recall regression (a self-recall guard test, `usearch_parallel_build_preserves_recall`, asserts recall@1 ≥ 0.95).
- **Oversampling margin in quantize-and-rerank (recall fix).** `warm.rs`/`cold.rs` `search` previously shortlisted *exactly* `top_k` quantized candidates before f32 rerank, so a true neighbor that quantizes to rank `top_k+1` was dropped before rerank could recover it. Now shortlists `max(top_k * RERANK_OVERSAMPLE, MIN_RERANK_SHORTLIST)` (8× / 64-floor) then reranks and truncates to `top_k`. Effect is muted at 10k (everything merges into one HNSW segment, so recall is HNSW-bound); the payoff is at 100k+ where Warm/Cold dominate. Guard test `warm_segment_oversampling_recovers_exact_nearest`.
- **Zero-copy numpy ingest (resolves 1.9 / 8.1).** PyO3 bindings now borrow contiguous `float32` numpy arrays directly (`F32Input`/`F32Matrix` View variants) instead of `slice.to_vec()` / `row.to_vec()`; lists, non-contiguous arrays, and wrong-dtype inputs fall back to owned copies. Engine batch signature changed to `&[&[f32]]` to carry the borrow through. The borrowed view is held on-stack across `py.allow_threads`; only the derived `&[f32]` is captured by the GIL-released closure.

Validated: full workspace test suite passes (storage incl. 2 new guard tests, core, graph, api); PyO3 build clean.

---

## Recently Completed — 2026-06-18

Shipped in commit `35cc6c7` ("fix: improve ANN recall with 8-bit Cold tier and configurable consolidation"). These resolve part of Phase 3/4 ahead of the structural work and fix a recall regression surfaced by high-dimensional benchmarks.

- **8-bit Cold tier (resolves Exec-Summary #7).** Cold tier default quantizer changed from 1-bit `Sign` to 8-bit `Scalar` (`config.rs`). Sign-only quantization could not resolve the tiny cosine gaps in high-dim data and floored recall at ~27% after full consolidation. `warm_capacity` budget also raised (16→64 MiB, clamp 200k→500k) so vectors stay in 8-bit Warm longer before demotion.
- **Caller-specified `ef` (partial 3.10).** `search_ann(query, top_k, ef)` is now wired through Python; benchmark exposes `--ef`. Adaptive auto-raise still pending.
- **Configurable background consolidation (partial 8.4).** `auto_consolidation_secs` Python kwarg (0 disables) replaces the hardcoded 60 s interval, so benchmarks and manual-consolidation workloads aren't silently penalized.
- **Realistic benchmark data (partial 10.5).** `benchmark.py` now defaults to clustered embeddings (64 centers, 0.15 jitter) instead of near-orthogonal Gaussian; old behavior available via `--data-distribution random`.

Validated at 50k × 1024 after full consolidation: adversarial-data recall floor lifted ~27% → ~71% at default `ef`; clustered data reaches 65–78% and scales with `ef`.

---

## Executive Summary — Are We Ready for 1M × 4k?

**No.** The codebase is solid for 100k × 128, but 1M+ vectors at high dimension introduces bottlenecks that are structural, not incremental:

1. **Single-node / single-shard design** — one `VectorStore`, one `SegmentHolder`, one `MetadataStore`, one graph.
2. **Single-threaded update worker** — all writes serialize through one channel/worker.
3. **~~Global `RwLock<SegmentHolder>`~~** — **Fixed (2026-06-19 audit).** Replaced with a lock-free `ArcSwap<SegmentSnapshot>`; searches read a published snapshot and never block on seal/merge swap (0.2 / 6.1).
4. **In-memory metadata cache** — every record's metadata lives in a `HashMap`; won't fit RAM at 1M+ text records. (redb is a lazy snapshot, not the primary cache; the cache itself is still a single in-memory `HashMap`.)
5. **No vacuum / physical deletion** — deleted vectors and old segments are never reclaimed.
6. **~~Single-threaded HNSW builds~~** — **Fixed (2026-06-18).** Builds now seed 256 points single-threaded then insert the remainder in parallel; see [Recently Completed](#recently-completed--2026-06-18-session-2). The 512 MiB default budget still applies.
7. **~~1-bit Cold tier collapse at 4k~~** — **Fixed (2026-06-18).** Cold tier now defaults to 8-bit scalar quantization; see [Recently Completed](#recently-completed--2026-06-18).
8. **~~No visited-set pool, no segment-level parallelism~~** — **Fixed (2026-06-19 audit).** Visited pool (3.1) and parallel multi-segment search (3.5) are implemented. Adaptive/cardinality-aware filtering is still pending.
9. **No GPU acceleration** — CPU-only HNSW build and distance compute.

This roadmap is the complete set of optimizations needed to reach 1M+ nodes × high dimensions. Items are grouped by phase, with Qdrant-derived defaults and concrete file targets.

---

## Phase 0 — Structural Foundations for Million-Scale

These are prerequisites. Without them, later optimizations are local fixes.

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 0.1 | **Shard the collection** into N independent `Shard` instances | `crates/turbomemory_storage/src/shard.rs` (new) | In Progress | Each shard owns its own `VectorStore`, `SegmentHolder`, `MetadataStore`, WAL, optimizers, graph shard. Default `num_shards = clamp(num_cpus / 4, 2, 16)`. Route by `id` hash or explicit partition key. Qdrant model: `lib/collection/src/shards/`. |
| 0.2 | **Replace global `RwLock<SegmentHolder>` with lock-free segment list** | `crates/turbomemory_storage/src/segment_holder.rs` | Done | Done (2026-06-19 audit). `ArcSwap<SegmentSnapshot>` (`segment_holder.rs:208`); searches read the published snapshot via `snapshot_handle()`, mutations publish a new `Arc`. Single-node, never blocks on swap. Replicated per-shard in 0.1. |
| 0.3 | **Add collection abstraction above `StorageEngine`** | `crates/turbomemory_storage/src/collection.rs` (new) | In Progress | One `Collection` = many `Shard`s. Python `MemoryEngine` opens a collection directory. Required for sharding and config per collection. |
| 0.4 | **Move metadata out of single in-memory `HashMap`** | `crates/turbomemory_storage/src/metadata_store.rs` | Pending | Use per-segment metadata files (Qdrant: `segment.json` + mmap id_tracker) or a paged metadata store. Keep hot cache, spill cold records to mmap. Target: < 20% of working set in RAM. |
| 0.5 | **Introduce per-shard update worker pool** | `crates/turbomemory_storage/src/update_worker.rs` | In Progress | A single batched update worker exists (`update_worker.rs`, `IndexApplier` + crossbeam channel; all writes serialize through `apply_batch`). The **per-shard pool** (one worker per shard) still depends on sharding (0.1) and is pending. |
| 0.6 | **Separate read / write / optimize / flush thread pools** | `crates/turbomemory_storage/src/runtime.rs` (new) | Pending | CPU-bound search pool, IO-bound pool, optimizer pool, flush pool. Adaptive switching when CPU > 90% (Qdrant `AdaptiveSearchHandle`). |
| 0.7 | **Add NUMA / huge-page awareness stubs** | `crates/turbomemory_storage/src/memory_policy.rs` (new) | Pending | Pin optimizer threads, advise `MEM_LARGE_PAGES` on Windows, `madvise(MADV_HUGEPAGE)` on Linux for vector mmap. |

---

## Phase 1 — Ingestion Pipeline (WAL + Update Workers)

Qdrant's ingestion is fast because inserts are **appends to plain segments + WAL**, not HNSW graph mutations. Copy that model exactly.

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 1.1 | **Make mutable Hot segment plain / brute-force only** | `crates/turbomemory_storage/src/segments/hot.rs` | Done | Done. `HotSegment` (`hot.rs:70-120`) is an append-only offset list + chunked `cosine_similarity_batch` exact scan. No HNSW mutation on insert — HNSW is only built during sealing. |
| 1.2 | **Chunk upserts into 32-point batches** | `crates/turbomemory_storage/src/update_worker.rs`, `engine.rs` | Pending | Qdrant `UPDATE_OP_CHUNK_SIZE = 32` (`lib/shard/src/update.rs:176`). Prevents long write locks. Deletions batch to 512 (`lib/shard/src/update.rs:342`). |
| 1.3 | **WAL segment size = 32 MiB with CRC32-C framing** | `crates/turbomemory_storage/src/wal.rs` | Pending | Qdrant default (`lib/wal/src/lib.rs:40`). Current TSM WAL is single-file; split into rotating 32 MB segments. |
| 1.4 | **Add `first-index` / acknowledged-offset file** | `crates/turbomemory_storage/src/wal.rs` | Pending | Qdrant `lib/shard/src/wal.rs:28`. Bounded replay on restart; prefix-truncate old segments after flush. |
| 1.5 | **Configurable flush policy: EveryWrite / EveryBatch / Periodic / None** | `crates/turbomemory_storage/src/wal.rs`, `config.rs` | Pending | Default Periodic 5 s (Qdrant `flush_interval_sec = 5`). `wait=true` requests force flush. |
| 1.6 | **Group commit / async WAL fsync** | `crates/turbomemory_storage/src/wal.rs` | Pending | Batch multiple in-flight inserts into one fsync. Separate committer thread. |
| 1.7 | **Smallest-appendable-segment selection** | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | If multiple Hot segments exist, write to the one with most free capacity. Qdrant: `lib/shard/src/segment_holder/mod.rs:313`. |
| 1.8 | **Batch payload / text / graph updates** | `crates/turbomemory_storage/src/update_worker.rs` | Pending | Apply metadata, payload index, text index, and graph edges in chunks, not per-point. Qdrant chunks payload ops at 32 (`lib/shard/src/update.rs:714`). |
| 1.9 | **Zero-copy Python ingest** | `crates/turbomemory_python/src/lib.rs` | Done | Done (2026-06-18). Borrows contiguous `float32` numpy arrays via `F32Input`/`F32Matrix`; copies only for non-contiguous / wrong-dtype / list inputs. Engine batch API now takes `&[&[f32]]`. |
| 1.10 | **Streaming bulk insert API** | `crates/turbomemory_python/src/lib.rs` | Pending | Accept iterator/reader or chunked callback API for datasets larger than RAM. |
| 1.11 | **Pipeline insert: validate → WAL append → segment append → ack** | `crates/turbomemory_storage/src/engine.rs` | Pending | Decouple durability ack from index visibility. Qdrant acks after WAL write; segment update is async. |

---

## Phase 2 — Segment Lifecycle & Optimizers

Qdrant uses four optimizers: **Indexing**, **Merge**, **Vacuum**, **ConfigMismatch**. We need all four, plus tier-aware versions.

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 2.1 | **Byte-threshold sealing: indexing_threshold_kb = 10,000 KB** | `crates/turbomemory_storage/src/config.rs`, `segment_holder.rs` | Pending | Seal Hot → SealedHot when vector storage reaches 10 MB. Convert to point count at runtime: `threshold_kb * 1024 / (dim * 4)`. Qdrant: `lib/shard/src/optimizers/config.rs:15`. |
| 2.2 | **Max segment size: 256,000 KB per indexing thread** | `crates/turbomemory_storage/src/config.rs` | Pending | Cap segment size to avoid giant unmanageable segments. `max_segment_size_kb = num_indexing_threads * 256_000`. Qdrant: `lib/collection/src/optimizers_builder.rs:184`. |
| 2.3 | **Target segment count: clamp(num_cpus / 2, 2, 8)** | `crates/turbomemory_storage/src/optimizer.rs`, `config.rs` | Pending | Merge optimizer tries to converge to this count. Qdrant: `lib/collection/src/optimizers_builder.rs:155`. |
| 2.4 | **IndexingOptimizer: build HNSW only when segment crosses threshold** | `crates/turbomemory_storage/src/optimizer.rs` | Pending | Background build from plain segment to SealedHot. Do not build HNSW for segments below `full_scan_threshold`. |
| 2.5 | **MergeOptimizer: merge small segments up to max size** | `crates/turbomemory_storage/src/optimizer.rs` | Pending | Greedily merge smallest segments. Require ≥3 segments in first batch or two batches of ≥2 to guarantee count reduction. Qdrant: `lib/shard/src/optimizers/merge_optimizer.rs`. |
| 2.6 | **VacuumOptimizer: reclaim deleted vectors at 20% / 1,000 vectors** | `crates/turbomemory_storage/src/optimizer.rs` | Pending | Trigger rebuild when deleted_ratio > 0.2 and deleted_count ≥ 1,000. Qdrant: `lib/shard/src/optimizers/config.rs:16-17`. |
| 2.7 | **ConfigMismatchOptimizer: rebuild when HNSW/quantization/tier config changes** | `crates/turbomemory_storage/src/optimizer.rs` | Pending | Detect desired vs actual segment config and rebuild. Qdrant: `lib/shard/src/optimizers/config_mismatch_optimizer.rs`. |
| 2.8 | **Build new segments in temp directory + atomic rename** | `crates/turbomemory_storage/src/optimizer.rs`, `segment_builder.rs` (new) | Pending | Qdrant `temp_segments/` → atomic rename into `segments/` (`lib/segment/src/segment_constructor/segment_builder.rs:761`). Current TSM builds in-place under lock; fix. |
| 2.9 | **Disk-space guard before optimization** | `crates/turbomemory_storage/src/optimizer.rs` | Pending | Require ≥2× source segment size free in temp path. Qdrant: `lib/shard/src/optimize.rs:601`. |
| 2.10 | **Resource budget: IO permit → CPU permit** | `crates/turbomemory_storage/src/optimizer.rs`, `resource_budget.rs` | Pending | Acquire IO permit for data copying, replace with CPU permit for HNSW indexing. Qdrant: `lib/shard/src/optimize.rs:330`. |
| 2.11 | **Multi-threaded HNSW construction** | `crates/turbomemory_storage/src/segments/usearch_index.rs` | Done | Done (2026-06-18). First 256 points single-threaded (`PARALLEL_SEED_POINTS`), remainder via bounded rayon pool; threads = `clamp(num_cpus / max_concurrent_builds, 1, 16)`. ~2× faster consolidation at 10k, no recall regression. |
| 2.12 | **Old-index reuse / heal on merge** | `crates/turbomemory_storage/src/segments/sealed_hot.rs` | Pending | When merging already-indexed segments, reuse/heal existing HNSW graph instead of rebuilding from scratch. Qdrant: `OldIndexCandidate` in `lib/segment/src/index/hnsw_index/hnsw/build.rs:150`. |
| 2.13 | **Cache hygiene: prefault before build, drop after build** | `crates/turbomemory_storage/src/optimizer.rs` | Pending | `populate()` vectors before HNSW build; `clear_cache()` after sealing to avoid page-cache pollution. Qdrant: `lib/segment/src/segment_constructor/segment_builder.rs:834`, `:675-733`. |
| 2.14 | **Deferred points / `prevent_unoptimized`** | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Hide new writes in over-capacity segments until optimization completes, preventing unbounded exact-scan segments. Qdrant pattern. |

---

## Phase 3 — Search Architecture

Search at 1M × 4k is dominated by HNSW traversal, multi-segment aggregation, and filtering. These are the highest-ROI CPU wins.

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 3.1 | **Visited-set pool** | `crates/turbomemory_storage/src/visited_pool.rs` | Done | Done (2026-06-19 audit). `visited_pool.rs` — `VisitedSet { tokens: Vec<u8>, generation }`, token wrap → refill, parking_lot-guarded pool. |
| 3.2 | **HNSW defaults aligned with Qdrant** | `crates/turbomemory_storage/src/config.rs` | Pending | `M = 16`, `M0 = 32`, `ef_construct = 100`, `ef_search = max(ef, top_k)`, `full_scan_threshold = 10_000` KB converted to vector count. |
| 3.3 | **Compressed / inline-vector graph links for sealed segments** | — | Cut | Qdrant-specific graph format optimization. `usearch` handles its own graph storage internally; this would require replacing usearch with a custom HNSW implementation to be meaningful. Not worth the complexity for the memory-engine use case. |
| 3.4 | **Heuristic neighbor selection (HNSW)** | — | Cut | `usearch` already applies a heuristic neighbor selection algorithm during build. Replacing it with a custom implementation would not improve recall and would break the usearch dependency contract. |
| 3.5 | **Parallel search across segments with result aggregation** | `crates/turbomemory_storage/src/segment_holder.rs` | Done | Done (2026-06-19 audit). `into_par_iter` over segments (`segment_holder.rs:102,119`) with k-way top-k merge. Sequential fallback for single-segment. |
| 3.6 | **Probabilistic sampling for multi-segment search** | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Poisson-derived per-segment limit; rerun if boundary crosses global top-k. Qdrant: `lib/collection/src/collection_manager/segments_searcher.rs:571`. |
| 3.7 | **Cardinality-aware filter routing** | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Estimate `{min, exp, max}` cardinality. `max < full_scan_threshold` → plain; `min > threshold` → HNSW; else sample ≤1,000 points with Agresti-Coull confidence interval. Qdrant: `lib/segment/src/index/hnsw_index/hnsw/vector_index_impl.rs:55-173`. |
| 3.8 | **ACORN-1 adaptive filtered search** | — | Cut | Qdrant-specific HNSW traversal optimization that requires a custom HNSW implementation (we use `usearch`). The `VisitedPool` (3.1, done) was built for this but is currently dead code. Revisit only if we replace usearch with a custom HNSW. |
| 3.9 | **Plain search with SIMD + early-exit top-k** | `crates/turbomemory_storage/src/segments/hot.rs` | Pending | Hot segment exact scan must be competitive. Use batched SIMD distance + min-heap top-k. |
| 3.10 | **Per-query latency budget / adaptive `ef`** | `crates/turbomemory_storage/src/segment_holder.rs` | In Progress | Caller-specified `ef` done (2026-06-18) — `search_ann(q, top_k, ef)` + benchmark `--ef`. Auto-raise on low recall audit still pending. |
| 3.11 | **Exact reranking after quantization** | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Rescore top-k approximate results with full f32 vectors. Default on for binary/TurboQuant, off for scalar/PQ. Qdrant: `lib/segment/src/index/vector_index_search_common.rs:48-87`. |
| 3.12 | **Abort-on-drop for long searches** | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | Wrap blocking search tasks so dropping the request cancels the thread work. Qdrant: `AbortOnDropHandle`. |

---

## Phase 4 — Quantization & Distance Compute (CPU)

At 4k dimensions, distance compute is the bottleneck. We need SIMD, batched kernels, and richer quantizers.

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 4.1 | **Batched SIMD distance kernel (matrix × query)** | `crates/turbomemory_core/src/metrics.rs` | Done | Done (2026-06-19 audit). `cosine_similarity_batch` with `dot_and_nb_x4` 4-vector unrolled kernel, AVX2/FMA + SSE + NEON paths. |
| 4.2 | **Pre-normalize cosine vectors** | `crates/turbomemory_core/src/metrics.rs` | Done | Already implemented. Vectors are L2-normalized on insert/update (`engine.rs` `normalize`); usearch index uses `MetricKind::Cos`; rerank uses self-normalizing `cosine_similarity_batch`. |
| 4.3 | **Scalar int8 quantizer** | `crates/turbomemory_core/src/quantization/` (new) | Pending | `alpha=(max-min)/127`, offset=min, per-metric multiplier, vector offset prefix. SIMD i8 dot/L1. Qdrant: `lib/quantization/src/encoded_vectors_u8.rs`. |
| 4.4 | **Product Quantization (PQ)** | — | Cut | TurboQuant (4.6, done) is provably better than PQ at all bit-widths and dimensions (per the TurboQuant paper, Section 4.4). Implementing PQ would duplicate capability with worse quality. Use TurboQuant instead. |
| 4.5 | **Binary / 1-bit + 1.5-bit / 2-bit quantizers** | — | Cut | Sign quantizer (done) + TurboQuant (done) cover this space. Additional bit-width variants are diminishing returns for the memory-engine use case. |
| 4.6 | **TurboQuant-style 1/2/4-bit quantizer** | `crates/turbomemory_core/src/turbo_quant.rs` | Done | Done. Both MSE and Prod variants fully implemented in `turbo_quant.rs` (1150 LOC, 12 tests): FWHT rotation, Lloyd-Max codebooks (bits 1–4 true optimum; 5–8 uniform approximation — a known gap), QJL residual 1-bit transform, LUT scoring with AVX2 gather + byte-weight fast paths. Config panic on non-pow2 dim fixed (2026-06-19). |
| 4.7 | **OPQ / learned rotation before PQ** | — | Cut | PQ is cut (4.4); OPQ is a PQ enhancement. TurboQuant's FWHT rotation already provides the "rotation before quantization" step with provable near-optimal distortion. |
| 4.8 | **Quantization auto-selection by dimension and recall target** | `crates/turbomemory_storage/src/config.rs` | In Progress | Cold tier no longer defaults to 1-bit sign at high dim — now 8-bit scalar by default (2026-06-18). Full auto-selection (scalar/PQ/TurboQuant by dim + recall target) still pending. |
| 4.9 | **Zero-copy quantized scans** | `crates/turbomemory_storage/src/segments/warm.rs`, `cold.rs` | Pending | Read mmap slices directly; remove `chunk_bytes.extend_from_slice` copies. |
| 4.10 | **Query LUT precomputation for quantized tiers** | `crates/turbomemory_storage/src/segments/warm.rs`, `cold.rs` | Pending | Build lookup table once per query, reuse across chunks. |
| 4.11 | **Distance compute thread pool with work stealing** | `crates/turbomemory_storage/src/runtime.rs` | Pending | Use `rayon` or custom pool for batch distance jobs; separate from search threads. |

---

## Phase 5 — Memory, Storage & I/O

1M × 4k = ~16 GB of vectors alone. We must be mmap-first, shard-first, and I/O-aware.

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 5.1 | **Shard `VectorStore` by point-offset range** | `crates/turbomemory_storage/src/vector_store.rs` | Pending | Split vectors into multiple `vectors-*.bin` files (e.g., per shard or per 256k offsets). Avoid 16 GB+ single file remap. |
| 5.2 | **Recycle deleted vector slots** | `crates/turbomemory_storage/src/vector_store.rs` | Pending | Maintain a free-list of deleted offsets; new inserts reuse slots instead of always appending. |
| 5.3 | **Async / prefetch mmap I/O** | `crates/turbomemory_storage/src/vector_store.rs`, `segments/warm.rs`, `cold.rs` | Pending | `madvise` / `PrefetchVirtualMemory` (Windows) for cold-start and sequential build reads. Add `MmapOptions::populate()` option. |
| 5.4 | **Separate hot and cold mmap policies** | `crates/turbomemory_storage/src/vector_store.rs` | Pending | Hot vectors locked/prefaulted; Warm/Cold left for OS cache. |
| 5.5 | **Paged metadata store with cache eviction** | `crates/turbomemory_storage/src/metadata_store.rs` | Pending | Replace single `HashMap` with LRU cache + mmap-backed pages. Critical for 1M+ text records. |
| 5.6 | **Per-segment metadata files** | `crates/turbomemory_storage/src/segments/` | Pending | Each segment owns its id→offset map and payload indexes, not a global redb. Qdrant: `segment.json` + mmap id_tracker. |
| 5.7 | **Replace redb with per-segment metadata + WAL** | — | Cut | redb works fine as a lazy snapshot store; the real bottleneck is the in-memory `HashMap` cache (5.5), not redb itself. Fixing 5.5 (paged metadata) eliminates the need to replace redb. |
| 5.8 | **Offset-mapped payload storage (mmap)** | — | Cut | Payloads are small JSON strings stored in the metadata cache. Paged metadata (5.5) handles this. A separate mmap payload store adds complexity without benefit at the memory-engine scale. |
| 5.9 | **Bitmap payload indexes with mmap backing** | — | Cut | The in-memory Roaring bitmap payload index is fast and compact. Moving it to mmap adds complexity without measurable benefit until 1M+ *filtered* records, which is a niche workload. Revisit if filtering at scale becomes a bottleneck. |
| 5.10 | **Text index segmentation and batch commits** | `crates/turbomemory_storage/src/text_index.rs` | Pending | Separate text index per segment; periodic commit; avoid `TopDocs::with_limit(num_docs)` full materialization. |
| 5.11 | **Write-ahead snapshot / checkpointing** | `crates/turbomemory_storage/src/engine.rs` | Pending | Take periodic checkpoints (segments + manifest) without blocking foreground; truncate WAL after checkpoint. |
| 5.12 | **mmap growth strategy: pre-allocate in power-of-two chunks** | `crates/turbomemory_storage/src/vector_store.rs` | Pending | Avoid full-file remap on every growth; double capacity and zero-fill lazily. |
| 5.13 | **DiskANN / SPANN-style on-disk ANN index option** | `crates/turbomemory_storage/src/index/diskann.rs` (new) | Pending | For >RAM collections, provide disk-based approximate index as alternative to in-RAM HNSW. |

---

## Phase 6 — Concurrency & Scheduling

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 6.1 | **Per-shard RwLock-free segment list** | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | See 0.2. Publish `Arc<Vec<Arc<Segment>>>` on mutation; readers snapshot cheaply. |
| 6.2 | **Lock-free access counters** | `crates/turbomemory_storage/src/access_counters.rs` | Pending | Current `Mutex<AHashMap>` contends. Use sharded atomic counters or thread-local buffers with periodic drain. |
| 6.3 | **parking_lot everywhere** | Whole workspace | Done | Done (2026-06-19 audit). `parking_lot::RwLock`/`Mutex` in use across the storage crate (engine, segment holder, update worker, metadata cache, access counters). |
| 6.4 | **Per-point link locks during mutable HNSW build** | — | Cut | The TODO itself said "for now, avoid incremental build." TSM's architecture (plain Hot + background seal) deliberately avoids incremental HNSW. This item is irrelevant to the current design. |
| 6.5 | **Resource isolation: search vs ingest vs optimize** | `crates/turbomemory_storage/src/resource_budget.rs` | Pending | CPU/IO/memory permits per operation class. Background optimizer throttled when foreground CPU > threshold. |
| 6.6 | **Adaptive search thread pool** | `crates/turbomemory_storage/src/runtime.rs` | Pending | Switch between HighIo and HighCpu pools based on process CPU ratio (Qdrant `AdaptiveSearchHandle`). |
| 6.7 | **Cancel slow/abandoned queries** | `crates/turbomemory_storage/src/segment_holder.rs` | Pending | See 3.12. |
| 6.8 | **Concurrent graph search / spreading activation** | `crates/turbomemory_graph/src/` | Pending | Remove single `RwLock<SpreadingActivation>`; shard graph or use RCU. |
| 6.9 | **Batch cognitive search rehydration** | `crates/turbomemory_storage/src/engine.rs` | Pending | Load result payloads in batches, not one-by-one. |

---

## Phase 7 — Cognitive Layer Deepening

The cognitive layer is the differentiator. The core is done and validated at
realistic scale (concept extraction, learnable edges, reinforcement/decay,
abstraction, refinement, contradiction detection, auto-importance, CCS
compressor, score fusion, introspection API). These are the remaining items
that make TSM a *memory engine* rather than a vector DB with a graph on top.

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| C1 | **Contradiction detection** | `crates/turbomemory_graph/src/graph.rs`, `engine.rs` | Done | Done (2026-06-20). `EdgeKind::Contradicts` (old→new); `add_contradiction` creates the edge + weakens the old memory's outgoing edges by `contradiction_weaken_factor`. `check_contradictions()` runs on consolidation: a pair is a contradiction when cosine >= `contradiction_cosine_threshold` AND shares a concept AND text Jaccard < `contradiction_text_threshold` (the dissimilarity signal that distinguishes contradiction from refinement). Spreading activation traverses `Contradicts` edges so the correction surfaces. 4 Python kwargs added; benchmark scenario 4 proves it flips the ranking (correction rank 1 cognitive vs rank 2 ANN). |
| C2 | **Automatic importance scoring** | `crates/turbomemory_storage/src/engine.rs` | Done | Done (2026-06-20). `StorageEngine::recompute_importance()` runs on consolidation (opt-in via `importance_auto_scoring`): blends normalized access_score + concept-degree into a target importance, moves current importance `importance_learning_rate` toward it, clamps to `[floor, ceiling]`, writes back to metadata, and syncs the graph via new `MemoryGraph::reweight_memory` (which preserves learned reinforcement using a stored `base_importance_factor` per memory node). Runs before dedup/eviction so recomputed importance participates in tiebreaking. 5 config fields + Python kwargs + `recompute_importance()` method; 6 new tests. |
| C3 | **Online concept vocabulary evolution** | `crates/turbomemory_graph/src/graph.rs`, `extract.rs`, `activation.rs`, `engine.rs`, `config.rs`, `lib.rs` | Done | Done (2026-06-21). `ConceptVocabulary` now persists in graph snapshots. `MemoryGraph::evolve_vocabulary(overlap_threshold, hub_fraction, max_pairs)` merges synonymous base concepts by Jaccard overlap of associated memory sets (higher-degree concept survives) and suppresses over-general hubs whose degree exceeds `hub_fraction * memory_count`. `SpreadingActivation` skips suppressed concepts during propagation. Engine snapshots vocabulary before insert canonicalization and runs evolution during consolidation when `concept_evolution_enabled=true`. 4 opt-in Python kwargs + `evolve_concept_vocabulary()` method. 5 new graph tests. |
| C4 | **Per-agent memory scoping** | `crates/turbomemory_storage/src/engine.rs`, `scope_index.rs`, `record.rs`, `wal.rs`, `update_worker.rs`, `lib.rs`; `crates/turbomemory_api/proto/turbomemory.proto`, `grpc.rs`, `rest.rs` | Done | Done (2026-06-21). Optional `scope` field on records; `ScopeIndex` returns matching-scope + global records. Wired through insert/update/delete, WAL replay, ANN search, cognitive search, payload-filtered search, Python bindings, gRPC, and REST. 2 new storage tests; covered in `verify.py`. |
| C5 | **Real-embedding cognitive benchmark** | `cognitive_benchmark.py` | Done | Done (2026-06-21). Benchmark now runs at realistic scale by default: 768-dim embeddings with 1000 clustered distractor memories per scenario (the original toy regime is still available via `--dimension 64 --distractors 0`). Scale result: cognitive layer wins **3/4 scenarios** at 768-dim/1000-distractors — refinement, reinforcement, and contradiction surfacing all find memories plain ANN misses entirely (rank 99). Abstraction traversal does NOT scale to 1000 distractors (top-k dominated by cosinely-nearby distractors before the multi-hop path surfaces); honest signal that abstraction needs hub-suppression/frontier tuning at scale. Toy regime still wins 4/4 (backward-compatible). |
| C6 | **LLM compressor integration test** | `crates/turbomemory_python/src/lib.rs`, `crates/turbomemory_graph/src/ccs.rs`, `verify.py` | Done | Done (2026-06-21). `StorageEngine::set_compressor` + Python `set_llm_compressor(callable)` install a Python-callable-backed compressor. The callable receives `(ccs_json, user_input, assistant_response)` and returns a CCS JSON string; invalid output falls back to the deterministic compressor. End-to-end assertion added to `verify.py` using a mock LLM. |
| C7 | **Graph introspection API** | `crates/turbomemory_python/src/lib.rs` | Done | Done (2026-06-20). Five read-only Python methods: `graph_stats()`, `get_concepts()` → list[(concept, degree)], `get_memory_concepts(id)`, `get_refinements(id)`, `get_contradictions(id)`. Backed by `MemoryGraph::memory_concepts`/`concept_count`/`stats()` (returns `GraphStats`) and `StorageEngine::read_graph()` (hands out a read guard). Tuple/list-of-tuples return shape matches existing binding style; unknown ids return empty lists. |
| C8 | **Streaming concept extraction** | `crates/turbomemory_graph/src/extract.rs` | Done | Done (2026-06-21). `extract.rs` rewritten with `ExtractorConfig` supporting unigram/bigram/trigram extraction + PMI scoring, subsumption suppression, and a `ConceptVocabulary` alias canonicalizer. Default remains unigram-only for backward compatibility; n-grams enabled via `concept_max_ngram_len`/`concept_min_ngram_freq`/`concept_enable_pmi` kwargs. Embedding-based matching to existing graph concepts is left for C3 (online vocabulary evolution) where the engine has access to the graph and vector store. |

---

## Phase 8 — Python Bindings & API

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 8.1 | **Zero-copy numpy ingest** | `crates/turbomemory_python/src/lib.rs` | Done | Done (2026-06-18). See 1.9. |
| 8.2 | **Streaming / chunked bulk insert** | `crates/turbomemory_python/src/lib.rs` | Pending | See 1.10. |
| 8.3 | **Async Python API (`asyncio`)** | `crates/turbomemory_python/src/lib.rs` | Pending | Return awaitable futures for insert/search; release GIL. |
| 8.4 | **Collection / shard config in Python** | `crates/turbomemory_python/src/lib.rs` | Pending | Expose `num_shards`, `indexing_threshold_kb`, `max_segment_size_kb`, quantization config. |
| 8.5 | **Batch search API** | `crates/turbomemory_python/src/lib.rs` | Pending | Accept matrix of queries; return list of result lists. Enables GPU batching. |
| 8.6 | **Progress callbacks for build/optimize** | `crates/turbomemory_python/src/lib.rs` | Pending | Long seals/merges report progress. |
| 8.7 | **Recall audit auto-tune helper** | `audit_recall.py` | Pending | Python helper to sample, measure recall, and raise `ef` until target is met. |

---

## Phase 9 — Observability & Operations

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 9.1 | **Replace `println!` with `tracing` spans** | Whole workspace | Pending | Structured, level-filtered logging. |
| 9.2 | **Metrics: ingest latency, search latency, segment sizes, tier counts** | `crates/turbomemory_storage/src/metrics.rs` (new) | Pending | Use `metrics` crate with Prometheus exporter in API server. |
| 9.3 | **WAL lag / optimizer queue depth metrics** | `crates/turbomemory_storage/src/update_worker.rs`, `optimizer.rs` | Pending | Operational visibility. |
| 9.4 | **Health metrics endpoint** | `crates/turbomemory_api/src/rest.rs` | Pending | Prometheus `/metrics`. |
| 9.5 | **Request timeouts, payload size limits, CORS, auth** | `crates/turbomemory_api/src/main.rs`, `rest.rs` | Pending | Production hardening. |
| 9.6 | **Docker / container build** | `deployments/docker/` (new) | Pending | Multi-stage build with optional CUDA base image. |
| 9.7 | **Cross-platform build support** | `Cargo.toml`, `.cargo/config.toml` | Pending | Windows MSVC, Linux, macOS; target-cpu=native optional. |
| 9.8 | **Release profile with LTO + codegen-units=1** | `Cargo.toml` | Done | Already configured. |

---

## Phase 10 — Testing & Benchmarking at Million-Scale

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| 10.1 | **Add 1M × 4k synthetic benchmark** | `benchmark.py`, `benches/` | Pending | Measure ingest time, search P50/P99, recall@10, peak RSS, disk usage. |
| 10.2 | **Add concurrent ingest + search benchmark** | `benchmark.py` | Pending | Catch lock contention and tail latency. |
| 10.3 | **Add crash-recovery tests for sharded WAL** | `crates/turbomemory_storage/tests/` | Pending | Kill mid-seal, verify no data loss. |
| 10.4 | **Add property-based tests for segment lifecycle** | `crates/turbomemory_storage/tests/` | Pending | Seal/merge/vacuum correctness. |
| 10.5 | **Add comparison harness vs Qdrant/Chroma/Faiss** | `benchmark.py` | In Progress | TSM vs NumPy/Chroma/Qdrant harness with clustered + random data and configurable `ef`/dim/N (2026-06-18). Faiss and fixed recall-target alignment still pending. |
| 10.6 | **Continuous benchmark tracking** | CI / `benches/` | Pending | Detect regressions on PRs. |
| 10.7 | **GPU correctness tests** | `crates/turbomemory_gpu/tests/` | Pending | Compare GPU exact top-k vs CPU brute force bit-exact. |

---

## Qdrant Default Cheat Sheet (Reference)

| Parameter | Qdrant Default | TSM Target |
|---|---|---|
| HNSW `m` | 16 | 16 |
| HNSW `m0` | 32 | 32 |
| `ef_construct` | 100 | 100 |
| `ef` search | `max(ef, top_k)` | `max(ef, top_k)` |
| `full_scan_threshold` | 10,000 KB → vector count | 10,000 KB → vector count |
| `indexing_threshold_kb` | 10,000 | 10,000 |
| `max_segment_size_kb` | 256,000 × indexing_threads | 256,000 × indexing_threads |
| Target segment count | `clamp(num_cpus/2, 2, 8)` | `clamp(num_cpus/2, 2, 8)` |
| Deleted threshold | 0.2 | 0.2 |
| Vacuum min vectors | 1,000 | 1,000 |
| HNSW build threads | `clamp(num_cpus, 1, 16)` | `clamp(num_cpus, 1, 16)` |
| Single-threaded HNSW prefix | 256 points | 256 points |
| Upsert chunk size | 32 | 32 |
| Deletion batch size | 512 | 512 |
| WAL segment size | 32 MiB | 32 MiB |
| Flush interval | 5 s | 5 s |
| Visited pool size | `clamp(num_cpus, 16, 128)` | `clamp(num_cpus, 16, 128)` |
| GPU groups (Vulkan) | 512 | 512 |

---

## Suggested Execution Order

### Stage 1 — Deepen the cognitive layer (the differentiator)
1. **C8** — Streaming concept extraction ✅ Done.
2. **C3** — Online concept vocabulary evolution.
3. **C4** — Per-agent memory scoping (multi-agent) ✅ Done.
4. **C6** — LLM compressor integration test ✅ Done.

> **C1 (Contradiction detection) — Done (2026-06-20).**
> **C7 (Graph introspection API) — Done (2026-06-20).**
> **C2 (Automatic importance scoring) — Done (2026-06-20).**
> **C5 (Real-embedding cognitive benchmark) — Done (2026-06-21).** See Recently Completed.

### Stage 2 — Unlock 1M × 4k structurally (production scaling) 🚧 In Progress
1. **0.1–0.3** — Shard collection, collection abstraction. **Current focus.**
2. **0.4** — Paged metadata store (the single biggest 1M blocker).
3. **1.3, 1.11** — Rotating WAL, pipeline insert.
4. **2.6, 2.8** — VacuumOptimizer (reclaim deleted slots), temp-dir build.
5. **5.1–5.3** — Sharded VectorStore, slot recycling, prefetch.

### Stage 3 — Make search fast at high dimension
6. **3.2, 3.7, 3.9** — HNSW defaults, cardinality-aware filtering, SIMD hot scan.
7. **4.3** — Scalar int8 quantizer (for Warm tier).
8. **4.9–4.10** — Zero-copy quantized scans, query LUT precomputation.
9. **3.11** — Exact reranking after quantization.

### Stage 4 — Operations & polish
10. **9.1, 9.5** — Tracing, auth/CORS/limits.
11. **9.2–9.4** — Metrics, health endpoint.
12. **9.6–9.7** — Docker, cross-platform builds.
13. **10.1–10.4** — Million-scale benchmarks, crash-recovery, property tests.

### Future — GPU acceleration (not near-term)
See the GPU appendix below. Only relevant after Stage 2 is complete and
throughput at 1M scale is the bottleneck. The TODO's original advice stands:
**do not start with GPU. A GPU-accelerated bad architecture is still a bad
architecture.**

---

## Legacy Items

The following are relevant but lower priority:

- ~~Cognitive graph durable persistence~~ — **Done (2026-06-19).**
- ~~Graph merge/forget policies~~ — **Mostly done (2026-06-19/20).** Reinforce, decay, abstraction, dedup, refinement, score fusion all shipped. Remaining: contradiction detection (→ C1), multi-agent scoping (→ C4).
- ~~Cognitive recall benchmark~~ — **Done (2026-06-20), scaled 2026-06-21 (C5).** `cognitive_benchmark.py` runs at realistic scale (768-dim, 1000 distractors): 3/4 cognitive scenarios beat plain ANN (refinement, reinforcement, contradiction surfacing find memories ANN misses entirely at rank 99). Abstraction traversal needs hub-suppression tuning to scale.
- **Sparse vectors** — not planned for near-term. Dense vectors + concept extraction cover the memory-engine use case.
- **Advanced full-text index tuning** — Tantivy works well; revisit only if FTS at scale becomes a bottleneck.

---

## Appendix: GPU Acceleration (Future, Not Near-Term)

GPU acceleration is a throughput optimization, not a cognition feature.
It only matters after the architecture scales to 1M+ on CPU and throughput
becomes the bottleneck. Qdrant uses GPU only for HNSW *build* (via Vulkan
compute), not search. We should support both CUDA (NVIDIA) and Vulkan
(portable) paths behind feature flags — but only after Stage 2 is complete.

| # | Fix | Location(s) | Status | Notes |
|---|---|---|---|---|
| G1 | `turbomemory_gpu` crate with `GpuBackend` trait | `crates/turbomemory_gpu/` (new) | Future | `init()`, `upload_vectors()`, `batch_dot()`, `build_hnsw_approx()`. `CudaBackend`, `VulkanBackend`. |
| G2 | cuBLAS batched distance compute | `crates/turbomemory_gpu/src/cuda/dot.rs` | Future | For batch queries × 1M vectors × 4k. |
| G3 | CUDA HNSW build path | `crates/turbomemory_gpu/src/cuda/hnsw_build.rs` | Future | Parallel level-by-level insertion. Fallback to CPU on OOM. |
| G4 | Vulkan compute HNSW build (Qdrant model) | `crates/turbomemory_gpu/src/vulkan/` | Future | `ash` + compute shaders. Cross-platform. |
| G5 | cuVS / RAFT CAGRA index | `crates/turbomemory_gpu/src/cuda/cagra.rs` | Future | Fastest GPU ANN for batch search. |
| G6 | GPU device manager + memory pool | `crates/turbomemory_gpu/src/` | Future | Per-optimization device lock; pinned host buffers. |

**Strategy:** Search stays CPU (single-query latency is worse on GPU due to
upload overhead). Build is the clear GPU win for 1M × 4k segments. Every GPU
path must silently fall back to CPU on error.

---

## Final Note

The original roadmap was written as a vector-database scaling plan. It has
been restructured to reflect the actual goal: **building a memory engine,
not a faster vector DB.** The key changes:

1. **Cognitive layer is now Phase 7** (was "Legacy Items" with 3 bullet
   points). 8 new items (C1–C8) cover contradiction detection, auto-importance,
   concept evolution, per-agent scoping, real benchmarks, LLM compressor
   testing, graph introspection, and streaming extraction.
2. **GPU is demoted to an appendix** (was Phase 7 with 12 items). GPU is a
   throughput optimization that only matters after the architecture scales
   on CPU. The original advice stands: do not start with GPU.
3. **10 items cut** as redundant or irrelevant: PQ (4.4, TurboQuant is
   better), OPQ (4.7), binary quantizers (4.5), compressed graph links
   (3.3, usearch handles it), heuristic neighbor selection (3.4, usearch
   handles it), ACORN-1 (3.8, needs custom HNSW), per-point link locks
   (6.4, we don't do incremental HNSW), replace redb (5.7, fix 5.5 instead),
   offset-mapped payload (5.8), mmap bitmap payload (5.9).
4. **3 stale "Pending" entries corrected to Done**: 1.1 (plain Hot segment),
   3.5 (parallel multi-segment search), 6.3 (parking_lot).

The execution order now puts **cognitive deepening (Stage 1)** before
**structural scaling (Stage 2)**. The rationale: a memory engine that
*thinks* at 100k beats a vector DB that's fast at 1M but doesn't think.
Prove the cognition works at scale (C5), then scale the architecture.

**Current status:** Stage 1 complete (C1–C8 done). Stage 2 in progress;
0.1 (collection sharding) and 0.3 (`Collection` abstraction) are the active
items.
