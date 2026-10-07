# TurboSuperMemory — Open Work

The roadmap is [`PLAN.md`](./PLAN.md) (Roadmap v2 / v2.1). The record of what
was built and measured is [`benchmarks/PHASE_PROGRESS.md`](./benchmarks/PHASE_PROGRESS.md).
This file is only the short list of open engineering work, grouped by area.

> **Trimmed 2026-10-05.** This file used to hold ~560 lines of dated
> "Recently Completed" logs and a ~110-item roadmap for a single 1M × 4k
> store. The logs duplicate git history and `PHASE_PROGRESS.md`, and
> `PLAN.md` (Phase C) retired the 1M × 4k axis in favor of many small,
> long-lived stores. Nothing was lost: `git show 39cbea4:TODO.md`.
>
> **2026-10-06.** A robustness pass fixed the durability, isolation,
> bad-input, concurrency and CUDA defects found by a from-scratch audit
> (see "Robustness pass" in `PHASE_PROGRESS.md`). What that audit found and
> this pass did NOT change is listed below, mostly under "Cognitive
> behaviour": those change what recall returns and need a judged re-run.
>
> **2026-10-07.** GPU search over device-resident vectors, `sync_writes`,
> a CI workflow, and verified belief revision landed (entry of that date in
> `PHASE_PROGRESS.md`). What each of them leaves open is listed under its
> heading below.
>
> **2026-10-07, later.** A judged end-to-end run of the shipped stack
> ("Judged end-to-end check" in `PHASE_PROGRESS.md`) found that no
> cognitive mechanism beats plain vector search with OpenAI embeddings, and
> why the shipped stack trailed it. The four items it produced lead the
> "SDK" and "Cognitive behaviour" lists below.

## Durability and operations

- **`sync_writes` has never met a real power cut.** It syncs the vector range
  and then the log record of every write before returning, and a test checks
  that every write path does so and recovers, but nothing here can cut the
  power, and a drive that acknowledges a flush it has not performed defeats
  it. It is off by default: without it a power loss (unlike a process kill)
  can still lose the writes since the last flush.
- **`sync_writes` pays one sync pair per write**: about 180 single inserts per
  second on an NVMe SSD against 11,000 without it; a batch of 100 costs about
  8 ms instead of 2.5 ms. Batch small writes. A middle policy (sync every N
  ms, or one sync shared by concurrent writers) is not implemented.
- **Blocking engine calls run on the async runtime** in every REST and gRPC
  handler. A slow `/flush` or consolidation stalls both transports, including
  `/health`. Move them to `spawn_blocking`.
- **No vacuum.** Deleted and evicted records leave their vectors in
  `vectors.bin`, and their offsets stay in immutable segments until those are
  rebuilt. Searches skip them correctly; a long-lived store under eviction
  pressure still grows on disk and searches wider pools.
- **Seal builds do not hold their resource-budget slot** (`optimizer.rs`: the
  guard from `try_acquire` is dropped at once), so `max_concurrent_builds` is
  not enforced for them. Holding it changes which tier a seal lands in when a
  build is already running; decide that deliberately.
- **Record offsets are truncated to 32 bits** wherever a roaring bitmap is
  used (`offset as u32`). Fine below 4.29 billion inserts per store; make it
  an explicit limit.
- **No format version for the store as a whole**, and `MetaRecord` is bincode
  (a new field breaks old snapshots). Only `vectors.bin`, the WAL and the
  graph snapshot are versioned.
- **A failed insert after the WAL append** (text-index failure) is reported as
  failed although the record is durable.
- **CI covers less than the local gate.** `.github/workflows/ci.yml` (fmt,
  clippy, the Rust suite and the SDK suite) passed on its first run on
  Linux, Windows and macOS on Apple Silicon: 294 Rust and 83 SDK tests on
  each. It has no GPU, datasets or models, so `make gate` and the CUDA tests
  stay local. The macOS job is still marked non-blocking; make it required
  once it has a few green runs. Linux on AArch64 is only linted
  (`turbomemory_core`, `turbomemory_graph`).
- Request timeouts, body-size configuration and CORS on the REST server.
- Metrics and tracing: the optimizer reports failures with `eprintln!`, and
  the Python extension initialises no logger, so the engine's warnings
  (recovery, GPU fallback, kept eviction victims) are invisible from Python.
- Container build, and a documented backup/restore procedure.

## Multi-tenancy

Scope and filter now bound every search stage, deduplication stays inside a
scope, and supersession cannot cross scopes. Still shared by design across the
scopes of one store:

- the `max_records` cap and its ranking (one scope's writes can evict
  another's records);
- importance normalization (store-wide maximum);
- concept vocabulary evolution (aliases and hub suppression);
- `step_session`'s working memory (one state per engine; over the API each
  caller's response contains fragments of other callers' turns).

The API does not enforce tenancy at all: `scope` is a caller-supplied field
under one shared key, `/get_payload`, `/delete` and `/update` act on any id,
and an omitted scope searches everything. In the SDK, `user_id=None` on `add`
or `recall` likewise means "everyone". PLAN Phase C (one engine per tenant) is
the intended answer; until then, treat a store as single-tenant unless these
are closed.

## SDK (`tsm/`)

- **Compacting as the conversation arrives scores lower than compacting
  once.** `Memory(max_user_tokens=256)` answers 0.464 of 112 judged
  questions when the store is compacted once after ingestion (the
  harness-built store: 0.482) and 0.411 when it is compacted four times
  along the way, which is what a deployment does (7 questions gained, 13
  lost, p=0.26). Each pass summarizes the previous pass's gists again;
  knowledge-update and multi-session questions are where it loses. To
  measure: keep earlier gists as they are until the gist share is full and
  only then merge the oldest; or compact only when a user is well over
  budget instead of at every `consolidate()`.
- **Gists carry no time cue.** `recall()` puts a date or a turn in front of
  each fact and leaves gists bare, so in a compacted store most of the
  history has no order. A gist could carry the span it covers
  (`[turns 3-41]`); `plan_compaction` would have to report which memories
  each gist was written from.
- **`max_user_tokens` only compacts in `consolidate()`, and only the users
  this process wrote to** (`compact()` covers every user). A store that is
  only ever added to grows until one of them is called.
- **The engine's `max_records` path is unchanged** and scores far lower on
  the same questions (0.148 with gists, 0.070 without, against 0.452 for
  `max_user_tokens`): one count for the whole store, the newest records
  kept whatever their role, one long gist per 24 evicted facts. Give it the
  same policy or keep steering conversational use to `max_user_tokens`, as
  the docs now do.
- **The judged head-to-head still measures `TSMAdapter`, not `tsm.Memory`.**
  The adapter has its own engine settings and adds keyword candidates.
  On the same questions the two are now level (adapter 0.583, `tsm.Memory`
  with everything on 0.591), so the published head-to-head and bounded
  tables can be re-run through `tsm.Memory` (`shipped_stack_eval.py` shows
  how) and the adapter retired.
- **`recall()` is not read-only.** A query reinforces what it returned, so
  the same question asked twice of one store comes back in a different
  order more often than not (13 of 19 questions checked, different members
  in 3). An evaluation has to build one store per arm, and a caller cannot
  rely on a repeated query. A read that does not reinforce is missing.
- **Extraction results cached before 2026-10-06 may be wrong.** The OpenAI
  extractor used to cache "no facts" for a reply that was cut off at 400
  tokens. It no longer does, but an existing `extract_<model>.json` cache can
  still hold such empty entries for long messages; they cannot be told apart
  from real "no facts" answers. Delete the cache to re-extract.
- **The caches hold every message, fact and query in plaintext**
  (`<db_path>/tsm_cache/`) and are never pruned, including after eviction.
  The embedding cache is still a pickle file, now read with an arrays-only
  loader; a plain format (npz) would be simpler.
- **`recall()` applies the role prior without re-sorting** unless a reranker or
  a budget is used, so on the default path the prior does not change the
  order and results are not sorted by the returned score. (The eval adapter
  does re-sort.)
- **Role prior is very broad.** `tsm/ranking.py` treats a query as first-person
  when it contains words like "what", "how", "have" or "did", so the user-fact
  boost applies to nearly every question. It is the cue set the evaluations
  ran with; narrowing it needs a judged re-run, not a guess.
- **Budget recall re-embeds its candidate pool on every call** (a paid API
  call with the OpenAI embedder). Have the engine return the stored vectors
  for the pool, or cache them.
- **Time tags were judged with turn numbers only.** LongMemEval messages
  carry no dates in this loader, so `[turn N]` is what gained 5 to 8
  questions of 115. `[YYYY-MM-DD]` is covered by unit tests, not by a judged
  run (LoCoMo has dated sessions). A tag costs about 3 tokens, one item of a
  150-token context; whether a shorter form does as well is not known.

## Cognitive behaviour (needs a judged re-run before changing)

Found by the audit, deliberately left as they are because fixing them changes
recall results:

- **Decide `cognitive_alpha` for a strong embedder (owner's call).** The
  conversational profile ranks by cosine plus a graph boost of up to 0.5
  (`cognitive_alpha=0.5`). On 115 judged LongMemEval questions with OpenAI
  embeddings, the default embedder, that scores 0.574; the same profile with
  `cognitive_alpha=1.0` (the graph still proposes candidates, ranking is by
  cosine) scores 0.609 and returns the plain vector pool in 114 of 115
  questions (7 gained, 3 lost, p=0.34; three other comparisons of cognitive
  search against the plain pool went 3/7, 2/8 and 3/8). No single
  comparison is significant; all point the same way. The 0.5 came from
  earlier runs, most on MiniLM embeddings, where the graph helped. Options:
  leave it; set 1.0 in the profile; choose by embedder. Whatever is chosen,
  describe graph expansion and belief revision as neutral for answer
  accuracy on strong embeddings, and lead with what was measured: packing
  to the budget, time tags, bounded compression.
- **Marking a superseded fact or demoting it.** Superseded facts are kept
  now, and a verifier's are marked for the reader, but the 0.4 score
  demotion keeps the marked fact out of a 150-token context almost always
  (2 of 115). Without the demotion (`supersession_demotion_factor=1.0`) a
  marked fact is in 62 contexts and the answers come out level (0.600
  against 0.591; 4 gained, 3 lost): better where the old fact holds a
  needed detail and on knowledge-update questions (12 against 10 of 18),
  worse where a marked fact takes a useful one's place or the supersession
  was wrong. Unverified detection (demoted, never marked) has lost 1 or 2
  questions and won none in three runs; whether it should demote at all is
  open.
- **Belief revision still retires some facts that are true.** Measured on
  the held-out half of `belief_pairs.jsonl` (54 updates, 58 pairs that both
  stay true, every pair in its own store): the LLM verifier with
  `gpt-4o-mini` catches 49 updates (the engine's own detection: 16) and
  retires 4 of the 58 still-true facts (8 are flagged without a verifier; 6
  are retired with NLI, 7 with a local 4.7B model). Three of the 4 are
  dated events ("I attended PyCon in 2022", "I attended PyCon in 2024").
  Open, in order of expected value:
  - the candidate floor: all 5 held-out misses had a MiniLM similarity below
    0.45 and were never shown to the model, which retired every update it
    did see. Lowering `candidate_min_cosine` costs more requests and more
    chances to be wrong. Tune it on `--split dev` only, and treat the
    held-out half as seen for this setting: its similarities have been
    looked at;
  - dated events in the prompt, and whether one pair per request is worth
    7 to 8 times the requests (on `dev`: 1 error instead of 2 and 52 updates
    instead of 51; not run on the held-out half);
  - pairs from real conversations. The 260 are single sentences written for
    the test.
- **No way to retract a supersession**, and a pair the verifier gives no
  verdict for is asked again at every consolidation.
- **The engine's own detection is unchanged** (without a verifier its
  results are now flagged, not hidden): it misses updates worded differently
  from the fact they replace, and its text gate ignores digits, short tokens
  and stop words, so "visited Paris in 2019" and "in 2023" look identical to
  it. The NLI verifier only sees what that detection proposes.
- **`incremental_supersession_detection` is not switched on with the LLM
  verifier**, so every consolidation looks at every record again. The
  verdict cache keeps that from costing requests, but the neighbour searches
  are repeated.
- **Maintenance reads count as accesses.** Deduplication and supersession
  detection (including the candidate search for an LLM verifier) search for
  neighbours through the normal path, which bumps access counters. Access-aware eviction and importance scoring therefore measure
  neighbourhood density and the consolidation schedule as well as real
  retrieval; with dedup on, everything sits inside the eviction grace window.
- **Access-aware eviction removes never-queried records first**, including
  just-inserted ones (they score zero and get no grace), and ignores
  supersession, so an unqueried correction can be evicted while the stale
  fact survives.
- **Deduplication keeps the older record on a tie**, so a near-duplicate
  correction can be merged away in favour of the stale version; and it runs
  before belief revision.
- **Gist records are not counted against `max_records`**, start with a score
  of zero (first in line next cycle), and superseded victims are gisted like
  any other.
- **The expansion can drop an ANN hit** when more graph candidates than the
  cut have a large delta, contrary to the "never drops an ANN hit" docs; the
  query-token concept seeding is uncapped.
- **Vocabulary evolution merges concepts that co-occur once** (no minimum
  support) and never lifts a hub suppression.
- **Demotion is not reverted** when the superseding record is deleted, and the
  temporal chain is not persisted across restarts.
- **Edge decay floors at 1.0**, below an important memory's baseline weight.
- Concept extraction is English-only in its stop words and opposition cues,
  and length-based on bytes.

## Engine

- **Remaining O(N) consolidation passes** (importance recompute, abstraction and
  vocabulary evolution, graph snapshot) — make them incremental with the
  sequence-cursor pattern already used for supersession detection. Profiler:
  `benchmarks/profile_consolidation.py`.
- **Per-query cost that grows with the store:** BM25 scans every document's
  tokens (no inverted index), and reinforcement scans all edges per hit under
  the graph write lock.
- **Graph growth:** concept nodes, co-occurrence counts, aliases and
  abstraction parents are never removed.
- **Quantizers** (all opt-in): RaBitQ's transform is not orthogonal at
  non-power-of-two dimensions (384 / 768 / 1536), which costs shortlist
  recall; the 3-bit Lloyd-Max table has two wrong centroids (±0.366, ±0.800
  instead of ±0.245, ±0.756); quantizer bit widths are only validated when a
  tier is first built; TurboQuant codes depend on the RNG stream of the
  pinned `rand` version. Fixing the table or the RNG dependence changes how
  existing segments decode: version the format first.
- **Multi-tenant serving** (PLAN Phase C): cheap per-scope engines, with
  pooling and idle eviction.
- **Metadata cache is fully in memory.** Fine for many small stores; a paged
  store is only needed if a single store has to hold millions of records.
- **`engine/mod.rs` is ~1,100 lines** and `open()` alone is ~330 (WAL
  replay, consistency checks, segment loading). `engine/tests.rs` is ~2,600
  lines and could follow the same split as the modules it tests.
- **Configuration surface:** 64 public config fields and ~60 `MemoryEngine`
  keyword arguments, most of them opt-in experiment flags. Group them into
  sub-structs and named presets.
- **`segment_holder.rs::build_records`** fills only the embedding (see its
  TODO); plumb `MetadataStore` in if a caller ever needs the other fields.
- **Gate determinism:** the LongMemEval smoke's edge count used to move
  between runs of the same code (248 to 251). Since score ties are broken
  deterministically (exact scan, expansion, fusion) it has been 247 on four
  consecutive runs. If that holds, the gate's edge window (10 to 800) can be
  tightened to catch real regressions.

## GPU

- **Resident search is measured on one card, up to 100,000 vectors** (RTX 3050
  Laptop, 4 GB: 4 to 7 times faster than the CPU path for one query, 5 to 17
  times in throughput under 8 threads;
  [`docs/gpu_acceleration.md`](./docs/gpu_acceleration.md)). Its cost grows
  with the bytes scanned (there: about 1.1 ms per 150 MB plus 0.35 ms per
  query), so a store several times larger hands the advantage back to an
  index. Measure before relying on it beyond a few hundred thousand records,
  and on a second card.
- **Device rows are full `f32`.** Half-precision or 8-bit rows would hold 2 to
  4 times as many records in the same memory and scan proportionally faster,
  with the head re-scored at full precision on the host.
- **A store that outgrows the memory budget turns the mirror off** for the
  life of the engine. Keeping part of it on the device (the newest rows, or
  one scope) is not implemented, and neither is turning it back on after a
  transient device error.
- **Without a GPU, search above 4,096 records is slower than a plain BLAS
  scan at these sizes** (100,000 x 384: 5.9 to 7.5 ms against 5.0 to 6.4 ms
  for NumPy's exact product) when segments are small (4,000 in the
  benchmark): the number of segments searched dominates, not the index.
  Larger or merged segments, or a BLAS exact scan up to a higher threshold,
  would help CPU-only builds.
- The quantized-scan and SpMV kernels in `turbomemory_gpu` are still unused;
  remove them or find them a caller.
- The CUDA build is tested by hand (`cargo test --features cuda`); no CI
  runner has a GPU.

## Evaluation and claims

- Confirm the bounded-compression result on the full LongMemEval set (~500
  conversations) and on LoCoMo before publishing any number (PLAN Phase D).
  On 120 conversations (2026-10-07) it reproduced against deletion at every
  budget; against Mem0 it is level at 64 and 128 stored tokens and ahead at
  256 (0.482 against 0.348), where Mem0's own store is smaller than the
  allowance. State the store sizes next to any Mem0 comparison.
- The 2026-10-07 judged run did not reproduce a lead of the stack over plain
  vector search (adapter 0.504, plain 0.539, 120 conversations, gpt-4.1-mini
  judge). The README's head-to-head table (50 conversations, gpt-4o-mini
  judge) shows one; re-run that table's exact configuration before relying
  on it, and say which system (adapter or SDK) each number measures.
- Re-run the retrieval-side levers on OpenAI embeddings; their original lifts
  were measured on MiniLM only.
- The README's benchmark tables are the owner's own runs. To let a reader
  check them, state the number of questions next to each table and commit the
  configuration and raw output: the LongMemEval head-to-head percentages
  (66.7% vs 55.6%) imply a small set and are identical at all three budgets,
  and the BEAM table is 20 questions. The 48-question multi-budget audit in
  `PHASE_PROGRESS.md` (56.2% vs 39.6% at 150 tokens) is a larger sample of
  the same comparison.

## Retired with the 1M × 4k axis

Not planned; revisit only if multi-tenant measurements demand it: collection
sharding and per-shard workers, NUMA / huge-page policy, WAL segmenting and
group commit, byte-threshold segment sizing, product-quantization and ACORN
variants, on-disk ANN (DiskANN / SPANN), an async Python API, and the
1M × 4k synthetic benchmark.
