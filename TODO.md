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

## Durability and operations

- **Power-loss durability.** The WAL is fsynced by `flush()`, not per write, so
  a power loss (unlike a process kill) can lose the writes since the last
  flush. Add a sync policy (per write / every N ms / on flush) if a deployment
  needs it.
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
- **No CI.** `make gate` is the merge gate and it only runs locally. At minimum
  run fmt, clippy, the Rust suite and `make test-python` on every PR.
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
- **Timestamp anchoring** (`[2024-01-15] fact`) exists only in the eval
  adapter's `search()`. Decide whether it belongs in `recall()`; it has no
  isolated measurement yet.

## Cognitive behaviour (needs a judged re-run before changing)

Found by the audit, deliberately left as they are because fixing them changes
recall results:

- **Belief revision precision.** With the default profile (MiniLM, no
  verifier) a 12-pair spot check retired 1 of 4 real updates and 2 of 8
  unrelated facts ("My sister lives in Toronto" was hidden by "My brother
  lives in Toronto"). The text gate ignores digits, short tokens and stop
  words, so "visited Paris in 2019" and "in 2023" look identical to it.
  The NLI-verified path was not measured. There is no way to retract a
  supersession.
- **Maintenance reads count as accesses.** Deduplication and supersession
  detection search for neighbours through the normal path, which bumps access
  counters. Access-aware eviction and importance scoring therefore measure
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
