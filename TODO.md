# TurboSuperMemory — Open Work

The roadmap is [`PLAN.md`](./PLAN.md) (Roadmap v2 / v2.1). The record of what
was built and measured is [`benchmarks/PHASE_PROGRESS.md`](./benchmarks/PHASE_PROGRESS.md).
This file is only the short list of open engineering work, grouped by area.

> **Trimmed 2026-10-05.** This file used to hold ~560 lines of dated
> "Recently Completed" logs and a ~110-item roadmap for a single 1M × 4k
> store. The logs duplicate git history and `PHASE_PROGRESS.md`, and
> `PLAN.md` (Phase C) retired the 1M × 4k axis in favor of many small,
> long-lived stores. Nothing was lost: `git show 39cbea4:TODO.md`.

## SDK (`tsm/`)

- **Role prior is very broad.** `tsm/ranking.py` treats a query as first-person
  when it contains words like "what", "how", "have" or "did", so the user-fact
  boost applies to nearly every question. It is the cue set the evaluations
  ran with; narrowing it needs a judged re-run, not a guess.
- **Budget recall re-embeds its candidate pool on every call** (a paid API
  call with the OpenAI embedder). Have the engine return the stored vectors
  for the pool, or cache them.
- **`add()` inserts one fact at a time.** Switch to `insert_batch`.
- **Single writer.** Concurrent `add()` calls on one `Memory` can race on id
  minting; either document it or guard it with a lock.
- **`embedder="openai"` is not accepted** as a string (only `None` selects
  OpenAI), and an unknown string fails late instead of raising at construction.
- **Timestamp anchoring** (`[2024-01-15] fact`) exists only in the eval
  adapter's `search()`. Decide whether it belongs in `recall()`; it has no
  isolated measurement yet.

## Engine

- **Remaining O(N) consolidation passes** (importance recompute, abstraction and
  vocabulary evolution, graph snapshot) — make them incremental with the
  sequence-cursor pattern already used for supersession detection. Profiler:
  `benchmarks/profile_consolidation.py`.
- **No vacuum.** Deleted and evicted records leave their vectors in
  `vectors.bin`, and merged-away segment directories are only cleaned up
  opportunistically. A long-lived store under eviction pressure grows on disk.
- **Multi-tenant serving** (PLAN Phase C): cheap per-scope engines, with
  pooling and idle eviction. `close()` now releases everything, which is the
  prerequisite.
- **Metadata cache is fully in memory.** Fine for many small stores; a paged
  store is only needed if a single store has to hold millions of records.
- **`engine/mod.rs` is still ~870 lines** and `open()` alone is ~260 (WAL
  replay, index rebuild, segment loading). `engine/tests.rs` is ~2,600 lines
  and could follow the same split as the modules it tests.
- **Configuration surface:** 64 public config fields and ~60 `MemoryEngine`
  keyword arguments, most of them opt-in experiment flags. Group them into
  sub-structs and named presets.
- **`segment_holder.rs::build_records`** fills only the embedding (see its
  TODO); plumb `MetadataStore` in if a caller ever needs the other fields.
- **Gate determinism:** the LongMemEval smoke's edge count is not stable
  between runs of the same code (248 to 251 observed over six runs). Find the
  time-dependent input before tightening any gate threshold.

## API and operations

- **No CI.** `make gate` is the merge gate and it only runs locally. At minimum
  run fmt, clippy, the Rust suite and `make test-python` on every PR.
- Request timeouts, payload size limits and CORS on the REST server
  (bearer-token auth exists).
- Metrics and tracing: the optimizer still reports failures with `eprintln!`;
  add ingest/search latency, optimizer queue depth, and a metrics endpoint.
- Container build, and a documented backup/restore procedure.

## Evaluation and claims

- Confirm the bounded-compression result on the full LongMemEval set (~500
  conversations) and on LoCoMo before publishing any number (PLAN Phase D).
- Re-run the retrieval-side levers on OpenAI embeddings; their original lifts
  were measured on MiniLM only.
- The README's benchmark tables are not the ones on record: the BEAM table is
  20 questions, and the LongMemEval head-to-head percentages (66.7% vs 55.6%)
  do not match the 48-question multi-budget audit in `PHASE_PROGRESS.md`
  (56.2% vs 39.6% at 150 tokens). Replace them with recorded, same-condition
  results, stating n.

## Retired with the 1M × 4k axis

Not planned; revisit only if multi-tenant measurements demand it: collection
sharding and per-shard workers, NUMA / huge-page policy, WAL segmenting and
group commit, byte-threshold segment sizing, product-quantization and ACORN
variants, on-disk ANN (DiskANN / SPANN), an async Python API, and the
1M × 4k synthetic benchmark.
