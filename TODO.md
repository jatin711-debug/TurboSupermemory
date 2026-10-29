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
> a CI workflow (removed again on 2026-10-21), and verified belief revision landed (entry of that date in
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
- **No CI.** The workflow was removed on 2026-10-21. `make gate`, run on
  this Windows machine, is the only check. Linux and macOS are not built or
  tested anywhere. The AArch64 NEON kernels in
  `turbomemory_core/src/metrics.rs` are only type-checked
  (`cargo clippy --target aarch64-unknown-linux-gnu`), never compiled into a
  binary or run. The last place they ran was that workflow's macOS job.
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

## Judged on 2026-10-29 (the messages as said, the packer in the engine)

`Memory` now keeps every message, in passages, next to its facts
(`keep_messages`, on by default unless `max_user_tokens` is set). On 29
LongMemEval-S questions at 2,000 tokens the facts alone answered 19, facts
plus messages 27 (8 / 0, p = 0.01) and the messages alone 26; reading the
whole 124k-token history had answered 22. The budget packer runs in the
engine on stored vectors. Numbers in `benchmarks/PHASE_PROGRESS.md`.

Open, in this order:

1. **More LongMemEval-S questions without paying for extraction.** The
   messages-only store needs no extraction, so it can be scored and judged on
   all 500 S questions for about $0.002 a question at 2,000 tokens. Done for
   questions 31-130 (free score): cognitive search 84 against plain search
   80 (4 / 0), and the assistant-question fix (3 -> 10 of 11). The rest of
   the 500, and a judged run of both, are still to do (needs the GPU for
   embedding, about 7 minutes per 20 new histories).
2. **`keep_messages` with `max_user_tokens`.** Compaction folds facts by
   speaker and does not know passages, so the two are refused together. A
   bounded store with messages needs a policy for folding passages (oldest
   first, into dated gists, as facts are).
3. **What still misses at 2,000 tokens.** After the packer fixes of
   2026-10-29 (near-duplicates only within one day, relevance first for
   counting questions, the prior never damping the top 3), 5 of 97 S
   questions lack evidence: 2 multi-session counts, 2 temporal, and 1 benign
   (the assistant's echo of a stated fact). All were in the candidate pool.
   Next candidates are a count over a typed list of events (User as Code)
   and a judged run to see what the reader does with what it now gets.
4. **Exact search on larger stores.** On the S30 run the CUDA build's exact
   device search and the CPU's tiered path chose identical contexts in all 8
   stores past 4,096 records (2026-10-29). Repeat on `--pool 20` stores with
   messages kept (tens of thousands of records), where the approximate path
   has room to differ.

## Judged on 2026-10-21 (control, held-out, read-time changes, LongMemEval-S)

The numbers are in `benchmarks/PHASE_PROGRESS.md`. In short:

- **Held out, the reading prompt at 600 tokens holds**: 296 of 355 against
  259 for the plain prompt at 150 (52 gained, 15 lost).
- **Graph search ties plain search** on the held-out answers (10 / 10).
- **The raw evidence sessions, read with the same prompt, score 104 of 115**
  where memory scores 98. On LongMemEval-S (29 questions) the raw 124k-token
  sessions score 22 and memory 20.
- **Not adopted**, all level or worse:
  - source turns as the read unit (`recall(sources="turn")`);
  - event dates (`event_dates=True`);
  - reader tools (`tsm.answering`);
  - two verifier-prompt rules for repeat occurrences (dev pairs only).

Still open, in this order:

1. **Decide `compaction="merge_oldest"` on the held-out questions.** Judged
   2026-10-28 on the 115 dev questions (`PHASE_PROGRESS.md`): 57 against 50
   for rolling compaction (p = 0.23) and 48 for compacting once (p = 0.11),
   the first rolling policy to beat compacting once. Not significant; the
   default stays `"refold"`. The same run found the dated stale mark (98 / 97)
   and dated gists (96 / 98 over two arms) level, so both stay on.
2. **LongMemEval-S on more questions.** The 29 here detect only large
   differences. The other 85 of the 115 need about 37k extraction requests,
   about $5, with `benchmarks/cognitive_eval/prefetch_extraction.py`.
3. **The `sources="turn"` excerpt picks sentences by word overlap** and drops
   the short sentence that answers. Choose by the facts' embeddings, and
   measure only at budgets where turns fit (1,500 tokens or more).
4. **Evidence is lost on the write path, not in ranking** (addressed
   2026-10-29 by keeping the messages, above)
   (`benchmarks/cognitive_eval/evidence_rank_report.py`, 2026-10-28 entry in
   `PHASE_PROGRESS.md`). At 600 tokens every question whose evidence turns
   all yielded a fact had all of it in the context. Every miss has a turn
   the extractor stored nothing from (34 of 115 dev questions, 80 of 355
   held out). Retrieval and graph work has little left to win on the oracle
   file; extraction coverage does.

## Judged on 2026-10-09 (the retrieval and input changes)

The same 115 questions, judged by gpt-4.1-mini: 989 judge requests, about
$0.13 at list prices. The arms and the table are in `benchmarks/PHASE_PROGRESS.md`.

- **Shipped default: 0.574 -> 0.783** (`sdk_cognitive`, 66 -> 90 of 115; 26
  questions gained, 2 lost, sign test p < 0.001). Most of it is the input fix,
  not retrieval. The retrieval change alone, on the old inputs, gives 0.635
  (13 gained, 6 lost, p = 0.17). The input fix (session dates, time-ordered
  ingestion and the question's date; this run cannot separate the three) adds
  17 more (19 gained, 2 lost, p < 0.001). Temporal questions 9 -> 21 of 28,
  knowledge updates 10 -> 15 of 18, multi-session 16 -> 22 of 31.
- **Minute tags lose.** `--time-tags minute` scored 84 against 90 with date
  tags (1 gained, 7 lost, p = 0.07). Date tags stay the default.
- **The graph term adds little on OpenAI embeddings.** Inside the new run it is
  3 questions ahead of cosine-only ranking (3 gained, 0 lost, p = 0.25).
- **Caveats.** The retrieval settings were chosen with the free score on these
  same 115 questions, so the retrieval share is optimistic. Per-type counts are
  9 to 31. One judge, one reader, one run each.
- **Held-out, coverage only (2026-10-12, free).** The cached extraction
  covers all 500 oracle conversations, so the settings were checked on the 355
  questions they were not chosen on. Questions with all their evidence at 150
  tokens: the 0.95 packer cut-off against 0.72 gains 11 and loses 1
  (p = 0.006); the calibrated graph term against the old fixed boost gains 11
  and loses 3 (p = 0.06); against plain search it is level (0 and 1). Judged
  answers on those 355 are still owed.
- **Still open.** A check on full histories needs more extraction (paid:
  about 14k requests for 30 LongMemEval-S questions, 51k for all 115). Date tags cannot order two sessions of one day (118 of the 500
  questions have such a pair); one regression from the input fix is such a
  pair. Tagging with both the date and the turn is untested. Validity intervals
  on facts and keeping the original turn beside its facts (item 3 of the plan)
  are not started.

## Judged on 2026-10-12 (the write path and the reading prompt)

The same 115 questions, `sdk_cognitive`, reader and judge gpt-4.1-mini: 2,856
extraction requests and 1,308 judge requests, about $0.96 at list prices. The
table is in `benchmarks/PHASE_PROGRESS.md`.

- **Best so far: 0.861** (99 of 115) with the two-pass store, a 600-token
  context and the `tsm.reading` prompt. The committed setup (first prompt
  only, 150 tokens, the harness's plain prompt) is 0.783 (90).
- **How the model is asked is the larger part.** On the same 600-token
  contexts the plain prompt answered 92 and the reading prompt 99. The plain
  prompt gets worse with more context on the same store (90 at 150 tokens,
  86 at 600), so a larger budget alone buys nothing.
- **Recovering the dropped turns is not settled**: 6 more answers at 600
  tokens, 8 fewer at 150, neither significant (SDK section below).
- **The evidence score misled here.** It counts a turn as found when any
  memory from it is in the context; the recovered memories are often not the
  one that answers. Do not tune extraction on it.
- **Caveats.** The reading prompt was written after reading what these 115
  questions got wrong, so 0.861 is an upper estimate. The 92 -> 99 step has
  no paired test: the per-question verdicts of the earlier runs were under
  `target/`, which was removed before the last run (results are now kept
  under `.cache/shipped_stack_eval`). One judge, one run each. Not in the
  numbers: the harness now lists one day's memories in the order they were
  said (it passed the reading prompt bare strings before, so they came by
  rank), and `statements` keeps a list number with its item.
- **Of the 16 still wrong**: 8 have the facts in the context and the reader
  still errs (two values said on one day, which date tags cannot order; a
  total and an average whose facts do not say which workshop or degree they
  are about; a subtraction; a count of flights); 3 miss a fact; 2 replies
  were cut off at 500 tokens before their answer line (the limit is 800
  now); 2 give the gold answer and were marked wrong by the judge; 1 asks
  about jewelry where the conversation has a chandelier.
- **Still open, in this order.** (1) ~~A held-out check~~ done 2026-10-21:
  296 of 355 with the reading prompt at 600 tokens, against 259 for the
  plain prompt at 150. (2) The paired grid on
  the 115 (store x budget x reader), which decides the extractor's defaults:
  about 1,600 requests. (3) The second prompt for the other 380 conversations
  (about 2,700 requests) before the two-pass store can be judged held-out.

## SDK (`tsm/`)

- **The extraction pass stores nothing for a third of what users say**
  (measured 2026-10-09, free; `benchmarks/PHASE_PROGRESS.md`). The extractor
  returned no fact for 470 of 1,403 user messages and 311 of 1,403 assistant
  messages. Most are a statement about the user followed by a question, which
  the prompt (the same one in `tsm/extractors.py`) tells it to ignore. 39 of
  229 evidence turns are lost this way, 34 of the 115 questions have one, and
  judged accuracy is 0.59 on those against 0.86 on the rest. This is the 0.70
  ceiling: 81 questions have all their evidence in the context at 300, 600
  and 1,500 tokens alike. Storing the user's own turn when extraction returns
  nothing lifts that to 99, 107 and 109 (103 and 98 at 600 tokens with 20 and
  40 conversations pooled). Built on 2026-10-12 as a second prompt for a
  message the first one finds nothing in, plus its statements stored as said
  (`tsm/extractors.py`), and judged: it does not pay the way coverage
  predicted. Questions with all their evidence in a 150-token context went
  from 76 to 92 and judged answers from 90 to 82 (4 gained, 12 lost,
  p = 0.08); at 600 tokens answers went from 86 to 92 (10 gained, 4 lost,
  p = 0.18). The recovered memories are mostly what the user wanted at the
  time; they match a question's wording and, in a small context, take the
  place of the fact that answers it. Both steps are off by default
  (`OpenAIExtractor(second_look=True, keep_unextracted=True)` turns them
  on); whether to ship them on waits for the paired grid, item (2) of the
  2026-10-12 list above. Still open
  from the diagnosis: facts taken from the recent context instead of the
  message, facts that do not say what they are about ("The workshop was
  free"), and the same fact stored several times in different words.
- **Readers cannot tell which of two values is current.** With all the
  evidence in the context, knowledge-update questions were answered right
  52% of the time (50% without it). Dates may help; a validity interval on
  the fact itself (true from, true until) is the direct fix.
- **The packer's weights are fixed amounts.** The new-turn bonus (0.20) and
  the redundancy penalty are added to relevance differences of about 0.01
  between neighbouring candidates. Normalizing relevance to the pool's
  spread was within one or two questions of not doing so on the free score,
  so it was left; it is the same scale problem the graph term had.
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
  The first is `compaction="merge_oldest"`: 57 against 50 judged
  (2026-10-28, not significant; item 1 of the open list).
- **Gists carry the dates they cover** (judged level 2026-10-28).
  `plan_compaction` reports each gist's span and `recall()` tags it
  (`[2023-01-05 to 2023-03-02] ...`). A gist written from undated memories
  is still bare: a span of turns (`[turns 3-41]`) is not stored.
- **`OpenAIGistSummarizer` defaults to gpt-4.1-nano, which returns 404 for
  the owner's account as of 2026-10-28** (it worked on 2026-10-21; still
  listed, not callable). Every
  gist then fails, nothing is removed, and `max_user_tokens` bounds nothing.
  Pick a default that is callable (the 2026-10-28 run used gpt-4o-mini).
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
- **Nothing calls `reinforce()` yet.** `recall()` no longer reinforces what
  it returns (the same question came back as a different list the second
  time in 13 of 19 checks); `Memory.reinforce(ids)` is there for memories
  that were used, and no example or integration uses it. Access counters
  are still bumped by every read.
- **Extraction results cached before 2026-10-06 may be wrong.** The OpenAI
  extractor used to cache "no facts" for a reply that was cut off at 400
  tokens. It no longer does, but an existing `extract_<model>.json` cache can
  still hold such empty entries for long messages; they cannot be told apart
  from real "no facts" answers. Delete the cache to re-extract.
- **The caches hold every message, fact and query in plaintext**
  (`<db_path>/tsm_cache/`) and are never pruned, including after eviction.
  The embedding cache is still a pickle file, now read with an arrays-only
  loader; a plain format (npz) would be simpler.
- **Role prior is very broad.** `tsm/ranking.py` treats a query as first-person
  when it contains words like "what", "how", "have" or "did", so the user-fact
  boost applies to nearly every question. It is the cue set the evaluations
  ran with; narrowing it needs a judged re-run, not a guess.
- **Budget recall re-embeds its candidate pool on every call** (a paid API
  call with the OpenAI embedder). Have the engine return the stored vectors
  for the pool, or cache them.
- **Time tags have been judged with turn numbers only** (5 to 8 questions
  of 115). Dated tags are wired through the eval now and wait for the
  judged run above. A tag costs about 3 tokens, one item of a 150-token
  context; whether a shorter form does as well is not known.

## Cognitive behaviour (needs a judged re-run before changing)

Found by the audit, deliberately left as they are because fixing them changes
recall results:

- **Cognitive search no longer costs answers; it has not been shown to add
  any.** The profile's fixed boost (`cognitive_alpha=0.5`) ranked by word
  overlap and scored 0.574 judged where cosine ranking scored 0.609. The
  profile now weights the graph against the query's cosine spread
  (`graph_weight=0.5`), drops the per-word concept bonus and normalizes by
  fan. On the free evidence score that is level with the plain vector pool
  at three store sizes (76, 74, 68 against 75, 74, 68) where the old
  settings, with the same packer, were 3 to 7 questions behind. Until something shows a gain,
  describe graph expansion and belief revision as neutral for answer
  accuracy and lead with what was measured: packing to the budget, time
  tags, bounded compression. Not measured: MiniLM, where the graph used to
  help and the new weights were not tried; the engine's own default
  (`cognitive_alpha=0.7`, the old seeding) is unchanged and is what the
  evaluation adapter and the gate use.
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
- **The graph step's costs grow with the store.** BM25 scores every document
  of every scope on each query, and the expansion stops at the first 50
  candidates it meets rather than the 50 strongest. (The ANN hits it used to
  drop at its cut are now always kept.)
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
- The CUDA build is tested by hand (`cargo test --features cuda`); `make
  gate` does not build it.

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
