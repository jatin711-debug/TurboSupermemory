# tsm — TurboSuperMemory Python SDK

One-flag conversational memory over the compiled `turbomemory` engine:
scoped fact storage, belief revision, verified supersession, budget recall.

## Install / requirements

Python 3.12 or newer (the extension is an `abi3-py312` wheel).

- Installed: `make dev` (`uv pip install -e .`) builds the extension with
  maturin and installs `tsm` plus `turbomemory` into the active environment;
  `make wheel` produces a redistributable wheel under `target/wheels/`.
- In-tree: `make build-python` then copy the artifact to the repo root as
  `turbomemory.pyd` / `.so` (what `make verify` / `make gate` do); `tsm` finds
  it there without installation.

The default embedder and extractor are OpenAI-backed: install the `openai`
extra (`pip install "tsm[openai]"`) and set `OPENAI_API_KEY`. Local models
(`embedder="local"`, `verifier="nli"`) need the `cpu` extra;
`embedder="gemma"` (EmbeddingGemma 2) needs `gemma` (sentence-transformers
6.1 or newer).

## Usage

```python
from tsm import Memory

with Memory("./my_db") as mem:                       # conversational profile
    mem.add([{"role": "user", "content": "I moved to Lisbon."}], user_id="alice")
    mem.recall("Where does Alice live?", user_id="alice")
    mem.recall("housing", user_id="alice", token_budget=64)   # MMR best set
    mem.consolidate()                                # verify + commit updates
```

- `profile=None` → plain vector store (engine defaults).
- Extra engine kwargs override the profile: `Memory(db, graph_weight=0.25)`.
- Plug in local backends via the `Embedder` / `Extractor` / `Verifier`
  protocols (`tsm.interfaces`); pass instances to `Memory(...)`.
- An embedder may add `encode_query` / `encode_document` (models that prompt a
  question differently from the text it searches, such as EmbeddingGemma 2);
  `Memory` embeds questions with the first and stored text with the second, and
  falls back to `encode` for both.
- A database belongs to the embedder that wrote it: the first `Memory` to write
  records the embedder's `fingerprint` in `tsm_embedder.json` and another embedder
  is refused with `EmbedderMismatchError` (vectors of two models are not
  comparable even at one dimension). `allow_embedder_change=True` overrides it for
  an embedder that makes the same vectors; see `docs/python_sdk.md`.
- `idle_consolidation_secs=N` consolidates in a background thread after N idle
  seconds with unwritten-to-consolidation changes (verification, supersession,
  `max_user_tokens` compaction), off the request path; off by default.
- Budget recall: `recall(..., token_budget=N)` returns the best set that fits
  (`tsm.budget.select_under_budget`, greedy MMR with a cross-turn bonus).
- Each result carries `context`, the text to put in a prompt: the fact with
  its date in front when the message was added with a `timestamp`
  (`[2024-03-02] ...`), otherwise its turn (`[turn 12] ...`), so a model can
  tell which of two facts came first. `time_tags=False` leaves that out;
  `time_tags="minute"` adds the time of day.
- Each result also carries `source`, the message the fact was taken from.
  Two read-time options built on it were measured and removed on 2026-11-09
  (`git show e79d75f:benchmarks/PHASE_PROGRESS.md`):
  - `sources="turn"` returned those messages in place of the facts. At a
    600-token budget it lost judged answers, because fewer turns than facts
    fit.
  - `event_dates=True` wrote the date after a relative time ("two weeks ago
    [week of 2023-05-08]"). Judged answers came out level.
- Asking a model to answer from what was recalled: `tsm.reading` has the
  prompt. `answer_prompt(question, results, today=...)` puts the memories in
  the order they were said and tells the model to list what bears on the
  question, settle what changed and what has to be counted or computed, and
  end with one `Answer:` line; `final_answer(reply)` takes that answer out.

  ```python
  from tsm.reading import answer_prompt, final_answer

  results = mem.recall(question, user_id="alice", token_budget=600)
  reply = client.chat.completions.create(
      model="gpt-4.1-mini", max_tokens=800, temperature=0,
      messages=answer_prompt(question, results, today="2024-03-09"))
  print(final_answer(reply.choices[0].message.content))
  ```

  A plain "answer from these snippets" prompt gets totals, changed values
  and date arithmetic wrong with the facts in front of it. On 355 judged
  LongMemEval questions that none of this was tuned on, this prompt at 600
  tokens answered 296 where the plain prompt at 150 answered 259
  (`git show e79d75f:benchmarks/PHASE_PROGRESS.md`, 2026-10-21).
  A variant that gave the model tools for arithmetic, dates and counting
  (`tsm.answering`) answered 94 of 115 against 98 and was removed on
  2026-11-09.
- `recall()` is a read: it does not change what the next call returns.
  `mem.reinforce(ids)` strengthens the memories that were actually used.
- `recall(search="vector")` is plain cosine search instead of the graph's,
  everything else unchanged. On the evidence score the graph was level with
  it on OpenAI vectors and found more on local ones (MiniLM 12 more
  questions, 1 fewer; EmbeddingGemma 4 / 1; 2026-11-11), so the graph stays
  the default.
- Compress instead of delete: with `max_records` set, pass
  `gist_summarizer=OpenAIGistSummarizer()` (or the model-free
  `ExtractiveGistSummarizer()`, both in `tsm.gist`) and eviction victims are
  folded into searchable gist records instead of being dropped. If the
  summarizer raises, the affected memories are kept and retried on the next
  eviction; only an empty summary drops them.
- Belief revision (`consolidate()`): pass `verifier="llm"` and a chat model
  decides, for each new fact and its closest older facts, whether the newer
  one replaces the older one (`tsm.verification.LLMVerifier`; any
  OpenAI-compatible endpoint, verdicts cached on disk). `verifier="nli"`
  uses a small local cross-encoder instead (`NLIVerifier`, needs `torch` +
  `transformers`): free, but it only vets the pairs the engine's lexical
  detection proposes and misses most reworded updates. Either way only
  accepted pairs are committed. A replaced fact is never removed from
  recall: it is ranked lower and returned with `superseded_by`. With a
  verifier, `recall()` also marks it in the result's `context`
  (`[earlier, since changed] ...`) and serves the fact that replaced it with
  it. `exclude_superseded=True` drops replaced facts instead.
- A token budget per user: `Memory(db, max_user_tokens=256,
  gist_summarizer=...)` keeps the newest facts as they are and folds
  everything older into a few short gists at `consolidate()`, again and
  again as the history grows (`tsm.compaction`).
- The engine is the only store: text, role, and scope are read back from it,
  and ids come from its durable insert sequence, so a reopened database keeps
  appending and recalls the same way it did before the restart.
- `close()` (or leaving the `with` block) flushes and releases the database;
  the same path can be reopened immediately. A closed `Memory` raises
  `RuntimeError` on further calls.
- A process that dies without `close()` loses nothing it was told was stored:
  the next open replays the write-ahead log (`mem.engine.recovery_report()`).
- `add()` is all-or-nothing per call and safe to call from several threads.
  Recall is scoped by the engine itself: other users' records never show up
  and never take result slots.

## The executive (a prototype, opt-in, not yet measured)

Control loops around the store. None of them changes what `recall()`
returns. None was measured: intentions were to be judged against a plain
to-do ledger and lessons against retrieving the raw past episodes
([LEARNINGS.md](../LEARNINGS.md)). The read-time monitor
(`checked_recall`) did not, and was removed on 2026-11-11: no second look,
free or judged, got evidence into a LongMemEval-S context that lacked it,
and both lost some (`git show e79d75f:benchmarks/PHASE_PROGRESS.md`).

```python
mem = Memory("./db", intentions=True, experience=True)

# Intentions: formed by add() from "remind me to ..." or "I need to ... by
# Friday", closed or replaced at consolidate(), due by time or brought to
# mind by the context. due() is a read.
mem.due("alice", now="2024-03-05", context="at the passport office")
mem.resolve_intention(intention_id, "done")

# Experience: outcomes in, lessons out (curated at consolidate(), counters
# only for lessons that were in front of the reader; "self" outcomes teach
# nothing).
items = mem.lessons(task, user_id="alice")
mem.feedback(task, "failure", memory_ids=[i["id"] for i in items],
             user_id="alice", lesson="Ask which date format the vendor uses.")

# Belief ledger.
mem.history(memory_id)        # the chain with valid_from / valid_until
mem.retract(old_id, new_id)   # take a supersession back, for good
mem.forget([memory_id], user_id="alice")   # cascades to the message; verified
```

Intentions and experience each keep their own engine under `db_path`
(`tsm_intentions/`, `tsm_experience/`). Their state changes are payload
patches (`MemoryEngine.set_payloads`), which keep a record's place, so
nothing is embedded twice and belief ordering is untouched. The cosine floors
they use (`tsm.intentions.COMPLETION_MIN_COSINE`, `RESCHEDULE_MIN_COSINE`,
`CUE_MIN_COSINE`, `tsm.experience.ITEM_MATCH_COSINE`) are untuned and belong to
an embedder's geometry.

`forget()` treats a forget request as a promise that the content is not
served again, so derived text goes too (`tsm/forgetting.py`):

- A gist that carries the forgotten content is **redacted**, not rebuilt. The
  sentences holding it are removed: the text itself, a number of three or
  more digits from it, an address, or most of its salient words. The rest is
  stored as a new gist *before* the old one is deleted. A gist left empty is
  deleted.
- Rebuilding with a model was rejected. The gist's sources are already gone,
  so there would be nothing to rebuild from, and a rewrite cannot be checked.
- `gists="keep"` leaves gists alone and lists the ones that may hold the
  content (`gists_may_hold`).
- The on-disk caches under `tsm_cache/` lose every entry that carries the
  content and are rewritten at once (`purge_caches=True`). These are the
  extractions, embeddings and judges' replies.
- `verified` in the result also checks that no redacted gist still carries
  it. The matching is rule-based and English-leaning, and errs toward
  removing.

## When a backend misbehaves

- A rejected API key or a malformed request fails at once with the HTTP
  status in the message; only transient errors (rate limits, timeouts, 5xx)
  are retried with backoff.
- The OpenAI extractor never drops a message because a request went wrong. A
  reply that is cut off or is not valid JSON is requested again with a
  larger budget, and if that fails the message itself is stored.
- A message the model found nothing in is another matter. People say
  things about themselves while asking for something else ("any collar for a
  Golden Retriever like Max?"), and the extraction prompt returned no fact
  for a third of the user messages it was measured on. Two steps recover
  such a message, both off by default: with
  `OpenAIExtractor(second_look=True)` a message of three words or more that
  comes back empty is asked about again with a second prompt (what the
  speaker reveals, also inside a question); with `keep_unextracted=True`, if
  that finds nothing either, its sentences that are not questions are stored
  as they were said, in pieces of about 80 tokens
  (`tsm.extractors.statements`). What this buys depends on how much you
  recall: with a 600-token context it answered more questions (92 of 115
  against 86), with a 150-token one fewer (82 against 90), because the
  recovered memories compete for the same few slots. Neither difference is
  statistically settled (`git show e79d75f:benchmarks/PHASE_PROGRESS.md`, 2026-10-12), which
  is why the default stores only the first prompt's facts.
- Only well-formed replies are cached, under a key that includes the prompt.
- `OpenAIEmbedder(dim=512)` asks the API for 512-dimensional vectors
  (text-embedding-3 models); a vector of an unexpected size is an error at
  the embedder, not at insert time.
- The embedding cache is read with a loader that accepts arrays only, so a
  cache file in a database directory you were given cannot run code.

## Tests

```
python -m unittest discover -s tsm/tests -t .   # from the repo root, no API key
make test-python                                # same, after rebuilding the extension
```
