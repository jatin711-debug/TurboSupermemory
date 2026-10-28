# Python SDK Reference (`tsm` & `turbomemory`)

TurboSuperMemory provides two layers of Python interfaces:
1. **High-Level Agent Framework Layer (`tsm`)**: Clean, turnkey API with automatic concept extraction, local/OpenAI embedders, and Stage-2 ColBERT late-interaction reranking.
2. **Low-Level Native Engine Layer (`turbomemory`)**: Direct PyO3 C-extension bindings with zero-copy NumPy buffers and sub-millisecond bare-metal control.

---

## 1. High-Level Agent Interface (`from tsm import Memory`)

The `Memory` class is the recommended entry point for AI agents, multi-agent frameworks, and chatbot backends.

### Quickstart

```python
from tsm import Memory

# 1. Initialize Memory with Stage-2 ColBERT Reranking
memory = Memory(
    db_path="./agent_memory_db",
    embedder="sentence_transformer",  # Uses local 'all-MiniLM-L6-v2' on CPU/GPU ($0.00 cost)
    extractor="passthrough",          # Store each message as-is (no LLM call on write)
    reranker="colbert",               # Uses 'LiquidAI/LFM2.5-ColBERT-350M' on CUDA
)

# 2. Add memories (accepts string, dict, or list of dicts)
memory.add("User's production telemetry port is 9443 on host alpha.prod.internal", user_id="alice")
memory.add("User updated database password policy to 16 characters minimum", user_id="alice")

# 3. Recall with cognitive fusion + ColBERT MaxSim reranking
results = memory.recall("What is the telemetry port?", user_id="alice", top_k=3)

for r in results:
    print(f"Memory: {r['text']} (Score: {r['score']:.4f})")

# 4. Flush and release the database (or use `with Memory(...) as memory:`)
memory.close()
```

With no `embedder` / `extractor` arguments, `Memory` uses OpenAI for both
(`pip install "tsm[openai]"`, `OPENAI_API_KEY`): one extraction call per added
message. `extractor="passthrough"` and `extractor="gliner"` keep writes local.

The OpenAI extractor asks the model for the facts a message asserts. That
prompt returns nothing for about a third of what users write, mostly a
statement about themselves followed by a request ("any collar for a Golden
Retriever like Max?"). Two steps recover such a message, and both are off
by default. With `OpenAIExtractor(second_look=True)` a message of three
words or more that comes back empty is asked about again with a second
prompt (what the speaker reveals, also inside a question); with
`keep_unextracted=True`, if that finds nothing either, its sentences that
are not questions are stored as they were said, in pieces of about 80
tokens. They are off because they did not pay when judged: the recovered
memories help when recall has room for them and compete with the others
when it does not. On 115 judged questions, 92 right against 86 with a
600-token context, 82 against 90 with a 150-token one, neither difference
statistically settled (`benchmarks/PHASE_PROGRESS.md`, 2026-10-12). Turn
them on when recall has room and nothing a user said may go unstored:
`Memory(extractor=OpenAIExtractor(second_look=True, keep_unextracted=True))`.

### Answering from memories

`tsm.reading` holds the prompt for the model that answers from what
`recall()` returned:

```python
from tsm.reading import answer_prompt, final_answer

results = memory.recall(question, user_id="alice", token_budget=600)
reply = client.chat.completions.create(
    model="gpt-4.1-mini", max_tokens=800, temperature=0,
    messages=answer_prompt(question, results, today="2024-03-09"))
print(final_answer(reply.choices[0].message.content))
```

`answer_prompt` lists the memories in the order they were said (each with
its date), and asks the model to name the ones that bear on the question,
to settle what they add up to (a later note replaces an earlier one, a
total counts each item once, "last week" counts from the note's date) and
to end with one line that starts with `Answer:`. `final_answer` returns
what follows that line. `today` is the date the question is asked; "a month
ago" has no meaning without it. Give the model about 800 reply tokens: a
reply cut off before its last line has no answer in it.

With the facts in front of it, a plain "answer from these snippets" prompt
still gets totals, changed values and date arithmetic wrong.

| Questions (LongMemEval) | Plain prompt, 150 tokens | This prompt, 600 tokens | Change |
|---|---|---|---|
| 115 the prompt was written on | 87 right | 98 right | |
| 355 held out (2026-10-21) | 259 (0.730) | 296 (0.834) | 52 gained, 15 lost |

So the gain holds on questions the prompt was not written against.

`tsm.answering.answer_with_tools(client, model, question, results, today)`
asks the same way and also gives the model four tools: arithmetic, the days
between two dates, a date some days away, and a count of distinct items.
It did not help. On the same 115 questions it answered 94 against 98 for the
plain reading prompt (3 gained, 7 lost). It is there for models that are
worse at arithmetic than gpt-4.1-mini.

### Results, budgets, and lifecycle

- `recall()` returns dicts, best first, with `id`, `text`, `context`,
  `score`, `role` (the stored source role), `turn_index`, `timestamp` and
  `source`.
  - `context` is the text to put in a prompt: `text` with when it was said
    in front (next point) and, for a fact that has since changed, a marker
    (the point after).
  - `source` is the message the memory was extracted from. It is `None` for
    gists.
- Source turns are opt-in. `recall(..., sources="turn")` searches on the
  facts and returns the turns they came from: one result per message, with
  `facts` and `ids` listing the facts it stands for, and the speaker named
  in `context`. A long turn is cut to its sentences that share words with
  those facts and the question (`source_tokens`, 120 by default). At a
  600-token budget this cost judged answers, 86 of 115 right against 98 for
  facts (6 gained, 18 lost), because far fewer turns than facts fit. Cut to
  30 tokens, turns fit as well as facts and scored 67: the cut often dropped
  the short sentence that held the answer.
- Event dates are opt-in. With `recall(..., event_dates=True)`, a relative
  time in what is shown is followed by the date it points to, counted from
  the day it was said: "two weeks ago [week of 2023-05-08]", "last Saturday
  [2023-05-20]", "on March 3rd [2023-03-03]". `tsm.when.annotate` does this
  with rules, no model. Only 2.4% of extracted facts have such a phrase. It
  was level on judged answers: 100 against 98 of 115, and 298 against 296 of
  355 held out (8 gained and 4 lost over both).
- Time tags: `context` starts with the memory's date when its message was
  added with one (`{"role": "user", "content": ..., "timestamp":
  "2024-03-02T18:00:00Z"}`; an ISO date or epoch seconds), as in
  `[2024-03-02] I joined a volleyball league`, and otherwise with its turn,
  `[turn 12] ...`, which only orders memories. A model asked which of two
  things came first, or which of two values is the newer one, has nothing
  else to go on: on 115 judged LongMemEval questions the tags gained 8
  answers and lost 1, on temporal and knowledge-update questions
  (`benchmarks/PHASE_PROGRESS.md`, 2026-10-07). A gist is tagged with the
  span of what it summarizes when that was dated (`[2023-01-05 to
  2023-03-02] ...`; added 2026-10-21, judged level 2026-10-28). The tags count against
  `token_budget`; `recall(..., time_tags=False)` leaves
  them out, and `time_tags="minute"` adds the time of day when the timestamp
  has one (`[2024-03-02 18:00] ...`), which is what tells two memories of
  one day apart.
- A fact that a newer one replaced is never removed from recall. It is
  ranked lower and carries `superseded_by` (the id of the current belief)
  and `chain`; a result without `superseded_by` is current. With a verifier
  installed, two more things happen: its `context` starts with
  `[earlier, changed by 2024-06-01] ` (the date its replacement was said,
  also in the result's `valid_until`; `[earlier, since changed] ` when that
  is not dated; the dated form was judged level on 2026-10-28),
  so a model reading it knows what it is and when it changed, and
  the fact that replaced it is ranked where the old one would have been and
  brought in if the search missed it (a question in the old wording often
  finds only the old fact). Pass `exclude_superseded=True` to drop
  superseded facts from recall instead: a question about what was true
  earlier then has nothing to go on. The lower rank (its score is multiplied
  by `supersession_demotion_factor`, 0.4) keeps a replaced fact out of a
  small context almost always; `supersession_demotion_factor=1.0` leaves it
  where it ranks, marked, next to the fact that replaced it. Judged answers
  came out level between the two.
- `recall(..., token_budget=N)` returns the best *set* that fits `N` tokens
  instead of the top-k: greedy MMR over a candidate pool, with a penalty for
  saying what is already there and a bonus for a turn not yet covered
  (`tsm.budget.select_under_budget`). Only a memory nearly identical to one
  already chosen is skipped outright: memories that merely read alike are
  often separate instances a question needs together (two classes in one
  week, a count before and after it changed). The budget is filled; pass
  `max_items=` to also limit the number of results.
- `recall()` is a read: asking does not change what the next call returns.
  `reinforce(memory_ids)` tells the memory which results were actually
  used, and their links in the concept graph get stronger. (Pass
  `reinforce_on_search=True` to reinforce everything every search returns,
  as older versions did; the same question then comes back differently the
  second time.)
- Ranking. The conversational profile ranks by embedding similarity plus a
  bounded vote from the concept graph and BM25: the vote can move a memory
  by at most half the spread of similarity among the candidates
  (`graph_weight=0.5`), so it settles close calls and cannot outvote the
  embedding. `graph_weight=0` is pure similarity ranking. The older fixed
  boost is still there (`Memory(db, cognitive_alpha=0.5)`); on OpenAI
  embeddings it ranked by word overlap and lost answers
  (`docs/cognitive_graph.md`, section 3).
- `consolidate()` runs the engine's consolidation cycle, including belief
  revision. What a superseded fact turns into depends on the verifier:
  - `verifier="llm"` (`tsm.verification.LLMVerifier`): every new fact is
    paired with its closest older facts (similarity at least 0.45, the
    closest one and at most one more about as close) and a chat model
    decides whether it replaces them. Accepted pairs are committed, and
    recall marks the older fact as earlier and serves the newer one with
    it. Any OpenAI-compatible endpoint works
    (`LLMVerifier(base_url="http://localhost:11434/v1", model=...)` for a
    local server); one short request per few pairs, verdicts cached under
    `<db_path>/tsm_cache`.
  - `verifier="nli"` (`NLIVerifier`): a small local cross-encoder vets the
    pairs the engine's own lexical detection proposes. Free and offline, but
    it only sees what that detection finds, and it cannot tell a changed
    value from a second person or a second item.
  - no verifier: the engine's detection runs unchecked, so its results
    are not shown to a model. A fact it marks as superseded is ranked lower
    and flagged with `superseded_by`; its `context` is left as it is.

  Measured on 127 held-out labeled pairs (`benchmarks/cognitive_eval/
  belief_pairs_eval.py --split test`, MiniLM embeddings; details in
  `benchmarks/PHASE_PROGRESS.md`):

  | verifier | real updates caught | still-true facts marked stale |
  |---|---|---|
  | none | 16 of 54 | 8 of 58 |
  | `"nli"` | 16 of 54 | 6 of 58 |
  | `"llm"` with `gpt-4o-mini`, the default | 49 to 50 of 54 | 4 to 8 of 58 |
  | `"llm"` with a local 4.7B model, `qwen3.5:4b` | 48 to 49 of 54 | 7 to 9 of 58 |
  | `"llm"` with local `Ternary-Bonsai-2-4B-PQ2_0` | 49 / 48 of 54 | 7 / 10 of 58 |
  | `"llm"` with local `Ternary-Bonsai-2-9B-PQ2_0` (2026-10-19) | 40 / 40 of 54 | 9 / 12 of 58 |

  Bonsai's slash-separated figures are isolated / shared scope, measured on
  2026-10-19 with the shipped prompt, non-thinking mode, greedy decoding,
  batches of 8 and a 1,024-token output limit. It remains opt-in: catching
  more updates does not eliminate false supersessions. See the Bonsai follow-up
  in `benchmarks/PHASE_PROGRESS.md` for the runtime and evaluation commands.
  The 9B model (same settings, MiniLM embeddings) caught fewer updates than
  the 4B and marked more true facts stale. It also failed the reply format on
  1 to 6 pairs per run. It needs about 3.2 GB of GPU memory with
  EmbeddingGemma 2 loaded. Use the 4B.

  In each range the first figure is with every pair in its own store and the
  second with all pairs in one store, where statements of different pairs
  collide as well: with `gpt-4o-mini`, 5 of the 8 came from another pair's
  statement, 4 of them statements that do conflict within one person (a
  second job, car, university and flat).

  The LLM verifier finds the updates the other two miss. `gpt-4o-mini`
  retired every update it was shown; the 5 it missed were worded so
  differently that their similarity fell below 0.45 and it was never asked
  (`candidate_min_cosine`). The "still true" pairs are built to look like
  updates, and what it still gets wrong among them is mostly dated events
  ("I attended PyCon in 2022", "I attended PyCon in 2024"). The small local
  model also confuses two things of one kind ("I play the guitar", "I play
  the piano").
- A token budget per user: `Memory(db, max_user_tokens=256,
  gist_summarizer=OpenAIGistSummarizer())` keeps each user's memory under
  that many tokens. `consolidate()` compacts the users written to since the
  last pass (`compact(user_id)` does one on request, `compact()` all of
  them): the newest facts stay as they are, what the user said before what
  the assistant said, and everything older is rewritten as a few short
  gists stored like any other memory (`role` `"summary"`). The next pass
  folds those gists again together with what has aged out since, so the
  store stays within its budget however long the history grows. Facts a
  newer one replaced are folded first. Gists are written before anything is
  removed, and a summarizer that fails removes nothing. Without a
  summarizer the older facts are simply dropped. This is the policy the
  bounded-storage evaluation measures (`tsm.compaction.plan_compaction`).
  On 112 judged LongMemEval questions a 256-token store answered 0.464
  when compacted once and 0.411 when compacted four times as the
  conversation arrived, against 0.348 for Mem0 at the same allowance, 0.098
  for keeping only the newest facts, and 0.59 for the unbounded store
  (`benchmarks/PHASE_PROGRESS.md`, 2026-10-07).
- The engine's own cap is a different thing: `Memory(db, max_records=500,
  gist_summarizer=...)` is one count for the whole store. It keeps whatever
  was used or written most recently, whoever said it, and folds eviction
  victims into gists of 24 facts (or the model-free
  `ExtractiveGistSummarizer()`, also in `tsm.gist`). A memory is only
  deleted once its gist is stored: if the summarizer raises, that chunk's
  memories stay and are tried again on the next eviction. A summarizer that
  returns an empty string is saying there is nothing worth keeping, and the
  chunk is dropped. Prefer `max_user_tokens` for conversational memory.
- `add()` stores the facts of one call as a single validated batch: it either
  stores all of them or raises having stored none, and it can be called from
  several threads.
- Scoping is enforced by the engine at every stage of a search, so one
  user's records never appear in, or use up slots of, another user's recall.
  `user_id=None` means "no scope" on both `add` and `recall`: such facts are
  visible to everyone, and such a recall sees everything.
- The engine is the only store. Reopening a `db_path` — in the same process
  or a later one — continues exactly where it left off: `add` keeps
  appending with fresh ids, and role, scope and text are read back from the
  database. `close()` flushes and releases the database lock, mmaps and
  worker threads; the path can be reopened immediately, and further calls on
  the closed object raise `RuntimeError`.
- Nothing acknowledged is lost if the process dies without `close()`: the
  next open replays the write-ahead log. (`mem.engine.recovery_report()` says
  what an open had to recover.) After a power loss, the writes since the last
  `flush()` / `close()` may be missing, unless the store was opened with
  `Memory(db, sync_writes=True)`: each `add()` is then synced to disk before
  it returns (about 5 ms per call on an NVMe SSD, whatever the number of
  facts in the call up to a few hundred).
- In a CUDA build (`make build-python FEATURES=cuda`) vector search runs on
  the GPU once a store holds more than 4,096 records; nothing changes in the
  API. `mem.engine.gpu_search_stats()` shows whether it is active. See
  [GPU acceleration](gpu_acceleration.md).

### Supported Embedders & Rerankers

```python
from tsm import Memory, SentenceTransformerEmbedder, ColBertReranker

# Local open-source embedder + ColBERT on CUDA
memory = Memory(
    db_path="./my_db",
    embedder=SentenceTransformerEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda"),
    reranker=ColBertReranker(model_name="LiquidAI/LFM2.5-ColBERT-350M", device="cuda"),
)

# EmbeddingGemma 2, local and offline once cached (pip install "tsm[gemma]")
memory = Memory("./my_db", embedder="gemma", extractor="passthrough")
```

**Local Bonsai verifier.** Serve the PQ2 model with the
[PrismML runtime](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b10777-e3de715)
and the alias `bonsai-2-4b-pq2` on `127.0.0.1:18080`. The live SDK smoke check
used `-ngl 99 -fa on -c 8192 -np 1 --jinja --cache-ram 128`. The cache bound
matters: the default is 8,192 MiB of host prompt cache, which grew the server's
private memory to about 11 GB in the longer evaluation. Then:

```python
from tsm import Memory
from tsm.verification import LLMVerifier

verifier = LLMVerifier(
    model="bonsai-2-4b-pq2",
    base_url="http://127.0.0.1:18080/v1",
    api_key="local",
    cache_dir="./gemma_db/tsm_cache",
    request_timeout=600.0,
    request_kwargs={
        "max_tokens": 1024,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}, "seed": 42},
    },
)
with Memory("./gemma_db", embedder="gemma", extractor="passthrough",
            verifier=verifier, idle_consolidation_secs=60) as memory:
    memory.add("My meeting is on Tuesday.", user_id="alice")
    memory.add("Correction: my meeting is on Wednesday.", user_id="alice")
    memory.consolidate()  # Explicit pass; otherwise the worker waits for idle time.
```

Compact PQ2/PTQ packs need the Prism fork. Non-thinking mode is explicit:
otherwise the model may spend thousands of tokens reasoning before returning
the verdict lines. Even with thinking disabled some batches are verbose or
unparseable; unresolved pairs are left uncommitted and retried later. Use a fresh
verdict cache when changing weights or decoding settings.

**Local embedders.** `SentenceTransformerEmbedder` follows each model's own card.
`encode` / `encode_document` embed stored text and `encode_query` embeds a
question, each with the model's task prompt (`title: none | text: ` and
`task: search result | query: ` for EmbeddingGemma 2); `Memory` asks for the right
one (`tsm.interfaces.AsymmetricEmbedder`), and an embedder with only `encode` works
unchanged. Per-model settings from the card are applied by name
(`tsm.embedders._MODEL_DEFAULTS`): for EmbeddingGemma 2 only the text model is
loaded (271M parameters, about 0.5 GB on the GPU in bfloat16, against 740M with
its image and audio encoders), the input is bounded at its 8,192-token context,
and float16 is refused because its activations overflow it (NaN, no error). A
load-time probe checks the model returns finite vectors of the declared size.
The cosine thresholds of belief detection are tuned on OpenAI vectors and do not
carry over: with EmbeddingGemma 2's vectors the profile's 0.85 / 0.75 marks 30 of 58
still-true facts stale (8 of 58 on MiniLM) and cost 10 of 77 questions their
evidence at one history (p = 0.012). An embedder can declare `engine_defaults`, which
`Memory` applies under the conversational profile and below anything passed to
`Memory(...)`; EmbeddingGemma 2 declares 0.92 / 0.86, which on the held-out pairs
marks 7 of 58 stale and costs 1 question (`benchmarks/PHASE_PROGRESS.md`,
2026-10-19). Give any other new embedder the same check
(`benchmarks/cognitive_eval/belief_pairs_eval.py --embed-model ...`) before trusting
the profile with it.
`truncate_dim=512` (also 256, 128) keeps the leading dimensions of a Matryoshka
model and re-normalizes; `query_prompt_name="QuestionAnswering"` picks another of
the model's prompts. Vectors are unit length, and the model is called under a lock
(HF tokenizers are not safe to share across threads).

**One embedder per database.** Two models of one dimension look the same to the
engine, and searching one's vectors with the other's queries returns plausible,
wrong results without an error. `Memory` therefore records the embedder's
`fingerprint` (model, commit, dimension, task prompts; for any other embedder its
class name and dimension: give it a `fingerprint` attribute if one class serves
several models) in `tsm_embedder.json` next to the engine's files and refuses to
open the database with a different one (`tsm.EmbedderMismatchError`, a
`ValueError`). An empty database takes any embedder; one written before this record
existed adopts the first embedder that opens it. `allow_embedder_change=True` is for
an embedder that produces the same vectors. Changing models means re-ingesting:
`update()` allocates a fresh insert sequence, which belief revision uses to order
old against new, so re-embedding in place would scramble the history.

**Idle consolidation.** `Memory(..., idle_consolidation_secs=60)` runs
`consolidate()` in a background thread once nothing has called `add()` or
`recall()` for that long and something was written since the last pass (and once
after opening a store that already holds memories): verification, supersession
commits and `max_user_tokens` compaction then happen off the request path. A pass
is not interrupted by a call (the engine serves it meanwhile; a manual
`consolidate()` waits for the pass), a failed pass or one a verifier could not
finish (`LLMVerifier.unresolved`) is tried again after another idle period, nothing
is deleted on failure, and `close()` stops the thread before it releases the
engine. It is separate from the engine's own `auto_consolidation_secs` timer,
which has no verifier and which the conversational profile leaves at 0.

---

## 2. Low-Level Native Engine (`import turbomemory`)

Direct PyO3 bindings for high-throughput vector ingestion, 3-tier TurboQuant hardware compression, and lock-free snapshot search.

### 3-Tier TurboQuant Configuration

```python
import numpy as np
import turbomemory

dim = 512  # Must be a power of 2 for Fast Walsh-Hadamard Transform (128, 256, 512, 1024)

engine = turbomemory.MemoryEngine(
    db_path="./turbo_db",
    dimension=dim,
    hot_capacity=1000,              # First 1,000 vectors stay in RAM (FP32)
    warm_capacity=10000,            # Next 10,000 vectors in TurboQuant-Prod (8-bit)
    warm_quantizer="turbo_prod8",   # 8-bit FWHT + QJL residual (3.6x compression)
    cold_quantizer="turbo_mse1",    # 1-bit FWHT sign quantization (32x compression)
    auto_consolidation_secs=60,
    outlier_count=0,
)

# Insert with zero-copy NumPy array
vector = np.random.randn(dim).astype(np.float32)
engine.insert(
    id="mem_001",
    text="Deployment configuration notes",
    embedding=vector,
    importance_score=1.0,
    concepts=["deployment", "config"],
    scope="team_alpha",
)

# Search across all tiers simultaneously
query_vec = np.random.randn(dim).astype(np.float32)
hits = engine.search(
    query_text="deployment notes",
    query_embedding=query_vec,
    top_k=5,
    scope="team_alpha",
)

# Check graph introspection and GPU status
print(f"GPU Accelerated: {engine.gpu_accelerated}")
print(f"Graph Stats: {engine.graph_stats()}")

engine.flush()
engine.close()   # releases the database; later calls raise RuntimeError
```

### Reading records back

```python
engine.get_records(["mem_001", "missing"])
# [{'id': 'mem_001', 'text': 'Deployment configuration notes', 'payload': None,
#   'scope': 'team_alpha', 'source_role': None, 'importance': 1.0,
#   'created_at': 1790000000, 'insert_seq': 1}, None]

engine.next_insert_seq()   # durable, monotonically increasing; never reused
engine.closed              # False until close()
engine.recovery_report()   # what this open had to repair; all zeros after a clean close
# {'wal_ops_replayed': 0, 'wal_inserts_without_vector': 0, 'wal_bytes_discarded': 0,
#  'segments_discarded': 0, 'segment_dirs_removed': 0, 'graph_nodes_pruned': 0}
```

### Input rules

- Vectors must have the store's dimension, be finite, and not be all zero;
  ids must be non-empty; payloads must be valid JSON. Anything else raises
  `ValueError` and changes nothing (a batch is rejected as a whole, and a
  rejected `update` leaves the existing record as it was).
- `top_k` is clamped to the number of records.
- `update(id, ...)` replaces the record atomically and returns `False` when
  the id does not exist.
- A store whose `vectors.bin` or `memory.redb` is missing or truncated raises
  `RuntimeError` ("corrupted store: ...") on open instead of opening empty.
