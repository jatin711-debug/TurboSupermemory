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

### Results, budgets, and lifecycle

- `recall()` returns dicts, best first, with `id`, `text`, `context`,
  `score`, `role` (the stored source role) and `turn_index`. `context` is
  the text to put in a prompt: the same as `text` unless the fact is marked
  (next point).
- A fact that a newer one replaced is never removed from recall. It is
  ranked lower and carries `superseded_by` (the id of the current belief)
  and `chain`; a result without `superseded_by` is current. With a verifier
  installed, two more things happen: its `context` starts with
  `[earlier, since changed] `, so a model reading it knows what it is, and
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
  instead of the top-k: greedy MMR over a candidate pool with a redundancy
  cutoff and a cross-turn coverage bonus (`tsm.budget.select_under_budget`).
  The budget is filled; pass `max_items=` to also limit the number of
  results.
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
```

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
