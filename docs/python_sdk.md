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

- `recall()` returns dicts with `id`, `text`, `score`, `role` (the stored
  source role) and `turn_index`; superseded facts whose current belief is not
  in the result set also carry `superseded_by` and `chain`.
- `recall(..., token_budget=N)` returns the best *set* that fits `N` tokens
  instead of the top-k: greedy MMR over a candidate pool with a redundancy
  cutoff, a cross-turn coverage bonus and an adaptive item cap
  (`tsm.budget.select_under_budget`).
- `consolidate()` runs the engine's consolidation cycle; with a `verifier`
  installed (`tsm.verification.NLIVerifier`) supersessions are proposed,
  vetted, and only then committed.
- Compress instead of delete: `Memory(db, max_records=500,
  gist_summarizer=OpenAIGistSummarizer())` (or the model-free
  `ExtractiveGistSummarizer()`, both in `tsm.gist`) folds eviction victims
  into searchable gist records. A memory is only deleted once its gist is
  stored: if the summarizer raises, that chunk's memories stay and are tried
  again on the next eviction. A summarizer that returns an empty string is
  saying there is nothing worth keeping, and the chunk is dropped.
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
