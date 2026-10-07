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
(`embedder="local"`, `reranker="colbert"`, `NLIVerifier`) need the `cpu` extra.

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
- Extra engine kwargs override the profile: `Memory(db, cognitive_alpha=0.7)`.
- Plug in local backends via the `Embedder` / `Extractor` / `Verifier`
  protocols (`tsm.interfaces`); pass instances to `Memory(...)`.
- Budget recall: `recall(..., token_budget=N)` returns the best set that fits
  (`tsm.budget.select_under_budget`, greedy MMR with a cross-turn bonus).
- Compress instead of delete: with `max_records` set, pass
  `gist_summarizer=OpenAIGistSummarizer()` (or the model-free
  `ExtractiveGistSummarizer()`, both in `tsm.gist`) and eviction victims are
  folded into searchable gist records instead of being dropped. If the
  summarizer raises, the affected memories are kept and retried on the next
  eviction; only an empty summary drops them.
- Verified supersession: pass `verifier=NLIVerifier()` (`tsm.verification`,
  needs `torch` + `transformers`) — consolidation then proposes, NLI-vets
  (accept contradiction/entailment, reject neutral), and commits only the
  survivors; stale facts are excluded from recall.
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

## When a backend misbehaves

- A rejected API key or a malformed request fails at once with the HTTP
  status in the message; only transient errors (rate limits, timeouts, 5xx)
  are retried with backoff.
- The OpenAI extractor never drops a message: a reply that is cut off or is
  not valid JSON is requested again with a larger budget, and if that fails
  the message itself is stored as one fact. Only well-formed replies are
  cached.
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
