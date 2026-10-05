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

## Tests

```
python -m unittest tsm.tests.test_memory -v   # from the repo root, no API key
make test-python                              # same, after rebuilding the extension
```
