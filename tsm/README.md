# tsm — TurboSuperMemory Python SDK

One-flag conversational memory over the compiled `turbomemory` engine:
scoped fact storage, belief revision, verified supersession, budget recall.

## Install / requirements

Build the extension first (`make build-python` → repo-root `turbomemory.pyd`),
keep the `tsm/` package next to it, `pip install numpy`. Defaults need
`pip install openai` and `OPENAI_API_KEY` set.

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
- The id→text map is in-memory only (per-process); engine data persists,
  result `text` fields don't across restarts. See `Memory` docstring.

## Tests

```
python -m unittest tsm.tests.test_memory -v   # from the repo root, no API key
```
