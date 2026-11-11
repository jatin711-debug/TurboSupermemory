# TurboSuperMemory (TSM)

[![CI](https://github.com/jatin711-debug/TurboSupermemory/actions/workflows/ci.yml/badge.svg)](https://github.com/jatin711-debug/TurboSupermemory/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rust-1.99%20(pinned)-orange.svg)](./rust-toolchain.toml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![Status: Beta](https://img.shields.io/badge/status-beta-yellow.svg)](#known-limits)

Long-term memory for AI agents. TSM is an embedded storage engine written in
Rust, with a Python SDK (`tsm`) that stores a conversation as it happens,
finds what bears on a question, and packs it into a token budget for the model
that answers.

- **`tsm.Memory`** (Python): `add()` messages, `recall()` a budgeted context,
  `consolidate()` to revise beliefs. Works with OpenAI or fully local models.
- **`turbomemory`** (Rust engine, PyO3 extension): crash-safe vector storage
  with tiered indexes, scoped search, a concept graph, and optional CUDA exact
  search.
- **`turbomemory-server`**: the same engine behind gRPC and REST.

Beta, one maintainer. Every number on this page comes from this repository's
own runs. Each has a dated entry with its command, sample size and caveats in
[`benchmarks/PHASE_PROGRESS.md`](./benchmarks/PHASE_PROGRESS.md), including the
experiments that failed.

---

## Measured results

### LongMemEval-S, every answerable question

Each question comes with a history of about 120,000 tokens. The history is
stored as messages (no model is asked at write time), `recall()` returns 2,000
tokens of it, and gpt-4.1-mini answers from those with the `tsm.reading`
prompt. gpt-4.1-mini judges. OpenAI `text-embedding-3-small` vectors, one
judged pass, 2026-11-05.

| Question type | Questions | Answered right | All evidence in the 2,000 tokens |
| :--- | ---: | ---: | ---: |
| knowledge-update | 72 | 70 (97.2%) | 70 |
| multi-session | 121 | 101 (83.5%) | 111 |
| single-session-assistant | 56 | 54 (96.4%) | 56 |
| single-session-preference | 30 | 20 (66.7%) | 29 |
| single-session-user | 64 | 61 (95.3%) | 62 |
| temporal-reasoning | 127 | 105 (82.7%) | 111 |
| **All** | **470** | **411 (87.4%)** | **439 (93.4%)** |

How to read it:

- The benchmark's 30 abstention questions are skipped, so this is not
  comparable with published scores over all 500, which also use other readers
  and judges.
- The reader is not deterministic. It moves about 1 question in 20 between
  passes on identical contexts, so differences of a few questions are noise.
- Of the 59 wrong answers, 41 had all of their evidence in the context. Most of
  what is left is the reader's, not retrieval's.

### What moved answers

| Change | Result | Sample |
| :--- | :--- | :--- |
| The reading prompt (`tsm.reading`) at 600 tokens, against a plain "answer from these snippets" prompt at 150 | 296 against 259 right (52 gained, 15 lost) | 355 held-out LongMemEval questions |
| Keeping every message next to the extracted facts (`keep_messages`, the default) | 27 against 19 right at 2,000 tokens (8 gained, 0 lost, p = 0.01) | 29 LongMemEval-S questions |
| Returning the assistant's reply whole when the question asks what the assistant said | 54 against 41 right (13 gained, 0 lost) | 56 LongMemEval-S questions |
| A date or turn in front of each memory (`time_tags`, the default) | 8 gained, 1 lost | 115 LongMemEval questions |
| Folding old memories into gists instead of deleting them, in a 256-token store | 0.482 against 0.098 (46 gained, 3 lost, p < 0.001) | 112 LongMemEval questions |

### What did not

| Idea | Result | Status |
| :--- | :--- | :--- |
| Concept-graph ("cognitive") search against plain vector search, OpenAI vectors | 357 against 360 right of 414 (p = 0.70); tied 10 / 10 on the held-out set | Still the default; adds nothing measurable here |
| Belief revision's effect on answers | Knowledge-update questions: 10 of 18 with it, 9 without | Kept; superseded facts are demoted and flagged, never dropped |
| A larger context: 8,000 tokens instead of 2,000 | Evidence rose from 91 to 96 of 96, right answers fell from 91 to 82 (p = 0.012) | Use about 2,000 |
| A typed event table for counting questions | 93.6 against 95.8 of 121, five reads each | Removed 2026-11-09 |
| Rewriting conversations into observations | Best 93.0 against 94.7 to 96.0 of 121 for the raw passages | Removed 2026-11-09 |
| Returning source turns in place of facts | 86 against 98 of 115 at 600 tokens | Removed 2026-11-09 |
| Writing the date after a relative time ("two weeks ago [...]") | 8 gained, 4 lost over 470 | Removed 2026-11-09 |
| Reader tools for arithmetic and dates | 94 against 98 of 115 | Removed 2026-11-09 |
| A ColBERT reranker (LFM2.5-ColBERT-350M) after search | Evidence score only: level on OpenAI and EmbeddingGemma vectors (5 gained, 3 lost on each), better on MiniLM's (19 / 6), where EmbeddingGemma alone did better still | Removed 2026-11-10 |
| Time of day in the tags (`time_tags="minute"`) | 84 against 90 of 115 (p = 0.07) | Not the default |
| A second extraction pass for messages that yielded no fact | 6 more answers at 600 tokens, 8 fewer at 150; neither significant | Off by default |

The graph's earlier gains were measured on MiniLM vectors only. Whether it
earns its place with a weak local embedder is open.

Unless a row says LongMemEval-S, a LongMemEval sample on this page is the
benchmark's oracle variant, where a history holds only the sessions with
evidence. Settings were chosen on 115 of its questions; the 355 "held-out"
ones are the rest.

### Controls and a second benchmark

- **Against reading everything.** On 29 LongMemEval-S questions, reading the
  whole 124,000-token history scored 22. Memory at 2,000 tokens scored 27 with
  facts and messages and 26 with messages alone (EmbeddingGemma 2 vectors).
- **Against the evidence itself.** On 115 questions where only the evidence
  sessions are given, reading them directly scored 104 and memory at 600
  tokens 98 (p = 0.24). The memory layer costs about six questions there.
- **LoCoMo**, ten two-person conversations, run as a check that the work was
  not fitted to LongMemEval. It found one thing that was: a ranking rule keyed
  to LongMemEval's question wording. Replacing it with a general rule took the
  messages-only evidence score from 1,177 to 1,560 of 1,977 and left
  LongMemEval unchanged. After that fix, all the evidence was in a 2,000-token
  context for 1,737 questions with facts and messages. On a judged sample of
  120, facts with messages answered 91 and messages alone 97, which is within
  the reader's noise.

### Bounded memory, and Mem0

`Memory(max_user_tokens=N)` keeps a user's store under a token budget by
folding older memories into short gists. Judged accuracy by stored size, 112
to 114 LongMemEval questions, 2026-10-07:

| Stored per conversation | Delete oldest | TSM compress | Mem0 |
| :--- | ---: | ---: | ---: |
| 64 tokens | 0.018 | 0.140 | 0.193 |
| 128 tokens | 0.036 | 0.250 | 0.268 |
| 256 tokens | 0.098 | 0.482 | 0.348 |

Compression beats deletion at every size. Against Mem0 it is level at 64 and
128 tokens (p = 0.33, 0.87) and ahead at 256 (p = 0.02). Read that last row
with the store sizes: Mem0's own memory averaged 170 tokens a conversation, so
the 256-token allowance did not constrain it, while TSM used 218. The TSM
column is the evaluation harness's store; `Memory(max_user_tokens=256)` itself
scored 0.464 on the same questions.

An earlier version of this page led with a Mem0 head-to-head (66.7% against
55.6%, the same at three budgets) and a 20-question BEAM table. Neither has an
entry in the progress log, so both were removed. The head-to-head that is
logged is a 48-question audit from 2026-08-14: 56.2% against 39.6% at 150
tokens of context. It measured an evaluation adapter, not `tsm.Memory`, and on
120 conversations that adapter did not beat plain vector search (0.504 against
0.539, 2026-10-07).

### Not measured

- TurboQuant and RaBitQ quantizers are implemented and unit-tested. Their
  recall figures in [`docs/core_quantization.md`](./docs/core_quantization.md)
  are approximate, and the defaults do not use them.
- Nothing above 100,000 records has been benchmarked.
- macOS and AArch64 are not built or tested anywhere.

---

## Quickstart

```python
from tsm import Memory
from tsm.reading import answer_prompt, final_answer

with Memory("./agent_memory") as memory:          # OpenAI by default: OPENAI_API_KEY
    memory.add(
        [{"role": "user", "content": "I moved to Lisbon in March.",
          "timestamp": "2024-03-02"}],
        user_id="alice",
    )

    question = "Where does Alice live?"
    results = memory.recall(question, user_id="alice", token_budget=2000)

    # results[i]["context"] is the text for a prompt: "[2024-03-02] user: I moved ..."
    messages = answer_prompt(question, results, today="2024-03-09")
    # reply = client.chat.completions.create(model=..., max_tokens=800, messages=messages)
    # print(final_answer(reply.choices[0].message.content))
```

Without any API key:

```python
memory = Memory("./agent_memory", embedder="gemma", extractor="none")
```

That stores the messages as said and embeds them locally with EmbeddingGemma 2
(`pip install "tsm[gemma]"`). It answered 26 of the 29 LongMemEval-S questions
above.

What the defaults do:

- **Write.** One extraction request per message (`OpenAIExtractor`), and every
  message is also kept as said, in passages of about two sentences.
  `extractor="none"` keeps only the passages and asks no model.
- **Read.** `recall(..., token_budget=N)` returns the best set that fits, each
  result with `context`: the text with its date or turn in front. `recall()`
  is a read; it changes nothing.
- **Time.** A question such as "what did I buy ten days ago" is matched against
  the dates memories were said. Pass `now=` when the question is not being
  asked today.
- **Beliefs.** `consolidate()` proposes which facts a newer one replaced. With
  `verifier="llm"` a chat model vets each pair, and `recall()` marks the old
  fact (`[earlier, changed by 2024-06-01] ...`) and serves its replacement.
  A replaced fact is never removed from recall.
- **Identity.** A database remembers which embedder wrote it and refuses
  another, because two models of one dimension return wrong results without
  an error.
- **Profile.** `profile=None` gives a plain scoped vector store. On the raw
  engine, belief revision, eviction, deduplication and auto-importance are
  all opt-in.

The full SDK reference is [`docs/python_sdk.md`](./docs/python_sdk.md) and
[`tsm/README.md`](./tsm/README.md).

Install: `make dev` builds the extension with maturin and installs `tsm` into
the active environment; `make wheel` builds a wheel. Python 3.12 or newer.

---

## The engine

`turbomemory.MemoryEngine` can be used on its own
([`docs/bindings_api.md`](./docs/bindings_api.md)).

```python
import turbomemory

engine = turbomemory.MemoryEngine(
    db_path="./turbo_db",
    dimension=768,
    hot_capacity=10_000,        # records kept in the mutable hot segment
    warm_capacity=100_000,      # records in the warm tier before compaction to cold
    warm_quantizer="scalar8",   # the default
    cold_quantizer="scalar8",   # the default; "rabitq1" stores 100 bytes per 768-d vector
)
```

### Search and tiers

- Up to 4,096 records, every search is an exact scan.
- Above that, immutable segments are searched (HNSW for sealed hot segments,
  quantized scans for warm and cold) and the candidates are reranked against
  the full float32 vectors. The regression gate holds ANN recall at 95% or
  better.
- Warm and cold tiers default to 8-bit scalar quantization (4x smaller).
  1-bit sign, RaBitQ and TurboQuant are selectable; TurboQuant needs a
  power-of-two dimension. See
  [`docs/core_quantization.md`](./docs/core_quantization.md).
- A scope or payload filter bounds every stage of a search, including the
  concept graph's expansion.

### Durability

- Every write is validated before anything changes. A rejected insert, batch
  or update leaves the store as it was, and a batch is all-or-nothing.
- Writes are logged with a checksum of their vector. If the process dies
  without `close()`, the next open replays the log: nothing acknowledged is
  lost. `engine.recovery_report()` says what an open had to repair.
- By default the log is fsynced by `flush()` and `close()`, not per write.
  After a power loss, writes since the last flush can be missing; they are
  never half-applied.
- `sync_writes=True` syncs each write before it returns. On an NVMe laptop SSD
  a single insert goes from 0.09 ms to 5.5 ms and a batch of 100 from about
  2.5 ms to 8 ms. It relies on the drive honouring flush requests; no
  power-cut test was run.
- Index segments are derived data. A damaged one is discarded on open and
  rebuilt from the vectors.
- A store whose vector file or metadata file is missing or truncated refuses
  to open, instead of opening empty.

The failures behind these rules are pinned in
[`crates/turbomemory_storage/tests/robustness.rs`](./crates/turbomemory_storage/tests/robustness.rs).
More in [`docs/storage_persistence.md`](./docs/storage_persistence.md).

### GPU (opt-in)

`make build-python FEATURES=cuda` keeps the store's vectors in device memory
and answers a search with one cuBLAS product over all of them. The result is
exact. It applies above 4,096 records while the store fits the device memory
budget; any CUDA error falls back to the CPU path.

Measured on an RTX 3050 Laptop (4 GB) against the CPU build on the same files:

| Store | Path | One query | Batch of 32, per query | 8 threads |
| :--- | :--- | :--- | :--- | :--- |
| 100,000 × 384-d | CPU | 5.9–7.5 ms | 6.0–9.2 ms | 128–174 /s |
| | CUDA | 1.35–1.55 ms | 0.34–0.39 ms | 1,436–1,754 /s |
| 100,000 × 768-d | CPU | 13.2–15.8 ms | 13.6–15.8 ms | 69–71 /s |
| | CUDA | 2.3–2.4 ms | 0.34–0.39 ms | 1,088–1,190 /s |

Method, limits and options are in
[`docs/gpu_acceleration.md`](./docs/gpu_acceleration.md). The SDK's local
models (embedder, verifier) use CUDA through PyTorch and need no
special build.

---

## Server

```bash
make build-api
TURBO_DB_PATH=./server_db TURBO_DIMENSION=768 make api-server   # gRPC :50051, REST :8080
```

- REST: `GET /health`, `POST /search` and the other routes in
  `crates/turbomemory_api/src/rest.rs`. gRPC: the service in
  `crates/turbomemory_api/proto/turbomemory.proto`.
- The store is flushed every `TURBO_FLUSH_INTERVAL_SECS` seconds (default 30)
  and on shutdown. `TURBO_SYNC_WRITES=1` syncs every write.
- `TURBO_API_KEY` enables bearer-token auth on both transports. Unset means
  open access.

---

## Workspace

| Crate | Responsibility |
| :--- | :--- |
| `turbomemory_core` | Vector math, SIMD kernels, quantizers, the token-budget packer. |
| `turbomemory_graph` | BM25, concept extraction, the concept graph and its one-pass expansion, belief semantics. |
| `turbomemory_storage` | The `StorageEngine`: persistence, WAL, tiered segments, filtering, background optimizer. |
| `turbomemory_gpu` | The GPU backend abstraction, CPU fallback, optional CUDA exact search. |
| `turbomemory_python` | The PyO3 `MemoryEngine` facade with zero-copy NumPy paths. |
| `turbomemory_api` | The gRPC (tonic) and REST (axum) server. |
| `tsm/` (Python) | The SDK: `Memory`, budget recall, the reading prompt, pluggable embedder / extractor / verifier. |

Design notes: [`docs/architecture.md`](./docs/architecture.md),
[`docs/cognitive_graph.md`](./docs/cognitive_graph.md).

---

## Build and verify

```bash
# Format, lint, Rust suite
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --exclude turbomemory_python

# Python extension and the SDK unit suite (fake embedder, real engine, no keys)
make build-python
make test-python

# All of the above plus the evaluation smoke tests: the merge gate
make gate
```

- The toolchain is pinned in `rust-toolchain.toml`. Building the API crate
  needs `protoc`.
- `make gate` needs the LongMemEval data (`make download-eval-data`) and a
  cached `all-MiniLM-L6-v2` model. It is run locally before merging.
- CI ([`.github/workflows/ci.yml`](./.github/workflows/ci.yml)) runs the
  format check, clippy, the Rust suite and the SDK unit suite on Linux. It
  runs no evaluations. Windows is covered by the local gate; the CUDA build is
  tested by hand.
- The evaluation harness and how to reproduce the numbers above:
  [`benchmarks/cognitive_eval/README.md`](./benchmarks/cognitive_eval/README.md).

---

## Known limits

The open work is listed in [`TODO.md`](./TODO.md). The ones a user should know
first:

- **Treat a store as single-tenant.** Scopes bound search, but the eviction
  cap, importance normalization and concept vocabulary are shared, and the
  server does not enforce tenancy: `scope` is a field the caller supplies.
- **No vacuum.** Deleted records leave their vectors on disk until segments
  are rebuilt.
- **No store-wide format version.** An upgrade that changes record metadata
  can break old snapshots.
- **`keep_messages` cannot be combined with `max_user_tokens` yet.**
- **Blocking engine calls run on the server's async runtime**, so a slow flush
  stalls both transports.

---

## License

Licensed under the **MIT License** — see [LICENSE](./LICENSE).

Copyright © 2026 jatin711-debug.
