# Evaluation harness

How the numbers in the top-level README and in
[`../PHASE_PROGRESS.md`](../PHASE_PROGRESS.md) are produced. Results are not
kept here: every run has a dated entry in `PHASE_PROGRESS.md` with its
command, sample size and caveats.

Run everything from the repository root with Python 3.12, after
`make build-python` has put the extension at the root (`make verify` and
`make gate` do that too).

## What measures what

| Script | Measures | Cost |
| :--- | :--- | :--- |
| `shipped_stack_eval.py` | `tsm.Memory` as it ships, on LongMemEval, one mechanism per arm | free evidence score; judged runs are paid |
| `locomo_shipped_eval.py` | `tsm.Memory` on LoCoMo (ten two-person conversations) | free evidence score; `--judge-sample` is paid |
| `reread.py` | saved contexts read several times, arms compared on the mean | paid |
| `pair_runs.py` | two arms question by question, from one run file or two | free |
| `prefetch_extraction.py` | extracts many histories in parallel into the runner's cache | paid; `--dry` prints the request count |
| `evidence_rank_report.py` | why a question's evidence did not reach the context | free |
| `belief_pairs_eval.py` | belief revision on 260 labeled sentence pairs (`belief_pairs.jsonl`) | free with `--verifier none` / `nli` |
| `head_to_head_eval.py`, `full_harness_audit.py` | the evaluation adapter (`adapters/tsm_adapter.py`) against naive RAG and Mem0 | paid |
| `bounded_head_to_head.py` | stores held to a token budget: compress, delete, Mem0 | paid |
| `run_belief_longmemeval.py`, `belief_revision.py` | belief revision in isolation, retrieval level; the gate runs both | free |
| `retention_eval.py`, `reinforcement_eval.py`, `run_abstraction_longmemeval.py`, `budget_recall_eval.py`, `compress_eval.py`, `verify_supersessions.py` | one engine mechanism each | see each docstring |

Two things to keep apart:

- **The adapter is not the SDK.** `TSMAdapter` has its own settings and
  additions. The head-to-head runners measure it; `shipped_stack_eval.py`
  measures `tsm.Memory`. On 120 conversations the adapter did not beat plain
  vector search (0.504 against 0.539, 2026-10-07).
- **Retrieval-level scores are not answers.** A rank win that a reader never
  sees does not change an answer, and the free evidence score says nothing
  about what a reader does with the context. Judge a change before claiming
  it, and compare arms with `reread.py --reads 5`: the reader alone flips 5 to
  6 verdicts of 121 on identical contexts.

## Data

```bash
make download-eval-data     # LongMemEval (oracle variant) and its official file with evidence labels
python benchmarks/cognitive_eval/benchmark_datasets/download.py --official s   # LongMemEval-S, 277 MB
```

- The default LongMemEval copy is the benchmark's oracle variant: each
  history holds only the sessions with evidence. `--variant s` runs the
  standard histories (about 120,000 tokens a question).
- LoCoMo is fetched as the docstring of `locomo_shipped_eval.py` says.
- Datasets are too large to check in. They live under `data/`, which is
  ignored.

## The shipped-stack runner

```bash
# plumbing check, no network: cached texts only, proxy judge
python benchmarks/cognitive_eval/shipped_stack_eval.py --offline --limit 5

# the free evidence score: for how many questions every evidence turn is in the context
python benchmarks/cognitive_eval/shipped_stack_eval.py --offline --limit 120 \
    --arms sdk_pack_tagged,sdk_cognitive --config gw25:graph_weight=0.25
python benchmarks/cognitive_eval/shipped_stack_eval.py --offline --limit 120 --pool 20 \
    --arms sdk_pack_tagged,sdk_cognitive

# the same score on a local embedder, no key
python benchmarks/cognitive_eval/shipped_stack_eval.py --offline --limit 120 \
    --arms sdk_pack_tagged,sdk_cognitive --local-embedder google/embeddinggemma-2

# judged
python benchmarks/cognitive_eval/shipped_stack_eval.py --limit 120 --arms sdk_cognitive \
    --token-budget 600 --reader notes --out "$TSM_EVAL_DIR/runs/notes600.json"

# the headline run: every LongMemEval-S question, messages only, 2,000 tokens
python benchmarks/cognitive_eval/shipped_stack_eval.py --variant s --limit 500 \
    --arms sdk_raw,sdk_raw_plain --token-budget 2000 --reader notes \
    --out "$TSM_EVAL_DIR/runs/s500_raw_2000.json"
```

- The arms are listed in the script's docstring. `--config NAME:KEY=VALUE`
  adds an arm with any engine setting, `--recall NAME:KEY=VALUE` one with any
  `recall()` argument (on the facts' store, or the one `--recall-store`
  names), and `--pool N` builds each history from N conversations so most
  of the store is distractors.
- `--reader notes` asks the answering model with the SDK's prompt
  (`tsm.reading`); the default is the harness's plain prompt.
- `--judge anthropic` reads and judges with Claude (key in `claude_key.txt`);
  embeddings still need OpenAI unless `--local-embedder` is given.
- An interrupted run started again with the same `--out` continues after the
  last finished history. `--reuse a.json,b.json` takes verdicts for identical
  contexts from earlier runs, and `--rejudge run.json` reads saved contexts
  again without rebuilding the stores.
- Paid results (verdicts, gists, the embedding overlay, run files) go to
  `$TSM_EVAL_DIR`, by default `~/.tsm_eval/shipped_stack_eval`, outside the
  checkout. Deleting that folder deletes what was paid for. Set the variable
  before using it in `--out` as above.

## Belief pairs

```bash
python benchmarks/cognitive_eval/belief_pairs_eval.py --verifier none --split dev
```

260 pairs of first-person statements written for this repository (110 updates,
120 pairs where both stay true, 30 rewordings), split in half into `dev` and
`test`. They are not sampled from real conversations. Tune on `dev`, quote
`test`. `--verifier llm` needs an OpenAI-compatible chat endpoint
(`--llm-base-url` for a local one).

## Tests of the harness itself

```bash
cd benchmarks && python -m unittest cognitive_eval.test_budgeting \
    cognitive_eval.test_diagnostics cognitive_eval.test_longmemeval_loader \
    cognitive_eval.test_role_aware
```

No dataset, model or key needed. Nothing else runs them: neither the gate nor
CI.

## Removed

- The recall@K runners (`run_longmemeval.py`, `run_locomo.py`,
  `run_comparison.py`, `report.py`, `metrics/`) and their Makefile targets,
  on 2026-11-09. They scored word-overlap retrieval from before the judged
  harness existed. `git show 55933a6:benchmarks/cognitive_eval/<path>`.
- The 48-question Mem0 audit tables that used to be on this page are in
  `PHASE_PROGRESS.md` (2026-08-14), with what they measured.
