# TurboSuperMemory — Roadmap v2: From Validated Mechanisms to a Product Wedge

> **Audience:** a fresh Claude session (or the maintainer) picking this up cold. Read fully
> before touching code. **Written:** 2026-07-11, at commit `c08f6aa`, branch `evaluation`.
> **Supersedes** the W1–W7 execution plan (completed; full record in
> `benchmarks/PHASE_PROGRESS.md`, prior plan in git history of this file).

---

## 1. Where the project stands (do not re-derive)

The W1–W7 effort is **complete**. Every cognitive mechanism has an honest, isolation-validated
real-data verdict on LongMemEval, the wins are first-class engine features (not eval hacks),
and `make gate` (7 checks) protects all of it. Commits `865224d..c08f6aa`.

| mechanism | retrieval-level verdict | gold-standard (LLM-judged) verdict |
|---|---|---|
| Belief revision (W1/W3) | **+0.11 hit@1** on knowledge-update, zero collateral, NLI-verified, role-scoped | **+0.00 at top_k=10** — rank win does NOT convert to answer accuracy at generous k |
| Retention/forgetting (W5) | **+0.41 gold survival** under budget eviction (3.3× vs FIFO) | **untested** under the judge (likeliest headline — an evicted fact is unanswerable at any k) |
| Reinforcement (P4/W5) | no ranking lift (honest negative); drives retention | n/a (folded into retention) |
| Abstraction (W4) | marginal/mixed (+0.06 temporal, −0.04 elsewhere, ~net neutral) | untested; low priority |

**The pivotal W6 finding:** belief revision genuinely re-ranks (the current fact surfaces at
rank 1, stale facts demoted) — but when the judge LLM sees the whole top-10, rank-within-context
is invisible, and answer accuracy is identical. The proxy and the gold standard measure
different things. **Full analysis: PHASE_PROGRESS.md "W6".**

**Also true:** the vector store underneath is solid but commodity (100% recall@100k, GPU,
tiers). It is not the differentiator and never will be — Qdrant/Milvus/LanceDB own that ground.

**Infrastructure in place (reuse, don't rebuild):** LLM-judged LongMemEval runner
(`run_belief_longmemeval.py --judge openai|ollama|auto --extractor openai|ollama|auto`,
concurrent, ~$0.25/120-conv run), NLI demotion verifier, retention eval, consolidation
profiler, regression gate, OpenAI key via gitignored `openai_key.txt`
(`cognitive_eval/_secrets.py`). Known defects: native-memory leak across engine
create/close cycles (flagged task); graph JSON snapshot ~24 KB/record; remaining O(N)
consolidation passes (importance, abstraction/vocab, snapshot) — see PHASE_PROGRESS "W7".

## 2. The strategic read (why this roadmap points where it points)

An evicted fact can't be answered from at any k; a badly-ranked fact still answers fine at
k=10. So the commercial value of cognitive memory, if it exists, lives in **constrained
regimes**: tight token budgets (k=1–3), long-lived stores under memory pressure, and
cost-per-query — which is exactly the regime real agent deployments run in. Nobody ships
top-10-full-context memory at scale. The roadmap's job: **prove the wedge in that regime for
~$2 (Phase A), then package it (B), productionize the right scale axis (C), and publish the
evidence (D).** Phase A is a genuine kill-gate: if nothing survives it, stop before B.

---

## Phase A — Prove (or kill) the wedge — ~1–2 weeks, ~$2 OpenAI

Four cheap, decisive experiments. All run on existing harness (A2 needs a small judge-wiring
addition to `retention_eval.py`; A4 needs a Mem0/naive-RAG adapter).

| # | Experiment | Command / build | Kill/keep signal |
|---|---|---|---|
| **A1** | Judged belief ON/OFF at **top_k ∈ {1, 3, 5}** | `run_belief_longmemeval.py --limit 120 --role-filtered --extractor openai --judge openai --top-k 3 --workers 12` (×3 k-values; reuse extraction cache if possible) | Judged KU lift > +0.05 at k≤3 → belief revision is product value; +0.00 again → demote to "hygiene feature", A2 becomes the story |
| **A2** | **Gold-standard judged retention** — rerun W5 with the LLM judge scoring answers post-eviction | Add `--judge` to `retention_eval.py` (same pattern as the belief runner; ~30 lines) | The +0.41 survival should convert almost by construction. If judged answer-accuracy lift ≥ +0.15 → **headline claim**. If it fails, hard rethink |
| **A3** | **Accuracy-per-token curve** — ON vs OFF judged accuracy at k=1/3/5/10 vs context tokens consumed | Derived from A1 runs + token counting (no new arms) | "Same accuracy at ⅓ the context cost" is a cost/latency pitch — often stronger than an accuracy pitch |
| **A4** | **Head-to-head vs Mem0** (and/or naive RAG over full history), same judge, same 120 convs | Mem0 adapter conforming to the `add/search` adapter interface in `cognitive_eval/adapters/` | The only number outsiders care about. Losing is also information (diagnose: usually extraction quality) |

**Decision gate (end of Phase A).** Pick the headline from whichever survived:
(i) *better answers under tight budgets* (A1), (ii) *retains the right memories under
pressure* (A2), (iii) *same accuracy at a fraction of the token cost* (A3) — each must also
hold or win vs Mem0 (A4) to be a market claim. **If none survive: stop.** The honest outcome
is portfolio-piece + eval-harness-as-asset; do not proceed to B on momentum.

**Methodology rules (unchanged from v1, they made this project work):** ON/OFF isolation on
identical corpora; real data decides; honest negatives are deliverables; small-n numbers
(KU n≈18 at limit 120) never go public without a full-set (~500 conv) confirmation run.

## Phase B — Package the wedge as the actual product — ~4–6 weeks

The engine is validated; the **product surface doesn't exist** — everything adoptable
currently lives in `benchmarks/` as eval scaffolding. Build order:

1. **Memory SDK** (`tsm` Python package): Mem0-shaped API — `add(messages, user_id)` /
   `recall(query, user_id, token_budget)`. Promote the adapter pipeline (LLM extraction →
   role tagging → NLI-verified belief revision → consolidation) from eval code to shipped
   code with tests.
2. **Budget-aware retrieval**: `recall(..., token_budget=N)` returns the best *set under a
   budget* — where demotion/supersession actually monetizes (choosing what to include when
   you can't include everything). Implements whatever A1/A3 proved.
3. **One-flag preset**: `profile="conversational"` = role-filtered + verified demotion +
   access-aware eviction + budget defaults (replaces today's ~12 kwargs).
4. **Fix the native-memory leak** (engine create/close cycles; already flagged) — a killer
   for the long-lived embedded use case this product is for.
5. **Extraction as a first-class interface** (ollama/openai/custom), moved out of benchmarks.

## Phase C — Production trust & the RIGHT scale axis — ~4–8 weeks

**Reframe scale:** the target is **not** 1M×4k in one shard (old TODO.md axis). It is
**many small stores** — 10k–100k users × ~1k memories each, long-lived. What that actually
requires:

- Remaining O(N) consolidation passes → incremental (importance recompute, abstraction/vocab
  evolution; seq-cursor pattern already proven 4.4× on detection; profiler exists).
- **Graph snapshot format**: JSON → binary/delta (the ~24 KB/record wall).
- Multi-tenant serving: cheap per-scope engine instances, engine pooling/eviction, WAL hygiene.
- Ops: tracing/metrics, auth, Docker, backup/restore; CI runs `make gate` on every PR.
- **Kill list:** mark ~80% of old TODO.md Phases 0–5 (sharding, NUMA, PQ variants, ACORN,
  32 MiB WAL segments…) **won't-do — wrong axis**. Revisit only what multi-tenant numbers
  demand (likely: vacuum, metadata paging).

## Phase D — Evidence & launch — ~2–4 weeks

- Publish the benchmark story **including the k=10 null result** — the honesty is a
  credibility asset competitors can't copy.
- README rebuilt around only-proven claims; demo agent that visibly forgets junk and
  survives corrections; LoCoMo + full LongMemEval (~500 convs) for generalization of any
  number going public.

---

## Risk register

| Risk | Response |
|---|---|
| A1 **and** A2 both fail under the judge | Stop at the gate; thesis falsified at the answer level for ~$2 — the cheapest possible way to learn it |
| Mem0 wins A4 | Diagnose why (usually extraction quality). "Their extractor, our engine" is a legitimate hybrid; losing on the *engine* axis is the real warning |
| Small-n noise (KU n=18 at limit 120) | Full ~500-conv confirmation before any public number; session teardown kills background runs — relaunch, don't trust partial logs |
| Leak resurfaces at multi-tenant scale | Phase B item 4 exists for exactly this reason; profiler + flagged task have the repro |

## Appendix — Environment & build (unchanged, follow exactly)

```powershell
$py = "C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe"   # MUST be 3.12
$env:PYO3_PYTHON = $py
$env:CARGO_PROFILE_DEV_DEBUG = "0"; $env:CARGO_PROFILE_TEST_DEBUG = "0"    # LNK1102 workaround
cargo build --release -p turbomemory_python
Copy-Item target\release\turbomemory.dll turbomemory.pyd -Force
$env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"                 # embeddings cached
```

- ONE python process may hold `turbomemory.pyd` — run evals sequentially.
- OpenAI key: gitignored `openai_key.txt` at repo root (never in chat/commands/logs).
- Ollama local: `qwen2.5:3b` pulled; `--extractor auto` / `--judge auto` prefer it (free) else OpenAI.
- `make gate` before every commit touching engine/cognitive code. Living record:
  `benchmarks/PHASE_PROGRESS.md` — append a section per phase, one commit per milestone.


---

# Roadmap v2.1 — Research-Grounded Improvement Plan (2026-07-11)

Survey of arXiv as of July 2026 (via the arXiv API). The current literature both
**validates our direction and hands us the next moves**:

| paper (arXiv, date) | what it says | what it means for us |
|---|---|---|
| **A-TMA: Decoupling State-Aware Memory Failures** (2607.01935, Jul 2) | Names "**ghost memory**": old/current/transition facts coexisting in retrieval mislead the answer model; systems must know *what is true now* | **Explains our A1 null.** Demotion changes rank; ghost facts still enter the context, and rank is invisible there. The fix is **state separation at answer time**, not ranking |
| **EMBER: Budgeted Evidence Retention** (2606.05894, Jun 4) | Retention under an explicit budget for long-horizon agents | Our A2 wedge (+0.40 judged) is exactly this axis — independently validated as a live research frontier |
| **What to Keep, What to Forget: Rate–Distortion View of Memory Compaction** (2607.08032, Jul 9) | Unifies eviction/pruning/summarization as rate–distortion: budget = rate, minimize task distortion | The theoretical frame for our retention product; motivates **compress-instead-of-delete** |
| **PACMS: Submodular Context Selection** (2606.20047, Jun 18) | Context selection under budget as submodular maximization (pluggable engine) | The right **algorithm for `recall(query, token_budget)`**: greedy facility-location/MMR, (1−1/e) guarantee |
| **Mandol: Agglomerative Agent Memory** (2606.29778, Jun 29) | Agglomerative (cluster-and-merge) memory; flags RAG "lack of token budget control" | Supports summary-hierarchy consolidation + budget-controlled recall |
| **MemDelta: Controlled Baselines and Hidden Confounds in Agent Memory Evaluation** (2606.29914, Jun 29) | Vary ONE component at a time on LongMemEval-S; reported memory gains often confound model/embedder changes | **Our ON/OFF isolation methodology, published.** Use its protocol for A4 head-to-head |
| **When Not to Write Memory** (2607.02579, Jun 30) | Refusing to write is a first-class decision; correlated traces cause false promotion | Generalizes our role-filter into a **write-gating policy** |
| **PLACEMEM** (2607.04089, Jul 5) / **MOSS** (2607.04391, Jul 5) | Memory as a compute-aware control plane; auditable memory | System-design direction: background consolidation jobs decoupled from serve path; audit log |
| Memory-poisoning attack papers (several, Jun–Jul 2026) | Agent memory is an attack surface | Our provenance (source_role) + NLI verify + write-gating double as **poisoning defenses** — a Phase C security story |

Plus cognitive science we should finally use properly: **ACT-R base-level activation**
(Anderson & Schooler rational analysis): recall probability follows a **power law over
spaced accesses**, `A_i = ln(sum_j t_j^(-d))` — better-grounded than our single
`access_count x 2^(-age/half_life)` heuristic.

## Track 1 — Algorithms (each item: motivation → change → kill/keep test on our harness)

**B1. Exclusion-not-demotion (state-aware answer context) — HIGHEST LEVERAGE, ~1 wk, ~$0.4**
- *Why:* A-TMA ghost-memory diagnosis fits our A1 data exactly (judged lift <= 0 at every k;
  ss-user negative — stale facts still reached the context and misled the model).
- *Change:* we already have the NLI-verified supersession graph. At recall, superseded
  memories are **excluded** from the returned set (or tagged `[OUTDATED as of ...]`), not
  just rank-demoted. Engine: filter/annotate by demotion state at hydrate/adapter level.
- *Test:* rerun the A1 multi-k judged eval with a third arm "belief-exclude". Keep if
  knowledge-update judged lift > +0.05 at any k with other types >= -0.05. This can
  **revive belief revision at the answer level** using infrastructure already built.

**B2. Budget-aware recall as submodular selection (PACMS) — ~1 wk, ~$0.4**
- *Change:* `recall(query, token_budget)`: greedy maximization of relevance + coverage -
  redundancy (facility location / MMR) under a token knapsack, over the fused candidate set.
- *Test:* judged accuracy at fixed budgets (50/100/200 tokens) vs naive top-k truncation
  (the A3 harness already measures exactly this). Keep if >= +0.05 judged at any budget.

**B3. ACT-R power-law activation for retention — ~1–2 wk, ~$0.3**
- *Change:* keep the last K=8 access timestamps per record (ring buffer);
  `activation = ln(sum t_j^(-d))`, d~0.5; use for eviction ranking (and optionally the
  importance blend). Spaced/repeated access beats one recent burst — matches human
  forgetting data and the EMBER budgeted-retention framing.
- *Test:* judged retention 3-arm: FIFO vs current-exponential vs ACT-R. Keep if ACT-R >=
  current (any gain is a bonus; parity still simplifies the theory/story).

**B4. Compress-instead-of-delete (rate–distortion consolidation; Mandol) — ~2 wk, ~$0.5**
- *Change:* eviction victims are clustered (existing dedup/concept machinery) and reduced to
  a **gist summary memory** (cheap extractive first; LLM summarizer optional) with lineage
  links; details deleted, gist retained within budget.
- *Test:* judged retention arms: delete vs compress. Keep if compress > delete on judged
  accuracy at the same budget. This is the rate–distortion product claim: "forget detail,
  keep gist."

**B5. Write-gating (When-Not-To-Write) — ~3 days, ~$0.2**
- *Change:* novelty gate at `add()`: near-duplicate (cosine + text) of an existing memory →
  reinforce the existing record instead of writing a new one. Reduces budget pressure and
  poisoning surface. (Role-filter already ships; this generalizes it.)
- *Test:* retention + belief evals unchanged-or-better with fewer stored records.

## Track 2 — Data structures

1. **Access-history ring buffer** per record (K=8 timestamps, ~64B) — powers B3. WAL-persisted.
2. **Lazy eviction heap** — O(log n) victim selection instead of the O(N) scan; uniform decay
   preserves relative order between access events, so re-heapify only on access. Matters at
   many-small-stores scale; the profiler validates.
3. **Binary/delta graph snapshot** — replace the ~24 KB/record JSON (bincode + periodic
   delta). Target <2 KB/record, O(delta) save. Kills the second W7 scale wall.
4. **Summary-hierarchy nodes** — gist ← detail lineage (for B4), enabling progressive
   disclosure at recall (expand gist to surviving details if budget allows).
5. **Per-scope store layout + engine pool (LRU of open scopes)** — the many-small-notebooks
   multi-tenant shape; pairs with the leak fix below.

## Track 3 — System design

1. **Fix the engine-lifecycle native leak** (flagged; blocker for engine pooling).
2. **Finish incremental consolidation** — importance recompute + abstraction/vocab on
   new-records-only (seq-cursor pattern proven 4.4x; profiler drives).
3. **Control-plane consolidation (PLACEMEM)** — detect/verify/compress as background jobs on
   an idle-time scheduler, decoupled from the serve path (we already split propose/commit —
   extend to a job queue). "Sleep-time" consolidation.
4. **Auditable memory (MOSS) + poisoning defenses** — expose a mutation audit log (the WAL is
   already the substrate); provenance + NLI-verify + write-gating documented as the
   security story for Phase C.
5. **MemDelta-conformant A4** — head-to-head vs Mem0/naive-RAG holding embedder + answer
   model fixed, varying only the memory system; LongMemEval-S protocol.

## Execution order & gates

| step | item | gate |
|---|---|---|
| 1 | B1 exclusion-not-demotion | KU judged lift > +0.05 at any k, no type < -0.05 |
| 2 | B2 submodular budget recall | >= +0.05 judged at any fixed budget vs truncation |
| 3 | B3 ACT-R activation (+ ring buffer, heap) | >= parity with current on judged retention |
| 4 | B4 compress-instead-of-delete (+ hierarchy) | > delete-arm on judged accuracy at same budget |
| 5 | B5 write-gating | no regression, fewer records |
| 6 | A4 MemDelta-style head-to-head | competitive with Mem0 under the same budget |
| (parallel) | leak fix, binary snapshot, incremental heads | profiler + gate green |

Each step lands as one commit with a PHASE_PROGRESS section; `make gate` before every
commit; judged runs use the disk-cached extractor (~free) + per-model quota flags. After
step 6, fold the winners into the SDK (Roadmap v2 Phase B) — by then `recall(query,
token_budget)` is not a wrapper but the algorithmically-differentiated core.
