# What TurboSuperMemory taught

TSM was built to learn how a long-term memory layer for AI agents works, from
the bytes on disk to the prompt a model answers from. This page is what the
experiments showed, layer by layer, and how to measure such a thing without
fooling yourself.

The evidence is in git. The lab notebook has every run with its command,
sample size and caveats, including the ideas that failed:
`git show e79d75f:benchmarks/PHASE_PROGRESS.md`. The harness that produced
the numbers is at `git show e79d75f:benchmarks/cognitive_eval/<path>`
(removed 2026-11-11). Paid results (verdicts, run files) are in
`~/.tsm_eval/shipped_stack_eval`, outside the repository.

How to read the numbers:

- They come from LongMemEval unless a line says otherwise. **Judged** means a
  model answered from the recalled context and another model graded the
  answer. The **evidence score** asks whether every turn the dataset marks as
  evidence made it into the context; it is free, and no model is asked.
- "12 / 1" means 12 questions gained and 1 lost against the comparison.

## The pipeline

1. A message comes in.
2. It is kept as said, cut into passages of about 100 tokens. Optionally a
   model also extracts facts from it.
3. Each passage or fact is embedded and stored with its date, its speaker and
   its scope (the user it belongs to).
4. A question gathers a pool of candidates: its nearest vectors, optionally
   widened through a concept graph.
5. The pool is ranked:
   - a speaker prior;
   - demotion of facts a newer one replaced;
   - a window of days, for a question like "ten days ago".
6. The ranked pool is packed greedily into a token budget, each memory with
   its date in front.
7. A reading prompt makes the model list what bears on the question, settle
   what changed, count or compute, and give one answer.

Most of the accuracy is decided in steps 2, 6 and 7, not in the clever part
of step 4.

## Layer by layer

### Storage: the part that has to be boringly right

- **Durability is an ordering problem.** A write goes in three steps:
  1. the vector bytes;
  2. a log record carrying the vector's checksum;
  3. a lazy snapshot.

  Recovery replays the log and checks each vector against its checksum. It
  never trusts the header count, which only `flush` writes.
- **Write crash tests that drop the engine without flushing.** Flushing
  stamps the header and hides exactly the bugs recovery has.
- **Validate everything before changing anything.** A rejected batch leaves
  the store as it was.
- **Exact search to about 4,096 records.** Up to that size an exact scan is
  fast and always right. Above it, the engine searches segments (HNSW for
  recent data, quantized scans for older data) and reranks the candidates on
  the full float32 vectors. A test of the tiered path needs more records than
  that line, or it silently tests the scan.
- **A filter has to bound every stage.** Scope and payload filters must
  restrict the graph expansion too, or other users' records take result
  slots.
- **State that changes later is a payload patch, not an update.** Statuses
  and counters are patched in place. An update moves the record to a new
  offset, and belief revision then reads it as "newer".
- **On a GPU, exact search over vectors kept on the device is what pays.** At
  100,000 × 384-d on an RTX 3050 it took about 1.4 ms a query against 6 to
  7 ms on the CPU, and 0.35 ms a query in batches. A CUDA HNSW build was
  removed: the graph it built was never searched.

### Writing: what to store

- **An extraction model drops a third of what users say.** With
  gpt-4.1-nano, 470 of 1,403 user messages gave no fact, and so did 39 of 229
  evidence turns. A store cannot return what it never held, so extraction is
  a ceiling.
- **Keep the messages as said.** On 29 LongMemEval-S questions at 2,000
  tokens:
  - facts alone answered 19;
  - facts plus messages answered 27 (8 / 0, p = 0.01);
  - messages alone answered 26.

  Messages alone ask no model at write time. Storing messages is the default.
- **A second extraction prompt is not worth it.** Asking it for the messages
  that yielded nothing raised the evidence score (76 to 92 questions) but not
  the answers (90 to 82 at 150 tokens). It stays off.

### Finding: vectors and the concept graph

- **The graph never beat plain vector search on OpenAI embeddings.** Judged:
  357 against 360 of 414 (p = 0.70), and 10 / 10 on the held-out set. The
  evidence score was level too.
- **Why it once lost.** Its boost was word overlap counted a second time,
  without IDF, at about eighteen times the cosine gap between neighbouring
  candidates. Scaling the boost to the spread of cosines (`graph_weight`)
  stopped the loss but did not create a win.
- **On weaker local embedders the graph finds more evidence:** MiniLM 12 / 1,
  EmbeddingGemma 4 / 1. Whether that becomes answers was never judged.
  Switching to a better embedder did more than the graph or a ColBERT
  reranker (removed) ever did.
- **Cosine thresholds belong to an embedder, not to the system.** Belief
  thresholds tuned on OpenAI vectors marked 30 of 58 still-true pairs stale
  on EmbeddingGemma's vectors.
- **A database must remember which embedder wrote it.** Two models with the
  same dimension return wrong results without an error.

### Packing: what goes into the budget

- **Fill a token budget, not an item count.** A fixed cap left half of a
  150-token budget unused and cost judged answers.
- **Skip only near-identical text.** At a redundancy cutoff of 0.72, 7 of
  115 questions lost their evidence. Memories that read alike are often the
  separate instances a counting question needs, so the cutoff is now 0.95.
- **More context is not better.** Going from 2,000 to 8,000 tokens raised the
  evidence score from 91 to 96 of 96 and dropped judged answers from 91 to 82
  (p = 0.012), all of them counts and dates.
- **Put a date in front of each memory:** 8 / 1. The time of day as well was
  worse (84 against 90 of 115).
- **Resolve a time in the question.** "Ten days ago" becomes a window of days
  whose memories are packed first. On 127 temporal questions it got all the
  evidence for 114 against 111. The 23 contexts it changed, read five times
  each, answered 15.8 against 13.4 (p = 0.38).
- **Weight by speaker.** Raise the user's own words for "I / my" questions.
  Leave them alone for questions about what the assistant said, and return
  the assistant's reply whole for those: 54 against 41 of 56.
- **The end result.** All 470 answerable LongMemEval-S questions, histories of
  about 120,000 tokens, 2,000 tokens recalled, gpt-4.1-mini reading: 411
  right (87.4%), with all the evidence in context for 439.

### Reading: the model is half the system

- **The reading prompt was the biggest single gain.** It has the model list
  what is relevant, settle changes, then answer. On 355 held-out questions it
  answered 296 at 600 tokens, where the plain prompt at 150 tokens answered
  259.
- **Most remaining errors are the reader's.** Of the 59 wrong answers on
  LongMemEval-S, 41 had all their evidence in the context. Every
  representation tried for multi-session questions (an event table, rewritten
  observations, raw passages) lands at the same accuracy for a given reader.
- **Things that did not help, all removed:**
  - reader tools for arithmetic and dates: 94 against 98;
  - a read-time monitor that took a second look when the context seemed
    insufficient: it found the lost evidence 0 times in 6 free and 0 in 4
    judged, and its signals fired as often on complete contexts as on
    incomplete ones.
- **Controls:**
  - Reading the whole 124,000-token history scored 22 of 29, against 27 for
    memory at 2,000 tokens.
  - Reading only the evidence sessions scored 104 of 115, against 98 for
    memory: the memory layer costs about six questions there.

### Belief revision: what changed

- **The first detector over-fired on real text.** It built 28,311 belief
  edges over 147 conversations and was net negative on every question type.
  Its cosine thresholds came from synthetic vectors.
- **Fixing detection.** Four changes cut the edges by 75%:
  - mutual-nearest-neighbour pairs;
  - one edge per new memory;
  - no pairs across scopes;
  - a floor on demotion.

  On knowledge-update questions, the gain in hit@1 went from +0.00 to +0.23
  (0.27 to 0.50).
- **Rank wins did not become answers.** Judged at 1, 3, 5 and 10 memories of
  context there was no lift: the current fact is usually in the context
  anyway, and a demotion sometimes buried a fact the answer needed.
- **What shipped:**
  - Pairs are proposed, an LLM verifier vets them, and only then are they
    committed.
  - A replaced fact is demoted and flagged, never dropped. Dropping lost
    answers about the past.
  - Knowledge-update questions: 10 of 18 with it, 9 without.

### Forgetting under a budget

- **Keeping what is used beats first-in-first-out.** Under a 10-record cap,
  judged accuracy was 0.438 against 0.045. The first measurement had
  rehearsed with the evaluation questions themselves, a leak. Without any
  re-access signal the survival gain fell from +0.41 to +0.08. With the user's own past messages
  as the honest signal it came back.
- **Fold old memories into short gists instead of deleting them.** At 256
  stored tokens: 0.482 against 0.098 (46 / 3). Against Mem0: level at 64 and
  128 tokens, ahead at 256, where Mem0's own store was not constrained.
- **ACT-R-style activation tied plain access counts** on this workload. It
  has no spaced repetition for the spacing effect to show up in.

### The executive: built, never measured

Opt-in prototypes with unit tests, each in its own engine so they cannot
change recall:

- intentions (`due`);
- lessons from outcomes (`feedback` / `lessons`);
- `history` / `retract` / `forget`.

`forget()` redacts gists sentence by sentence and purges caches. The plan to
measure them lived in a research report kept outside the repository.

## How to measure without fooling yourself

1. **Free proxy first, judge second, and never read the proxy as answers.**
   The 8,000-token budget and the second extraction prompt both raised the
   evidence score and lowered judged answers.
2. **Look for leaks:**
   - Rehearsing with the test questions inflated retention.
   - A ranking rule keyed to LongMemEval's question wording was found only by
     a second benchmark. On LoCoMo the general replacement took the
     messages-only evidence score from 1,177 to 1,560 of 1,977, with
     LongMemEval unchanged.
3. **Know which data you test on.** LongMemEval's oracle copy holds only the
   evidence sessions, about 6,700 tokens against 122,000 in the standard
   copy. Everything looks good on it.
4. **The reader is noisy.** Five or six of 121 verdicts flip between passes on
   identical contexts. Compare arms on the same questions, read them five
   times, and report gained / lost with a p-value.
5. **Run controls:** no memory (read everything) and an oracle (read only the
   evidence).
6. **Hold out.** Settings were chosen on 115 questions and checked on the
   other 355.
7. **Keep paid results outside the checkout.** They were wiped three times:
   - under `target/`;
   - under `.cache/`;
   - by a forced `git worktree remove` on Windows, which deleted through the
     directory junctions pointing at the caches (2026-11-11).
8. **Delete what measures level,** and leave a `git show` pointer behind.

## Removed, and where it is

| What | Removed | Code |
| :--- | :--- | :--- |
| Evaluation harness, datasets loader, judges | 2026-11-11 | `git show e79d75f:benchmarks/cognitive_eval/<path>` |
| Lab notebook, open-work list, original plan | 2026-11-11 | `git show e79d75f:benchmarks/PHASE_PROGRESS.md`, `e79d75f:TODO.md`, `e79d75f:PLAN.md` |
| One-off scripts (leak repro, quantizer audits, batch test, GPU benchmark) | 2026-11-11 | `git show e79d75f:benchmarks/<name>` |
| Read-time monitor (`checked_recall`) | 2026-11-11 | `git show 940ef42:tsm/monitor.py` |
| ColBERT reranker | 2026-11-10 | `git show cbc2f6f:tsm/rerankers.py` |
| Event table, observations, `sources="turn"`, event dates, reader tools | 2026-11-09 | `git show 26092df:tsm/<events, observe, sources, answering>.py` |

## Questions left open

- Does the graph's extra evidence on local embedders turn into answers? That
  needs a judged run.
- The executive's parts were never measured against their baselines:
  - intentions against a plain to-do list;
  - lessons against retrieving raw past episodes.
- Can a different store, budget or reader recover the third of user messages
  that extraction drops? Storing the messages as said is the current answer.
- `compaction="merge_oldest"` scored 57 against 50 for the default (p =
  0.23). A held-out run would decide it.
- Engine limits that remain:
  - no vacuum;
  - no store-wide format version;
  - `sync_writes` never tested against a real power cut;
  - a store is single-tenant in practice;
  - `keep_messages` cannot be combined with `max_user_tokens`.
