#!/usr/bin/env python3
"""Extract the facts of many histories in parallel, into the cache that
shipped_stack_eval.py reads, so a later run finds them instead of extracting
one message at a time.

The runner extracts message by message as it builds each history; for
LongMemEval-S (about 480 messages a question) that is hours. This sends the
same requests through a thread pool. A message is keyed exactly as the runner
keys it (the message and the three before it in its conversation, file
order), and replies the earlier first-prompt cache already holds are copied
instead of being paid for again.

    # what a run would cost: messages not yet cached
    python benchmarks/cognitive_eval/prefetch_extraction.py --variant s --limit 30 --dry
    python benchmarks/cognitive_eval/prefetch_extraction.py --variant s --limit 115 \\
        --model gpt-4.1-nano --workers 16

Use the same --model with the runner's --extractor-model.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from cognitive_eval._secrets import ensure_openai_key, key_file_hint  # noqa: E402
from cognitive_eval.benchmark_datasets.longmemeval import load_longmemeval  # noqa: E402
from cognitive_eval.run_belief_longmemeval import _msg_content  # noqa: E402


def pending_messages(extraction, histories):
    """The (message, context) pairs no cache holds yet, deduplicated; replies
    the earlier first-prompt cache holds are copied in on the way."""
    extractor = extraction.instance
    seen, pending = set(), []
    for conv in histories:
        context = []
        for m in conv.messages:
            content = _msg_content(m)
            if not content or not content.strip():
                continue
            recent = context[-3:]
            context.append(content)
            key = extractor.cache_key(content, recent)
            if key in seen:
                continue
            seen.add(key)
            if key in extractor._cache:
                continue
            if extraction._earlier:
                paid = extraction._earlier.get(extraction._earlier_key(content, recent))
                if paid is not None:
                    extractor._cache[key] = paid
                    continue
            pending.append((content, recent))
    return seen, pending


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["oracle", "s", "m"], default="s")
    ap.add_argument("--skip", type=int, default=0, help="as in shipped_stack_eval.py")
    ap.add_argument("--limit", type=int, default=30, help="as in shipped_stack_eval.py")
    ap.add_argument("--model", default="gpt-4.1-nano", help="the runner's --extractor-model")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry", action="store_true", help="only count what is not cached")
    args = ap.parse_args()
    if not ensure_openai_key():
        sys.exit(key_file_hint())
    import shipped_stack_eval as harness
    from tsm._retry import QuotaExhausted

    convs = load_longmemeval(None)[args.skip:args.limit]
    ids = [c.queries[0].query_id for c in convs if c.queries]
    histories = load_longmemeval(None, variant=args.variant, question_ids=ids)
    extraction = harness.CachedExtraction(args.model, offline=False, mode="shipped")
    seen, pending = pending_messages(extraction, histories)
    tokens = sum(len(c) // 4 + sum(len(x) // 4 for x in r) for c, r in pending)
    print(f"{len(histories)} histories, {len(seen)} distinct messages, {len(pending)} not "
          f"cached (about {tokens / 1e6:.1f}M tokens of message and context before the "
          f"prompt)", flush=True)
    if args.dry or not pending:
        extraction.flush()
        return

    started, done, failed = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(extraction.instance.extract_facts, c, r) for c, r in pending]
        try:
            for future in as_completed(futures):
                try:
                    future.result()
                except QuotaExhausted:
                    raise
                except Exception as e:  # noqa: BLE001 — left uncached; a rerun asks again
                    failed += 1
                    print(f"failed: {type(e).__name__}", flush=True)
                done += 1
                if done % 500 == 0:
                    extraction.flush()
                    print(f"{done}/{len(pending)} ({done / (time.time() - started):.1f}/s, "
                          f"{extraction.instance.calls} requests)", flush=True)
        except QuotaExhausted as e:
            for future in futures:
                future.cancel()
            extraction.flush()
            sys.exit(f"stopped: {e}; {done} messages done and saved")
    extraction.flush()
    print(f"done: {done} messages ({failed} failed), {extraction.instance.calls} requests, "
          f"{time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
