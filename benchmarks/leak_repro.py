"""Repro for the flagged native-memory leak across engine create/close cycles.

Creates and closes many MemoryEngine instances in one process (the eval
harness pattern: one engine per conversation) and reports process RSS and
thread count per cycle. A leak shows up as monotonically growing RSS (or a
growing thread count) after Python-level GC.

Usage:
    python benchmarks/leak_repro.py [--cycles 40] [--records 200] [--keep-db]
"""

import argparse
import gc
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import turbomemory  # noqa: E402

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def process_stats():
    if not HAS_PSUTIL:
        return None, None
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / (1024.0 * 1024.0), p.num_threads()


def run(cycles: int, records: int, dim: int, keep_db: bool, cognitive: bool,
        consolidate: bool, searches: int) -> None:
    root = tempfile.mkdtemp(prefix="tsm_leak_repro_")
    rng = np.random.default_rng(0)
    print(f"db root: {root} (cycles={cycles}, records={records}, dim={dim}, "
          f"cognitive={cognitive}, consolidate={consolidate}, searches={searches})")
    if not HAS_PSUTIL:
        print("psutil not installed; RSS/thread metrics unavailable")

    baseline_rss, baseline_threads = process_stats()
    print(f"baseline: rss={baseline_rss:.1f} MB threads={baseline_threads}")

    kwargs = dict(
        dimension=dim,
        outlier_count=0,
        auto_consolidation_secs=0,  # background worker off: isolate lifecycle
    )
    if cognitive:
        # Mirrors cognitive_eval/adapters/tsm_adapter.py defaults.
        kwargs.update(
            importance_auto_scoring=True,
            refinement_cosine_threshold=0.85,
            contradiction_cosine_threshold=0.75,
            cognitive_alpha=0.5,
            max_concepts=10,
            concept_max_ngram_len=2,
        )

    try:
        for cycle in range(cycles):
            db_path = os.path.join(root, f"db_{cycle}")
            engine = turbomemory.MemoryEngine(db_path=db_path, **kwargs)
            for i in range(records):
                engine.insert(
                    id=f"mem_{cycle}_{i}",
                    text=f"conversation {cycle} turn {i} about topic {i % 7}",
                    embedding=rng.standard_normal(dim).astype(np.float32),
                    importance_score=1.0,
                    concepts=[f"topic_{i % 7}"],
                )
            if consolidate:
                engine.trigger_consolidation()
            for q in range(searches):
                engine.search(
                    query_text=f"topic {q % 7} conversation {cycle}",
                    query_embedding=rng.standard_normal(dim).astype(np.float32),
                    top_k=5,
                )
            engine.flush()
            engine.close()
            del engine
            gc.collect()

            if (cycle + 1) % 5 == 0 or cycle == 0:
                rss, threads = process_stats()
                delta = rss - baseline_rss if rss is not None else float("nan")
                print(
                    f"cycle {cycle + 1:>4}: rss={rss:.1f} MB "
                    f"(+{delta:.1f}) threads={threads}"
                )
            if not keep_db:
                shutil.rmtree(db_path, ignore_errors=True)
    finally:
        if not keep_db:
            shutil.rmtree(root, ignore_errors=True)

    rss, threads = process_stats()
    print(
        f"final: rss={rss:.1f} MB (+{rss - baseline_rss:.1f}) "
        f"threads={threads} (baseline {baseline_threads})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=40)
    parser.add_argument("--records", type=int, default=200)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--keep-db", action="store_true")
    parser.add_argument("--cognitive", action="store_true",
                        help="enable the cognitive-layer features (adapter parity)")
    parser.add_argument("--consolidate", action="store_true",
                        help="call trigger_consolidation() each cycle (harness parity)")
    parser.add_argument("--searches", type=int, default=0,
                        help="cognitive searches per cycle (covers the read path)")
    args = parser.parse_args()
    run(args.cycles, args.records, args.dim, args.keep_db, args.cognitive,
        args.consolidate, args.searches)
