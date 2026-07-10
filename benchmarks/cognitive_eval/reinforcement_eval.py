#!/usr/bin/env python3
"""
TurboSuperMemory — Reinforcement Isolation Eval (Phase 4).

Isolates the value of RETRIEVAL REINFORCEMENT (rehearsal strengthens a memory's
concept edges, so a frequently-recalled memory outranks an equally-relevant but
never-recalled one).

Design (controlled-cosine geometry, per Phase 1 — avoids the near-orthogonal
make_close_vec trap):
  - Per probe, a shared private concept and:
      * anchor : HIGH cosine to the query (an ANN seed) tagged [concept].
      * mem_r  : LOW cosine to the query (NOT an ANN seed), tagged [concept]  -> "rehearsed"
      * mem_c  : LOW cosine to the query at the SAME cosine as mem_r, [concept] -> "cold"
    mem_r and mem_c are reachable only through anchor -> concept -> {mem_r, mem_c}
    (2-hop association spread). They sit at equal cosine, so cosine alone cannot
    separate them — only reinforcement of mem_r's concept edge can.

Two arms (identical corpus; the ONLY difference is whether mem_r is rehearsed):
  - CogOFF : mem_r is NOT rehearsed -> mem_r and mem_c have equal edges -> a coin-flip.
  - CogON  : mem_r IS rehearsed (searched N times) -> its concept edge strengthens ->
             mem_r should outrank mem_c.

Metric: rehearsed_wins = P(mem_r ranks ABOVE mem_c).  feature_lift = CogON - CogOFF.

Usage:
    python benchmarks/cognitive_eval/reinforcement_eval.py
    python benchmarks/cognitive_eval/reinforcement_eval.py --distractors 100 1000 --rehearse 8
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import belief_revision as br  # reuse setup_extension + geometry helpers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("ReinforcementEval")


def build_and_score(tsm, dim, probes_n, distractors, rehearse, n_rehearse, seed):
    rng = np.random.RandomState(seed)
    tmp = tempfile.mkdtemp(prefix="tsm_reinf_")
    try:
        eng = tsm.MemoryEngine(
            db_path=os.path.join(tmp, "e"), dimension=dim, auto_consolidation_secs=0,
            max_concepts=5, cognitive_alpha=0.5,
            edge_decay_half_life_secs=0,  # reinforcement must persist (no decay)
        )
        try:
            br.insert_distractors(eng, dim, distractors, rng, "bg")
            probes = []
            for p in range(probes_n):
                concept = f"rtopic{p}"
                center = br.unit_vec(dim, rng)
                anchor_v = br.jitter_vec(center, 0.03, rng)   # near query -> ANN seed
                query_v = br.jitter_vec(center, 0.03, rng)
                # Three SIBLING memories at the same low cosine to the query (all
                # non-ANN, all reachable only via anchor -> concept). They are
                # symmetric, so cold retrieval orders them purely by insertion
                # tie-break. We rehearse the LAST-inserted sibling (the cold
                # tie-break LOSER), so any lift is attributable to reinforcement,
                # not geometry.
                sibs = []
                for s in range(3):
                    v = br.vec_at_cosine(query_v, 0.12, rng)
                    mid = f"mem_{p}_{s}"
                    eng.insert(mid, f"sibling note {s} about {concept}", v, 1.0, [concept])
                    sibs.append((mid, v))
                eng.insert(f"anchor_{p}", f"anchor context for {concept}", anchor_v, 1.0, [concept])
                r_id, r_v = sibs[-1]
                probes.append((concept, r_id, [m for m, _ in sibs], query_v, r_v))
            eng.trigger_consolidation()

            if rehearse:
                # Rehearse r_id only: query at its OWN vector (top hit) so retrieval
                # reinforces its concept edge; the other siblings stay cold.
                for _ in range(n_rehearse):
                    for (concept, r_id, sib_ids, q_v, r_v) in probes:
                        eng.search(concept, r_v, top_k=1)

            wins = 0
            for (concept, r_id, sib_ids, q_v, r_v) in probes:
                res = eng.search(concept, q_v, top_k=10)
                ids = [x[0] for x in res] if res else []
                rank = {m: (ids.index(m) if m in ids else 10**9) for m in sib_ids}
                if all(rank[r_id] < rank[m] for m in sib_ids if m != r_id):
                    wins += 1
            return wins / probes_n
        finally:
            eng.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="TSM reinforcement isolation eval")
    ap.add_argument("--dimension", type=int, default=768)
    ap.add_argument("--probes", type=int, default=24)
    ap.add_argument("--distractors", type=int, nargs="+", default=[100, 1000])
    ap.add_argument("--rehearse", type=int, default=8, help="rehearsal searches per probe")
    ap.add_argument("--seed", type=int, default=20260710)
    args = ap.parse_args()

    tsm = br.setup_extension()
    dim = args.dimension
    logger.info("=" * 74)
    logger.info("Reinforcement Isolation Eval — dim=%d probes=%d rehearse=%d", dim, args.probes, args.rehearse)
    logger.info("Metric: P(rehearsed sibling outranks its 2 cold siblings). lift = CogON - CogOFF")
    logger.info("=" * 74)
    lifts = []
    for d in args.distractors:
        off = build_and_score(tsm, dim, args.probes, d, False, args.rehearse, args.seed)
        on = build_and_score(tsm, dim, args.probes, d, True, args.rehearse, args.seed)
        lift = on - off
        lifts.append(lift)
        logger.info("distractors=%-6d  CogOFF(no rehearsal)=%.2f  CogON(rehearsed)=%.2f  lift=%+.2f",
                    d, off, on, lift)
    mean_lift = sum(lifts) / len(lifts) if lifts else 0.0
    logger.info("-" * 74)
    logger.info("Mean feature lift (reinforcement): %+.2f", mean_lift)
    if mean_lift > 0.10:
        logger.info("VERDICT: rehearsal reinforcement MEASURABLY re-ranks a rehearsed memory up.")
    elif mean_lift > 0.0:
        logger.info("VERDICT: reinforcement adds MARGINAL lift.")
    else:
        logger.info("VERDICT: reinforcement adds NO isolated lift (mechanism inert in retrieval).")
    logger.info("=" * 74)


if __name__ == "__main__":
    main()
