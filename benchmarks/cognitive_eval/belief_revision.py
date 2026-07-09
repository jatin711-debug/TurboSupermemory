#!/usr/bin/env python3
"""
TurboSuperMemory — Belief-Revision Evaluation Harness (Phase 1).

WHY THIS EXISTS
---------------
The existing `cognitive_benchmark.py` is a *wiring test*: 4 hand-built scenarios
that prove the cognitive path is connected. The frozen baseline (benchmarks/BASELINE.md)
exposed its blind spot — at realistic scale it reports "3/4 won", but in every win the
**CogOFF arm wins too**. The targets are surfaced by generic BM25 lexical + concept
expansion, NOT by the specific belief-revision edge under test. So the benchmark cannot
distinguish "the graph helped" from "good lexical recall would have helped anyway."

This harness is built to *fail* when cognition adds no value. Its discriminating metric
is **belief accuracy**, not recall:

    belief_accuracy = P( current belief ranks ABOVE its superseded version )

Plain ANN ranks the STALE fact first (it is the cosine-nearest to a query phrased the way
the agent originally learned the fact). Generic lexical expansion will *surface* the
correction as a candidate — but typically below the stale fact. Only the Contradicts /
Refines edge + weakening should FLIP the order so the current belief wins. The metric that
isolates the mechanism is therefore:

    feature_lift = belief_accuracy(CogON) - belief_accuracy(CogOFF)

THREE ARMS (identical config except the belief-revision feature)
----------------------------------------------------------------
  - ANN     : search_ann()             — vector only.
  - CogOFF  : search()  belief edges OFF (refinement/contradiction thresholds = None).
  - CogON   : search()  belief edges ON  (the mechanism under test).

THREE SCALING AXES
------------------
  - distractors      : corpus noise (100 -> 10k). Does the win survive scale?
  - temporal gap     : # filler memories inserted BETWEEN stale fact and its correction.
                       Stresses temporal/insert-order dependence of edge building.
  - subtlety (jitter): how far the correction sits from the stale fact in vector space.
                       Larger jitter => ANN buries the correction deeper => harder.

Usage:
    python benchmarks/cognitive_eval/belief_revision.py                 # default sweep
    python benchmarks/cognitive_eval/belief_revision.py --mode refinement
    python benchmarks/cognitive_eval/belief_revision.py --distractors 100 1000 10000
    python benchmarks/cognitive_eval/belief_revision.py --probes 40 --gap 50 --quick
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("BeliefRevisionEval")


# ---------------------------------------------------------------------------
# Extension loading (mirrors cognitive_benchmark.py)
# ---------------------------------------------------------------------------

def setup_extension():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    ext = ".pyd" if sys.platform.startswith("win") else ".so"
    pyd = os.path.join(project_root, f"turbomemory{ext}")
    dll = os.path.join(project_root, "target", "release", "turbomemory.dll")
    if sys.platform.startswith("win") and os.path.exists(dll) and not os.path.exists(pyd):
        shutil.copy2(dll, pyd)
    if not os.path.exists(pyd):
        logger.error("turbomemory extension not found at %s. Run 'make build-python' first.", pyd)
        sys.exit(1)
    sys.path.insert(0, project_root)
    import turbomemory
    return turbomemory


# ---------------------------------------------------------------------------
# Deterministic clustered embeddings (models real text-embedding manifolds)
# ---------------------------------------------------------------------------

def _normalize(v):
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)


def unit_vec(dim, rng):
    return _normalize(rng.randn(dim).astype(np.float32))


def jitter_vec(base, jitter, rng):
    return _normalize(base + jitter * rng.randn(len(base)).astype(np.float32))


# ---------------------------------------------------------------------------
# Topic vocabulary — used to build texts whose overlap controls whether the
# pair is detected as a REFINEMENT (high overlap) or CONTRADICTION (low overlap).
# ---------------------------------------------------------------------------

_TOPICS = [
    ("python", "Python", "interpreter", "compilation", "bytecode", "runtime"),
    ("database", "the database", "indexing", "sharding", "replication", "transactions"),
    ("rust", "Rust", "borrow checker", "ownership", "lifetimes", "concurrency"),
    ("http", "the HTTP protocol", "caching", "headers", "streaming", "compression"),
    ("ml", "the model", "training", "inference", "quantization", "distillation"),
    ("storage", "the storage engine", "compaction", "flushing", "snapshots", "recovery"),
    ("network", "the network stack", "routing", "congestion", "buffering", "framing"),
    ("auth", "authentication", "tokens", "sessions", "rotation", "expiry"),
]

_DISTRACTOR_BANK = [
    ("weather forecasting and atmospheric models", ["weather", "forecasting"]),
    ("ocean current circulation patterns", ["ocean", "currents"]),
    ("medieval castle architecture and design", ["castles", "architecture"]),
    ("photosynthesis light reaction stages", ["photosynthesis", "biology"]),
    ("ancient roman road construction methods", ["rome", "history"]),
    ("jazz improvisation and chord theory", ["jazz", "music"]),
    ("plate tectonics and earthquake causes", ["geology", "earthquakes"]),
    ("baroque painting brushwork techniques", ["baroque", "art"]),
    ("sourdough fermentation chemistry", ["baking", "fermentation"]),
    ("victorian literature recurring themes", ["literature", "victorian"]),
]


def stale_text(topic):
    """Original (later superseded) claim about a topic."""
    key, subj, a, b, _, _ = topic
    return f"{subj} relies on {a} and {b} for its core behavior"


def correction_text(topic, mode):
    """The corrected/refined claim.

    refinement  -> high lexical overlap with the stale text (same wording, updated).
    contradiction -> disjoint vocabulary about the same topic (low Jaccard).
    """
    key, subj, a, b, c, d = topic
    if mode == "refinement":
        # Reuse most of the stale wording; only the tail changes (high overlap).
        return f"{subj} relies on {a} and {b}, specifically using {c} at runtime"
    # contradiction: same subject, otherwise disjoint vocabulary (low overlap).
    return f"{subj} actually depends on {c} plus {d} instead"


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------

class Probe:
    __slots__ = ("topic_key", "stale_id", "correction_id", "query_vec")

    def __init__(self, topic_key, stale_id, correction_id, query_vec):
        self.topic_key = topic_key
        self.stale_id = stale_id
        self.correction_id = correction_id
        self.query_vec = query_vec


def insert_distractors(tsm, dim, n, rng, seq_label):
    for i in range(n):
        center = unit_vec(dim, rng)
        vec = jitter_vec(center, 0.20, rng)
        text, concepts = _DISTRACTOR_BANK[i % len(_DISTRACTOR_BANK)]
        tsm.insert(f"dist_{seq_label}_{i}", f"{text} note {i}", vec, 0.5, concepts)


def build_corpus(tsm, dim, probes_n, distractors, gap, subtlety, mode, rng):
    """Insert distractors + belief pairs with a temporal gap between each
    stale fact and its correction. Returns the list of Probe objects.

    Insertion order per probe:  stale fact  ->  `gap` filler memories  ->  correction.
    This forces old->new ordering (Refines/Contradicts are directed old->new) and
    inserts temporal distance between the two halves of the belief.
    """
    # Background noise first (lowest insert_seq), so belief pairs are "newer".
    insert_distractors(tsm, dim, distractors, rng, seq_label="bg")

    probes = []
    for p in range(probes_n):
        topic = _TOPICS[p % len(_TOPICS)]
        key = f"{topic[0]}_{p}"
        concept = topic[0]

        # Topic cluster center; stale fact sits at the center, query near it.
        center = unit_vec(dim, rng)
        stale_vec = jitter_vec(center, 0.03, rng)
        # Correction is the SAME topic but pushed `subtlety` away in vector space,
        # so a query near the stale fact ranks the stale fact first under pure ANN.
        correction_vec = jitter_vec(center, subtlety, rng)
        # Query is phrased the way the agent originally learned it -> near stale.
        query_vec = jitter_vec(stale_vec, 0.03, rng)

        stale_id = f"stale_{key}"
        corr_id = f"corr_{key}"

        tsm.insert(stale_id, stale_text(topic), stale_vec, 1.0, [concept])

        # Temporal gap: filler memories between stale fact and correction.
        for g in range(gap):
            fc = unit_vec(dim, rng)
            fv = jitter_vec(fc, 0.20, rng)
            text, concepts = _DISTRACTOR_BANK[(p + g) % len(_DISTRACTOR_BANK)]
            tsm.insert(f"gap_{key}_{g}", f"{text} filler {g}", fv, 0.4, concepts)

        # Correction is newer (higher insert_seq) and more important.
        tsm.insert(corr_id, correction_text(topic, mode), correction_vec, 1.5, [concept])

        probes.append(Probe(concept, stale_id, corr_id, query_vec))

    tsm.trigger_consolidation()
    return probes


# ---------------------------------------------------------------------------
# Arm configs
# ---------------------------------------------------------------------------

def base_cog_config():
    """Shared config for both CogON and CogOFF — the ONLY difference between the
    two arms is whether the belief-revision edges are built. cognitive_alpha,
    concepts, lexical weighting etc. are identical so the lift is attributable."""
    return {
        "max_concepts": 10,
        "cognitive_alpha": 0.5,
    }


def cogon_config(mode):
    cfg = base_cog_config()
    if mode == "refinement":
        cfg["refinement_cosine_threshold"] = 0.5
    else:  # contradiction
        cfg["contradiction_cosine_threshold"] = 0.5
        cfg["contradiction_text_threshold"] = 0.3
        cfg["contradiction_weaken_factor"] = 0.5
        cfg["refinement_cosine_threshold"] = 0.95  # high => stays a contradiction
    return cfg


# ---------------------------------------------------------------------------
# Probing + metrics
# ---------------------------------------------------------------------------

def rank_of(results, target_id):
    ids = [r[0] for r in results] if results else []
    return ids.index(target_id) + 1 if target_id in ids else None


def score_arm(tsm, probes, mode_search, top_k, query_text_fn):
    """Run all probes for one arm. Returns aggregate metrics.

    mode_search: "ann" -> search_ann ; "cog" -> search.
    """
    n = len(probes)
    belief_correct = 0       # correction ranks ABOVE stale (the headline metric)
    correction_found = 0     # correction appears in top_k at all
    stale_suppressed = 0     # stale fact is NOT rank 1
    for pr in probes:
        if mode_search == "ann":
            res = tsm.search_ann(pr.query_vec, top_k=top_k)
        else:
            res = tsm.search(query_text_fn(pr), pr.query_vec, top_k=top_k)

        c_rank = rank_of(res, pr.correction_id)
        s_rank = rank_of(res, pr.stale_id)

        if c_rank is not None:
            correction_found += 1
        # Belief is "current" iff the correction is present AND outranks the stale.
        s_eff = s_rank if s_rank is not None else 10**9
        c_eff = c_rank if c_rank is not None else 10**9
        if c_rank is not None and c_eff < s_eff:
            belief_correct += 1
        if s_rank != 1:
            stale_suppressed += 1

    return {
        "belief_accuracy": belief_correct / n,
        "correction_recall": correction_found / n,
        "stale_suppression": stale_suppressed / n,
    }


def run_cell(tsm_module, dim, probes_n, distractors, gap, subtlety, mode, top_k, seed):
    """Build one corpus and score all three arms against it."""
    rng = np.random.RandomState(seed)
    tmp = tempfile.mkdtemp(prefix="tsm_belief_")
    try:
        # The corpus (vectors + insertion order) must be identical across arms;
        # only the engine config differs. We rebuild per arm from the same seed.
        def query_text(pr):
            return pr.topic_key

        arms = {}

        # ANN arm — config is irrelevant to search_ann ranking, use defaults.
        eng = tsm_module.MemoryEngine(db_path=os.path.join(tmp, "ann"), dimension=dim,
                                      auto_consolidation_secs=0, **base_cog_config())
        try:
            probes = build_corpus(eng, dim, probes_n, distractors, gap, subtlety, mode,
                                  np.random.RandomState(seed))
            arms["ANN"] = score_arm(eng, probes, "ann", top_k, query_text)
        finally:
            eng.close()

        # CogOFF arm.
        eng = tsm_module.MemoryEngine(db_path=os.path.join(tmp, "off"), dimension=dim,
                                      auto_consolidation_secs=0, **base_cog_config())
        try:
            probes = build_corpus(eng, dim, probes_n, distractors, gap, subtlety, mode,
                                  np.random.RandomState(seed))
            arms["CogOFF"] = score_arm(eng, probes, "cog", top_k, query_text)
        finally:
            eng.close()

        # CogON arm.
        eng = tsm_module.MemoryEngine(db_path=os.path.join(tmp, "on"), dimension=dim,
                                      auto_consolidation_secs=0, **cogon_config(mode))
        try:
            probes = build_corpus(eng, dim, probes_n, distractors, gap, subtlety, mode,
                                  np.random.RandomState(seed))
            arms["CogON"] = score_arm(eng, probes, "cog", top_k, query_text)
        finally:
            eng.close()

        return arms
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main / sweep
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="TSM belief-revision eval")
    ap.add_argument("--dimension", type=int, default=768)
    ap.add_argument("--mode", choices=["contradiction", "refinement"], default="contradiction")
    ap.add_argument("--probes", type=int, default=24, help="belief pairs per cell")
    ap.add_argument("--distractors", type=int, nargs="+", default=[100, 1000, 10000],
                    help="corpus-noise sweep")
    ap.add_argument("--gap", type=int, default=20, help="filler memories between stale fact and correction")
    ap.add_argument("--subtlety", type=float, default=0.25,
                    help="vector jitter of the correction relative to the topic center")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260629)
    ap.add_argument("--quick", action="store_true", help="single small cell (smoke test)")
    args = ap.parse_args()

    tsm = setup_extension()
    dim = args.dimension

    if args.quick:
        distractor_sweep = [args.distractors[0] if args.distractors else 100]
        probes_n = min(args.probes, 8)
    else:
        distractor_sweep = args.distractors
        probes_n = args.probes

    logger.info("=" * 78)
    logger.info("Belief-Revision Eval — mode=%s dim=%d probes=%d gap=%d subtlety=%.2f top_k=%d",
                args.mode, dim, probes_n, args.gap, args.subtlety, args.top_k)
    logger.info("Headline metric: belief_accuracy = P(correction outranks the stale fact)")
    logger.info("Feature lift     = belief_accuracy(CogON) - belief_accuracy(CogOFF)")
    logger.info("=" * 78)

    header = (f"{'distractors':>11} | {'arm':<7} | {'belief_acc':>10} | "
              f"{'corr_recall':>11} | {'stale_suppr':>11}")
    rows = []
    verdicts = []
    for d in distractor_sweep:
        logger.info("Building cell: distractors=%d ...", d)
        arms = run_cell(tsm, dim, probes_n, d, args.gap, args.subtlety, args.mode,
                        args.top_k, args.seed)
        logger.info("-" * 78)
        logger.info(header)
        logger.info("-" * 78)
        for name in ("ANN", "CogOFF", "CogON"):
            m = arms[name]
            logger.info("%11d | %-7s | %10.2f | %11.2f | %11.2f",
                        d, name, m["belief_accuracy"], m["correction_recall"],
                        m["stale_suppression"])
        lift = arms["CogON"]["belief_accuracy"] - arms["CogOFF"]["belief_accuracy"]
        vs_ann = arms["CogON"]["belief_accuracy"] - arms["ANN"]["belief_accuracy"]
        logger.info("  -> feature lift (CogON - CogOFF) = %+.2f   |   vs ANN = %+.2f", lift, vs_ann)
        rows.append((d, arms, lift, vs_ann))
        verdicts.append(lift)

    logger.info("=" * 78)
    logger.info("SUMMARY (%s)", args.mode)
    logger.info("=" * 78)
    for d, arms, lift, vs_ann in rows:
        logger.info("  distractors=%-6d  belief_acc ANN=%.2f CogOFF=%.2f CogON=%.2f  "
                    "lift=%+.2f", d, arms["ANN"]["belief_accuracy"],
                    arms["CogOFF"]["belief_accuracy"], arms["CogON"]["belief_accuracy"], lift)

    mean_lift = sum(verdicts) / len(verdicts) if verdicts else 0.0
    logger.info("-" * 78)
    logger.info("Mean feature lift across sweep: %+.2f", mean_lift)
    if mean_lift > 0.10:
        logger.info("VERDICT: belief-revision edges add MEASURABLE value beyond generic recall.")
    elif mean_lift > 0.0:
        logger.info("VERDICT: belief-revision edges add MARGINAL value — investigate per-cell.")
    else:
        logger.info("VERDICT: belief-revision edges add NO value over generic lexical/association "
                    "expansion. This is the failing signal the eval is designed to catch.")
    logger.info("=" * 78)


if __name__ == "__main__":
    main()
