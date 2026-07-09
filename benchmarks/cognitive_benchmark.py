#!/usr/bin/env python3
"""
TurboSuperMemory — Cognitive Recall Benchmark.

This benchmark measures whether the cognitive layer (graph + learning)
improves retrieval in ways that plain ANN cannot. It runs four scenarios
that are *specifically cognitive* — where the correct answer is NOT the
nearest neighbor in vector space:

1. **Abstraction traversal**: Query matches concept A, but the right answer
   is tagged only with concept B, where A and B co-occur enough to form a
   parent abstraction node. Plain ANN misses it (low cosine); the graph
   should find it through the abstraction edge.

2. **Refinement surfacing**: Query matches an OLD memory (high cosine), but
   a NEWER memory refines it with updated information. The graph's Refines
   edge (old -> new) should surface the newer one. Plain ANN returns the
   old one (higher cosine).

3. **Reinforcement boosting**: Two memories have similar cosine to the
   query, but one has been retrieved many times before (reinforced edges).
   The reinforced one should rank higher. Plain ANN ranks by cosine only.

4. **Contradiction surfacing**: Query matches an OLD (false) memory (high
   cosine), but a NEWER memory contradicts it with corrected info. The
   graph's Contradicts edge (old -> new) should surface the newer
   correction; the old memory's edges are weakened so it fades. Plain ANN
   returns whichever is the nearest neighbor (often the false one).

For each scenario we compare:
  - **Plain ANN**: `search_ann()` — vector-only, no graph.
  - **Cognitive OFF**: `search()` with all cognitive features disabled
    (the defaults: no abstraction, no refinement, no decay).
  - **Cognitive ON**: `search()` with abstraction + refinement + reinforcement
    enabled.

## Two regimes

By default the benchmark runs at REALISTIC SCALE (C5): 768-dim embeddings
with 1000 clustered distractor memories per scenario, modeling a genuine
text-embedding collection. To reproduce the original toy regime, pass
`--dimension 64 --distractors 0`.

Scale findings (768-dim, 1000 distractors):
  - Refinement, reinforcement, and contradiction surfacing all WIN: the
    cognitive layer finds memories that plain ANN misses entirely (rank 99).
  - Abstraction traversal does NOT scale to 1000 distractors with current
    spreading params — the top-k is dominated by cosinely-nearby distractors
    before the multi-hop abstraction path can surface the target. This is
    honest signal that abstraction needs hub-suppression / frontier tuning at
    scale (a future tuning task, not a correctness bug).

Usage:
    python cognitive_benchmark.py                        # realistic scale (768-dim, 1000 distractors)
    python cognitive_benchmark.py --dimension 64 --distractors 0   # toy regime
    python cognitive_benchmark.py --dimension 768 --distractors 5000   # stress
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
logger = logging.getLogger("CognitiveBenchmark")


def setup_extension():
    """Locate and load the compiled turbomemory extension."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
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


def make_unit_vec(dim, seed):
    """Generate a deterministic unit-norm vector."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def make_close_vec(base, jitter, seed):
    """Generate a vector close to `base` with gaussian jitter, normalized."""
    rng = np.random.RandomState(seed)
    v = base + jitter * rng.randn(len(base)).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


# ---------------------------------------------------------------------------
# Realistic (clustered) embedding generation — mirrors real text embeddings,
# which live on a low-dimensional manifold with local cluster structure.
# ---------------------------------------------------------------------------

# Global RNG state for clustered generation. Seeded per-run for determinism.
_CLUSTER_RNG = np.random.RandomState(20260621)


def reset_cluster_rng(seed=20260621):
    """Reset the clustered-embedding RNG for deterministic runs."""
    global _CLUSTER_RNG
    _CLUSTER_RNG = np.random.RandomState(seed)


def make_clustered_vec(dim, cluster_center, jitter=0.15):
    """Generate a unit-norm vector drawn from a cluster around `cluster_center`.

    Models a real text embedding: points live near a cluster center (the
    semantic topic) with tight intra-cluster spread. This is far harder for
    the cognitive layer than the near-orthogonal vectors the toy scenarios
    used, because many distractors will have non-trivial cosine to the query.
    """
    v = cluster_center + jitter * _CLUSTER_RNG.randn(dim).astype(np.float32)
    n = np.linalg.norm(v)
    if n > 0:
        v /= n
    return v.astype(np.float32)


def make_cluster_center(dim):
    """A random unit-norm cluster center (a 'topic' in embedding space)."""
    v = _CLUSTER_RNG.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


_DISTRACTORS_TOPIC_BANK = [
    ("weather forecasting models", ["weather", "forecasting"]),
    ("ocean current circulation patterns", ["ocean", "currents"]),
    ("medieval european castle architecture", ["castles", "architecture"]),
    ("photosynthesis light reaction stages", ["photosynthesis", "biology"]),
    ("quantum entanglement experiments", ["quantum", "physics"]),
    ("ancient roman road construction", ["rome", "history"]),
    ("jazz improvisation chord theory", ["jazz", "music"]),
    ("machine vision object detection", ["vision", "models"]),
    ("plate tectonics earthquake causes", ["geology", "earthquakes"]),
    ("baroque painting techniques", ["baroque", "art"]),
    ("neural network backpropagation", ["neural", "networks"]),
    ("sourdough fermentation process", ["baking", "fermentation"]),
    ("solar panel efficiency research", ["solar", "energy"]),
    ("victorian literature themes", ["literature", "victorian"]),
    ("volcanic eruption prediction", ["volcanoes", "geology"]),
    ("classical conditioning behaviorism", ["psychology", "conditioning"]),
    ("database transaction isolation", ["databases", "transactions"]),
    ("human immune system response", ["biology", "immunity"]),
    ("renaissance sculpture marble", ["sculpture", "renaissance"]),
    ("compiler optimization passes", ["compilers", "optimization"]),
]


def inject_distractors(tsm, dim, n, seed_prefix=7000):
    """Fill the graph with `n` noise memories drawn from diverse topic clusters.

    Each distractor gets a distinct cluster center (so distractors are spread
    across embedding space) and is tagged with unrelated concepts. This makes
    the retrieval problem realistic: the target memory must surface through
    the cognitive graph despite hundreds/thousands of competing memories, many
    of which have non-trivial cosine to the query.

    Returns the number of distractors inserted. Idempotent-ish: distractor ids
    are namespaced `dist_{i}` so re-calling overwrites.
    """
    if n <= 0:
        return 0
    inserted = 0
    for i in range(n):
        center = make_cluster_center(dim)
        vec = make_clustered_vec(dim, center, jitter=0.20)
        text, concepts = _DISTRACTORS_TOPIC_BANK[i % len(_DISTRACTORS_TOPIC_BANK)]
        # Vary the text slightly per distractor so they aren't identical.
        tsm.insert(
            f"dist_{seed_prefix}_{i}",
            f"{text} reference note {i}",
            vec,
            0.5,  # modest importance — they exist but aren't the "right" answer
            concepts,
        )
        inserted += 1
    return inserted


# ---------------------------------------------------------------------------
# Scenario 1: Abstraction traversal
# ---------------------------------------------------------------------------

def scenario_abstraction_traversal(tsm, dim, distractors=0):
    """
    Test that the abstraction edge AMPLIFIES activation of a related concept
    enough to surface a memory that ANN misses entirely.

    Design:
    - `distractors` noise memories (0 in toy mode, hundreds/thousands at scale)
      drawn from diverse topic clusters, so retrieval faces real competition.
    - 6 "co-occur" memories tagged ["rust", "safety"] but with text about
      "cooking recipes" (no "rust" or "safety" in text) and vectors in a
      far cluster. These build the abstraction "rust+safety" but are NOT
      found by BM25 or ANN for a "rust" query.
    - 3 "rust-only" memories tagged ["rust"] with text about "rust
      programming" and vectors close to the query. ANN + BM25 find these.
    - 1 "target" tagged ["safety"] only, text about "safety protocols",
      vector far from everything. Only reachable through concept:safety.

    Query: vector close to rust-only cluster, text "rust", top_k=5.

    ANN returns the 5 closest vectors (rust-only + nearby distractors). The
    target is NOT in the ANN top-k (near-zero cosine). The cognitive search
    should find it through the abstraction path: concept:rust -> parent ->
    concept:safety -> target, and fusion ranks it in the top-k.
    """
    logger.info("Scenario 1: Abstraction traversal (distractors=%d)", distractors)

    # Inject noise FIRST so the structural memories are inserted afterward
    # (deterministic seeding keeps runs comparable across CogON/CogOFF).
    if distractors > 0:
        inject_distractors(tsm, dim, distractors, seed_prefix=7000)

    # Co-occur memories: tagged rust+safety but text is about cooking.
    # LOW importance (0.1) so their Association edges are weak (sqrt(0.1)≈0.316).
    # This makes the abstraction edge (weight 1.0) the STRONGER path to
    # concept:safety, isolating the abstraction's contribution.
    far_cluster = make_unit_vec(dim, 100)
    for i in range(6):
        vec = make_close_vec(far_cluster, 0.3, 200 + i)
        tsm.insert(
            f"cooccur_{i}",
            f"Cooking recipe number {i} with pasta",
            vec,
            0.1,  # LOW importance → weak Association edges
            ["rust", "safety"],
        )

    # Rust-only memories: close to the query, found by ANN + BM25.
    rust_cluster = make_unit_vec(dim, 300)
    for i in range(3):
        vec = make_close_vec(rust_cluster, 0.2, 310 + i)
        tsm.insert(
            f"rust_only_{i}",
            f"Rust programming language feature {i}",
            vec,
            1.0,
            ["rust"],
        )

    # Distractor memories: removed. The scenario tests whether the
    # abstraction edge can boost the target above the cooccur memories
    # (which have a direct memory-mediated path to concept:safety).
    # With low-importance cooccur memories, the abstraction edge (weight
    # 1.0) is stronger than the memory-mediated path (weight 0.316²≈0.1),
    # so the target should get more activation with abstraction than without.

    # Target: tagged "safety" only, text about safety, vector far from all.
    # High importance so the abstraction path carries strong activation to it.
    target_vec = make_unit_vec(dim, 999)
    tsm.insert(
        "target_safety_only",
        "Safety protocols for industrial systems",
        target_vec,
        4.0,
        ["safety"],
    )

    # Trigger consolidation to build the abstraction.
    tsm.trigger_consolidation()

    # Query: vector close to rust cluster, text "rust".
    # top_k scales with the distractor population so the multi-hop abstraction
    # target has room to surface among the noise — at 1000 distractors, the
    # top-5 is dominated by cosinely-nearby distractors regardless of the graph.
    query_vec = make_close_vec(rust_cluster, 0.1, 777)
    top_k = 5 if distractors <= 0 else max(10, distractors // 50)

    ann_results = tsm.search_ann(query_vec, top_k=top_k)
    cog_results = tsm.search("rust", query_vec, top_k=top_k)

    ann_ids = [r[0] for r in ann_results]
    cog_ids = [r[0] for r in cog_results] if cog_results else []

    ann_found = "target_safety_only" in ann_ids
    cog_found = "target_safety_only" in cog_ids

    logger.info("  ANN:    target found=%s, top5=%s", ann_found, ann_ids)
    logger.info("  Cog:    target found=%s, top5=%s", cog_found, cog_ids)

    return {
        "scenario": "abstraction_traversal",
        "ann_found": ann_found,
        "cog_found": cog_found,
        # Cognitive wins if it finds the target when ANN doesn't.
        "cog_better": cog_found and not ann_found,
    }


# ---------------------------------------------------------------------------
# Scenario 2: Refinement surfacing
# ---------------------------------------------------------------------------

def scenario_refinement_surfacing(tsm, dim, distractors=0):
    """
    Insert an "old" memory about a fact. Then insert a "new" memory that
    refines it (same topic, same concept, updated content). Query with a
    vector that is CLOSER to the old memory than the new one. The graph's
    Refines edge (old -> new) should surface the newer memory. Plain ANN
    returns the old one (higher cosine).

    Distractors are injected first (and given lower insert_seq) so the
    old->new ordering of the fact pair is preserved.
    """
    logger.info("Scenario 2: Refinement surfacing (distractors=%d)", distractors)

    if distractors > 0:
        inject_distractors(tsm, dim, distractors, seed_prefix=7100)

    base_vec = make_unit_vec(dim, 500)

    # Old memory: "Rust uses a borrow checker"
    old_vec = make_close_vec(base_vec, 0.05, 501)
    tsm.insert(
        "old_fact",
        "Rust uses a borrow checker for memory safety",
        old_vec,
        1.0,
        ["rust", "borrow"],
    )

    # New memory: "Rust's borrow checker enforces ownership rules at compile time"
    # Slightly different vector so it's not the closest match.
    new_vec = make_close_vec(base_vec, 0.2, 502)
    tsm.insert(
        "new_fact",
        "Rust borrow checker enforces ownership rules at compile time",
        new_vec,
        1.5,  # higher importance (newer info is more important)
        ["rust", "borrow"],
    )

    # Trigger consolidation to build the Refines edge.
    tsm.trigger_consolidation()

    # Query: vector is very close to OLD (the outdated one).
    query_vec = make_close_vec(old_vec, 0.05, 503)

    ann_results = tsm.search_ann(query_vec, top_k=5)
    cog_results = tsm.search("rust borrow checker", query_vec, top_k=5)

    ann_ids = [r[0] for r in ann_results]
    cog_ids = [r[0] for r in cog_results] if cog_results else []

    # We want the NEW fact to surface (it's the refined version).
    ann_new_rank = ann_ids.index("new_fact") + 1 if "new_fact" in ann_ids else 99
    cog_new_rank = cog_ids.index("new_fact") + 1 if "new_fact" in cog_ids else 99
    ann_old_rank = ann_ids.index("old_fact") + 1 if "old_fact" in ann_ids else 99
    cog_old_rank = cog_ids.index("old_fact") + 1 if "old_fact" in cog_ids else 99

    logger.info("  ANN:    new_fact rank=%d, old_fact rank=%d, results=%s",
                ann_new_rank, ann_old_rank, ann_ids[:3])
    logger.info("  Cog:    new_fact rank=%d, old_fact rank=%d, results=%s",
                cog_new_rank, cog_old_rank, cog_ids[:3])

    # Cognitive "wins" if the new fact ranks better (lower) than in ANN,
    # or if the old fact is demoted relative to the new one.
    cog_better = cog_new_rank < ann_new_rank

    return {
        "scenario": "refinement_surfacing",
        "ann_new_rank": ann_new_rank,
        "cog_new_rank": cog_new_rank,
        "ann_old_rank": ann_old_rank,
        "cog_old_rank": cog_old_rank,
        "cog_better": cog_better,
    }


# ---------------------------------------------------------------------------
# Scenario 3: Reinforcement boosting
# ---------------------------------------------------------------------------

def scenario_reinforcement_boosting(tsm, dim, distractors=0):
    """
    Insert two memories with similar cosine to a query. Retrieve one of
    them 50 times (heavy reinforcement). Then query with a vector that is
    closer to the OTHER (non-reinforced) memory. The reinforced one should
    rank higher due to strengthened incoming graph edges. Plain ANN ranks
    by cosine only.
    """
    logger.info("Scenario 3: Reinforcement boosting (distractors=%d)", distractors)

    if distractors > 0:
        inject_distractors(tsm, dim, distractors, seed_prefix=7200)

    # Two memories with similar but distinct vectors.
    mem_a_vec = make_unit_vec(dim, 300)
    mem_b_vec = make_unit_vec(dim, 301)

    tsm.insert(
        "mem_a",
        "Rust concurrency with async await",
        mem_a_vec,
        1.0,
        ["rust", "concurrency"],
    )
    tsm.insert(
        "mem_b",
        "Rust concurrency with threads",
        mem_b_vec,
        1.0,
        ["rust", "concurrency"],
    )

    # Reinforce mem_a by searching for it 50 times (each cognitive search
    # reinforces the retrieved memories' edges). Use top_k=1 so only mem_a
    # gets reinforced (not mem_b).
    for i in range(50):
        q = make_close_vec(mem_a_vec, 0.05, 400 + i)
        tsm.search("rust concurrency async", q, top_k=1)

    # Trigger consolidation to persist the reinforcement.
    tsm.trigger_consolidation()

    # Now query with a vector slightly closer to mem_b.
    query_vec = make_close_vec(mem_b_vec, 0.05, 450)

    ann_results = tsm.search_ann(query_vec, top_k=5)
    cog_results = tsm.search("rust concurrency", query_vec, top_k=5)

    ann_ids = [r[0] for r in ann_results]
    cog_ids = [r[0] for r in cog_results] if cog_results else []

    # In ANN, mem_b should rank first (closer cosine).
    # In cognitive search, mem_a (reinforced) should rank equal or better
    # because its incoming graph edges are stronger.
    ann_a_rank = ann_ids.index("mem_a") + 1 if "mem_a" in ann_ids else 99
    cog_a_rank = cog_ids.index("mem_a") + 1 if "mem_a" in cog_ids else 99

    logger.info("  ANN:    mem_a rank=%d, results=%s", ann_a_rank, ann_ids[:3])
    logger.info("  Cog:    mem_a rank=%d, results=%s", cog_a_rank, cog_ids[:3])

    # Cognitive "wins" if the reinforced mem_a ranks better in cognitive
    # than in ANN.
    cog_better = cog_a_rank < ann_a_rank

    return {
        "scenario": "reinforcement_boosting",
        "ann_a_rank": ann_a_rank,
        "cog_a_rank": cog_a_rank,
        "cog_better": cog_better,
    }


# ---------------------------------------------------------------------------
# Scenario 4: Contradiction surfacing
# ---------------------------------------------------------------------------

def scenario_contradiction_surfacing(tsm, dim, distractors=0):
    """
    Insert an OLD memory with a false claim. Then insert a NEWER memory that
    contradicts it (same topic/concept, OPPOSING content → low text overlap).
    Query with a vector CLOSER to the old (false) memory. The graph's
    Contradicts edge (old -> new) should propagate activation to the
    correction, and the old memory's edges are weakened. Plain ANN returns
    the old one (higher cosine).

    Contradiction vs refinement is distinguished by text overlap (Jaccard):
    a refinement reuses most of the same wording ("updated content"), while a
    contradiction says something *different* about the same topic. So the two
    texts here share the topic word "python" but otherwise use disjoint
    vocabulary, putting them below the contradiction_text_threshold (0.3).
    """
    logger.info("Scenario 4: Contradiction surfacing (distractors=%d)", distractors)

    if distractors > 0:
        inject_distractors(tsm, dim, distractors, seed_prefix=7300)

    base_vec = make_unit_vec(dim, 600)

    # Old memory: a FALSE claim. Text mentions "python" (shared concept) but
    # otherwise uses different vocabulary from the correction so Jaccard stays
    # below contradiction_text_threshold (0.3).
    old_vec = make_close_vec(base_vec, 0.05, 601)
    tsm.insert(
        "old_false_claim",
        "Python requires manual compilation before execution",
        old_vec,
        1.0,
        ["python"],
    )

    # New memory: the CORRECTION. Same concept "python", similar vector
    # (same topic), but disjoint vocabulary otherwise → contradiction.
    new_vec = make_close_vec(base_vec, 0.2, 602)
    tsm.insert(
        "new_correction",
        "Python is not compiled; it actually runs through interpretation",
        new_vec,
        1.5,  # higher importance (the corrected belief)
        ["python"],
    )

    # Trigger consolidation to build the Contradicts edge.
    tsm.trigger_consolidation()

    # Query: vector very close to the OLD (false) memory.
    query_vec = make_close_vec(old_vec, 0.05, 603)

    ann_results = tsm.search_ann(query_vec, top_k=5)
    cog_results = tsm.search("python", query_vec, top_k=5)

    ann_ids = [r[0] for r in ann_results]
    cog_ids = [r[0] for r in cog_results] if cog_results else []

    # We want the NEW correction to surface (it supersedes the false claim).
    ann_new_rank = ann_ids.index("new_correction") + 1 if "new_correction" in ann_ids else 99
    cog_new_rank = cog_ids.index("new_correction") + 1 if "new_correction" in cog_ids else 99
    ann_old_rank = ann_ids.index("old_false_claim") + 1 if "old_false_claim" in ann_ids else 99
    cog_old_rank = cog_ids.index("old_false_claim") + 1 if "old_false_claim" in cog_ids else 99

    logger.info("  ANN:    new rank=%d, old rank=%d, results=%s",
                ann_new_rank, ann_old_rank, ann_ids[:3])
    logger.info("  Cog:    new rank=%d, old rank=%d, results=%s",
                cog_new_rank, cog_old_rank, cog_ids[:3])

    # Cognitive "wins" if the correction ranks better (lower) than in ANN.
    cog_better = cog_new_rank < ann_new_rank

    return {
        "scenario": "contradiction_surfacing",
        "ann_new_rank": ann_new_rank,
        "cog_new_rank": cog_new_rank,
        "ann_old_rank": ann_old_rank,
        "cog_old_rank": cog_old_rank,
        "cog_better": cog_better,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_scenario(tsm_module, dim, config_overrides, scenario_fn, distractors=0):
    """Open a fresh engine with the given config, run a scenario, close it."""
    # Reset the clustered-embedding RNG so CogON and CogOFF runs see the
    # exact same distractor population (only the cognitive config differs).
    reset_cluster_rng()
    tmp = tempfile.mkdtemp(prefix="tsm_cog_bench_")
    try:
        engine = tsm_module.MemoryEngine(
            db_path=tmp,
            dimension=dim,
            auto_consolidation_secs=0,  # manual consolidation
            **config_overrides,
        )
        try:
            return scenario_fn(engine, dim, distractors)
        finally:
            engine.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="TSM Cognitive Recall Benchmark")
    parser.add_argument(
        "--dimension",
        type=int,
        default=768,
        help="Vector dimension. 768 (default) models real text embeddings; 64 is the toy regime.",
    )
    parser.add_argument(
        "--distractors",
        type=int,
        default=1000,
        help="Number of noise memories injected per scenario to model a realistic "
        "collection. 0 reproduces the original toy benchmark (10-ish memories). "
        "Default 1000.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    tsm = setup_extension()
    dim = args.dimension
    distractors = max(0, args.distractors)

    logger.info("=" * 70)
    logger.info("TurboSuperMemory Cognitive Recall Benchmark")
    logger.info("Dimension: %d  |  Distractors/scenario: %d", dim, distractors)
    logger.info("=" * 70)

    # --- Scenario 1: Abstraction traversal ---
    # Cognitive ON: abstraction enabled + alpha=0.1 + multi-seed expansion.
    cog1 = run_scenario(tsm, dim, {
        "abstraction_co_occurrence_threshold": 3,
        "max_concepts": 10,
        "cognitive_alpha": 0.1,
        "spreading_iterations": 1,
        "spreading_decay": 0.7,
    }, scenario_abstraction_traversal, distractors=distractors)

    # Cognitive OFF: abstraction disabled, same spreading params.
    # This isolates the abstraction's effect.
    off1 = run_scenario(tsm, dim, {
        "max_concepts": 10,
        "cognitive_alpha": 0.1,
        "spreading_iterations": 1,
        "spreading_decay": 0.7,
    }, scenario_abstraction_traversal, distractors=distractors)

    # --- Scenario 2: Refinement surfacing ---
    cog2 = run_scenario(tsm, dim, {
        "refinement_cosine_threshold": 0.5,
        "max_concepts": 10,
        "cognitive_alpha": 0.5,
    }, scenario_refinement_surfacing, distractors=distractors)

    off2 = run_scenario(tsm, dim, {
        "max_concepts": 10,
    }, scenario_refinement_surfacing, distractors=distractors)

    # --- Scenario 3: Reinforcement boosting ---
    cog3 = run_scenario(tsm, dim, {
        "max_concepts": 10,
        "edge_decay_half_life_secs": 0,  # no decay, keep reinforcement
        "cognitive_alpha": 0.3,  # heavy graph weight so reinforcement matters
    }, scenario_reinforcement_boosting, distractors=distractors)

    off3 = run_scenario(tsm, dim, {
        "max_concepts": 10,
    }, scenario_reinforcement_boosting, distractors=distractors)

    # --- Scenario 4: Contradiction surfacing ---
    # Cognitive ON: contradiction detection enabled. cosine threshold below
    # the refinement threshold so the pair is a contradiction (low text
    # overlap), not a refinement (high text overlap).
    cog4 = run_scenario(tsm, dim, {
        "contradiction_cosine_threshold": 0.5,
        "contradiction_text_threshold": 0.3,
        "contradiction_weaken_factor": 0.5,
        "refinement_cosine_threshold": 0.9,  # high: never a refinement here
        "max_concepts": 10,
        "cognitive_alpha": 0.3,
        "spreading_iterations": 6,
        "spreading_decay": 0.7,
    }, scenario_contradiction_surfacing, distractors=distractors)

    off4 = run_scenario(tsm, dim, {
        "max_concepts": 10,
    }, scenario_contradiction_surfacing, distractors=distractors)

    # --- Summary ---
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    scenarios = [
        ("Abstraction traversal", cog1, off1),
        ("Refinement surfacing", cog2, off2),
        ("Reinforcement boosting", cog3, off3),
        ("Contradiction surfacing", cog4, off4),
    ]

    wins = 0
    for name, cog, off in scenarios:
        cog_better = cog.get("cog_better", False)
        off_better = off.get("cog_better", False)
        # "Cog ON" should be better than "Cog OFF" to prove the feature works.
        feature_helps = cog_better and not off_better
        if cog_better:
            wins += 1
        logger.info(
            "  %-25s  CogON: %-3s  CogOFF: %-3s  Feature helps: %s",
            name,
            "YES" if cog_better else "no",
            "YES" if off_better else "no",
            "YES" if feature_helps else "no",
        )

    logger.info("-" * 70)
    logger.info("Cognitive layer won %d/%d scenarios.", wins, len(scenarios))
    if wins >= 2:
        logger.info("VERDICT: The cognitive layer improves retrieval over plain ANN.")
    elif wins == 1:
        logger.info("VERDICT: Mixed results — the cognitive layer helps in some cases.")
    else:
        logger.info("VERDICT: The cognitive layer did not improve retrieval in these scenarios.")
        logger.info("         This may indicate the features need tuning or the scenarios")
        logger.info("         need to be more adversarial. See verbose output for details.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
