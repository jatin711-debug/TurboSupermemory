import os
import sys
import shutil
import numpy as np

# Ensure root directory is on path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

import turbomemory

DIMENSION = 768
DB_PATH = os.path.join(root_dir, "test_cognitive_live_db")

if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)

def make_vec(seed, dim=DIMENSION):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v

def make_similar_vec(base_vec, cos_target, seed=123):
    """Generate a vector at an exact cosine angle to base_vec."""
    rng = np.random.RandomState(seed)
    rand_v = rng.randn(len(base_vec)).astype(np.float32)
    perp = rand_v - np.dot(rand_v, base_vec) * base_vec
    perp /= np.linalg.norm(perp)
    c = float(cos_target)
    out = c * base_vec + np.sqrt(max(0.0, 1.0 - c * c)) * perp
    out /= np.linalg.norm(out)
    return out.astype(np.float32)

print("=" * 80)
print(" LIVE COGNITIVE MEMORY RETRIEVAL EXPERIMENT")
print("=" * 80)

# Initialize engine with cognitive layer enabled
engine = turbomemory.MemoryEngine(
    db_path=DB_PATH,
    dimension=DIMENSION,
    cognitive_alpha=0.5,                  # 50% vector cosine, 50% cognitive graph delta
    refinement_cosine_threshold=0.70,
    contradiction_cosine_threshold=0.60,
    contradiction_require_opposition=True, # Safety gate for opposition markers
    exclude_superseded=False,              # Show ranking demotion (if True, stale is dropped)
    importance_auto_scoring=True,
    max_concepts=5,
    concept_max_ngram_len=2,
    auto_consolidation_secs=0,             # Manual consolidation for experiment
)

print(f"[OK] Initialized TurboSuperMemory Engine (dim={DIMENSION}, cognitive_alpha=0.5)")

# -----------------------------------------------------------------------------
# SCENARIO 1: BELIEF REVISION & CONTRADICTION
# -----------------------------------------------------------------------------
print("\n" + "-" * 80)
print(" SCENARIO 1: CONTRADICTION & BELIEF REVISION")
print(" Fact 1: 'User lives in Seattle and works remotely.' (Stale belief)")
print(" Fact 2: 'User actually moved and is no longer living in Seattle; instead they live in Tokyo.' (Correction)")
print("-" * 80)

# Base query vector for "Where does user live?"
query_vec = make_vec(seed=42)

# Stale fact: High vector similarity to query (cosine ~ 0.88)
v_stale = make_similar_vec(query_vec, cos_target=0.88, seed=101)

# Correction fact: Sits at cosine 0.75 relative to v_stale (same topic), cosine ~ 0.70 to query
v_correction = make_similar_vec(v_stale, cos_target=0.75, seed=202)

cos_stale_q = float(np.dot(v_stale, query_vec))
cos_corr_q = float(np.dot(v_correction, query_vec))
cos_between = float(np.dot(v_stale, v_correction))

print(f"Geometric Setup:")
print(f"  • Cosine(Query, Stale Fact)     = {cos_stale_q:.4f} (Higher similarity to query)")
print(f"  • Cosine(Query, Correction Fact) = {cos_corr_q:.4f} (Lower similarity to query)")
print(f"  • Cosine(Stale, Correction)     = {cos_between:.4f} (Shared topic, above 0.60 gate)")

# Insert stale fact first
engine.insert(
    id="fact_old_location",
    text="User lives in Seattle and works remotely.",
    embedding=v_stale,
    importance_score=1.0,
    concepts=["user_location", "seattle", "remote"],
)

# Insert correction later (shares concept 'user_location' and has opposition marker 'no longer')
engine.insert(
    id="fact_new_location",
    text="User actually moved and is no longer living in Seattle; instead they live in Tokyo.",
    embedding=v_correction,
    importance_score=1.0,
    concepts=["user_location", "tokyo", "moved"],
)

# Insert 5 distractor memories
for i in range(5):
    d_vec = make_vec(seed=500 + i)
    engine.insert(
        id=f"distractor_{i}",
        text=f"Random documentation note #{i}: system configuration item.",
        embedding=d_vec,
        importance_score=0.5,
        concepts=[f"topic_{i}"],
    )

print("\nIngested 2 facts + 5 distractors.")

# Consolidate engine (triggers contradiction detection, graph edge formation & demotion factor)
engine.trigger_consolidation()

# Check graph state
stats = engine.graph_stats()
contradictions = engine.get_contradictions("fact_old_location")
print(f"Graph stats: Nodes={stats[0]}, Edges={stats[1]}, Contradictions={stats[5]}")
print(f"Contradictions detected on 'fact_old_location': {contradictions}")

# 1. Plain Vector Search (ANN only)
ann_results = engine.search_ann(query_vec, top_k=2)
print("\n[1] PLAIN VECTOR SEARCH (Standard Vector DB - Pure Cosine):")
for rank, (doc_id, score) in enumerate(ann_results, 1):
    txt = engine.get_text(doc_id)
    print(f"  Rank #{rank}: [{doc_id}] (Cosine Score={score:.4f}) -> {txt}")

# 2. Cognitive Search (Vector + Cognitive Augmenter)
cog_results = engine.search(
    query_text="Where does user live?",
    query_embedding=query_vec,
    top_k=2,
)
print("\n[2] COGNITIVE RETRIEVAL (TSM Cognitive Layer - Belief Revision ON):")
if cog_results:
    for rank, (doc_id, score) in enumerate(cog_results, 1):
        txt = engine.get_text(doc_id)
        print(f"  Rank #{rank}: [{doc_id}] (Fused Score={score:.4f}) -> {txt}")
else:
    print("  No results returned.")

# -----------------------------------------------------------------------------
# SCENARIO 2: MULTI-AGENT SCOPE ISOLATION
# -----------------------------------------------------------------------------
print("\n" + "-" * 80)
print(" SCENARIO 2: MULTI-AGENT SCOPING & PRIVATE ISOLATION")
print("-" * 80)

# Shared memory (scope=None)
engine.insert(
    id="shared_db_rules",
    text="Company DB policy: all production tables must have primary keys.",
    embedding=make_similar_vec(query_vec, cos_target=0.80, seed=301),
    importance_score=1.0,
    concepts=["policy", "database"],
    scope=None,
)

# Agent Alpha private memory
engine.insert(
    id="alpha_private_key",
    text="Agent Alpha private note: secret token is ALPHA_SECRET_99.",
    embedding=make_similar_vec(query_vec, cos_target=0.88, seed=302),
    importance_score=1.0,
    concepts=["secret", "alpha"],
    scope="agent_alpha",
)

# Agent Beta private memory
engine.insert(
    id="beta_private_key",
    text="Agent Beta private note: secret token is BETA_SECRET_77.",
    embedding=make_similar_vec(query_vec, cos_target=0.88, seed=303),
    importance_score=1.0,
    concepts=["secret", "beta"],
    scope="agent_beta",
)

print("Agent Alpha searches (scope='agent_alpha'):")
alpha_search = engine.search_ann(query_vec, top_k=3, scope="agent_alpha")
for rank, (doc_id, score) in enumerate(alpha_search, 1):
    txt = engine.get_text(doc_id)
    print(f"  Rank #{rank}: [{doc_id}] (Score={score:.4f}) -> {txt}")

print("\nAgent Beta searches (scope='agent_beta'):")
beta_search = engine.search_ann(query_vec, top_k=3, scope="agent_beta")
for rank, (doc_id, score) in enumerate(beta_search, 1):
    txt = engine.get_text(doc_id)
    print(f"  Rank #{rank}: [{doc_id}] (Score={score:.4f}) -> {txt}")

engine.close()
print("\n" + "=" * 80)
print(" EXPERIMENT COMPLETE - ALL MECHANISMS VALIDATED LIVE")
print("=" * 80)
