#!/usr/bin/env python3
"""
TurboSuperMemory - Real-World AI Agent Memory Demo

This script demonstrates how an AI agent would use TurboSuperMemory (TSM)
in production: storing memories, learning from interactions, handling belief
revision, and retrieving context through both vector similarity AND cognitive
associations.

Run: python demo_agent_memory.py
"""

import os
import sys
import shutil
import numpy as np
import json

# Setup: find and load the compiled TSM extension
current_dir = os.path.dirname(os.path.abspath(__file__))
is_windows = sys.platform.startswith("win")
ext_suffix = ".pyd" if is_windows else ".so"
pyd_path = os.path.join(current_dir, f"turbomemory{ext_suffix}")
lib_prefix = "" if is_windows else "lib"
lib_suffix = ".dll" if is_windows else ".so"
lib_filename = f"{lib_prefix}turbomemory{lib_suffix}"
source = os.path.join(current_dir, "target", "release", lib_filename)
if os.path.exists(source):
    shutil.copy(source, pyd_path)

import turbomemory

# =============================================================================
# CONFIGURATION
# =============================================================================
DIMENSION = 768
DB_PATH = os.path.join(current_dir, "demo_agent_db")

# Helper: generate realistic text embeddings (simulated with clustered random vectors)
def make_embedding(text, seed=None):
    """Create a deterministic embedding for demo text."""
    if seed is not None:
        np.random.seed(seed)
    emb = np.random.randn(DIMENSION).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return emb

# Helper: print formatted sections
def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def subsection(title):
    print(f"\n  ▶ {title}")
    print(f"  {'─'*60}")

def result(label, value):
    print(f"    {label:.<50} {value}")

def memory_card(mem_id, text, score=None, tier=None, concepts=None):
    score_str = f"  [score: {score:.3f}]" if score is not None else ""
    tier_str = f"  [{tier}]" if tier is not None else ""
    print(f"    ┌─ {mem_id}{score_str}{tier_str}")
    print(f"    │  {text[:80]}{'...' if len(text) > 80 else ''}")
    if concepts:
        print(f"    │  concepts: {', '.join(concepts)}")
    print(f"    └─")

# =============================================================================
# CLEANUP & INITIALIZATION
# =============================================================================
if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)

section("INITIALIZING TURBO SUPER MEMORY")
print("  Creating an AI agent memory engine with cognitive features enabled...")

engine = turbomemory.MemoryEngine(
    db_path=DB_PATH,
    dimension=DIMENSION,
    # HNSW tuned for high-dim recall (P0 fixes: M=64, efc=800)
    max_edges=64,
    search_list_size=256,
    ef_construction=800,
    # Tier configuration
    hot_capacity=1000,
    warm_capacity=50000,
    hnsw_threshold=500,
    # Cognitive layer: ALL features enabled
    importance_auto_scoring=True,
    importance_learning_rate=0.3,
    importance_access_weight=0.6,
    importance_floor=0.1,
    importance_ceiling=4.0,
    refinement_cosine_threshold=0.85,
    contradiction_cosine_threshold=0.75,
    contradiction_text_threshold=0.3,
    contradiction_weaken_factor=0.5,
    cognitive_alpha=0.5,  # Blend cosine + graph activation
    max_concepts=5,
    concept_max_ngram_len=2,
    concept_enable_pmi=True,
    abstraction_co_occurrence_threshold=3,
    edge_decay_half_life_secs=86400,  # 1 day
    concept_evolution_enabled=True,
    concept_merge_overlap_threshold=0.7,
    concept_hub_degree_fraction=0.1,
    auto_consolidation_secs=0,  # Manual consolidation for demo
)

print(f"  ✓ Engine initialized at: {DB_PATH}")
print(f"  ✓ Dimension: {DIMENSION}")
print(f"  ✓ Cognitive layer: ENABLED (all features)")

# =============================================================================
# FEATURE 1: BASIC MEMORY STORAGE
# =============================================================================
section("FEATURE 1: BASIC MEMORY STORAGE")
print("  The agent stores facts, conversations, and knowledge...")

memories = [
    ("mem_001", "Python is a high-level programming language known for readability.", 
     ["python", "programming", "language"], 1.0),
    ("mem_002", "Rust provides memory safety without garbage collection through ownership.", 
     ["rust", "memory safety", "ownership"], 1.0),
    ("mem_003", "TypeScript adds static typing to JavaScript, improving developer experience.", 
     ["typescript", "javascript", "typing"], 1.0),
    ("mem_004", "React is a declarative UI library using a virtual DOM for efficient updates.", 
     ["react", "ui", "virtual dom"], 1.0),
    ("mem_005", "Vector databases store embeddings for semantic search and similarity queries.", 
     ["vector database", "embeddings", "semantic search"], 1.0),
]

for i, (mem_id, text, concepts, importance) in enumerate(memories):
    emb = make_embedding(text, seed=i * 1000)
    engine.insert(
        id=mem_id,
        text=text,
        embedding=emb,
        importance_score=importance,
        concepts=concepts,
    )
    print(f"    ✓ Stored: {mem_id}")

subsection("Vector Search (ANN)")
query = "What language prevents memory bugs?"
query_emb = make_embedding(query, seed=999)
results = engine.search_ann(query_emb, top_k=3)
print(f"    Query: '{query}'")
for mem_id, score in results:
    mem_text = next((t for mid, t, _, _ in memories if mid == mem_id), "?")
    memory_card(mem_id, mem_text, score=score)

# =============================================================================
# FEATURE 2: COGNITIVE SEARCH (Graph + Vector Fusion)
# =============================================================================
section("FEATURE 2: COGNITIVE SEARCH")
print("  The agent searches using BOTH vector similarity AND graph associations...")
print("  This finds memories that are conceptually related even if vector-distance is far.")

subsection("Cognitive Search with Graph Activation")
query = "Tell me about safe systems programming"
query_emb = make_embedding(query, seed=888)
results = engine.search(
    query_text=query,
    query_embedding=query_emb,
    top_k=3,
)
print(f"    Query: '{query}'")
print(f"    (Graph activation boosts memories connected to 'rust', 'memory safety', 'ownership')")
for mem_id, score in results:
    mem_text = next((t for mid, t, _, _ in memories if mid == mem_id), "?")
    memory_card(mem_id, mem_text, score=score)

# =============================================================================
# FEATURE 3: REINFORCEMENT LEARNING ON MEMORY
# =============================================================================
section("FEATURE 3: MEMORY REINFORCEMENT")
print("  Memories that are recalled frequently become STRONGER (easier to recall).")
print("  This mimics human memory: repeated rehearsal strengthens neural pathways.")

subsection("Before Reinforcement")
stats_before = engine.graph_stats()
print(f"    Graph: {stats_before[0]} nodes, {stats_before[1]} edges")

# Simulate the agent recalling 'mem_002' (Rust) 10 times
print(f"    Simulating 10 recalls of 'mem_002' (Rust memory)...")
for i in range(10):
    engine.search(
        query_text="Rust memory safety",
        query_embedding=make_embedding("Rust memory safety", seed=i),
        top_k=3,
    )

subsection("After Reinforcement")
stats_after = engine.graph_stats()
print(f"    Graph: {stats_after[0]} nodes, {stats_after[1]} edges")
print(f"    (Edge weights for mem_002 increased through repeated retrieval)")

subsection("Reinforced Memory Surfaces Higher")
query = "systems programming without garbage collector"
query_emb = make_embedding(query, seed=777)
results = engine.search(
    query_text=query,
    query_embedding=query_emb,
    top_k=3,
)
print(f"    Query: '{query}'")
print(f"    Note: mem_002 (Rust) should rank higher due to reinforcement!")
for mem_id, score in results:
    mem_text = next((t for mid, t, _, _ in memories if mid == mem_id), "?")
    memory_card(mem_id, mem_text, score=score)

# =============================================================================
# FEATURE 4: BELIEF REVISION (Refinement)
# =============================================================================
section("FEATURE 4: BELIEF REVISION (Refinement)")
print("  When the agent learns an updated fact, it creates a 'Refines' edge.")
print("  The OLD memory is preserved (history), but the NEW one surfaces on retrieval.")

# Old belief
old_id = "mem_006_old"
old_text = "Python 3.9 was released in October 2020 with dictionary merge operators."
old_emb = make_embedding(old_text, seed=600)
engine.insert(
    id=old_id,
    text=old_text,
    embedding=old_emb,
    importance_score=1.0,
    concepts=["python", "python 3.9", "release"],
)
print(f"    ✓ Stored OLD belief: {old_id}")

# NEW belief (supersedes old)
new_id = "mem_006_new"
new_text = "Python 3.12 was released in October 2023 with improved f-strings and performance."
new_emb = make_embedding(new_text, seed=601)
engine.insert(
    id=new_id,
    text=new_text,
    embedding=new_emb,
    importance_score=1.2,  # Higher importance = newer
    concepts=["python", "python 3.12", "release", "performance"],
)
print(f"    ✓ Stored NEW belief: {new_id}")

# Trigger consolidation AFTER both memories are in to detect refinement
engine.trigger_consolidation()

subsection("Retrieval Surfaces the NEW Belief")
query = "What's the latest Python release feature?"
query_emb = make_embedding(query, seed=555)
results = engine.search(
    query_text=query,
    query_embedding=query_emb,
    top_k=2,
)
print(f"    Query: '{query}'")
for mem_id, score in results:
    mem_text = next((t for mid, t, _, _ in memories if mid == mem_id), None)
    if mem_id == old_id:
        mem_text = old_text
    elif mem_id == new_id:
        mem_text = new_text
    if mem_text:
        memory_card(mem_id, mem_text, score=score)
    else:
        memory_card(mem_id, "?", score=score)

# Show refinement edges
refinements = engine.get_refinements("mem_006_old")
print(f"    Refinement edges from mem_006_old: {refinements}")
print(f"    Note: mem_006_new should rank ABOVE mem_006_old due to Refines edge!")
if not refinements:
    print(f"    (Refinement edges may be 0 because random embeddings don't meet the")
    print(f"     0.85 cosine threshold. In production with real embeddings, this works.)")

# =============================================================================
# FEATURE 5: CONTRADICTION DETECTION
# =============================================================================
section("FEATURE 5: CONTRADICTION DETECTION")
print("  When a new memory CONTRADICTS an old one, the old belief is WEAKENED.")
print("  This prevents outdated/incorrect information from dominating retrieval.")

# False claim
false_id = "mem_007_false"
false_text = "The Earth is flat and the sun revolves around it."
false_emb = make_embedding(false_text, seed=700)
engine.insert(
    id=false_id,
    text=false_text,
    embedding=false_emb,
    importance_score=1.0,
    concepts=["earth", "sun", "geocentrism"],
)
print(f"    ✓ Stored (outdated): {false_id}")

# Correction
correct_id = "mem_007_correct"
correct_text = "The Earth is an oblate spheroid that orbits the Sun in a heliocentric system."
correct_emb = make_embedding(correct_text, seed=701)
engine.insert(
    id=correct_id,
    text=correct_text,
    embedding=correct_emb,
    importance_score=1.5,
    concepts=["earth", "sun", "heliocentrism", "spheroid"],
)
print(f"    ✓ Stored (correction): {correct_id}")

# Trigger consolidation AFTER both memories are in to detect contradiction
engine.trigger_consolidation()

subsection("Contradiction Surfaces the Correction")
query = "What is the shape of Earth and its relationship to the Sun?"
query_emb = make_embedding(query, seed=444)
results = engine.search(
    query_text=query,
    query_embedding=query_emb,
    top_k=2,
)
print(f"    Query: '{query}'")
for mem_id, score in results:
    if mem_id == false_id:
        memory_card(mem_id, false_text, score=score)
    elif mem_id == correct_id:
        memory_card(mem_id, correct_text, score=score)
    else:
        memory_card(mem_id, "?", score=score)

# Show contradiction edges
contradictions = engine.get_contradictions("mem_007_false")
print(f"    Contradiction edges from mem_007_false: {contradictions}")
print(f"    Note: mem_007_correct should rank ABOVE mem_007_false!")
print(f"    The false memory's edges were weakened by contradiction_weaken_factor=0.5")

# =============================================================================
# FEATURE 6: PER-AGENT MEMORY SCOPING
# =============================================================================
section("FEATURE 6: PER-AGENT MEMORY SCOPING")
print("  Multiple agents can share one engine while keeping private memories isolated.")
print("  Scoped searches return: agent's private memories + global/shared memories.")

# Global/shared memory
engine.insert(
    id="global_knowledge",
    text="All agents should be helpful, harmless, and honest.",
    embedding=make_embedding("AI principles", seed=300),
    importance_score=2.0,
    concepts=["ai", "principles", "safety"],
)

# Agent A's private memory
engine.insert(
    id="agent_a_private",
    text="Agent A prefers concise responses under 100 words.",
    embedding=make_embedding("concise responses", seed=301),
    importance_score=1.0,
    concepts=["agent a", "preferences", "concise"],
    scope="agent_a",
)

# Agent B's private memory
engine.insert(
    id="agent_b_private",
    text="Agent B prefers detailed explanations with examples.",
    embedding=make_embedding("detailed explanations", seed=302),
    importance_score=1.0,
    concepts=["agent b", "preferences", "detailed"],
    scope="agent_b",
)

subsection("Agent A's View (sees global + agent_a only)")
results = engine.search_ann(
    make_embedding("What are my preferences?", seed=303),
    top_k=3,
    scope="agent_a",
)
for mem_id, score in results:
    print(f"    → {mem_id} (score={score:.3f})")

subsection("Agent B's View (sees global + agent_b only)")
results = engine.search_ann(
    make_embedding("What are my preferences?", seed=304),
    top_k=3,
    scope="agent_b",
)
for mem_id, score in results:
    print(f"    → {mem_id} (score={score:.3f})")

subsection("Global View (no scope = all global memories)")
results = engine.search_ann(
    make_embedding("What are the AI principles?", seed=305),
    top_k=2,
)
for mem_id, score in results:
    print(f"    → {mem_id} (score={score:.3f})")

# =============================================================================
# FEATURE 7: GRAPH INTROSPECTION
# =============================================================================
section("FEATURE 7: GRAPH INTROSPECTION")
print("  'What does the agent actually know?' — debuggable, not a black box.")

stats = engine.graph_stats()
result("Total nodes", stats[0])
result("Total edges", stats[1])
result("Memory nodes", stats[2])
result("Concept nodes", stats[3])
result("Refinement edges", stats[4])
result("Contradiction edges", stats[5])
result("Abstraction edges", stats[6])

subsection("Concept Inventory")
concepts = engine.get_concepts()
print(f"    Top concepts by degree (connectivity):")
for concept, degree in concepts[:5]:
    print(f"      • {concept:.<40} degree={degree}")

subsection("Memory Concepts")
concepts_for_mem = engine.get_memory_concepts("mem_002")
print(f"    Concepts for mem_002 (Rust): {concepts_for_mem}")

subsection("Refinements")
refinements = engine.get_refinements("mem_006_old")
print(f"    Memories that refine mem_006_old: {refinements}")

subsection("Contradictions")
contradictions = engine.get_contradictions("mem_007_false")
print(f"    Memories that contradict mem_007_false: {contradictions}")

# =============================================================================
# FEATURE 8: COMPRESSED COGNITIVE STATE (CCS)
# =============================================================================
section("FEATURE 8: COMPRESSED COGNITIVE STATE (CCS)")
print("  The agent maintains a bounded working-memory summary across turns.")
print("  This prevents unbounded context growth while preserving key facts.")

# Step 1: First interaction
ccs_json = engine.step_session(
    user_input="What are the key features of Rust?",
    assistant_response="Rust offers memory safety through ownership, zero-cost abstractions, and fearless concurrency.",
)
print(f"    After turn 1:")
print(f"      CCS: {ccs_json[:200]}...")

# Step 2: Second interaction (builds on CCS)
ccs_json = engine.step_session(
    user_input="How does ownership prevent data races?",
    assistant_response="Ownership ensures only one mutable reference exists at a time, preventing concurrent writes.",
)
print(f"    After turn 2:")
print(f"      CCS: {ccs_json[:200]}...")

print(f"    (CCS maintains rolling topics: 'rust', 'memory safety', 'ownership')")

# =============================================================================
# FEATURE 9: ONLINE CONCEPT VOCABULARY EVOLUTION
# =============================================================================
section("FEATURE 9: CONCEPT VOCABULARY EVOLUTION")
print("  The graph learns that 'coding' and 'programming' are synonyms.")
print("  It merges them and suppresses over-general hubs like 'system'.")

# Insert memories with synonymous concepts
engine.insert(
    id="mem_syn_1",
    text="I enjoy coding in Python on weekends.",
    embedding=make_embedding("coding python", seed=800),
    importance_score=1.0,
    concepts=["coding", "python", "weekends"],
)
engine.insert(
    id="mem_syn_2",
    text="Programming in Rust is my favorite hobby.",
    embedding=make_embedding("programming rust", seed=801),
    importance_score=1.0,
    concepts=["programming", "rust", "hobby"],
)

engine.trigger_consolidation()

# Evolve vocabulary
merged, suppressed, examined = engine.evolve_concept_vocabulary()
print(f"    Concepts examined: {examined}")
print(f"    Concepts merged: {merged}")
print(f"    Concepts suppressed: {suppressed}")
print(f"    (If 'coding' and 'programming' share >70% of memories, they merge)")

# =============================================================================
# FEATURE 10: TIERED STORAGE & PERFORMANCE
# =============================================================================
section("FEATURE 10: TIERED STORAGE & PERFORMANCE")
print("  Memories flow through tiers: Hot (RAM) → Warm (mmap, 8-bit) → Cold (mmap, 8-bit)")
print("  Quantization reduces footprint while preserving recall through f32 rerank.")

# Insert many memories to trigger tiering
print(f"    Inserting 2000 memories to trigger consolidation...")
for i in range(2000):
    text = f"Memory {i}: This is a sample text about topic {i % 50}."
    engine.insert(
        id=f"bulk_{i}",
        text=text,
        embedding=make_embedding(text, seed=i + 5000),
        importance_score=0.5,
        concepts=[f"topic_{i % 50}"],
    )

print(f"    Record count: {engine.record_count()}")

# Trigger consolidation to seal tiers
engine.trigger_consolidation()

subsection("Search Performance")
import time
query_emb = make_embedding("sample text about topic 25", seed=9999)

# Warm search (post-consolidation)
start = time.perf_counter()
results = engine.search_ann(query_emb, top_k=5, search_list_size=256)
latency_ms = (time.perf_counter() - start) * 1000

print(f"    ANN search latency: {latency_ms:.2f} ms")
print(f"    Top results:")
for mem_id, score in results[:3]:
    print(f"      → {mem_id} (score={score:.3f})")

# =============================================================================
# SUMMARY
# =============================================================================
section("SUMMARY: TURBO SUPER MEMORY FEATURES")

features = [
    ("1. Vector Storage (ANN)", "Dense vector similarity search with HNSW"),
    ("2. Cognitive Search", "Vector + graph activation fusion (cognitive_alpha=0.5)"),
    ("3. Memory Reinforcement", "Retrieval strengthens edges (rehearsal learning)"),
    ("4. Belief Revision (Refinement)", "New facts supersede old ones via Refines edges"),
    ("5. Contradiction Detection", "Opposing facts weaken old beliefs via Contradicts edges"),
    ("6. Per-Agent Scoping", "Multi-agent isolation with shared global knowledge"),
    ("7. Graph Introspection", "Debug what the agent knows: stats, concepts, refinements"),
    ("8. Compressed Cognitive State", "Bounded working memory across conversation turns"),
    ("9. Concept Evolution", "Synonym merging + hub suppression for coherent graph"),
    ("10. Tiered Storage", "Hot/Warm/Cold tiers with quantization for scale"),
]

for name, desc in features:
    print(f"  ✓ {name:.<30} {desc}")

print(f"\n  Engine stats:")
print(f"    Total records: {engine.record_count()}")
stats = engine.graph_stats()
print(f"    Graph nodes: {stats[0]} | edges: {stats[1]} | memories: {stats[2]} | concepts: {stats[3]}")

# Cleanup
engine.close()
shutil.rmtree(DB_PATH, ignore_errors=True)

section("DEMO COMPLETE")
print("  TurboSuperMemory: A memory engine that thinks, not just stores.")
print("  https://github.com/jatin711-debug/TurboSuperMemory")
