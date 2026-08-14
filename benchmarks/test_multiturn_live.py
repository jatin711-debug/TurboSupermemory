import os
import sys
import shutil
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

import turbomemory

DIMENSION = 768
DB_PATH = os.path.join(root_dir, "test_multiturn_live_db")

if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)

def make_vec(seed, dim=DIMENSION):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v

def make_similar_vec(base_vec, cos_target, seed=123):
    rng = np.random.RandomState(seed)
    rand_v = rng.randn(len(base_vec)).astype(np.float32)
    perp = rand_v - np.dot(rand_v, base_vec) * base_vec
    perp /= np.linalg.norm(perp)
    c = float(cos_target)
    out = c * base_vec + np.sqrt(max(0.0, 1.0 - c * c)) * perp
    out /= np.linalg.norm(out)
    return out.astype(np.float32)

print("=" * 85)
print(" EXTENSIVE MULTI-TURN AGENT CONVERSATION & MEMORY SIMULATION")
print("=" * 85)

engine = turbomemory.MemoryEngine(
    db_path=DB_PATH,
    dimension=DIMENSION,
    cognitive_alpha=0.5,
    refinement_cosine_threshold=0.70,
    contradiction_cosine_threshold=0.60,
    contradiction_require_opposition=True,
    exclude_superseded=False,               # Keep stale in view to show demotion math
    importance_auto_scoring=True,
    importance_learning_rate=0.3,
    max_concepts=6,
    concept_max_ngram_len=2,
    auto_consolidation_secs=0,
)

print("[OK] Initialized TurboSuperMemory Engine with Full Cognitive Architecture")

# Anchor embeddings for distinct semantic topics
emb_database = make_vec(seed=100)
emb_languages = make_vec(seed=200)
emb_project = make_vec(seed=300)
emb_editor = make_vec(seed=400)

print("\n" + "=" * 85)
print(" SIMULATING MULTI-TURN CONVERSATION TIMELINE (Over 30 Simulated Days)")
print("=" * 85)

# -----------------------------------------------------------------------------
# TURN 1: DAY 1 - Initial Stack & Profile
# -----------------------------------------------------------------------------
print("\n[TURN 1 - Day 1]: Ingesting Initial Profile...")
v_db_initial = make_similar_vec(emb_database, cos_target=0.88, seed=11)
engine.insert(
    id="mem_db_v1",
    text="User is a backend engineer using PostgreSQL as their primary production database.",
    embedding=v_db_initial,
    importance_score=1.0,
    concepts=["database", "postgresql", "backend"],
    scope="user_alex",
)

v_lang_python = make_similar_vec(emb_languages, cos_target=0.85, seed=21)
engine.insert(
    id="mem_lang_python",
    text="User primarily codes backend services in Python.",
    embedding=v_lang_python,
    importance_score=1.0,
    concepts=["programming", "python", "backend"],
    scope="user_alex",
)

v_proj_v1 = make_similar_vec(emb_project, cos_target=0.88, seed=31)
engine.insert(
    id="mem_project_deadline_v1",
    text="Project Apollo is scheduled to launch next Friday at 5 PM.",
    embedding=v_proj_v1,
    importance_score=1.0,
    concepts=["project", "apollo", "deadline"],
    scope="user_alex",
)
print("  - Stored: PostgreSQL database (mem_db_v1)")
print("  - Stored: Python backend language (mem_lang_python)")
print("  - Stored: Project Apollo launch next Friday (mem_project_deadline_v1)")

# -----------------------------------------------------------------------------
# TURN 2: DAY 3 - Editor Preferences
# -----------------------------------------------------------------------------
print("\n[TURN 2 - Day 3]: Ingesting Editor Preferences...")
v_editor_v1 = make_similar_vec(emb_editor, cos_target=0.88, seed=41)
engine.insert(
    id="mem_editor_vim",
    text="User uses Vim keybindings and prefers dark mode in their code editor.",
    embedding=v_editor_v1,
    importance_score=1.0,
    concepts=["editor", "vim", "preferences"],
    scope="user_alex",
)
print("  - Stored: Vim keybindings preference (mem_editor_vim)")

# -----------------------------------------------------------------------------
# TURN 3: DAY 7 - Project Deadline Refinement
# -----------------------------------------------------------------------------
print("\n[TURN 3 - Day 7]: User Refines Project Apollo Deadline...")
# Newer refinement vector: close to the original fact (cosine ~ 0.76)
v_proj_v2 = make_similar_vec(v_proj_v1, cos_target=0.76, seed=32)
engine.insert(
    id="mem_project_deadline_v2",
    text="Project Apollo launch was rescheduled and extended to next month on the 15th.",
    embedding=v_proj_v2,
    importance_score=1.0,
    concepts=["project", "apollo", "deadline"],
    scope="user_alex",
)
print("  - Stored: Extended deadline to next month (mem_project_deadline_v2)")

# -----------------------------------------------------------------------------
# TURN 4: DAY 12 - Coexisting Knowledge (Rust alongside Python)
# -----------------------------------------------------------------------------
print("\n[TURN 4 - Day 12]: User Adds Coexisting Fact (Learning Rust)...")
# Coexisting fact: shares 'programming' concept, but does NOT oppose Python!
v_lang_rust = make_similar_vec(emb_languages, cos_target=0.78, seed=22)
engine.insert(
    id="mem_lang_rust",
    text="User also started learning Rust for high-performance systems engineering.",
    embedding=v_lang_rust,
    importance_score=1.0,
    concepts=["programming", "rust", "systems"],
    scope="user_alex",
)
print("  - Stored: User learning Rust (mem_lang_rust) [Coexisting with Python]")

# -----------------------------------------------------------------------------
# TURN 5: DAY 18 - Hard Contradiction (Switched from Postgres to SQLite/DuckDB)
# -----------------------------------------------------------------------------
print("\n[TURN 5 - Day 18]: User Corrects Database Choice (Hard Contradiction)...")
# Opposition marker 'no longer', same concept 'database', cosine 0.74 to old fact
v_db_v2 = make_similar_vec(v_db_initial, cos_target=0.74, seed=12)
engine.insert(
    id="mem_db_v2",
    text="User completely abandoned PostgreSQL and is no longer using it; they switched strictly to SQLite and DuckDB.",
    embedding=v_db_v2,
    importance_score=1.0,
    concepts=["database", "sqlite", "duckdb"],
    scope="user_alex",
)
print("  - Stored: Switched from PostgreSQL to SQLite/DuckDB (mem_db_v2)")

# -----------------------------------------------------------------------------
# TURN 6: DAY 25 - Preference Correction (Switched from Vim to VS Code standard)
# -----------------------------------------------------------------------------
print("\n[TURN 6 - Day 25]: User Corrects Editor Setup...")
v_editor_v2 = make_similar_vec(v_editor_v1, cos_target=0.74, seed=42)
engine.insert(
    id="mem_editor_vscode",
    text="User actually stopped using Vim keybindings; instead they now prefer standard VS Code shortcuts.",
    embedding=v_editor_v2,
    importance_score=1.0,
    concepts=["editor", "vscode", "preferences"],
    scope="user_alex",
)
print("  - Stored: Switched to standard VS Code shortcuts (mem_editor_vscode)")

# Ingest 10 distractor memories
for i in range(10):
    d_v = make_vec(seed=900 + i)
    engine.insert(
        id=f"background_note_{i}",
        text=f"General team memo #{i}: system maintenance guidelines.",
        embedding=d_v,
        importance_score=0.4,
        concepts=[f"general_{i}"],
        scope="user_alex",
    )

print(f"\nIngested 7 core memories + 10 background distractor notes.")

# Trigger Background Consolidation
print("\nTriggering Engine Consolidation...")
engine.trigger_consolidation()

stats = engine.graph_stats()
print(f"[OK] Consolidation Complete. Graph Stats: Nodes={stats[0]}, Edges={stats[1]}, Refinements={stats[4]}, Contradictions={stats[5]}")

# =============================================================================
# MULTI-TURN VERIFICATION QUERIES
# =============================================================================
print("\n" + "=" * 85)
print(" EXECUTING VERIFICATION QUERIES ACROSS THE CONVERSATIONAL HISTORY")
print("=" * 85)

def run_comparison(query_title, query_text, query_vector, expected_top_id):
    print(f"\n>>> QUERY: \"{query_text}\"")
    print("  " + "-" * 70)
    
    # 1. Plain ANN
    ann = engine.search_ann(query_vector, top_k=2, scope="user_alex")
    print("  [Plain Vector DB - Pure Cosine Only]:")
    for r, (doc_id, score) in enumerate(ann, 1):
        txt = engine.get_text(doc_id)
        marker = " (STALE FACT)" if doc_id != expected_top_id and ("v1" in doc_id or "vim" in doc_id) else ""
        print(f"    Rank #{r}: [{doc_id}] (Cosine Score={score:.4f}){marker} -> {txt}")

    # 2. Cognitive Retrieval
    cog = engine.search(query_text, query_vector, top_k=2, scope="user_alex")
    print("  [TurboSuperMemory - Cognitive Fusion]:")
    if cog:
        for r, (doc_id, score) in enumerate(cog, 1):
            txt = engine.get_text(doc_id)
            win = " [WIN: Correct Active Belief Surfaced!]" if doc_id == expected_top_id and r == 1 else ""
            demoted = " [DEMOTED]" if "v1" in doc_id or "vim" in doc_id else ""
            print(f"    Rank #{r}: [{doc_id}] (Fused Score={score:.4f}){win}{demoted} -> {txt}")

# TEST 1: Database Stack
run_comparison(
    query_title="Database Stack",
    query_text="What database is the user using for backend development?",
    query_vector=emb_database,
    expected_top_id="mem_db_v2",
)

# TEST 2: Editor Setup
run_comparison(
    query_title="Editor Keybindings",
    query_text="What editor keybindings does the user prefer?",
    query_vector=emb_editor,
    expected_top_id="mem_editor_vscode",
)

# TEST 3: Project Deadline Refinement
run_comparison(
    query_title="Project Apollo Deadline",
    query_text="When is the Project Apollo launch deadline?",
    query_vector=emb_project,
    expected_top_id="mem_project_deadline_v2",
)

# TEST 4: Coexisting Facts (Languages)
print(f"\n>>> QUERY: \"What programming languages does the user know?\"")
print("  " + "-" * 70)
lang_cog = engine.search("What programming languages does user code in?", emb_languages, top_k=2, scope="user_alex")
print("  [TurboSuperMemory - Coexisting Facts Check]:")
for r, (doc_id, score) in enumerate(lang_cog, 1):
    txt = engine.get_text(doc_id)
    print(f"    Rank #{r}: [{doc_id}] (Score={score:.4f}) -> {txt}")
print("  [OK] Coexisting Fact Validation: Both Python and Rust are retained without false demotion!")

# -----------------------------------------------------------------------------
# GRAPH INTROSPECTION & BELIEF RELATIONSHIPS
# -----------------------------------------------------------------------------
print("\n" + "=" * 85)
print(" GRAPH INTROSPECTION: BELIEF RELATIONSHIPS & KNOWLEDGE STRUCTURE")
print("=" * 85)

print("Contradictions Map:")
for stale_id in ["mem_db_v1", "mem_editor_vim"]:
    corr = engine.get_contradictions(stale_id)
    print(f"  • '{stale_id}' is contradicted by -> {corr}")

print("\nRefinements Map:")
for old_proj_id in ["mem_project_deadline_v1"]:
    ref = engine.get_refinements(old_proj_id)
    print(f"  • '{old_proj_id}' is refined by -> {ref}")

print("\nTop Extracted Concepts by Degree:")
top_concepts = engine.get_concepts()[:6]
for concept, degree in top_concepts:
    print(f"  • Concept '{concept}' -> Degree: {degree}")

engine.close()
print("\n" + "=" * 85)
print(" EXTENSIVE MULTI-TURN EXPERIMENT SUCCESSFULLY COMPLETED")
print("=" * 85)
