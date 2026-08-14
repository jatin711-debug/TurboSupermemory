import os
import sys
import shutil
import time
import numpy as np
from openai import OpenAI

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

# Load OpenAI key
key_path = os.path.join(root_dir, "openai_key.txt")
if not os.path.exists(key_path):
    print("Error: openai_key.txt not found!")
    sys.exit(1)

with open(key_path, "r", encoding="utf-8") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)
import turbomemory

DIMENSION = 1536  # Native dimension for text-embedding-3-small
DB_PATH = os.path.join(root_dir, "test_openai_embed_db")

if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)

print("=" * 90)
print(" LIVE TURBOSUPERMEMORY TEST WITH REAL 'openai/text-embedding-3-small' EMBEDDINGS")
print("=" * 90)

# Initialize TSM engine for 1536-dim OpenAI embeddings
engine = turbomemory.MemoryEngine(
    db_path=DB_PATH,
    dimension=DIMENSION,
    cognitive_alpha=0.5,
    refinement_cosine_threshold=0.75,
    contradiction_cosine_threshold=0.65,
    contradiction_require_opposition=True,
    exclude_superseded=False,
    importance_auto_scoring=True,
    max_concepts=6,
    concept_max_ngram_len=2,
    auto_consolidation_secs=0,
)
print(f"[OK] Initialized TurboSuperMemory Engine with dim={DIMENSION}")

def get_embedding(text: str) -> np.ndarray:
    """Generate real normalized 1536-dim embedding using text-embedding-3-small."""
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    vec = np.array(res.data[0].embedding, dtype=np.float32)
    vec /= np.linalg.norm(vec)
    return vec

# Realistic multi-turn conversation stream with real natural language
conversation_stream = [
    # Turn 1: Initial Profile & Location
    {
        "id": "mem_location_v1",
        "text": "I live in Seattle, Washington and work as a backend engineer.",
        "concepts": ["user_location", "seattle", "backend_job"],
    },
    # Turn 2: Database choice
    {
        "id": "mem_database_v1",
        "text": "Our production backend relies heavily on PostgreSQL for storing user profiles.",
        "concepts": ["database", "postgresql", "backend"],
    },
    # Turn 3: Editor preference
    {
        "id": "mem_editor_v1",
        "text": "I write all my code in Vim and prefer dark mode theme.",
        "concepts": ["editor_setup", "vim", "preferences"],
    },
    # Turn 4: Project launch deadline
    {
        "id": "mem_apollo_v1",
        "text": "Project Apollo backend launch is scheduled for next Friday at 5 PM.",
        "concepts": ["project_apollo", "launch_date", "deadline"],
    },
    # Turn 5: Coexisting languages
    {
        "id": "mem_languages",
        "text": "I primarily code in Python, and I also started learning Rust for high performance systems.",
        "concepts": ["programming", "python", "rust"],
    },
    # Turn 6: Distractor turns
    {
        "id": "mem_distractor_1",
        "text": "We had a team sync meeting about quarterly OKRs and budget allocation.",
        "concepts": ["meeting", "okrs", "team"],
    },
    {
        "id": "mem_distractor_2",
        "text": "The office coffee machine was replaced with a new espresso maker on the 3rd floor.",
        "concepts": ["office", "coffee", "amenities"],
    },
    # Turn 7: UPDATE / CONTRADICTION 1 (Location)
    {
        "id": "mem_location_v2",
        "text": "I actually moved out of Seattle last week and no longer live there; instead I now live in Tokyo, Japan.",
        "concepts": ["user_location", "tokyo", "moved"],
    },
    # Turn 8: UPDATE / CONTRADICTION 2 (Database)
    {
        "id": "mem_database_v2",
        "text": "We completely abandoned PostgreSQL and are no longer using it; instead we migrated strictly to SQLite and DuckDB.",
        "concepts": ["database", "sqlite", "duckdb"],
    },
    # Turn 9: UPDATE / CONTRADICTION 3 (Editor)
    {
        "id": "mem_editor_v2",
        "text": "I stopped using Vim; instead I now use standard VS Code keybindings.",
        "concepts": ["editor_setup", "vscode", "preferences"],
    },
    # Turn 10: UPDATE / REFINEMENT (Project Apollo Deadline)
    {
        "id": "mem_apollo_v2",
        "text": "Project Apollo launch was rescheduled and extended to next month on the 15th.",
        "concepts": ["project_apollo", "launch_date", "deadline"],
    },
]

print("\n[PHASE 1]: Generating Real OpenAI Embeddings and Ingesting into TSM...")
t0 = time.time()
for item in conversation_stream:
    t_emb_0 = time.time()
    vec = get_embedding(item["text"])
    t_emb = time.time() - t_emb_0
    
    t_ins_0 = time.time()
    engine.insert(
        id=item["id"],
        text=item["text"],
        embedding=vec,
        importance_score=1.0,
        concepts=item["concepts"],
        scope="user_alex",
    )
    t_ins = time.time() - t_ins_0
    print(f"  • Inserted '{item['id']}': OpenAI Embed={t_emb*1000:.1f}ms | TSM Rust Write={t_ins*1000:.3f}ms")

t_total_ingest = time.time() - t0
print(f"[OK] Ingested all memories in {t_total_ingest:.2f}s.")

# Trigger Background Consolidation
print("\n[PHASE 2]: Triggering TSM Engine Consolidation...")
t0 = time.time()
engine.trigger_consolidation()
t_cons = time.time() - t0

stats = engine.graph_stats()
print(f"[OK] Consolidation completed in {t_cons*1000:.2f} ms.")
print(f"  • Live Graph Nodes: {stats[0]}")
print(f"  • Live Graph Edges: {stats[1]}")
print(f"  • Detected Refinements: {stats[4]}")
print(f"  • Detected Contradictions: {stats[5]}")

print("\nContradiction Map:")
for stale_id in ["mem_location_v1", "mem_database_v1", "mem_editor_v1"]:
    corr = engine.get_contradictions(stale_id)
    print(f"  - '{stale_id}' is contradicted by -> {corr}")

print("\nRefinement Map:")
for old_id in ["mem_apollo_v1"]:
    ref = engine.get_refinements(old_id)
    print(f"  - '{old_id}' is refined by -> {ref}")

# =============================================================================
# EVALUATION WITH REAL QUERY EMBEDDINGS
# =============================================================================
print("\n" + "=" * 90)
print(" [PHASE 3]: EXECUTING REAL-TIME QUERIES WITH OPENAI EMBEDDINGS")
print("=" * 90)

evaluation_queries = [
    {
        "query_text": "Where does the user currently live?",
        "expected_active_id": "mem_location_v2",
        "stale_id": "mem_location_v1",
        "desc": "Location Update (Seattle -> Tokyo)",
    },
    {
        "query_text": "What database does the user use for production?",
        "expected_active_id": "mem_database_v2",
        "stale_id": "mem_database_v1",
        "desc": "Database Migration (Postgres -> SQLite/DuckDB)",
    },
    {
        "query_text": "What editor keybindings does the user use?",
        "expected_active_id": "mem_editor_v2",
        "stale_id": "mem_editor_v1",
        "desc": "Editor Preference (Vim -> VS Code)",
    },
    {
        "query_text": "When is Project Apollo scheduled to launch?",
        "expected_active_id": "mem_apollo_v2",
        "stale_id": "mem_apollo_v1",
        "desc": "Launch Deadline Extension (Friday -> Next month 15th)",
    },
    {
        "query_text": "What programming languages does the user code in?",
        "expected_active_id": "mem_languages",
        "stale_id": "none",
        "desc": "Coexisting Languages (Python + Rust)",
    },
]

for q in evaluation_queries:
    print(f"\n>>> QUERY: \"{q['query_text']}\" [{q['desc']}]")
    print("  " + "-" * 75)
    
    # Embed the query with text-embedding-3-small
    q_vec = get_embedding(q["query_text"])
    
    # 1. Plain Vector Search (Cosine with OpenAI embeddings)
    ann = engine.search_ann(q_vec, top_k=2, scope="user_alex")
    print("  [Plain Vector DB (Raw OpenAI Cosine)]:")
    for r, (doc_id, score) in enumerate(ann, 1):
        txt = engine.get_text(doc_id)
        tag = " (STALE FACT)" if doc_id == q["stale_id"] else ""
        print(f"    Rank #{r}: [{doc_id}] (Cosine={score:.4f}){tag} -> {txt[:75]}...")

    # 2. TSM Cognitive Search (OpenAI Cosine + Saturating Graph Delta + Demotion)
    cog = engine.search(q["query_text"], q_vec, top_k=2, scope="user_alex")
    print("\n  [TurboSuperMemory (Cognitive Layer + Belief Revision)]:")
    for r, (doc_id, score) in enumerate(cog, 1):
        txt = engine.get_text(doc_id)
        win = " [WIN: Correct Active Fact!]" if doc_id == q["expected_active_id"] and r == 1 else ""
        demoted = " [DEMOTED]" if doc_id == q["stale_id"] else ""
        print(f"    Rank #{r}: [{doc_id}] (Fused Score={score:.4f}){win}{demoted} -> {txt[:75]}...")

engine.close()
print("\n" + "=" * 90)
print(" OPENAI EMBEDDINGS EVALUATION SUCCESSFULLY COMPLETED")
print("=" * 90)
