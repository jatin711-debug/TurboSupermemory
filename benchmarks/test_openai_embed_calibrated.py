import os
import sys
import shutil
import time
import numpy as np
from openai import OpenAI

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

key_path = os.path.join(root_dir, "openai_key.txt")
with open(key_path, "r", encoding="utf-8") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)
import turbomemory

DIMENSION = 1536
DB_PATH = os.path.join(root_dir, "test_openai_embed_calibrated_db")

if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)

def get_embedding(text: str) -> np.ndarray:
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    vec = np.array(res.data[0].embedding, dtype=np.float32)
    vec /= np.linalg.norm(vec)
    return vec

print("=" * 90)
print(" CALIBRATED TEST: TSM WITH REAL 'openai/text-embedding-3-small' EMBEDDINGS")
print("=" * 90)

# Calibrated thresholds for OpenAI's embedding distribution (mean cosine between sentences is ~0.45-0.55)
engine = turbomemory.MemoryEngine(
    db_path=DB_PATH,
    dimension=DIMENSION,
    cognitive_alpha=0.5,
    refinement_cosine_threshold=0.50,
    contradiction_cosine_threshold=0.40,
    contradiction_require_opposition=True,
    exclude_superseded=False,
    importance_auto_scoring=True,
    max_concepts=6,
    concept_max_ngram_len=2,
    auto_consolidation_secs=0,
)
print(f"[OK] Initialized TSM Engine (dim=1536, contradiction_thresh=0.40, refinement_thresh=0.50)")

conversation_stream = [
    {
        "id": "mem_location_v1",
        "text": "I live in Seattle, Washington and work as a backend engineer.",
        "concepts": ["user_location", "seattle", "backend_job"],
    },
    {
        "id": "mem_database_v1",
        "text": "Our production backend relies heavily on PostgreSQL for storing user profiles.",
        "concepts": ["database", "postgresql", "backend"],
    },
    {
        "id": "mem_editor_v1",
        "text": "I write all my code in Vim and prefer dark mode theme.",
        "concepts": ["editor_setup", "vim", "preferences"],
    },
    {
        "id": "mem_apollo_v1",
        "text": "Project Apollo backend launch is scheduled for next Friday at 5 PM.",
        "concepts": ["project_apollo", "launch_date", "deadline"],
    },
    {
        "id": "mem_languages",
        "text": "I primarily code in Python, and I also started learning Rust for high performance systems.",
        "concepts": ["programming", "python", "rust"],
    },
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
    {
        "id": "mem_location_v2",
        "text": "I actually moved out of Seattle last week and no longer live there; instead I now live in Tokyo, Japan.",
        "concepts": ["user_location", "tokyo", "moved"],
    },
    {
        "id": "mem_database_v2",
        "text": "We completely abandoned PostgreSQL and are no longer using it; instead we migrated strictly to SQLite and DuckDB.",
        "concepts": ["database", "sqlite", "duckdb"],
    },
    {
        "id": "mem_editor_v2",
        "text": "I stopped using Vim; instead I now use standard VS Code keybindings.",
        "concepts": ["editor_setup", "vscode", "preferences"],
    },
    {
        "id": "mem_apollo_v2",
        "text": "Project Apollo launch was rescheduled and extended to next month on the 15th.",
        "concepts": ["project_apollo", "launch_date", "deadline"],
    },
]

print("\nIngesting real OpenAI embeddings into TSM...")
for item in conversation_stream:
    vec = get_embedding(item["text"])
    engine.insert(
        id=item["id"],
        text=item["text"],
        embedding=vec,
        importance_score=1.0,
        concepts=item["concepts"],
        scope="user_alex",
    )

print("Ingestion complete. Triggering Consolidation...")
engine.trigger_consolidation()

stats = engine.graph_stats()
print(f"[OK] Graph Consolidation Stats:")
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
print(" EXECUTING REAL-TIME QUERIES WITH OPENAI EMBEDDINGS")
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
    
    q_vec = get_embedding(q["query_text"])
    
    # 1. Plain Vector Search
    ann = engine.search_ann(q_vec, top_k=2, scope="user_alex")
    print("  [Plain Vector DB (Raw OpenAI Cosine)]:")
    for r, (doc_id, score) in enumerate(ann, 1):
        txt = engine.get_text(doc_id)
        tag = " (STALE FACT)" if doc_id == q["stale_id"] else ""
        print(f"    Rank #{r}: [{doc_id}] (Cosine={score:.4f}){tag} -> {txt[:70]}...")

    # 2. TSM Cognitive Search
    cog = engine.search(q["query_text"], q_vec, top_k=2, scope="user_alex")
    print("  [TurboSuperMemory (Cognitive Layer + Belief Revision)]:")
    for r, (doc_id, score) in enumerate(cog, 1):
        txt = engine.get_text(doc_id)
        win = " [WIN: Active Belief Surfaced!]" if doc_id == q["expected_active_id"] and r == 1 else ""
        demoted = " [DEMOTED]" if doc_id == q["stale_id"] else ""
        print(f"    Rank #{r}: [{doc_id}] (Fused Score={score:.4f}){win}{demoted} -> {txt[:70]}...")

engine.close()
print("\n" + "=" * 90)
print(" TEST COMPLETE")
print("=" * 90)
