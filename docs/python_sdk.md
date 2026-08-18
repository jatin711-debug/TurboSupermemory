# Python SDK Reference (`tsm` & `turbomemory`)

TurboSuperMemory provides two layers of Python interfaces:
1. **High-Level Agent Framework Layer (`tsm`)**: Clean, turnkey API with automatic concept extraction, local/OpenAI embedders, and Stage-2 ColBERT late-interaction reranking.
2. **Low-Level Native Engine Layer (`turbomemory`)**: Direct PyO3 C-extension bindings with zero-copy NumPy buffers and sub-millisecond bare-metal control.

---

## 1. High-Level Agent Interface (`from tsm import Memory`)

The `Memory` class is the recommended entry point for AI agents, multi-agent frameworks, and chatbot backends.

### Quickstart

```python
from tsm import Memory

# 1. Initialize Memory with Stage-2 ColBERT Reranking
memory = Memory(
    db_path="./agent_memory_db",
    embedder="sentence_transformer",  # Uses local 'all-MiniLM-L6-v2' on CPU/GPU ($0.00 cost)
    reranker="colbert",               # Uses 'LiquidAI/LFM2.5-ColBERT-350M' on CUDA
)

# 2. Add memories (accepts string, dict, or list of dicts)
memory.add("User's production telemetry port is 9443 on host alpha.prod.internal", user_id="alice")
memory.add("User updated database password policy to 16 characters minimum", user_id="alice")

# 3. Recall with cognitive fusion + ColBERT MaxSim reranking
results = memory.recall("What is the telemetry port?", user_id="alice", top_k=3)

for r in results:
    print(f"Memory: {r['text']} (Score: {r['score']:.4f})")
```

### Supported Embedders & Rerankers

```python
from tsm import Memory, SentenceTransformerEmbedder, ColBertReranker

# Local open-source embedder + ColBERT on CUDA
memory = Memory(
    db_path="./my_db",
    embedder=SentenceTransformerEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda"),
    reranker=ColBertReranker(model_name="LiquidAI/LFM2.5-ColBERT-350M", device="cuda"),
)
```

---

## 2. Low-Level Native Engine (`import turbomemory`)

Direct PyO3 bindings for high-throughput vector ingestion, 3-tier TurboQuant hardware compression, and lock-free snapshot search.

### 3-Tier TurboQuant Configuration

```python
import numpy as np
import turbomemory

dim = 512  # Must be a power of 2 for Fast Walsh-Hadamard Transform (128, 256, 512, 1024)

engine = turbomemory.MemoryEngine(
    db_path="./turbo_db",
    dimension=dim,
    hot_capacity=1000,              # First 1,000 vectors stay in RAM (FP32)
    warm_capacity=10000,            # Next 10,000 vectors in TurboQuant-Prod (8-bit)
    warm_quantizer="turbo_prod8",   # 8-bit FWHT + QJL residual (3.6x compression)
    cold_quantizer="turbo_mse1",    # 1-bit FWHT sign quantization (32x compression)
    auto_consolidation_secs=60,
    outlier_count=0,
)

# Insert with zero-copy NumPy array
vector = np.random.randn(dim).astype(np.float32)
engine.insert(
    id="mem_001",
    text="Deployment configuration notes",
    embedding=vector,
    importance_score=1.0,
    concepts=["deployment", "config"],
    scope="team_alpha",
)

# Search across all tiers simultaneously
query_vec = np.random.randn(dim).astype(np.float32)
hits = engine.search(
    query_text="deployment notes",
    query_embedding=query_vec,
    top_k=5,
    scope="team_alpha",
)

# Check graph introspection and GPU status
print(f"GPU Accelerated: {engine.gpu_accelerated}")
print(f"Graph Stats: {engine.graph_stats()}")

engine.flush()
engine.close()
```
