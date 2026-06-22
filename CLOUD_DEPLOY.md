# Cloud Deployment Guide for TurboSuperMemory Benchmarks

This guide helps you deploy TSM benchmarks on cloud GPU instances.

## Recommended Cloud Providers

| Provider | Instance | VRAM | Cost/hr | Best For |
|----------|----------|------|---------|----------|
| **RunPod** | RTX 3090 | 24GB | ~$0.50 | Best value |
| **Lambda Labs** | A10 | 24GB | ~$0.80 | Reliable |
| **Vast.ai** | RTX 3090 | 24GB | ~$0.40 | Cheapest |
| **TensorDock** | RTX 3090 | 24GB | ~$0.45 | Flexible |

## Quick Start

### Option 1: Using setup.sh (Recommended)

```bash
# 1. SSH into your cloud instance
ssh user@your-instance-ip

# 2. Clone the repo
git clone https://github.com/YOUR_USERNAME/TurboSuperMemory.git
cd TurboSuperMemory

# 3. Run setup (takes ~10-15 minutes)
sudo bash setup.sh

# 4. Run benchmarks
./run_quick_test.sh
```

### Option 2: Using cloud-init (AWS/GCP/Azure)

```bash
# Launch instance with cloud-config.yaml as user-data
curl -X POST "https://api.cloud-provider.com/launch" \
  -d "user_data=$(cat cloud-config.yaml | base64)"

# Wait 10-15 minutes for setup, then SSH
ssh tsm@your-instance-ip

# Benchmarks are ready to run!
./run_longmemeval_full.sh
```

### Option 3: Docker (Coming Soon)

```bash
docker run -it --gpus all turbosupermemory/benchmarks:latest
```

## Benchmark Scripts

After setup, these scripts are available:

| Script | Time | Description |
|--------|------|-------------|
| `./run_quick_test.sh` | ~2 min | Quick validation (5 conversations + comparison) |
| `./run_longmemeval_full.sh` | ~6 min | Full LongMemEval (500 conversations) |
| `./run_locomo.sh` | ~10 min | LoCoMo sampled (100 queries) |
| `./run_recall_audit.sh` | ~5 min | Recall audit (100K vectors) |

## Manual Commands

```bash
# Activate environment
source .venv/bin/activate

# Quick test
python benchmarks/cognitive_eval/run_longmemeval.py \
    --quick --quick-n 5 --lightweight --batch-size 64

# Full benchmark with BGE-Large
python benchmarks/cognitive_eval/run_longmemeval.py \
    --embedding-model BAAI/bge-large-en-v1.5 --batch-size 32

# Compare ANN vs Cognitive
python benchmarks/cognitive_eval/run_longmemeval.py \
    --quick --quick-n 3 --lightweight --compare-cognitive

# Full LoCoMo (warning: takes hours)
python benchmarks/cognitive_eval/run_locomo.py \
    --lightweight --batch-size 64
```

## Expected Results (RTX 3090)

| Benchmark | Time | Recall@10 |
|-----------|------|-----------|
| LongMemEval quick (5 conv) | ~1 min | 100% |
| LongMemEval full (500 conv) | ~6 min | 95-100% |
| LoCoMo quick (100 queries) | ~10 min | TBD |
| Recall audit (100K vectors) | ~5 min | 100% |

## Troubleshooting

### CUDA not found
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### TSM build fails
```bash
# Make sure Python 3.12 is active
source .venv/bin/activate
export PYO3_PYTHON=$(which python)

# Rebuild
cargo clean
cargo build --workspace --release --features cuda
```

### Out of memory
```bash
# Use smaller batch size
python benchmarks/cognitive_eval/run_longmemeval.py \
    --lightweight --batch-size 32

# Or use CPU
python benchmarks/cognitive_eval/run_longmemeval.py \
    --lightweight --batch-size 16
```

### Model download fails
```bash
# Set HuggingFace token for higher rate limits
export HF_TOKEN=your_token_here

# Or download manually
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2
```

## Cost Estimates

| Task | Time | Cost (RTX 3090) |
|------|------|-----------------|
| Setup + Quick test | 15 min | ~$0.15 |
| Full LongMemEval | 6 min | ~$0.05 |
| Full LoCoMo | 4 hours | ~$2.00 |
| Complete benchmark suite | 5 hours | ~$2.50 |

## Next Steps

After running benchmarks:

1. Check results in `results/` directory
2. Compare with Mem0 claims
3. Tune hyperparameters if needed
4. Scale to 1M+ vectors

## Support

- GitHub Issues: https://github.com/YOUR_USERNAME/TurboSuperMemory/issues
- Benchmark README: `benchmarks/cognitive_eval/README.md`
