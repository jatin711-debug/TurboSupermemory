#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# TurboSuperMemory Cloud Setup Script
# =============================================================================
# This script prepares a Linux environment for running TSM benchmarks
# on cloud GPU instances (RunPod, Lambda Labs, Vast.ai, etc.)
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# Requirements:
#   - Ubuntu 22.04+ or 24.04 (Noble) - or compatible Debian-based distro
#   - NVIDIA GPU with CUDA 12.x
#   - 16GB+ VRAM for full benchmarks (8GB+ for quick tests)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="turbosupermemory"
PYTHON_VERSION="3.12"
CUDA_VERSION="12.8"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# 1. SYSTEM UPDATE & BASE DEPENDENCIES
# =============================================================================
setup_system() {
    log_info "Updating system packages..."
    apt-get update && apt-get upgrade -y
    
    log_info "Installing base dependencies..."
    apt-get install -y \
        build-essential \
        curl \
        wget \
        git \
        git-lfs \
        cmake \
        pkg-config \
        libssl-dev \
        libffi-dev \
        protobuf-compiler \
        python3-dev \
        python3-pip \
        python3-venv \
        unzip \
        jq \
        htop \
        nvtop \
        tmux \
        vim \
        tree \
        parallel
    
    log_success "System dependencies installed"
}

# =============================================================================
# 2. CUDA SETUP (if not already present)
# =============================================================================
setup_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        log_info "CUDA already installed:"
        nvidia-smi
        return 0
    fi
    
    log_info "Installing CUDA ${CUDA_VERSION}..."
    
    # Detect Ubuntu version for correct CUDA repo
    UBUNTU_CODENAME=$(lsb_release -cs 2>/dev/null || echo "ubuntu2204")
    if [[ "$UBUNTU_CODENAME" == "noble" ]]; then
        UBUNTU_REPO="ubuntu2404"
    else
        UBUNTU_REPO="ubuntu2204"
    fi
    
    wget https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_REPO}/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update
    apt-get install -y cuda-toolkit-${CUDA_VERSION//./-}
    
    # Add CUDA to PATH
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    
    log_success "CUDA ${CUDA_VERSION} installed"
    nvidia-smi
}

# =============================================================================
# 3. RUST SETUP
# =============================================================================
setup_rust() {
    if command -v rustc &> /dev/null; then
        log_info "Rust already installed: $(rustc --version)"
        return 0
    fi
    
    log_info "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    
    # Add target for optimization
    rustup target add x86_64-unknown-linux-gnu
    
    log_success "Rust installed: $(rustc --version)"
}

# =============================================================================
# 4. PYTHON SETUP (3.12)
# =============================================================================
setup_python() {
    log_info "Setting up Python ${PYTHON_VERSION}..."
    
    # Add deadsnakes PPA for Python 3.12
    apt-get install -y software-properties-common
    add-apt-repository ppa:deadsnakes/ppa -y
    apt-get update
    
    # Install Python 3.12 (distutils is deprecated in 24.04, use venv instead)
    apt-get install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv
    
    # Create virtual environment
    VENV_PATH="${SCRIPT_DIR}/.venv"
    python${PYTHON_VERSION} -m venv "${VENV_PATH}"
    source "${VENV_PATH}/bin/activate"
    
    # Upgrade pip (handle system-managed packages on Ubuntu 24.04)
    pip install --upgrade pip setuptools wheel --break-system-packages 2>/dev/null || \
    pip install --upgrade pip setuptools wheel --user 2>/dev/null || \
    log_warn "Could not upgrade pip - using system version"
    
    log_success "Python ${PYTHON_VERSION} virtual environment created at ${VENV_PATH}"
}

# =============================================================================
# 5. PYTHON DEPENDENCIES
# =============================================================================
install_python_deps() {
    log_info "Installing Python dependencies..."
    
    source "${SCRIPT_DIR}/.venv/bin/activate"
    
    # Ensure pip is up to date first
    python -m pip install --upgrade pip --break-system-packages 2>/dev/null || \
    python -m pip install --upgrade pip --user 2>/dev/null || true
    
    # Core dependencies
    pip install \
        torch torchvision torchaudio \
        transformers \
        sentence-transformers \
        numpy \
        pandas \
        pyarrow \
        scipy \
        scikit-learn \
        tqdm \
        matplotlib \
        seaborn \
        jupyter \
        ipython
    
    # Benchmark-specific dependencies
    pip install \
        datasets \
        huggingface-hub \
        tokenizers
    
    # Optional: Mem0 for comparison (if needed)
    # pip install mem0ai
    
    log_success "Python dependencies installed"
}

# =============================================================================
# 6. TURBOsuperMemory BUILD
# =============================================================================
build_tsm() {
    log_info "Building TurboSuperMemory..."
    
    source "$HOME/.cargo/env"
    source "${SCRIPT_DIR}/.venv/bin/activate"
    
    cd "${SCRIPT_DIR}"
    
    # Set PyO3 Python path to use the venv
    export PYO3_PYTHON="${SCRIPT_DIR}/.venv/bin/python"
    export PYTHON="${SCRIPT_DIR}/.venv/bin/python"
    
    # Build with GPU support
    log_info "Building TSM with GPU support..."
    cargo build --workspace --release --features cuda
    
    # Copy the Python extension
    cp target/release/libturbomemory.so turbomemory.so 2>/dev/null || \
    cp target/release/libturbomemory*.so turbomemory.so 2>/dev/null || \
    true
    
    log_success "TSM built successfully"
    
    # Verify
    python -c "import turbomemory; print(f'TSM version: {turbomemory.__version__}')" || \
    log_warn "Could not import turbomemory - may need manual setup"
}

# =============================================================================
# 7. DOWNLOAD BENCHMARK DATASETS
# =============================================================================
download_datasets() {
    log_info "Downloading benchmark datasets..."
    
    source "${SCRIPT_DIR}/.venv/bin/activate"
    
    cd "${SCRIPT_DIR}/benchmarks/cognitive_eval"
    
    # Download LongMemEval and LoCoMo
    python -m datasets.download --dataset all
    
    log_success "Datasets downloaded"
    
    # Verify
    DATA_DIR="${SCRIPT_DIR}/benchmarks/cognitive_eval/data"
    echo "Dataset contents:"
    ls -lah "${DATA_DIR}" 2>/dev/null || log_warn "Data directory not found"
}

# =============================================================================
# 8. PRE-CACHE EMBEDDING MODELS
# =============================================================================
cache_models() {
    log_info "Pre-caching embedding models..."
    
    source "${SCRIPT_DIR}/.venv/bin/activate"
    
    python -c "
from transformers import AutoModel, AutoTokenizer
import torch

models = [
    'sentence-transformers/all-MiniLM-L6-v2',
    'BAAI/bge-large-en-v1.5',
]

for model_name in models:
    print(f'Downloading {model_name}...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    print(f'  Done: {sum(p.numel() for p in model.parameters())} parameters')
    del model, tokenizer
    torch.cuda.empty_cache()

print('All models cached!')
"
    
    log_success "Embedding models cached"
}

# =============================================================================
# 9. VERIFY INSTALLATION
# =============================================================================
verify_setup() {
    log_info "Verifying installation..."
    
    source "${SCRIPT_DIR}/.venv/bin/activate"
    source "$HOME/.cargo/env"
    
    echo ""
    echo "=== System Info ==="
    echo "OS: $(lsb_release -d 2>/dev/null || cat /etc/os-release | head -1)"
    echo "CPU: $(nproc) cores"
    echo "RAM: $(free -h | awk '/^Mem:/ {print $2}')"
    
    echo ""
    echo "=== GPU Info ==="
    nvidia-smi || echo "No GPU detected"
    
    echo ""
    echo "=== Python ==="
    python --version
    pip --version
    
    echo ""
    echo "=== Rust ==="
    rustc --version
    cargo --version
    
    echo ""
    echo "=== CUDA ==="
    nvcc --version 2>/dev/null || echo "NVCC not found"
    
    echo ""
    echo "=== PyTorch ==="
    python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
    
    echo ""
    echo "=== TSM ==="
    python -c "import turbomemory; print('TSM imported successfully')" 2>/dev/null || echo "TSM not yet built"
    
    log_success "Verification complete"
}

# =============================================================================
# 10. CREATE RUNNER SCRIPTS
# =============================================================================
create_runners() {
    log_info "Creating benchmark runner scripts..."
    
    # Quick test runner
    cat > "${SCRIPT_DIR}/run_quick_test.sh" << 'EOF'
#!/bin/bash
set -e
source "$(dirname "$0")/.venv/bin/activate"
cd "$(dirname "$0")"

echo "=== Quick LongMemEval Test (5 conversations) ==="
python benchmarks/cognitive_eval/run_longmemeval.py \
    --quick --quick-n 5 \
    --lightweight --batch-size 64 \
    --data-dir benchmarks/cognitive_eval/data

echo ""
echo "=== Quick ANN vs Cognitive Comparison ==="
python benchmarks/cognitive_eval/run_longmemeval.py \
    --quick --quick-n 3 \
    --lightweight --batch-size 64 \
    --compare-cognitive \
    --data-dir benchmarks/cognitive_eval/data
EOF
    chmod +x "${SCRIPT_DIR}/run_quick_test.sh"
    
    # Full LongMemEval runner
    cat > "${SCRIPT_DIR}/run_longmemeval_full.sh" << 'EOF'
#!/bin/bash
set -e
source "$(dirname "$0")/.venv/bin/activate"
cd "$(dirname "$0")"

echo "=== Full LongMemEval Benchmark (500 conversations) ==="
python benchmarks/cognitive_eval/run_longmemeval.py \
    --lightweight --batch-size 64 \
    --data-dir benchmarks/cognitive_eval/data \
    --output results/longmemeval_full.json

echo "Results saved to results/longmemeval_full.json"
EOF
    chmod +x "${SCRIPT_DIR}/run_longmemeval_full.sh"
    
    # LoCoMo runner
    cat > "${SCRIPT_DIR}/run_locomo.sh" << 'EOF'
#!/bin/bash
set -e
source "$(dirname "$0")/.venv/bin/activate"
cd "$(dirname "$0")"

echo "=== LoCoMo Benchmark (sampled) ==="
python benchmarks/cognitive_eval/run_locomo.py \
    --quick --quick-n 100 \
    --lightweight --batch-size 64 \
    --data-dir benchmarks/cognitive_eval/data \
    --output results/locomo_quick.json

echo "Results saved to results/locomo_quick.json"
EOF
    chmod +x "${SCRIPT_DIR}/run_locomo.sh"
    
    # Recall audit
    cat > "${SCRIPT_DIR}/run_recall_audit.sh" << 'EOF'
#!/bin/bash
set -e
source "$(dirname "$0")/.venv/bin/activate"
cd "$(dirname "$0")"

echo "=== Recall Audit (100K vectors) ==="
python benchmarks/audit_recall.py \
    --num-items 100000 \
    --dimension 384 \
    --num-queries 100 \
    --top-k 10
EOF
    chmod +x "${SCRIPT_DIR}/run_recall_audit.sh"
    
    # Create results directory
    mkdir -p "${SCRIPT_DIR}/results"
    
    log_success "Runner scripts created"
}

# =============================================================================
# MAIN
# =============================================================================
main() {
    echo "======================================================================"
    echo "  TurboSuperMemory Cloud Setup"
    echo "======================================================================"
    echo ""
    
    # Check if running as root (needed for apt)
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
    
    # Run all setup steps
    setup_system
    setup_cuda
    setup_rust
    setup_python
    install_python_deps
    build_tsm
    download_datasets
    cache_models
    create_runners
    verify_setup
    
    echo ""
    echo "======================================================================"
    echo "  Setup Complete!"
    echo "======================================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Quick test:     ./run_quick_test.sh"
    echo "  2. Full benchmark: ./run_longmemeval_full.sh"
    echo "  3. LoCoMo test:    ./run_locomo.sh"
    echo "  4. Recall audit:     ./run_recall_audit.sh"
    echo ""
    echo "To activate the environment:"
    echo "  source .venv/bin/activate"
    echo ""
    echo "To rebuild TSM after code changes:"
    echo "  make build-python"
    echo ""
}

# Run main if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
