#!/bin/bash
# Installation script for vLLM-Mirage integration

set -e

echo "=================================="
echo "vLLM-Mirage Installation Script"
echo "=================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running from correct directory
if [ ! -f "install.sh" ]; then
    echo -e "${RED}Error: Please run this script from the vllm-mirage directory${NC}"
    exit 1
fi

# Step 1: Clone Mirage
echo -e "${GREEN}[1/4] Cloning Mirage repository...${NC}"
if [ ! -d "mirage" ]; then
    git clone --recursive https://github.com/mirage-project/mirage.git
    cd mirage
    git submodule update --init --recursive
    cd ..
else
    echo "Mirage directory already exists, skipping..."
fi

# Step 2: Build and install Mirage
echo -e "${GREEN}[2/4] Building and installing Mirage...${NC}"
cd mirage
if [ ! -f "build/CMakeCache.txt" ]; then
    mkdir -p build
    cd build
    cmake ..
    make -j$(nproc)
    cd ..
fi
pip install -e .
cd ..

# Step 3: Clone and setup vLLM
echo -e "${GREEN}[3/4] Cloning vLLM repository...${NC}"
if [ ! -d "vllm" ]; then
    git clone https://github.com/vllm-project/vllm.git
    cd vllm
    # Checkout a stable version (adjust as needed)
    # git checkout v0.6.0
    cd ..
else
    echo "vLLM directory already exists, skipping..."
fi

# Step 4: Apply patches
echo -e "${GREEN}[4/4] Applying Mirage patches to vLLM...${NC}"
cp patches/vllm/model_executor/layers/fused_mirage_ops.py vllm/vllm/model_executor/layers/
cp patches/vllm/model_executor/models/qwen3.py vllm/vllm/model_executor/models/

# Install vLLM
cd vllm
pip install -e .
cd ..

echo ""
echo -e "${GREEN}=================================="
echo "Installation completed successfully!"
echo "==================================${NC}"
echo ""
echo "Next steps:"
echo "1. Set environment variables:"
echo "   export PATH=/usr/local/cuda/bin:\$PATH"
echo "   export PYTHONPATH=\$(pwd)/vllm:\$PYTHONPATH"
echo "   export VLLM_USE_MIRAGE=1"
echo ""
echo "2. Run tests:"
echo "   python test_mirage_qwen3.py --skip-baseline --verbose"
echo ""
echo "3. Run benchmarks:"
echo "   python benchmark_mirage.py --prompts test_prompts.txt"
echo ""
