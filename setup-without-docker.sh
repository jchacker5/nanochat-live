#!/bin/bash
# Alternative setup script without Docker
# Creates a Python virtual environment and installs all dependencies

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Setting up SRGI without Docker ===${NC}"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${YELLOW}⚠️  Warning: Python 3.10+ recommended for THRML support${NC}"
    echo -e "${YELLOW}   Current version: Python $PYTHON_VERSION${NC}"
    read -p "Continue anyway? (y/n): " continue
    if [ "$continue" != "y" ]; then
        exit 0
    fi
fi

# Create virtual environment
echo -e "${GREEN}Creating virtual environment...${NC}"
python3 -m venv venv

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${GREEN}Upgrading pip...${NC}"
pip install --upgrade pip

# Install core dependencies
echo -e "${GREEN}Installing core dependencies...${NC}"
pip install torch>=2.8.0
pip install datasets>=4.0.0
pip install fastapi>=0.117.1
pip install tiktoken>=0.11.0
pip install tokenizers>=0.22.0
pip install wandb>=0.21.3
pip install numpy scipy matplotlib

# Install JAX ecosystem (for THRML)
echo -e "${GREEN}Installing JAX ecosystem...${NC}"
pip install jax jaxlib equinox

# Install testing dependencies
echo -e "${GREEN}Installing testing dependencies...${NC}"
pip install pytest>=8.0.0 pytest-cov

# Try to install THRML (optional)
echo -e "${GREEN}Attempting to install THRML...${NC}"
pip install git+https://github.com/extropic-ai/thrml.git || echo -e "${YELLOW}THRML installation failed (requires Python 3.10+), will use PyTorch fallback${NC}"

# Install optional dependencies
echo -e "${GREEN}Installing optional dependencies...${NC}"
pip install geoopt || echo -e "${YELLOW}geoopt optional, skipping${NC}"

# Install project in development mode
echo -e "${GREEN}Installing project...${NC}"
pip install -e .

echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo -e "${BLUE}To activate the environment:${NC}"
echo "  source venv/bin/activate"
echo ""
echo -e "${BLUE}To test installation:${NC}"
echo "  python -c 'from nanochat.ebm_hopfield import EBMHopfieldMemory; print(\"✓ EBM module works!\")'"
echo ""
echo -e "${BLUE}To run tests:${NC}"
echo "  pytest tests/test_ebm_hopfield.py -v"

