#!/bin/bash
# Virtual environment setup script for CoRSAL-STT
# This script creates a Python virtual environment and installs dependencies

set -e  # Exit on error

echo "=========================================="
echo "CoRSAL-STT Virtual Environment Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
REQUIRED_VERSION="3.10"

echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 is not installed or not in PATH${NC}"
    exit 1
fi

# Compare versions (simple check)
PYTHON_MAJOR=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1)
PYTHON_MINOR=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo -e "${YELLOW}Warning: Python 3.10+ recommended. Found Python $(python3 --version)${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Virtual environment name
VENV_NAME="venv"

# Check if venv already exists
if [ -d "$VENV_NAME" ]; then
    echo -e "${YELLOW}Virtual environment '$VENV_NAME' already exists.${NC}"
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_NAME"
    else
        echo "Using existing virtual environment."
        echo ""
        echo -e "${GREEN}To activate the environment, run:${NC}"
        echo "  source $VENV_NAME/bin/activate"
        exit 0
    fi
fi

# Create virtual environment
echo -e "${GREEN}Creating virtual environment: $VENV_NAME${NC}"
python3 -m venv "$VENV_NAME"

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source "$VENV_NAME/bin/activate"

# Upgrade pip
echo -e "${GREEN}Upgrading pip...${NC}"
pip install --upgrade pip

# Install base requirements
echo -e "${GREEN}Installing base requirements from requirements.txt...${NC}"
pip install -r requirements.txt

# Install additional dependencies for ASR models
echo -e "${GREEN}Installing ASR model dependencies...${NC}"
pip install transformers>=4.30.0
pip install datasets>=2.14.0
pip install torchaudio>=2.0.0
pip install jiwer  # For WER calculation
pip install accelerate  # For distributed training

# Try to install evaluate (HuggingFace evaluation library)
echo -e "${GREEN}Installing evaluate library...${NC}"
if pip install evaluate 2>/dev/null; then
    echo "✓ evaluate installed successfully"
else
    echo -e "${YELLOW}Warning: evaluate package installation failed. This is optional and can be installed later if needed.${NC}"
    echo "  You can try: pip install evaluate"
fi

# Optional: Install deepspeed for large model training
read -p "Install deepspeed for large model training? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install deepspeed
fi

# Create directories
echo -e "${GREEN}Creating directory structure...${NC}"
mkdir -p models/checkpoints
mkdir -p models/xlsr
mkdir -p models/mml
mkdir -p models/omnilingual
mkdir -p data/raw
mkdir -p data/processed
mkdir -p scripts
mkdir -p logs
mkdir -p outputs

# Verify installation
echo -e "${GREEN}Verifying installation...${NC}"
python -c "import torch; import transformers; import datasets; print('✓ Core packages imported successfully')" || {
    echo -e "${RED}Error: Some core packages failed to import${NC}"
    exit 1
}

# Try to verify evaluate (optional)
python -c "import evaluate" 2>/dev/null && echo "✓ evaluate package available" || echo -e "${YELLOW}Note: evaluate package not available (optional)${NC}"

# Print summary
echo ""
echo -e "${GREEN}=========================================="
echo "Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "Virtual environment: $VENV_NAME"
echo "Python version: $(python --version)"
echo ""
echo -e "${GREEN}To activate the environment, run:${NC}"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
echo "Next steps:"
echo "1. Activate the environment: source $VENV_NAME/bin/activate"
echo "2. Download model checkpoints (optional): python scripts/download_models.py"
echo "3. Prepare your CoRSAL data in the data/ directory"
echo "4. Run training scripts from the scripts/ directory"
echo ""
echo "For more information, see README.md"
echo ""
