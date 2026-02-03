#!/bin/bash
# Model setup script for CoRSAL-STT
# This script sets up the environment and downloads model checkpoints

set -e  # Exit on error

echo "=========================================="
echo "CoRSAL-STT Model Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    echo "Please install Miniconda or Anaconda first."
    exit 1
fi

# Environment name
ENV_NAME="corsal-stt"
PYTHON_VERSION="3.10"  # Python 3.10-3.11 recommended for best compatibility
                      # Python 3.13 works but has limited PyTorch support (Linux-only for 2.5.x)
                      # Python 3.8 is outdated (EOL Oct 2024)

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}Environment ${ENV_NAME} already exists.${NC}"
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Using existing environment."
        echo "Activating environment: ${ENV_NAME}"
        eval "$(conda shell.bash hook)"
        conda activate ${ENV_NAME}
    fi
fi

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${GREEN}Creating conda environment: ${ENV_NAME}${NC}"
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
fi

# Activate environment
echo -e "${GREEN}Activating environment: ${ENV_NAME}${NC}"
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Upgrade pip
echo -e "${GREEN}Upgrading pip...${NC}"
pip install --upgrade pip

# Install base requirements
echo -e "${GREEN}Installing base requirements...${NC}"
pip install -r requirements.txt

# Install additional dependencies for ASR models
echo -e "${GREEN}Installing ASR model dependencies...${NC}"
pip install transformers>=4.30.0
pip install datasets>=2.14.0
pip install torchaudio>=2.0.0
pip install jiwer  # For WER calculation
pip install evaluate  # HuggingFace evaluation library
pip install accelerate  # For distributed training
pip install deepspeed  # Optional: for large model training

# Install model-specific dependencies
echo -e "${GREEN}Installing model-specific dependencies...${NC}"

# wav2vec XLS-R dependencies (already in transformers)
echo "  - wav2vec XLS-R: Using transformers library"

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

# Download model checkpoints (optional - can be done on-demand)
echo -e "${YELLOW}Model checkpoint downloads:${NC}"
echo "  Model checkpoints will be downloaded automatically when first used."
echo "  To pre-download checkpoints, run:"
echo "    python scripts/download_models.py"

# Create download script if it doesn't exist
if [ ! -f scripts/download_models.py ]; then
    echo -e "${GREEN}Creating model download script...${NC}"
    cat > scripts/download_models.py << 'EOF'
#!/usr/bin/env python3
"""Download pre-trained model checkpoints for CoRSAL-STT"""

import os
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor
)

def download_xlsr():
    """Download wav2vec XLS-R model"""
    print("Downloading wav2vec XLS-R model...")
    model_name = "facebook/wav2vec2-xlsr-53"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    print(f"✓ Downloaded {model_name}")

def download_omnilingual():
    """Download Omnilingual ASR model (if available)"""
    print("Note: Omnilingual ASR models are large and should be downloaded on-demand.")
    print("Check https://github.com/facebookresearch/omnilingual-asr for download instructions.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download ASR model checkpoints")
    parser.add_argument("--model", choices=["xlsr", "omnilingual", "all"], 
                       default="all", help="Model to download")
    args = parser.parse_args()
    
    if args.model in ["xlsr", "all"]:
        download_xlsr()
    if args.model in ["omnilingual", "all"]:
        download_omnilingual()
    
    print("\n✓ Model download complete!")
EOF
    chmod +x scripts/download_models.py
fi

# Verify installation
echo -e "${GREEN}Verifying installation...${NC}"
python -c "import torch; import transformers; import datasets; print('✓ All packages imported successfully')" || {
    echo -e "${RED}Error: Some packages failed to import${NC}"
    exit 1
}

# Print summary
echo ""
echo -e "${GREEN}=========================================="
echo "Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "Environment: ${ENV_NAME}"
echo "Python version: $(python --version)"
echo ""
echo "Next steps:"
echo "1. Activate the environment: conda activate ${ENV_NAME}"
echo "2. Download model checkpoints (optional): python scripts/download_models.py"
echo "3. Prepare your CoRSAL data in the data/ directory"
echo "4. Run training scripts from the scripts/ directory"
echo ""
echo "For more information, see README.md"
echo ""
