# CoRSAL-STT: Speech Recognition for South Asian Languages

This repository provides training templates and tools for automatic speech recognition (ASR) on data from the [Computational Resource on South Asian Languages (CoRSAL)](https://digital.library.unt.edu/explore/collections/CORSAL/) archive hosted by the UNT Digital Library.

CoRSAL is a digital archive containing source audio, video, and text materials for minority languages of South Asia, covering over 80 languages including Lamkang, Akha, Mankiyali, Burushaski, Boro, and many others. This repository aims to facilitate ASR research and development for these under-resourced languages using ongoing community efforts on Lamkang. 

## Overview

This repository provides training templates for three multilingual ASR models:

1. **wav2vec XLS-R** - Cross-lingual speech representation learning at scale
2. **Massively Multilingual ASR (MML)** - Large-scale multilingual ASR with 50+ languages
3. **Omnilingual ASR** - Open-source multilingual ASR supporting 1,600+ languages

## Models

### 1. wav2vec XLS-R

**Paper:** Babu, A., Wang, C., Tjandra, A., Lakhotia, K., Xu, Q., Goyal, N., Singh, K., von Platen, P., Saraf, Y., Pino, J., Baevski, A., Conneau, A., & Auli, M. (2022). XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale. *Interspeech 2022*, 2278-2282. doi: 10.21437/Interspeech.2022-143

**arXiv:** [2111.09296](https://arxiv.org/abs/2111.09296)

XLS-R is a large-scale self-supervised model for cross-lingual speech representation learning based on wav2vec 2.0. The model scales to 2 billion parameters and is trained on nearly 500,000 hours of publicly available speech audio across 128 languages. It demonstrates strong performance on speech translation, recognition, and language identification tasks, especially for low-resource languages.

### 2. Massively Multilingual ASR (MML)

**Paper:** Pratap, V., Tjandra, A., Shi, B., Tomasello, P., Babu, A., Kundu, S., Elkahky, A., Ni, Z., Vyas, A., Fazel-Zarandi, M., Baevski, A., Adi, Y., Auli, M., & Conneau, A. (2020). Massively Multilingual ASR: 50 Languages, 1 Model, 1 Billion Parameters. *Interspeech 2020*.

**arXiv:** [2007.03001](https://arxiv.org/abs/2007.03001)

This work demonstrates training a single ASR model that handles 50 languages with 1 billion parameters. The model shows that scaling up multilingual training can achieve competitive or better performance compared to language-specific models while being more parameter-efficient.

### 3. Omnilingual ASR

**Paper:** Meta AI Research (2024). Omnilingual ASR: Open-Source Multilingual Speech Recognition for 1600+ Languages.

**arXiv:** [2511.09690](https://arxiv.org/abs/2511.09690)

Omnilingual ASR is a groundbreaking open-source multilingual speech recognition system that supports over 1,600 languages, including more than 500 languages never previously served by any ASR system. The system scales self-supervised pre-training to 7 billion parameters and uses an encoder-decoder architecture designed for zero-shot generalization. It enables communities to add unserved languages with minimal data samples.

## Setup

### Prerequisites
- Python 3.10+ (3.10 or 3.11 recommended for best compatibility; 3.13 supported but with limited PyTorch support on some platforms)
- CUDA-capable GPU (recommended)
- Sufficient disk space for model checkpoints and data

### Installation

Run the setup script to install all dependencies and download model checkpoints:

```bash
bash model_setup.sh
```

This script will:
1. Create a conda environment (`corsal-stt`)
2. Install all required Python packages
3. Download pre-trained model checkpoints
4. Set up directory structure for training

### Manual Setup

**Option 1: Using Conda (Recommended for cluster environments)**

```bash
# Create conda environment
conda create -n corsal-stt python=3.10
conda activate corsal-stt

# Install dependencies
pip install -r requirements.txt

# Install additional model-specific dependencies
pip install transformers datasets torchaudio
```

**Option 2: Using Python venv**

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install additional model-specific dependencies
pip install transformers>=4.30.0 datasets>=2.14.0 torchaudio>=2.0.0
```

**Quick Setup Script (venv)**

Alternatively, use the provided setup script:

```bash
bash setup_venv.sh
```

This will create a `venv` directory and install all dependencies automatically.

## Data Preparation

CoRSAL data can be accessed through the [UNT Digital Library](https://digital.library.unt.edu/explore/collections/CORSAL/). Contact the team to request resources for any language. After downloading audio files and transcripts, in our case Lamkange, we adopt the following directory structure:

1. Organize data in the following structure:
```
data/
├── language1/
│   ├── audio/
│   └── transcripts/
├── language2/
│   ├── audio/
│   └── transcripts/
└── ...
```

2. Convert audio files to 16kHz mono WAV format
3. Prepare manifest files (JSONL format) with paths to audio files and transcripts

## Training

Each model has its own training script. See the `scripts/` directory for model-specific training templates:

- `train_xlsr.py` - Training script for wav2vec XLS-R
- `train_mml.py` - Training script for Massively Multilingual ASR
- `train_omnilingual.py` - Training script for Omnilingual ASR

Example usage:

```bash
python scripts/train_xlsr.py \
    --data_dir data/ \
    --output_dir models/xlsr_corsal/ \
    --num_epochs 10 \
    --batch_size 8
```

## Evaluation

Evaluate trained models using standard ASR metrics (WER, CER):

```bash
python scripts/evaluate.py \
    --model_path models/xlsr_corsal/checkpoint-1000 \
    --test_manifest data/test.jsonl
```

## Citation

If you use this repository or CoRSAL data in your research, please cite:

```bibtex
@misc{corsal-stt,
  title={CoRSAL-STT: Speech Recognition for South Asian Languages},
  author={Melissa Lieffers, Victor Shi, Phakphum Artkaew, Shobhana Chelliah},
  year={2026},
  url={https://github.com/yourusername/corsal-stt}
}

@misc{corsal,
  title={Computational Resource on South Asian Languages (CoRSAL)},
  publisher={UNT Digital Library},
  url={https://digital.library.unt.edu/explore/collections/CORSAL/}
}
```

## License

This repository is licensed under the MIT License. Please check individual model licenses for their respective pre-trained checkpoints.

## Acknowledgments

- CoRSAL archive maintained by UNT Digital Library
- Model developers: Meta AI (wav2vec XLS-R, Omnilingual ASR, MML)
- South Asian language communities who contributed data to CoRSAL

## Contact

For questions or issues, please open an issue on GitHub or contact any of the co-authors. 
