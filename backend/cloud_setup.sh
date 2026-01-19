#!/bin/bash
# =============================================================================
# XR2Text Cloud GPU Setup Script
# Run this after connecting to your cloud GPU via SSH
# =============================================================================

echo "=========================================="
echo "XR2Text Cloud GPU Setup"
echo "=========================================="

# Step 1: Clone or upload your project
echo "[1/6] Setting up project directory..."
mkdir -p /workspace/xr2text
cd /workspace/xr2text

# Create all required directories with proper permissions
echo "Creating directories with write permissions..."
mkdir -p /workspace/xr2text/backend/checkpoints
mkdir -p /workspace/xr2text/backend/logs
mkdir -p /workspace/xr2text/backend/data/figures
mkdir -p /workspace/xr2text/backend/data/statistics
mkdir -p /workspace/xr2text/backend/data/human_evaluation
mkdir -p /workspace/xr2text/backend/data/ablation_results
chmod -R 777 /workspace/xr2text/backend/checkpoints 2>/dev/null || true
chmod -R 777 /workspace/xr2text/backend/logs 2>/dev/null || true
chmod -R 777 /workspace/xr2text/backend/data 2>/dev/null || true
echo "Directories created!"

# Step 2: Install dependencies
echo "[2/6] Installing Python dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.35.0
pip install datasets>=2.14.0
pip install Pillow>=10.0.0
pip install pyyaml>=6.0
pip install loguru>=0.7.0
pip install tqdm>=4.66.0
pip install nltk>=3.8.0
pip install rouge-score>=0.1.2
pip install pycocoevalcap>=1.2
pip install scikit-learn>=1.3.0
pip install pandas>=2.0.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
pip install huggingface-hub>=0.17.0
pip install accelerate>=0.24.0
pip install safetensors>=0.4.0
pip install sentencepiece>=0.1.99
pip install timm>=0.9.12
pip install albumentations>=1.3.1
pip install opencv-python>=4.8.0
pip install jupyter notebook ipywidgets
pip install bert-score>=0.3.13
pip install radgraph>=0.1.0
pip install f1chexbert>=0.0.1

# Step 3: Download NLTK data
echo "[3/6] Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Step 4: Login to Hugging Face (for dataset access)
echo "[4/6] Hugging Face login..."
echo "Run: huggingface-cli login"
echo "Enter your HF token when prompted"

# Step 5: Verify GPU
echo "[5/6] Verifying GPU..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"

echo "[6/6] Setup complete!"
echo ""
echo "=========================================="
echo "Next steps:"
echo "1. Upload your project files (see instructions below)"
echo "2. Run: cd /workspace/xr2text/backend"
echo "3. Run: python train.py"
echo "=========================================="
