# Cloud GPU Training Guide for XR2Text

## Supported Cloud Providers

| Provider | Recommended GPU | Price/hr | Est. Cost (50 epochs) |
|----------|----------------|----------|----------------------|
| **RunPod** | A40 (48GB) | $0.40 | ~$5-7 |
| **RunPod** | RTX 4090 (24GB) | $0.35 | ~$6-8 |
| Vast.ai | RTX 4090 (24GB) | $0.30-0.50 | ~$5-8 |
| Lambda Labs | A100 (40GB) | $1.10 | ~$12-15 |

---

## Quick Start (RunPod) - RECOMMENDED

### 1. Create Account & Add Credits
1. Go to https://runpod.io
2. Sign up with email
3. Click "Billing" -> Add $10-15 credits

### 2. Deploy a GPU Pod
1. Click "Deploy" -> "GPU Pods"
2. Select: **1x A40 (48GB VRAM)** - Best value!
3. Choose **On-Demand** ($0.40/hr) for reliable training
4. Template: Select **RunPod PyTorch 2.1**
5. Check "Start Jupyter notebook"
6. Click "Deploy On-Demand"

### 3. Connect to Your Pod
1. Go to "My Pods" in RunPod dashboard
2. Click "Connect" on your pod
3. Options:
   - **Jupyter Lab**: Click "Connect to Jupyter Lab" (easiest)
   - **SSH**: Click "SSH Terminal" for command line
   - **Web Terminal**: Click "Connect to Web Terminal"

### 4. Setup (in Jupyter or Terminal)
```bash
# Clone or upload your project
cd /workspace

# Option A: Upload via Jupyter (drag & drop your zip file)
# Then: unzip backend.zip

# Option B: Git clone
git clone https://github.com/yourusername/MajorProject.git
cd MajorProject/backend

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

# Login to Hugging Face
huggingface-cli login
```

### 5. Start Training with A40 Config
```bash
# Use the A40-optimized config
python train.py --config configs/runpod_a40.yaml
```

### 6. Expected Training Time on A40
- **Batch size 8** with grad_accum 16 = effective batch 128
- **~60 steps/epoch** (vs ~240 on RTX 4060)
- **Total time: ~12-16 hours** (vs 65 hours locally!)
- **Cost: ~$5-7**

---

## Quick Start (Vast.ai)

### 1. Create Account & Add Credits
1. Go to https://vast.ai
2. Sign up with email
3. Click "Billing" -> Add $10-20 credits (card or crypto)

### 2. Rent a GPU
1. Click "Search" in the top menu
2. Set filters:
   - GPU: RTX 4000 Ada or RTX 4090
   - Min VRAM: 20 GB
   - Image: PyTorch 2.0+
3. Sort by price (cheapest first)
4. Click "RENT" on a machine with green "High" availability

### 3. Connect via SSH
After renting, you'll see your instance. Click "Connect" to get:
```
ssh -p <PORT> root@<IP_ADDRESS>
```

Or use the web terminal (click "Open" button).

### 4. Upload Your Project

**Option A: Using SCP (from your Windows terminal)**
```bash
# First, zip your backend folder
# Then upload:
scp -P <PORT> backend.zip root@<IP_ADDRESS>:/workspace/
```

**Option B: Using Git (if you have a repo)**
```bash
git clone https://github.com/yourusername/MajorProject.git
cd MajorProject/backend
```

**Option C: Using rsync (recommended)**
```bash
rsync -avz -e "ssh -p <PORT>" ./backend/ root@<IP_ADDRESS>:/workspace/xr2text/backend/
```

### 5. Setup Environment
```bash
# On the cloud GPU:
cd /workspace/xr2text/backend

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

# Login to Hugging Face (for dataset)
huggingface-cli login
# Enter your HF token: https://huggingface.co/settings/tokens
```

### 6. Start Training
```bash
# Use cloud-optimized config
python train.py --config configs/cloud_gpu.yaml
```

### 7. Monitor Training
```bash
# In another terminal or use tmux:
tail -f logs/training.log

# Or check GPU usage:
watch -n 1 nvidia-smi
```

### 8. Download Results
After training completes:
```bash
# From your local machine:
scp -P <PORT> -r root@<IP_ADDRESS>:/workspace/xr2text/backend/checkpoints ./
scp -P <PORT> -r root@<IP_ADDRESS>:/workspace/xr2text/backend/data/statistics ./
```

---

## Expected Training Time

| GPU | VRAM | Time for 50 epochs | Cost |
|-----|------|-------------------|------|
| RTX 4000 Ada | 20GB | ~20-24 hours | ~$5 |
| RTX 4090 | 24GB | ~12-15 hours | ~$7 |
| A100 | 40GB | ~8-10 hours | ~$12 |

---

## Troubleshooting

### "CUDA out of memory"
Edit `configs/cloud_gpu.yaml`:
```yaml
training:
  batch_size: 2  # Reduce from 4
  gradient_accumulation_steps: 64  # Increase
```

### "Dataset not found"
Make sure you're logged in to Hugging Face:
```bash
huggingface-cli login
```

### Connection drops
Use tmux to keep training running:
```bash
tmux new -s training
python train.py --config configs/cloud_gpu.yaml
# Press Ctrl+B, then D to detach
# Reconnect later with: tmux attach -t training
```

### Resume training after disconnect
Training auto-resumes from latest checkpoint:
```bash
python train.py --config configs/cloud_gpu.yaml
# It will automatically find checkpoint_epoch_X.pt
```

---

## Files to Upload

**Required:**
```
backend/
├── configs/
│   └── cloud_gpu.yaml
├── src/
│   ├── models/
│   ├── training/
│   ├── data/
│   └── utils/
├── train.py
└── requirements.txt
```

**Optional (if resuming):**
```
backend/
└── checkpoints/
    └── checkpoint_epoch_12.pt  # Your latest checkpoint
```

---

## Quick Commands Reference

```bash
# Connect
ssh -p PORT root@IP

# Upload project
rsync -avz -e "ssh -p PORT" ./backend/ root@IP:/workspace/backend/

# Start training (in tmux)
tmux new -s train
cd /workspace/backend && python train.py --config configs/cloud_gpu.yaml

# Check progress
tail -f logs/training.log

# GPU status
nvidia-smi

# Download checkpoints
scp -P PORT -r root@IP:/workspace/backend/checkpoints ./

# Stop instance (save money!)
# Go to vast.ai dashboard -> click "Destroy" when done
```

---

## Cost Estimate

- RTX 4000 Ada @ $0.20/hr x 24 hours = **$4.80**
- RTX 4090 @ $0.50/hr x 15 hours = **$7.50**

Add $2-3 buffer for setup/testing. Total: **~$10 max**
