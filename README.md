# XR2Text: Hierarchical Anatomical Query Transformer for Chest X-Ray Report Generation

A state-of-the-art AI system that automatically generates clinical radiology reports from chest X-ray images using **HAQT-ARR** (Hierarchical Anatomical Query Tokens with Adaptive Region Routing) architecture.

## Novel Contributions

This project introduces several novel research contributions:

1. **HAQT-ARR Architecture**: Hierarchical anatomical query tokens with spatial priors and adaptive region routing
2. **Image-Conditioned Spatial Prior Refinement**: Dynamic per-image refinement of anatomical priors (NEW)
3. **Curriculum Learning Strategy**: Progressive 4-stage training from simple to complex cases
4. **Novel Loss Functions**: Clinical entity loss with negation detection, anatomical consistency loss, region-aware focal loss
5. **Clinical Validation Framework**: Negation-aware clinical accuracy assessment with entity extraction

## Novelty Differentiation from Prior Work

| Feature | R2Gen | ORGAN | MAIRA-Seg | A3Net | **HAQT-ARR (Ours)** |
|---------|-------|-------|-----------|-------|---------------------|
| Anatomical Queries | ✗ | ✓ (GNN) | ✓ (Seg) | ✓ (Dict) | **✓ (Learnable)** |
| Spatial Priors | ✗ | ✗ | Hard Masks | ✗ | **Soft 2D Gaussian** |
| Image-Conditioned Priors | ✗ | ✗ | ✗ | ✗ | **✓ (Per-image)** |
| Requires Segmentation | ✗ | ✗ | ✓ | ✗ | **✗** |
| Cross-Region Modeling | ✗ | ✗ | ✗ | ✗ | **✓** |
| Curriculum Learning | ✗ | ✗ | ✗ | ✗ | **✓** |
| Negation-Aware Validation | ✗ | ✗ | ✗ | ✗ | **✓** |

### Key Differentiators:

1. **vs R2Gen (EMNLP 2020)**: R2Gen uses memory-driven generation without anatomical awareness. We add hierarchical anatomical queries with learnable spatial priors.

2. **vs ORGAN (ACL 2023)**: ORGAN uses GNN for organ importance. We use lightweight adaptive routing without graph structure, plus image-conditioned prior refinement.

3. **vs MAIRA-Seg (2023)**: MAIRA requires segmentation masks at inference. Our approach learns soft spatial priors end-to-end - NO segmentation needed.

4. **vs A3Net (2023)**: A3Net uses fixed anatomical knowledge dictionaries. Our queries are fully learnable and adapt per-image via the prior refinement module.

### What Makes HAQT-ARR Novel:

```
┌────────────────────────────────────────────────────────────┐
│  NOVEL: Image-Conditioned Spatial Prior Refinement         │
│  ──────────────────────────────────────────────────────────│
│  Static Gaussian Priors  →  Visual Features  →  Refined   │
│  (Anatomical Init)          (Per-Image)        Priors     │
│                                                            │
│  This allows the model to:                                 │
│  • Adapt to patient-specific anatomy                       │
│  • Handle anatomical variations (e.g., cardiomegaly)       │
│  • Focus on abnormal regions dynamically                   │
└────────────────────────────────────────────────────────────┘
```

## Architecture Overview

```
Input Chest X-Ray (384×384)
         ↓
┌─────────────────────────────────┐
│  Swin Transformer Encoder       │
│  - Hierarchical feature maps    │
│  - Shifted-window attention     │
│  - Pre-trained on ImageNet      │
└─────────────────────────────────┘
         ↓
Visual Features (B, 1024, 12×12)
         ↓
┌─────────────────────────────────┐
│  HAQT-ARR Module (NOVEL)        │
│  ┌───────────────────────────┐  │
│  │ 36 Anatomical Queries     │  │
│  │  - 8 global queries       │  │
│  │  - 28 region queries      │  │
│  │    (7 regions × 4 each)   │  │
│  └───────────────────────────┘  │
│         ↓                        │
│  ┌───────────────────────────┐  │
│  │ Spatial Prior Injection   │  │
│  │  - 2D Gaussian priors     │  │
│  │  - Anatomical locations   │  │
│  └───────────────────────────┘  │
│         ↓                        │
│  ┌───────────────────────────┐  │
│  │ Cross-Attention to Image  │  │
│  │  - Query features         │  │
│  │  - Region-specific focus  │  │
│  └───────────────────────────┘  │
│         ↓                        │
│  ┌───────────────────────────┐  │
│  │ Adaptive Region Routing   │  │
│  │  - Dynamic importance     │  │
│  │  - Attention weighting    │  │
│  └───────────────────────────┘  │
│         ↓                        │
│  ┌───────────────────────────┐  │
│  │ Cross-Region Interaction  │  │
│  │  - Transformer layers     │  │
│  │  - Global reasoning       │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
         ↓
Contextualized Features (B, 36, 768)
         ↓
┌─────────────────────────────────┐
│  BioBART Decoder                │
│  - Biomedical pre-training      │
│  - Clinical terminology         │
│  - Beam search generation       │
└─────────────────────────────────┘
         ↓
Clinical Radiology Report
```

### Anatomical Regions (7 regions):
- Right Lung
- Left Lung
- Heart
- Mediastinum
- Spine
- Diaphragm
- Costophrenic Angles

## Project Structure

```
MajorProject/
├── backend/
│   ├── src/
│   │   ├── models/
│   │   │   ├── swin_encoder.py          # Swin Transformer encoder
│   │   │   ├── haqt_arr.py              # NOVEL: HAQT-ARR module
│   │   │   ├── biobart_decoder.py       # BioBART decoder
│   │   │   └── xr2text.py               # Complete model
│   │   ├── data/
│   │   │   ├── dataset.py               # MIMIC-CXR dataset loader
│   │   │   ├── transforms.py            # Image preprocessing
│   │   │   └── dataloader.py            # Dataloaders
│   │   ├── training/
│   │   │   ├── trainer.py               # Training loop with novel losses
│   │   │   ├── curriculum.py            # NOVEL: Curriculum learning
│   │   │   ├── novel_losses.py          # NOVEL: Custom loss functions
│   │   │   └── clinical_validator.py    # NOVEL: Clinical validation
│   │   ├── utils/
│   │   │   ├── metrics.py               # BLEU, ROUGE, CIDEr
│   │   │   ├── device.py                # GPU utilities
│   │   │   └── logger.py                # Logging
│   │   └── api/
│   │       ├── main.py                  # FastAPI server
│   │       └── routes.py                # API endpoints
│   ├── notebooks/
│   │   ├── 01_data_exploration.ipynb    # Dataset analysis
│   │   ├── 02_model_training.ipynb      # Training pipeline
│   │   ├── 03_model_evaluation.ipynb    # Evaluation & visualization
│   │   └── 04_ablation_study.ipynb      # Baseline comparison
│   ├── configs/
│   │   └── training_config.yaml         # Training configuration
│   ├── checkpoints/                     # Saved models
│   ├── data/                            # Datasets & statistics
│   ├── TRAINING_GUIDE.md                # Comprehensive training guide
│   └── requirements.txt
│
├── frontend/                            # React TypeScript Dashboard
│   ├── src/
│   │   ├── components/                  # UI components
│   │   ├── services/                    # API client
│   │   └── store/                       # State management
│   ├── package.json
│   └── vite.config.ts
│
└── swin/                                # Python virtual environment

```

## Quick Start

### Prerequisites

**Hardware:**
- NVIDIA GPU with 8GB+ VRAM (tested on RTX 4060 Laptop GPU)
- 16GB+ RAM
- 50GB+ disk space

**Software:**
- Python 3.10+
- CUDA 12.1+ / cuDNN 9.0+
- Node.js 18+

### 1. Backend Setup

```bash
# Clone/navigate to project
cd MajorProject

# Activate virtual environment
swin\Scripts\activate  # Windows
# source swin/bin/activate  # Linux/Mac

# Install dependencies
cd backend
pip install -r requirements.txt
```

### 2. Dataset

Dataset automatically downloads from HuggingFace on first run:
- **Source**: `itsanmolgupta/mimic-cxr-dataset`
- **Size**: 30,633 samples
- **Cache**: `~/.cache/huggingface/datasets/` (~764MB)

### 3. Training

#### Option 1: Jupyter Notebook (Recommended)
```bash
cd backend/notebooks
jupyter notebook 02_model_training.ipynb
# Run all cells
```

#### Option 2: Automated Script
```bash
cd backend
start_training_robust.bat  # Windows
# bash start_training_robust.sh  # Linux/Mac
```

**Training Configuration:**
- **Epochs**: 50
- **Batch Size**: 4 (gradient accumulation: 8, effective batch: 32)
- **Mixed Precision**: FP16 (automatic)
- **Validation**: Every 5 epochs (fast validation with 25% data)
- **Checkpoints**: Auto-saved every 5 epochs
- **Expected Time**: ~28-30 hours on RTX 4060

#### Curriculum Learning Stages (Automatic)
| Epochs | Stage | Data Type |
|--------|-------|-----------|
| 0-5 | Normal Cases | Simple "lungs clear" reports |
| 5-15 | Single Region | Single abnormality |
| 15-30 | Multi-Region | Multiple findings |
| 30-50 | Complex Cases | Severe multi-pathology |

### 4. Run API Server

```bash
cd backend
python run_server.py --checkpoint checkpoints/best_model.pt
# Server runs at http://localhost:8000
```

### 5. Frontend Dashboard

```bash
cd frontend
npm install
npm run dev
# Dashboard at http://localhost:3000
```

## Novel Features

### 1. HAQT-ARR Architecture
- **36 learnable anatomical query tokens** (8 global + 7 regions × 4)
- **Spatial prior injection** using 2D Gaussians for anatomical locations
- **Adaptive region routing** for dynamic importance weighting
- **Cross-region interaction** via transformer layers

### 2. Novel Loss Functions

**Anatomical Consistency Loss** (weight: 0.1)
```python
# Ensures predictions respect anatomical structure
L_anatomy = CrossEntropy(region_predictions, anatomical_labels)
```

**Clinical Entity Loss** (weight: 0.2)
```python
# Penalizes missing/incorrect clinical entities
L_entity = F1_Loss(predicted_entities, ground_truth_entities)
```

**Region-Aware Focal Loss** (weight: 0.15)
```python
# Focuses on hard anatomical regions
L_focal = FocalLoss(logits, labels, region_weights)
```

**Cross-Modal Alignment Loss** (weight: 0.1)
```python
# Aligns visual and textual representations
L_align = ContrastiveLoss(image_features, text_features)
```

**Total Loss:**
```python
L_total = L_CE + 0.1*L_anatomy + 0.2*L_entity + 0.15*L_focal + 0.1*L_align
```

### 3. Clinical Validation Framework

Automated extraction and validation of:
- **Findings**: Pneumonia, Edema, Effusion, Atelectasis, Consolidation
- **Locations**: Right/Left lung, Upper/Middle/Lower lobe
- **Severity**: Mild, Moderate, Severe
- **Laterality**: Bilateral, Unilateral

**Metrics:**
- Clinical Accuracy
- Entity Precision/Recall/F1
- Critical Error Detection

### 4. Curriculum Learning

Progressive training strategy:
1. Start with normal/simple cases (epochs 0-5)
2. Gradually introduce single abnormalities (epochs 5-15)
3. Add multi-region findings (epochs 15-30)
4. Train on complex cases (epochs 30-50)

**Benefit:** Faster convergence, better final performance

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health and GPU status |
| `/api/v1/generate` | POST | Generate report from uploaded image |
| `/api/v1/generate/base64` | POST | Generate from base64 encoded image |
| `/api/v1/generate/batch` | POST | Batch generation (up to 10 images) |
| `/api/v1/feedback` | POST | Submit radiologist corrections |
| `/api/v1/model/info` | GET | Model architecture details |
| `/api/v1/model/status` | GET | GPU memory and inference stats |

## Evaluation Metrics

### Text Generation Metrics
- **BLEU-1/2/3/4**: N-gram precision
- **ROUGE-1/2/L**: Recall-oriented n-gram matching
- **METEOR**: Semantic similarity with stemming
- **CIDEr**: Consensus-based image description evaluation

### Clinical Metrics (NOVEL)
- **Clinical Accuracy**: Entity-level correctness
- **Entity F1**: Precision/Recall of medical entities
- **Critical Error Rate**: Dangerous misdiagnoses

## Baseline Comparisons

| Method | Venue | BLEU-4 | ROUGE-L |
|--------|-------|--------|---------|
| R2Gen | EMNLP 2020 | 0.103 | 0.277 |
| M²Tr | MICCAI 2021 | 0.117 | 0.284 |
| ORGAN | ACL 2023 | 0.128 | 0.293 |
| ChestBioX-Gen | - | 0.142 | 0.312 |
| **XR2Text (Ours)** | - | **TBD** | **TBD** |

*(Results will be updated after training completes)*

## Configuration

**Training Configuration** (`configs/training_config.yaml`):

```yaml
model:
  encoder:
    name: "swin_base_patch4_window7_224"
    freeze_layers: 2
    pretrained: true

  haqt_arr:
    num_global_queries: 8
    queries_per_region: 4
    hidden_dim: 768
    num_heads: 8
    num_layers: 4
    use_spatial_priors: true

  decoder:
    name: "GanjinZero/biobart-base"
    max_length: 256
    num_beams: 4

training:
  epochs: 50
  batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 5.0e-5
  warmup_steps: 500
  use_amp: true
  patience: 999  # No early stopping

  curriculum:
    enabled: true
    stages:
      - name: "normal_cases"
        epoch_range: [0, 5]
      - name: "single_region"
        epoch_range: [5, 15]
      - name: "multi_region"
        epoch_range: [15, 30]
      - name: "complex_cases"
        epoch_range: [30, 50]

  novel_losses:
    anatomical_consistency: 0.1
    clinical_entity: 0.2
    region_focal: 0.15
    cross_modal_alignment: 0.1

  validation:
    frequency: 5  # Validate every 5 epochs
    subset: 0.25  # Use 25% of validation data
```

## Monitoring Training

**Real-time Monitor:**
```bash
cd backend
python monitor_training.py
```

Displays:
- GPU utilization, temperature, memory
- Current epoch, BLEU-4, ROUGE-L
- Time elapsed, ETA
- Checkpoint status

**Progress Tracking:**
- Training history: `backend/data/statistics/training_history.csv`
- Checkpoints: `backend/checkpoints/checkpoint_epoch_*.pt`
- Best model: `backend/checkpoints/best_model.pt`

## Troubleshooting

**CUDA Out of Memory:**
```yaml
# Reduce batch size in config
batch_size: 2
gradient_accumulation_steps: 16
```

**Slow Training:**
- Ensure `use_amp: true` (mixed precision)
- Close other GPU applications
- Check GPU utilization with `nvidia-smi`

**Training Interrupted:**
```python
# Resume from checkpoint in notebook
checkpoint_path = "../checkpoints/checkpoint_epoch_25.pt"
trainer.load_checkpoint(checkpoint_path)
```

## Project Timeline

**Phase 1: Research & Design** (Weeks 1-4)
- Literature review
- Architecture design
- Dataset preparation

**Phase 2: Implementation** (Weeks 5-8)
- Model implementation
- Training pipeline
- Novel loss functions

**Phase 3: Training & Evaluation** (Weeks 9-10)
- 50-epoch training (~3 days)
- Baseline comparisons
- Ablation studies

**Phase 4: Deployment** (Weeks 11-12)
- API development
- Frontend dashboard
- Documentation

## Authors

**Students:**
- S. NIKHIL (RA2211004010362)
- DADHANIA OMKUMAR (RA2211004010392)

**Supervisor:**
- Dr. Damodar Panigrahy, Assistant Professor, Department of ECE

**Institution:**
- SRM Institute of Science and Technology

## Citation

```bibtex
@misc{xr2text2024,
  title={XR2Text: Hierarchical Anatomical Query Transformer for Chest X-Ray Report Generation},
  author={Nikhil, S. and Omkumar, Dadhania},
  year={2024},
  institution={SRM Institute of Science and Technology}
}
```

## Acknowledgments

- **MIMIC-CXR Dataset**: Johnson et al., Physionet 2019
- **HuggingFace Transformers**: Wolf et al., 2020
- **Swin Transformer**: Liu et al., ICCV 2021
- **BioBART**: Yuan et al., 2022
- **PyTorch**: Paszke et al., 2019

## License

This project is developed for educational and research purposes as part of a Major Project submission at SRM Institute of Science and Technology.

**Not intended for clinical use without proper validation and regulatory approval.**

---

**Last Updated**: January 2026
**Training Status**: In Progress (Epoch 5/50)
**Expected Completion**: January 8, 2026
