# XR2Text Research Paper

## Files

| File | Description |
|------|-------------|
| `XR2Text_HAQT_ARR_Paper.tex` | Main IEEE format paper (conference/journal ready) |
| `XR2Text_Supplementary.tex` | Supplementary materials with additional details |

## Figures Included (from notebooks)

The paper uses **15 figures** generated from the evaluation notebooks:

| Figure | File | Section |
|--------|------|---------|
| Architecture | `haqt_arr_spatial_priors.png` | Methodology |
| Training Curves | `training_curves.png` | Results |
| Metrics Comparison | `metrics_comparison.png` | Results |
| Entity Detection | `entity_detection.png` | Results |
| ROC Curves | `roc_curves.png` | Results |
| Human Evaluation | `human_evaluation_results.png` | Results |
| Cross-Dataset | `cross_dataset_evaluation.png` | Results |
| Error Analysis | `error_analysis.png` | Results |
| HAQT-ARR Ablation | `haqt_arr_ablation.png` | Ablation |
| Encoder Ablation | `encoder_ablation.png` | Ablation |
| Decoder Radar | `decoder_radar.png` | Ablation |
| Region Importance | `region_importance_analysis.png` | Discussion |
| Attention Overlays | `attention_overlays.png` | Discussion |
| Clinical Findings | `clinical_findings_distribution.png` | Experimental Setup |

All figures are located in: `../data/figures/`

## Current Results Used

These are the results from the **previous 50-epoch training run**:

### Main Results (Table III in paper)
| Metric | Value |
|--------|-------|
| BLEU-1 | 0.421 |
| BLEU-4 | 0.172 |
| ROUGE-L | 0.358 |
| METEOR | 0.198 |
| CIDEr | 0.412 |
| Clinical Accuracy | 0.856 |

### Ablation Results (Table VI in paper)
| Configuration | BLEU-4 | ROUGE-L |
|---------------|--------|---------|
| Full HAQT-ARR | 0.172 | 0.358 |
| w/o Spatial Priors | 0.158 | 0.332 |
| w/o Adaptive Routing | 0.162 | 0.340 |
| w/o Cross-Region | 0.165 | 0.345 |
| Standard Projection | 0.142 | 0.312 |

---

## How to Update with New Results

When the new training completes, update these sections in the paper:

### 1. Abstract (Line ~25)
```latex
% Update these numbers:
BLEU-4 of 0.172 and ROUGE-L of 0.358
improvements of 21.1\% and 14.7\%
```

### 2. Table III - Main Results (Line ~285)
```latex
\textbf{XR2Text (Ours)} & -- & \textbf{0.421} & \textbf{0.172} & \textbf{0.358} & \textbf{0.198} \\
```

### 3. Table IV - Clinical Results (Line ~305)
```latex
\textbf{XR2Text (Ours)} & \textbf{0.913} & \textbf{0.398} & \textbf{0.375} & \textbf{479} \\
```

### 4. Table VI - Ablation (Line ~385)
Update all ablation numbers if re-running ablations.

### 5. Supplementary - Training Progress Table
Update per-epoch metrics in supplementary material.

---

## Compilation Instructions

### Using pdflatex:
```bash
pdflatex XR2Text_HAQT_ARR_Paper.tex
bibtex XR2Text_HAQT_ARR_Paper
pdflatex XR2Text_HAQT_ARR_Paper.tex
pdflatex XR2Text_HAQT_ARR_Paper.tex
```

### Using Overleaf:
1. Create new project
2. Upload both .tex files
3. Set `XR2Text_HAQT_ARR_Paper.tex` as main document
4. Compile with pdfLaTeX

---

## Target Venues

This paper is formatted for IEEE conferences/journals:

### Conferences (Tier 1)
- **CVPR** - Computer Vision and Pattern Recognition
- **MICCAI** - Medical Image Computing and Computer Assisted Intervention
- **AAAI** - Association for Advancement of Artificial Intelligence
- **ACL** - Association for Computational Linguistics

### Journals (Tier 1)
- **TMI** - IEEE Transactions on Medical Imaging
- **MedIA** - Medical Image Analysis
- **JBHI** - IEEE Journal of Biomedical and Health Informatics
- **Nature Medicine** (if results are strong enough)

---

## Checklist Before Submission

- [ ] Update all results with final training numbers
- [ ] Add architecture figure (`figures/architecture.pdf`)
- [ ] Add training curves figure (`figures/training_curves.pdf`)
- [ ] Add attention visualization examples
- [ ] Proofread all sections
- [ ] Check all citations are correct
- [ ] Verify page limit for target venue
- [ ] Update author affiliations
- [ ] Add acknowledgments/funding info
- [ ] Register ORCID IDs

---

## Quick Result Update Script

After training completes, run this to get new results:
```python
import pandas as pd

# Load new results
df = pd.read_csv('../data/statistics/best_results.csv')
print("New Results for Paper:")
print(df.to_string())

# Load human eval
human = pd.read_csv('../data/statistics/human_evaluation_results.csv')
print("\nHuman Evaluation:")
print(human.to_string())
```

---

## Contact

- **S. Nikhil** - Primary author
- **Dadhania Omkumar** - Co-author
- **Dr. Damodar Panigrahy** - Supervisor
