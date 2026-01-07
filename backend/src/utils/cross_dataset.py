"""
Cross-Dataset Evaluation Module

This module provides tools for evaluating models across multiple datasets
to demonstrate generalization capability.

Supported Datasets:
1. MIMIC-CXR (primary)
2. IU-XRay (Indiana University Chest X-Ray)
3. CheXpert (Stanford)
4. PadChest (optional)

Essential for publication - demonstrates model generalization.

Authors: S. Nikhil, Dadhania Omkumar
Supervisor: Dr. Damodar Panigrahy
"""

import os
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from loguru import logger


class IUXRayDataset(Dataset):
    """
    Indiana University Chest X-Ray Dataset.
    
    Dataset Structure:
    - images/: DICOM or PNG X-ray images
    - reports/: XML files with findings and impressions
    
    Reference: https://openi.nlm.nih.gov/
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'test',
        transform=None,
        max_samples: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Load metadata
        self.samples = self._load_samples()
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        logger.info(f"Loaded {len(self.samples)} IU-XRay {split} samples")
        
    def _load_samples(self) -> List[Dict]:
        """Load sample metadata."""
        samples = []
        
        # Check for preprocessed CSV
        csv_path = os.path.join(self.data_dir, f'{self.split}.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                samples.append({
                    'image_path': os.path.join(self.data_dir, 'images', row['image_id']),
                    'report': row.get('findings', '') + ' ' + row.get('impression', ''),
                    'findings': row.get('findings', ''),
                    'impression': row.get('impression', ''),
                })
        else:
            # Create placeholder for dataset structure
            logger.warning(f"IU-XRay CSV not found at {csv_path}. Using placeholder.")
            samples = self._create_placeholder_samples()
        
        return samples
    
    def _create_placeholder_samples(self) -> List[Dict]:
        """Create placeholder samples for demonstration."""
        return [
            {
                'image_path': 'placeholder',
                'report': 'Heart size is normal. Lungs are clear.',
                'findings': 'Heart size is normal. Lungs are clear.',
                'impression': 'No acute cardiopulmonary process.',
            }
            for _ in range(100)
        ]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        if os.path.exists(sample['image_path']):
            image = Image.open(sample['image_path']).convert('RGB')
        else:
            # Placeholder image
            image = Image.new('RGB', (384, 384), color='gray')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'report': sample['report'],
            'findings': sample['findings'],
            'impression': sample['impression'],
        }


class CheXpertDataset(Dataset):
    """
    CheXpert Dataset (Stanford).
    
    Note: CheXpert doesn't have text reports, only labels.
    We use this for label prediction evaluation.
    
    Reference: https://stanfordmlgroup.github.io/competitions/chexpert/
    """
    
    LABELS = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
        'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
        'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
        'Pleural Other', 'Fracture', 'Support Devices',
    ]
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'valid',
        transform=None,
        max_samples: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        self.samples = self._load_samples()
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        logger.info(f"Loaded {len(self.samples)} CheXpert {split} samples")
    
    def _load_samples(self) -> List[Dict]:
        """Load sample metadata."""
        csv_path = os.path.join(self.data_dir, f'{self.split}.csv')
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            samples = []
            for _, row in df.iterrows():
                labels = {label: row.get(label, 0) for label in self.LABELS}
                samples.append({
                    'image_path': os.path.join(self.data_dir, row['Path']),
                    'labels': labels,
                })
            return samples
        else:
            logger.warning(f"CheXpert CSV not found at {csv_path}. Using placeholder.")
            return self._create_placeholder_samples()
    
    def _create_placeholder_samples(self) -> List[Dict]:
        """Create placeholder samples."""
        return [
            {
                'image_path': 'placeholder',
                'labels': {label: np.random.choice([0, 1, -1]) for label in self.LABELS},
            }
            for _ in range(100)
        ]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if os.path.exists(sample['image_path']):
            image = Image.open(sample['image_path']).convert('RGB')
        else:
            image = Image.new('RGB', (384, 384), color='gray')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'labels': sample['labels'],
        }


class CrossDatasetEvaluator:
    """
    Evaluate models across multiple datasets.
    """
    
    def __init__(
        self,
        model,
        device: str = 'cuda',
        batch_size: int = 8,
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.results = {}
        
    def evaluate_iu_xray(
        self,
        data_dir: str,
        transform=None,
        max_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate on IU-XRay dataset."""
        from src.utils.metrics import compute_metrics
        
        logger.info("Evaluating on IU-XRay dataset...")
        
        dataset = IUXRayDataset(data_dir, 'test', transform, max_samples)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        predictions = []
        references = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                images = batch['image'].to(self.device)
                
                _, generated, _ = self.model.generate(
                    images=images,
                    max_length=256,
                    num_beams=4,
                )
                
                predictions.extend(generated)
                references.extend(batch['report'])
        
        metrics = compute_metrics(predictions, references, include_all=True)
        self.results['iu_xray'] = metrics
        
        logger.info(f"IU-XRay Results: BLEU-4={metrics['bleu_4']:.4f}, ROUGE-L={metrics['rouge_l']:.4f}")
        
        return metrics
    
    def evaluate_chexpert_labels(
        self,
        data_dir: str,
        transform=None,
        max_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate label extraction from generated reports on CheXpert.
        
        This tests whether the model's generated text contains the correct findings.
        """
        logger.info("Evaluating on CheXpert dataset (label extraction)...")
        
        dataset = CheXpertDataset(data_dir, 'valid', transform, max_samples)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        all_predictions = []
        all_labels = []
        
        # Keywords for each label
        label_keywords = {
            'Cardiomegaly': ['cardiomegaly', 'enlarged heart', 'cardiac enlargement'],
            'Edema': ['edema', 'pulmonary edema', 'vascular congestion'],
            'Consolidation': ['consolidation', 'consolidative'],
            'Pneumonia': ['pneumonia', 'infectious', 'infection'],
            'Atelectasis': ['atelectasis', 'collapse'],
            'Pneumothorax': ['pneumothorax'],
            'Pleural Effusion': ['effusion', 'pleural fluid'],
        }
        
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                images = batch['image'].to(self.device)
                labels = batch['labels']
                
                _, generated, _ = self.model.generate(
                    images=images,
                    max_length=256,
                    num_beams=4,
                )
                
                # Extract labels from generated text
                for text in generated:
                    text_lower = text.lower()
                    pred_labels = {}
                    for label, keywords in label_keywords.items():
                        pred_labels[label] = any(kw in text_lower for kw in keywords)
                    all_predictions.append(pred_labels)
                
                all_labels.extend([dict(l) for l in labels])
        
        # Calculate metrics per label
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        metrics = {}
        for label in label_keywords.keys():
            y_true = [l.get(label, 0) == 1 for l in all_labels]
            y_pred = [p.get(label, False) for p in all_predictions]
            
            if sum(y_true) > 0:
                metrics[f'{label}_precision'] = precision_score(y_true, y_pred, zero_division=0)
                metrics[f'{label}_recall'] = recall_score(y_true, y_pred, zero_division=0)
                metrics[f'{label}_f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Average metrics
        precisions = [v for k, v in metrics.items() if 'precision' in k]
        recalls = [v for k, v in metrics.items() if 'recall' in k]
        f1s = [v for k, v in metrics.items() if 'f1' in k]
        
        metrics['avg_precision'] = np.mean(precisions) if precisions else 0
        metrics['avg_recall'] = np.mean(recalls) if recalls else 0
        metrics['avg_f1'] = np.mean(f1s) if f1s else 0
        
        self.results['chexpert'] = metrics
        
        logger.info(f"CheXpert Results: Avg F1={metrics['avg_f1']:.4f}")
        
        return metrics
    
    def evaluate_domain_shift(
        self,
        source_metrics: Dict[str, float],
        target_metrics: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Analyze domain shift between source and target datasets.
        
        Returns degradation percentages and transfer scores.
        """
        results = {}
        
        for metric in ['bleu_4', 'rouge_l', 'meteor']:
            if metric in source_metrics and metric in target_metrics:
                source_val = source_metrics[metric]
                target_val = target_metrics[metric]
                
                degradation = (source_val - target_val) / source_val * 100 if source_val > 0 else 0
                transfer_score = target_val / source_val if source_val > 0 else 0
                
                results[f'{metric}_degradation_%'] = degradation
                results[f'{metric}_transfer_score'] = transfer_score
        
        # Overall transfer score
        transfer_scores = [v for k, v in results.items() if 'transfer_score' in k]
        results['overall_transfer_score'] = np.mean(transfer_scores) if transfer_scores else 0
        
        return results
    
    def generate_cross_dataset_report(self) -> str:
        """Generate comprehensive cross-dataset evaluation report."""
        report = """
# Cross-Dataset Evaluation Report

## Summary

This report presents evaluation results across multiple chest X-ray datasets
to demonstrate the generalization capability of our model.

"""
        
        for dataset, metrics in self.results.items():
            report += f"### {dataset.upper()}\n\n"
            for metric, value in metrics.items():
                report += f"- {metric}: {value:.4f}\n"
            report += "\n"
        
        return report
    
    def generate_latex_table(self) -> str:
        """Generate LaTeX table for cross-dataset results."""
        latex = """
\\begin{table}[h]
\\centering
\\caption{Cross-Dataset Evaluation Results}
\\label{tab:cross_dataset}
\\begin{tabular}{l|cccc}
\\hline
\\textbf{Dataset} & \\textbf{BLEU-4} & \\textbf{ROUGE-L} & \\textbf{METEOR} & \\textbf{CIDEr} \\\\
\\hline
"""
        
        for dataset, metrics in self.results.items():
            latex += f"{dataset} & "
            latex += f"{metrics.get('bleu_4', 0):.3f} & "
            latex += f"{metrics.get('rouge_l', 0):.3f} & "
            latex += f"{metrics.get('meteor', 0):.3f} & "
            latex += f"{metrics.get('cider', 0):.3f} \\\\\n"
        
        latex += """\\hline
\\end{tabular}
\\end{table}
"""
        return latex


def run_cross_dataset_evaluation(
    model,
    mimic_test_loader,
    iu_xray_dir: Optional[str] = None,
    chexpert_dir: Optional[str] = None,
    device: str = 'cuda',
) -> Dict[str, Dict]:
    """
    Run comprehensive cross-dataset evaluation.
    
    Args:
        model: The trained model
        mimic_test_loader: Test loader for MIMIC-CXR
        iu_xray_dir: Directory containing IU-XRay data
        chexpert_dir: Directory containing CheXpert data
        device: Device to run on
        
    Returns:
        Dictionary with results for each dataset
    """
    evaluator = CrossDatasetEvaluator(model, device)
    
    results = {}
    
    # Evaluate on MIMIC-CXR (primary)
    logger.info("Evaluating on MIMIC-CXR (primary dataset)...")
    from src.utils.metrics import compute_metrics
    
    predictions = []
    references = []
    
    model.eval()
    with torch.no_grad():
        for batch in mimic_test_loader:
            images = batch['images'].to(device)
            _, generated, _ = model.generate(images=images, max_length=256, num_beams=4)
            predictions.extend(generated)
            references.extend(batch['raw_texts'])
    
    results['mimic_cxr'] = compute_metrics(predictions, references, include_all=True)
    evaluator.results['mimic_cxr'] = results['mimic_cxr']
    
    # Evaluate on IU-XRay if available
    if iu_xray_dir and os.path.exists(iu_xray_dir):
        from src.data.transforms import get_val_transforms
        transform = get_val_transforms(384)
        results['iu_xray'] = evaluator.evaluate_iu_xray(iu_xray_dir, transform)
        
        # Calculate domain shift
        results['domain_shift_iu'] = evaluator.evaluate_domain_shift(
            results['mimic_cxr'], results['iu_xray']
        )
    
    # Evaluate on CheXpert if available
    if chexpert_dir and os.path.exists(chexpert_dir):
        from src.data.transforms import get_val_transforms
        transform = get_val_transforms(384)
        results['chexpert'] = evaluator.evaluate_chexpert_labels(chexpert_dir, transform)
    
    # Generate report
    print("\n" + "=" * 60)
    print("CROSS-DATASET EVALUATION RESULTS")
    print("=" * 60)
    
    for dataset, metrics in results.items():
        print(f"\n{dataset.upper()}:")
        for metric, value in list(metrics.items())[:8]:  # Show first 8 metrics
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
    
    return results
