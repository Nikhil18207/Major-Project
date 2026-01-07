"""
Curriculum Learning Strategy for XR2Text

NOVEL: Anatomical-Aware Curriculum Learning

This module implements a novel curriculum learning strategy that progressively
teaches the model from simple to complex cases, with a focus on anatomical regions.

The curriculum is organized by:
1. Difficulty: Normal cases -> Abnormal cases
2. Anatomical complexity: Single region -> Multiple regions
3. Clinical severity: Mild findings -> Severe findings

This is novel because:
- Prior work doesn't use curriculum learning for radiology report generation
- We incorporate anatomical awareness into the curriculum
- Progressive training improves model convergence and performance

Authors: S. Nikhil, Dadhania Omkumar
Supervisor: Dr. Damodar Panigrahy
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Callable
import numpy as np
from loguru import logger
import re


class AnatomicalCurriculumScheduler:
    """
    NOVEL: Anatomical-Aware Curriculum Learning Scheduler
    
    Organizes training samples by difficulty and anatomical complexity.
    """
    
    def __init__(
        self,
        stages: List[Dict] = None,
        difficulty_scorer: Optional[Callable] = None,
    ):
        """
        Initialize curriculum scheduler.
        
        Args:
            stages: List of curriculum stages with criteria
            difficulty_scorer: Function to score sample difficulty
        """
        if stages is None:
            # Default curriculum stages
            self.stages = [
                {
                    'name': 'normal_cases',
                    'epoch_start': 0,
                    'epoch_end': 5,
                    'criteria': {'max_findings': 1, 'severity': 'normal'},
                },
                {
                    'name': 'single_region',
                    'epoch_start': 5,
                    'epoch_end': 15,
                    'criteria': {'max_regions': 1, 'max_findings': 3},
                },
                {
                    'name': 'multi_region',
                    'epoch_start': 15,
                    'epoch_end': 30,
                    'criteria': {'max_regions': 3, 'max_findings': 5},
                },
                {
                    'name': 'complex_cases',
                    'epoch_start': 30,
                    'epoch_end': 50,
                    'criteria': {},  # All cases
                },
            ]
        else:
            self.stages = stages
        
        self.difficulty_scorer = difficulty_scorer or self._default_difficulty_scorer
        self.current_stage = 0
    
    def _default_difficulty_scorer(self, text: str) -> Dict[str, float]:
        """
        Score sample difficulty based on text content.
        
        Returns:
            Dictionary with difficulty metrics
        """
        text_lower = text.lower()
        
        # Count clinical findings
        findings_keywords = [
            'cardiomegaly', 'pneumonia', 'effusion', 'edema', 'consolidation',
            'atelectasis', 'pneumothorax', 'infiltrate', 'mass', 'nodule',
            'opacity', 'fracture', 'fibrosis'
        ]
        
        num_findings = sum(1 for keyword in findings_keywords if keyword in text_lower)
        
        # Count anatomical regions mentioned
        region_keywords = [
            'lung', 'heart', 'mediastinum', 'spine', 'diaphragm',
            'costophrenic', 'pleural', 'cardiac', 'pulmonary'
        ]
        
        num_regions = len(set(kw for kw in region_keywords if kw in text_lower))
        
        # Check for severity indicators
        severity_keywords = ['severe', 'extensive', 'large', 'massive', 'marked']
        has_severe = any(kw in text_lower for kw in severity_keywords)
        
        # Check if normal
        is_normal = 'normal' in text_lower or 'no acute' in text_lower or 'clear' in text_lower
        
        return {
            'num_findings': num_findings,
            'num_regions': num_regions,
            'has_severe': has_severe,
            'is_normal': is_normal,
            'difficulty': num_findings * 0.4 + num_regions * 0.3 + (has_severe * 0.3),
        }
    
    def get_current_stage(self, epoch: int) -> Dict:
        """Get current curriculum stage for given epoch."""
        for stage in self.stages:
            if stage['epoch_start'] <= epoch < stage['epoch_end']:
                return stage
        
        # Return last stage if epoch exceeds all
        return self.stages[-1]
    
    def should_include_sample(self, epoch: int, text: str) -> bool:
        """
        Determine if a sample should be included in current curriculum stage.
        
        Args:
            epoch: Current training epoch
            text: Sample text
            
        Returns:
            True if sample should be included
        """
        stage = self.get_current_stage(epoch)
        criteria = stage.get('criteria', {})
        
        if not criteria:  # Empty criteria = include all
            return True
        
        difficulty = self.difficulty_scorer(text)
        
        # Check criteria
        if 'max_findings' in criteria:
            if difficulty['num_findings'] > criteria['max_findings']:
                return False
        
        if 'max_regions' in criteria:
            if difficulty['num_regions'] > criteria['max_regions']:
                return False
        
        if 'severity' in criteria:
            if criteria['severity'] == 'normal' and not difficulty['is_normal']:
                return False
        
        return True
    
    def get_stage_name(self, epoch: int) -> str:
        """Get name of current curriculum stage."""
        stage = self.get_current_stage(epoch)
        return stage.get('name', 'unknown')


class CurriculumDataset(Dataset):
    """
    Dataset wrapper that implements curriculum learning.
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        curriculum_scheduler: AnatomicalCurriculumScheduler,
        current_epoch: int = 0,
    ):
        self.base_dataset = base_dataset
        self.curriculum_scheduler = curriculum_scheduler
        self.current_epoch = current_epoch
        
        # Filter samples based on current curriculum stage
        self.valid_indices = self._filter_samples()
        
        logger.info(
            f"Curriculum stage '{self.curriculum_scheduler.get_stage_name(current_epoch)}': "
            f"{len(self.valid_indices)}/{len(base_dataset)} samples"
        )
    
    def _filter_samples(self) -> List[int]:
        """Filter samples based on current curriculum stage."""
        valid_indices = []
        
        for idx in range(len(self.base_dataset)):
            # Get sample text
            sample = self.base_dataset[idx]
            text = sample.get('raw_text', '') or sample.get('findings', '') or ''
            
            if self.curriculum_scheduler.should_include_sample(self.current_epoch, text):
                valid_indices.append(idx)
        
        return valid_indices
    
    def update_epoch(self, epoch: int):
        """Update current epoch and re-filter samples."""
        self.current_epoch = epoch
        self.valid_indices = self._filter_samples()
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int):
        actual_idx = self.valid_indices[idx]
        return self.base_dataset[actual_idx]


def create_curriculum_dataloader(
    base_dataset: Dataset,
    curriculum_scheduler: AnatomicalCurriculumScheduler,
    current_epoch: int,
    batch_size: int,
    **dataloader_kwargs,
) -> DataLoader:
    """
    Create a DataLoader with curriculum learning.
    
    Args:
        base_dataset: Base dataset
        curriculum_scheduler: Curriculum scheduler
        current_epoch: Current training epoch
        batch_size: Batch size
        **dataloader_kwargs: Additional DataLoader arguments
        
    Returns:
        Curriculum-aware DataLoader
    """
    curriculum_dataset = CurriculumDataset(
        base_dataset=base_dataset,
        curriculum_scheduler=curriculum_scheduler,
        current_epoch=current_epoch,
    )
    
    return DataLoader(
        curriculum_dataset,
        batch_size=batch_size,
        **dataloader_kwargs,
    )

