"""
Curriculum Learning Strategy for XR2Text

NOVEL: Anatomical-Aware Curriculum Learning with Pre-computed Difficulty Scores

This module implements a novel curriculum learning strategy that progressively
teaches the model from simple to complex cases, with a focus on anatomical regions.

The curriculum is organized by:
1. Difficulty: Normal cases -> Abnormal cases
2. Anatomical complexity: Single region -> Multiple regions
3. Clinical severity: Mild findings -> Severe findings

Key Optimizations:
- Pre-computes difficulty scores once for the entire dataset
- Uses indexed sampling instead of per-epoch filtering
- Efficient O(1) lookup for sample difficulty

This is novel because:
- Prior work doesn't use curriculum learning for radiology report generation
- We incorporate anatomical awareness into the curriculum
- Progressive training improves model convergence and performance

Authors: S. Nikhil, Dadhania Omkumar
Supervisor: Dr. Damodar Panigrahy
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Dict, Optional, Callable, Tuple
import numpy as np
from loguru import logger
import re
from functools import lru_cache


class AnatomicalCurriculumScheduler:
    """
    NOVEL: Anatomical-Aware Curriculum Learning Scheduler

    Organizes training samples by difficulty and anatomical complexity.
    Uses pre-computed difficulty scores for efficiency.
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
            # IMPROVED: Extended curriculum stages for better convergence
            # Longer stages allow the model to fully learn each difficulty level
            self.stages = [
                {
                    'name': 'warmup',
                    'epoch_start': 0,
                    'epoch_end': 10,
                    'criteria': {'max_findings': 1, 'severity': 'normal'},
                    'description': 'Normal/clear cases only - learn basic patterns',
                },
                {
                    'name': 'easy',
                    'epoch_start': 10,
                    'epoch_end': 25,
                    'criteria': {'max_findings': 2, 'max_regions': 2},
                    'description': 'Simple abnormalities - single region, few findings',
                },
                {
                    'name': 'medium',
                    'epoch_start': 25,
                    'epoch_end': 50,
                    'criteria': {'max_findings': 4, 'max_regions': 4},
                    'description': 'Moderate complexity - multiple regions, several findings',
                },
                {
                    'name': 'hard',
                    'epoch_start': 50,
                    'epoch_end': 80,
                    'criteria': {},  # All cases including complex multi-finding
                    'description': 'Full complexity - all samples including complex cases',
                },
                {
                    'name': 'finetune',
                    'epoch_start': 80,
                    'epoch_end': 100,
                    'criteria': {},  # All cases with focus on difficult ones
                    'description': 'Fine-tuning on full dataset',
                },
            ]
        else:
            self.stages = stages

        self.difficulty_scorer = difficulty_scorer or self._default_difficulty_scorer
        self.current_stage = 0

        # Cache for pre-computed indices per stage
        self._stage_indices_cache: Dict[str, List[int]] = {}
        self._difficulty_scores_cache: Dict[int, Dict] = {}

    def _default_difficulty_scorer(self, text: str) -> Dict[str, float]:
        """
        Score sample difficulty based on text content.

        Returns:
            Dictionary with difficulty metrics
        """
        text_lower = text.lower()

        # Count clinical findings (with negation awareness)
        findings_keywords = [
            'cardiomegaly', 'pneumonia', 'effusion', 'edema', 'consolidation',
            'atelectasis', 'pneumothorax', 'infiltrate', 'mass', 'nodule',
            'opacity', 'fracture', 'fibrosis', 'emphysema', 'hernia'
        ]

        # Negation patterns
        negation_patterns = ['no ', 'no evidence', 'without', 'absence', 'negative', 'unremarkable']

        num_findings = 0
        for keyword in findings_keywords:
            if keyword in text_lower:
                # Check if negated
                keyword_pos = text_lower.find(keyword)
                context_start = max(0, keyword_pos - 30)
                context = text_lower[context_start:keyword_pos]
                is_negated = any(neg in context for neg in negation_patterns)
                if not is_negated:
                    num_findings += 1

        # Count anatomical regions mentioned
        region_keywords = [
            'lung', 'heart', 'mediastinum', 'spine', 'diaphragm',
            'costophrenic', 'pleural', 'cardiac', 'pulmonary'
        ]

        num_regions = len(set(kw for kw in region_keywords if kw in text_lower))

        # Check for severity indicators
        severity_keywords = ['severe', 'extensive', 'large', 'massive', 'marked', 'significant']
        severity_score = sum(1 for kw in severity_keywords if kw in text_lower)

        # Check if normal
        normal_indicators = ['normal', 'no acute', 'clear', 'unremarkable', 'negative']
        is_normal = any(ind in text_lower for ind in normal_indicators) and num_findings == 0

        return {
            'num_findings': num_findings,
            'num_regions': num_regions,
            'severity_score': severity_score,
            'has_severe': severity_score > 0,
            'is_normal': is_normal,
            'difficulty': num_findings * 0.4 + num_regions * 0.3 + severity_score * 0.3,
        }

    def precompute_difficulty_scores(self, dataset: Dataset) -> Dict[int, Dict]:
        """
        Pre-compute difficulty scores for all samples in dataset.
        This is called once and cached for efficiency.

        Args:
            dataset: The training dataset

        Returns:
            Dictionary mapping sample index to difficulty scores
        """
        if self._difficulty_scores_cache:
            return self._difficulty_scores_cache

        logger.info(f"Pre-computing difficulty scores for {len(dataset)} samples...")

        for idx in range(len(dataset)):
            try:
                sample = dataset[idx]
                text = sample.get('raw_text', '') or sample.get('findings', '') or ''
                self._difficulty_scores_cache[idx] = self.difficulty_scorer(text)
            except Exception as e:
                # Default to medium difficulty if scoring fails
                self._difficulty_scores_cache[idx] = {
                    'num_findings': 1,
                    'num_regions': 1,
                    'severity_score': 0,
                    'has_severe': False,
                    'is_normal': False,
                    'difficulty': 0.5,
                }

        logger.info(f"Pre-computed {len(self._difficulty_scores_cache)} difficulty scores")
        return self._difficulty_scores_cache

    def get_stage_indices(self, dataset: Dataset, stage_name: str) -> List[int]:
        """
        Get pre-computed indices for a curriculum stage.

        Args:
            dataset: The training dataset
            stage_name: Name of the curriculum stage

        Returns:
            List of sample indices for this stage
        """
        # Include dataset identity in cache key to prevent stale indices
        # when dataset is recreated with same length but different samples
        cache_key = f"{stage_name}_{len(dataset)}_{id(dataset)}"

        if cache_key in self._stage_indices_cache:
            return self._stage_indices_cache[cache_key]

        # Ensure difficulty scores are pre-computed
        if not self._difficulty_scores_cache:
            self.precompute_difficulty_scores(dataset)

        # Find the stage
        stage = next((s for s in self.stages if s['name'] == stage_name), None)
        if stage is None:
            return list(range(len(dataset)))

        criteria = stage.get('criteria', {})
        if not criteria:
            # No criteria = all samples
            indices = list(range(len(dataset)))
        else:
            indices = []
            for idx, scores in self._difficulty_scores_cache.items():
                if self._sample_matches_criteria(scores, criteria):
                    indices.append(idx)

        self._stage_indices_cache[cache_key] = indices
        return indices

    def _sample_matches_criteria(self, scores: Dict, criteria: Dict) -> bool:
        """Check if a sample's scores match the curriculum criteria."""
        if 'max_findings' in criteria:
            if scores['num_findings'] > criteria['max_findings']:
                return False

        if 'max_regions' in criteria:
            if scores['num_regions'] > criteria['max_regions']:
                return False

        if 'severity' in criteria:
            if criteria['severity'] == 'normal' and not scores['is_normal']:
                return False

        return True
    
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
    Dataset wrapper that implements curriculum learning with O(1) sample lookup.

    Uses pre-computed difficulty scores and cached stage indices for efficiency.
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

        # Pre-compute difficulty scores once (cached in scheduler)
        self.curriculum_scheduler.precompute_difficulty_scores(base_dataset)

        # Get valid indices using pre-computed scores (O(1) lookup after first call)
        stage_name = self.curriculum_scheduler.get_stage_name(current_epoch)
        self.valid_indices = self.curriculum_scheduler.get_stage_indices(
            base_dataset, stage_name
        )

        logger.info(
            f"Curriculum stage '{stage_name}': "
            f"{len(self.valid_indices)}/{len(base_dataset)} samples"
        )

    def update_epoch(self, epoch: int):
        """Update current epoch and get new stage indices (O(1) if cached)."""
        self.current_epoch = epoch
        stage_name = self.curriculum_scheduler.get_stage_name(epoch)
        self.valid_indices = self.curriculum_scheduler.get_stage_indices(
            self.base_dataset, stage_name
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        actual_idx = self.valid_indices[idx]
        return self.base_dataset[actual_idx]

    def get_difficulty_distribution(self) -> Dict[str, int]:
        """Get distribution of difficulty levels in current stage."""
        distribution = {'easy': 0, 'medium': 0, 'hard': 0}
        scores = self.curriculum_scheduler._difficulty_scores_cache

        for idx in self.valid_indices:
            if idx in scores:
                difficulty = scores[idx]['difficulty']
                if difficulty < 0.3:
                    distribution['easy'] += 1
                elif difficulty < 0.7:
                    distribution['medium'] += 1
                else:
                    distribution['hard'] += 1

        return distribution


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

