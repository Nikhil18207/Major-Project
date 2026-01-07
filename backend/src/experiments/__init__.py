"""
Experiment utilities for XR2Text research.
"""

from .ablation_runner import (
    AblationRunner,
    AblationResult,
    PUBLISHED_BASELINES,
    create_baseline_comparison_table,
)

__all__ = [
    'AblationRunner',
    'AblationResult',
    'PUBLISHED_BASELINES',
    'create_baseline_comparison_table',
]
