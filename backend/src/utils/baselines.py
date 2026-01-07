"""
Baseline Models for Fair Comparison

This module implements baseline models (R2Gen, Show-Attend-Tell, etc.) 
for rigorous comparison with our HAQT-ARR approach.

Authors: S. Nikhil, Dadhania Omkumar
Supervisor: Dr. Damodar Panigrahy
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger


class BaselineProjection(nn.Module):
    """Standard linear projection layer (BLIP-2 style baseline)."""
    
    def __init__(
        self,
        visual_dim: int = 1024,
        language_dim: int = 768,
        num_queries: int = 32,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, language_dim))
        self.projection = nn.Linear(visual_dim, language_dim)
        self.cross_attn = nn.MultiheadAttention(language_dim, 8, batch_first=True)
        self.norm = nn.LayerNorm(language_dim)
        
    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: (B, N, visual_dim) from encoder
        Returns:
            projected: (B, num_queries, language_dim)
        """
        B = visual_features.shape[0]
        
        # Project visual features
        projected_visual = self.projection(visual_features)
        
        # Expand query tokens for batch
        queries = self.query_tokens.expand(B, -1, -1)
        
        # Cross-attention
        attended, _ = self.cross_attn(queries, projected_visual, projected_visual)
        output = self.norm(attended + queries)
        
        return output


class R2GenStyleProjection(nn.Module):
    """
    R2Gen-style projection with memory matrix.
    
    Reference: Chen et al., "Generating Radiology Reports via Memory-driven Transformer"
    """
    
    def __init__(
        self,
        visual_dim: int = 1024,
        language_dim: int = 768,
        num_memories: int = 64,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_memories = num_memories
        
        # Memory matrix (key-value pairs)
        self.memory_keys = nn.Parameter(torch.randn(num_memories, language_dim))
        self.memory_values = nn.Parameter(torch.randn(num_memories, language_dim))
        
        # Projection
        self.visual_proj = nn.Linear(visual_dim, language_dim)
        
        # Memory attention
        self.memory_attn = nn.MultiheadAttention(language_dim, num_heads, batch_first=True)
        self.visual_attn = nn.MultiheadAttention(language_dim, num_heads, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(language_dim * 2, language_dim),
            nn.LayerNorm(language_dim),
            nn.ReLU(),
        )
        
    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: (B, N, visual_dim)
        Returns:
            output: (B, N, language_dim)
        """
        B, N, _ = visual_features.shape
        
        # Project visual features
        visual_proj = self.visual_proj(visual_features)
        
        # Memory retrieval
        memory_keys = self.memory_keys.unsqueeze(0).expand(B, -1, -1)
        memory_values = self.memory_values.unsqueeze(0).expand(B, -1, -1)
        
        memory_output, _ = self.memory_attn(visual_proj, memory_keys, memory_values)
        
        # Combine visual and memory
        combined = torch.cat([visual_proj, memory_output], dim=-1)
        output = self.output_proj(combined)
        
        return output


class ShowAttendTellBaseline(nn.Module):
    """
    Show, Attend and Tell baseline for image captioning.
    
    Reference: Xu et al., "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"
    """
    
    def __init__(
        self,
        visual_dim: int = 1024,
        language_dim: int = 768,
        attention_dim: int = 512,
    ):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, language_dim)
        self.attention = nn.Sequential(
            nn.Linear(language_dim * 2, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )
        self.context_proj = nn.Linear(language_dim, language_dim)
        
    def forward(
        self, 
        visual_features: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            visual_features: (B, N, visual_dim)
            hidden_state: (B, language_dim) decoder hidden state
        Returns:
            context: (B, language_dim)
            attention_weights: (B, N)
        """
        B, N, _ = visual_features.shape
        
        # Project visual features
        visual_proj = self.visual_proj(visual_features)  # (B, N, language_dim)
        
        if hidden_state is None:
            hidden_state = visual_proj.mean(dim=1)  # (B, language_dim)
        
        # Compute attention
        hidden_expanded = hidden_state.unsqueeze(1).expand(-1, N, -1)
        attn_input = torch.cat([visual_proj, hidden_expanded], dim=-1)
        attn_weights = self.attention(attn_input).squeeze(-1)  # (B, N)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # Weighted sum
        context = (visual_proj * attn_weights.unsqueeze(-1)).sum(dim=1)
        context = self.context_proj(context)
        
        return context, attn_weights


class BaselineModelFactory:
    """Factory to create baseline models for comparison."""
    
    BASELINES = {
        'standard': BaselineProjection,
        'r2gen': R2GenStyleProjection,
        'show_attend_tell': ShowAttendTellBaseline,
    }
    
    @classmethod
    def create(
        cls,
        baseline_name: str,
        visual_dim: int = 1024,
        language_dim: int = 768,
        **kwargs,
    ) -> nn.Module:
        """Create a baseline model by name."""
        if baseline_name not in cls.BASELINES:
            raise ValueError(f"Unknown baseline: {baseline_name}. Available: {list(cls.BASELINES.keys())}")
        
        return cls.BASELINES[baseline_name](
            visual_dim=visual_dim,
            language_dim=language_dim,
            **kwargs,
        )


def run_baseline_evaluation(
    model,
    test_loader,
    device: str = 'cuda',
    baseline_name: str = 'standard',
) -> Dict[str, float]:
    """
    Run evaluation with a specific baseline configuration.
    
    Args:
        model: The XR2Text model
        test_loader: Test data loader
        device: Device to run on
        baseline_name: Name of baseline configuration
        
    Returns:
        Dictionary of metrics
    """
    from src.utils.metrics import compute_metrics
    
    model.eval()
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['images'].to(device)
            raw_texts = batch['raw_texts']
            
            # Generate predictions
            _, generated_texts, _ = model.generate(
                images=images,
                max_length=256,
                num_beams=4,
            )
            
            all_predictions.extend(generated_texts)
            all_references.extend(raw_texts)
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_references, include_all=True)
    return metrics
