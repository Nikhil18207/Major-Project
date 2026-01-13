"""
Self-Supervised Pre-Training Module for XR2Text

This module implements self-supervised pre-training objectives that can
be used to improve feature representations before supervised fine-tuning.

Novel Contributions:
1. Contrastive Image-Text Learning - SimCLR-style contrastive pre-training
2. Masked Region Prediction - Predict masked anatomical regions
3. Report Reconstruction - Reconstruct reports from masked inputs
4. Cross-Modal Matching - Learn aligned image-text representations

This is NOVEL because:
- Self-supervised pre-training is standard in vision-language models
- Medical domain pre-training improves downstream performance
- Anatomical-aware masking leverages domain knowledge
- Combined objectives create robust representations

Authors: S. Nikhil, Dadhania Omkumar
Supervisor: Dr. Damodar Panigrahy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from loguru import logger
import numpy as np


@dataclass
class PreTrainingOutput:
    """Container for pre-training outputs."""
    contrastive_loss: torch.Tensor
    mrp_loss: torch.Tensor  # Masked Region Prediction
    reconstruction_loss: Optional[torch.Tensor]
    total_loss: torch.Tensor
    # Metrics
    contrastive_accuracy: float
    mrp_accuracy: float


class ContrastiveLearningLoss(nn.Module):
    """
    NOVEL: Contrastive Learning for Vision-Language Alignment

    Uses SimCLR/CLIP-style contrastive learning to align
    visual and textual representations in a shared semantic space.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        normalize: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute contrastive loss.

        Args:
            image_features: Image embeddings (B, D)
            text_features: Text embeddings (B, D)

        Returns:
            Tuple of (loss, accuracy)
        """
        batch_size = image_features.shape[0]
        device = image_features.device

        # Normalize features
        if self.normalize:
            image_features = F.normalize(image_features, p=2, dim=-1)
            text_features = F.normalize(text_features, p=2, dim=-1)

        # Compute similarity matrix
        # (B, D) @ (D, B) -> (B, B)
        similarity_matrix = torch.matmul(
            image_features, text_features.t()
        ) / self.temperature

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=device)

        # Bidirectional contrastive loss
        loss_i2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2i = F.cross_entropy(similarity_matrix.t(), labels)

        loss = (loss_i2t + loss_t2i) / 2.0

        # Compute accuracy
        preds_i2t = similarity_matrix.argmax(dim=-1)
        preds_t2i = similarity_matrix.t().argmax(dim=-1)
        accuracy = (
            (preds_i2t == labels).float().mean().item() +
            (preds_t2i == labels).float().mean().item()
        ) / 2.0

        return loss, accuracy


class MaskedRegionPrediction(nn.Module):
    """
    NOVEL: Masked Region Prediction for Visual Pre-Training

    Masks anatomical regions and predicts their features.
    Similar to BERT's masked language modeling but for images.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_regions: int = 7,
        mask_ratio: float = 0.15,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_regions = num_regions
        self.mask_ratio = mask_ratio

        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        region_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply masking and predict original features.

        Args:
            visual_features: Visual features (B, N, D)
            region_masks: Optional pre-computed masks

        Returns:
            Tuple of (loss, predicted_features, accuracy)
        """
        batch_size, num_patches, hidden_dim = visual_features.shape
        device = visual_features.device

        # Generate random masks if not provided
        if region_masks is None:
            num_mask = int(num_patches * self.mask_ratio)
            mask_indices = torch.rand(batch_size, num_patches, device=device).argsort(dim=-1)[:, :num_mask]
            mask = torch.zeros(batch_size, num_patches, device=device)
            mask.scatter_(1, mask_indices, 1.0)
            mask = mask.bool()
        else:
            mask = region_masks.bool()

        # Store original features for targets
        target_features = visual_features.clone()

        # Apply masking
        masked_features = visual_features.clone()
        mask_expanded = mask.unsqueeze(-1).expand_as(masked_features)
        masked_features = torch.where(
            mask_expanded,
            self.mask_token.expand(batch_size, num_patches, -1),
            masked_features
        )

        # Predict original features
        predicted_features = self.prediction_head(masked_features)

        # Compute loss only on masked positions
        loss = F.mse_loss(
            predicted_features[mask],
            target_features[mask],
        )

        # Compute accuracy (cosine similarity > threshold)
        with torch.no_grad():
            pred_norm = F.normalize(predicted_features[mask], dim=-1)
            target_norm = F.normalize(target_features[mask], dim=-1)
            cosine_sim = (pred_norm * target_norm).sum(dim=-1)
            accuracy = (cosine_sim > 0.5).float().mean().item()

        return loss, predicted_features, accuracy


class ReportReconstructionLoss(nn.Module):
    """
    NOVEL: Report Reconstruction Pre-Training

    Reconstructs masked portions of reports to learn
    language patterns specific to radiology.
    """

    def __init__(
        self,
        vocab_size: int = 50265,  # BioBART vocab size
        hidden_dim: int = 768,
        mask_ratio: float = 0.15,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.mask_ratio = mask_ratio

        # Prediction head
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute masked language modeling loss.

        Args:
            hidden_states: Decoder hidden states (B, seq_len, D)
            labels: Target token IDs (B, seq_len)
            attention_mask: Attention mask

        Returns:
            Reconstruction loss
        """
        # Predict logits
        logits = self.lm_head(hidden_states)

        # Compute cross-entropy loss
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            labels.view(-1),
            ignore_index=-100,  # Ignore padding
        )

        return loss


class CrossModalMatchingLoss(nn.Module):
    """
    NOVEL: Cross-Modal Matching

    Binary classification task: does the image match the report?
    Used to learn fine-grained alignment.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute matching loss.

        Args:
            image_features: Pooled image features (B, D)
            text_features: Pooled text features (B, D)
            labels: Binary labels (1 = match, 0 = no match)

        Returns:
            Binary cross-entropy loss
        """
        # Concatenate features
        combined = torch.cat([image_features, text_features], dim=-1)

        # Predict matching score
        logits = self.classifier(combined).squeeze(-1)

        # Binary cross-entropy
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        return loss


class MultimodalPreTrainer(nn.Module):
    """
    NOVEL: Unified Multimodal Pre-Training Module

    Combines multiple self-supervised objectives for
    comprehensive pre-training of the XR2Text model.
    """

    def __init__(
        self,
        visual_dim: int = 1024,
        language_dim: int = 768,
        temperature: float = 0.07,
        mask_ratio: float = 0.15,
        use_mrp: bool = True,
        use_contrastive: bool = True,
        use_matching: bool = False,
    ):
        super().__init__()

        self.visual_dim = visual_dim
        self.language_dim = language_dim
        self.use_mrp = use_mrp
        self.use_contrastive = use_contrastive
        self.use_matching = use_matching

        # Visual projection (for contrastive)
        self.visual_projector = nn.Sequential(
            nn.Linear(visual_dim, language_dim),
            nn.GELU(),
            nn.Linear(language_dim, language_dim),
        )

        # Text projection (for contrastive)
        self.text_projector = nn.Sequential(
            nn.Linear(language_dim, language_dim),
            nn.GELU(),
            nn.Linear(language_dim, language_dim),
        )

        # Contrastive learning
        if use_contrastive:
            self.contrastive_loss = ContrastiveLearningLoss(temperature)

        # Masked region prediction
        if use_mrp:
            self.mrp = MaskedRegionPrediction(visual_dim, mask_ratio=mask_ratio)

        # Cross-modal matching
        if use_matching:
            self.matching_loss = CrossModalMatchingLoss(language_dim)

        logger.info(
            f"MultimodalPreTrainer initialized: "
            f"contrastive={use_contrastive}, mrp={use_mrp}, matching={use_matching}"
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        matching_labels: Optional[torch.Tensor] = None,
    ) -> PreTrainingOutput:
        """
        Compute all pre-training losses.

        Args:
            visual_features: Visual features (B, N, D_v)
            text_features: Text features (B, seq_len, D_t)
            matching_labels: Optional labels for matching task

        Returns:
            PreTrainingOutput with all losses
        """
        device = visual_features.device
        losses = []

        # Pool features for contrastive learning
        visual_pooled = visual_features.mean(dim=1)  # (B, D_v)
        text_pooled = text_features.mean(dim=1)  # (B, D_t)

        # Project to shared space
        visual_proj = self.visual_projector(visual_pooled)
        text_proj = self.text_projector(text_pooled)

        # Contrastive loss
        contrastive_loss = torch.tensor(0.0, device=device)
        contrastive_acc = 0.0
        if self.use_contrastive:
            contrastive_loss, contrastive_acc = self.contrastive_loss(
                visual_proj, text_proj
            )
            losses.append(contrastive_loss)

        # Masked region prediction
        mrp_loss = torch.tensor(0.0, device=device)
        mrp_acc = 0.0
        if self.use_mrp:
            mrp_loss, _, mrp_acc = self.mrp(visual_features)
            losses.append(mrp_loss * 0.5)  # Lower weight

        # Matching loss
        if self.use_matching and matching_labels is not None:
            matching_loss = self.matching_loss(
                visual_proj, text_proj, matching_labels
            )
            losses.append(matching_loss * 0.3)

        # Total loss
        total_loss = sum(losses) if losses else torch.tensor(0.0, device=device)

        return PreTrainingOutput(
            contrastive_loss=contrastive_loss,
            mrp_loss=mrp_loss,
            reconstruction_loss=None,
            total_loss=total_loss,
            contrastive_accuracy=contrastive_acc,
            mrp_accuracy=mrp_acc,
        )


class PreTrainingTrainer:
    """
    Trainer class for self-supervised pre-training.
    """

    def __init__(
        self,
        model: nn.Module,
        pretrainer: MultimodalPreTrainer,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ):
        self.model = model
        self.pretrainer = pretrainer
        self.optimizer = optimizer
        self.device = device

    def pretrain_step(
        self,
        images: torch.Tensor,
        text_features: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single pre-training step.

        Args:
            images: Input images (B, C, H, W)
            text_features: Text embeddings (B, seq_len, D)

        Returns:
            Dictionary with loss values
        """
        self.model.train()
        self.pretrainer.train()

        # Move to device
        images = images.to(self.device)
        text_features = text_features.to(self.device)

        # Forward pass through encoder
        with torch.no_grad():
            visual_features = self.model.encoder(images)

        # Pre-training losses
        outputs = self.pretrainer(visual_features, text_features)

        # Backward pass
        self.optimizer.zero_grad()
        outputs.total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': outputs.total_loss.item(),
            'contrastive_loss': outputs.contrastive_loss.item(),
            'mrp_loss': outputs.mrp_loss.item(),
            'contrastive_acc': outputs.contrastive_accuracy,
            'mrp_acc': outputs.mrp_accuracy,
        }

    def pretrain_epoch(
        self,
        dataloader,
    ) -> Dict[str, float]:
        """
        Run one epoch of pre-training.

        Args:
            dataloader: Pre-training data loader

        Returns:
            Average metrics for the epoch
        """
        total_metrics = {}
        num_batches = 0

        for batch in dataloader:
            images = batch['images']
            # Assume text_features are pre-computed or compute from text
            text_features = batch.get('text_features')

            if text_features is None:
                # Skip if no text features
                continue

            metrics = self.pretrain_step(images, text_features)

            for key, value in metrics.items():
                total_metrics[key] = total_metrics.get(key, 0.0) + value
            num_batches += 1

        # Average
        if num_batches > 0:
            for key in total_metrics:
                total_metrics[key] /= num_batches

        return total_metrics


def build_pretrainer(config: Dict) -> MultimodalPreTrainer:
    """Factory function to build pretrainer from config."""
    return MultimodalPreTrainer(
        visual_dim=config.get('visual_dim', 1024),
        language_dim=config.get('language_dim', 768),
        temperature=config.get('contrastive_temperature', 0.07),
        mask_ratio=config.get('mask_ratio', 0.15),
        use_mrp=config.get('use_mrp', True),
        use_contrastive=config.get('use_contrastive', True),
        use_matching=config.get('use_matching', False),
    )
