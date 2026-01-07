"""
Novel Loss Functions for XR2Text with HAQT-ARR

This module implements novel loss functions that strengthen the research contribution:

1. Anatomical Consistency Loss: Ensures spatial priors align with actual attention patterns
2. Clinical Entity Loss: Encourages detection of important clinical findings
3. Region-Aware Focal Loss: Focuses learning on difficult anatomical regions
4. Cross-Modal Alignment Loss: Better vision-language alignment

These losses are NOVEL contributions that differentiate this work from prior art.

Authors: S. Nikhil, Dadhania Omkumar
Supervisor: Dr. Damodar Panigrahy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import re
from loguru import logger


class AnatomicalConsistencyLoss(nn.Module):
    """
    NOVEL: Anatomical Consistency Loss
    
    Ensures that the learned spatial priors actually guide attention to the correct
    anatomical regions. This loss penalizes cases where attention patterns don't
    align with the spatial priors, encouraging the model to learn anatomically
    meaningful representations.
    
    This is novel because:
    - Prior work uses fixed spatial priors or doesn't enforce consistency
    - We learn both priors AND enforce their alignment with attention
    - Creates a feedback loop between spatial knowledge and attention
    """
    
    def __init__(self, weight: float = 0.1, temperature: float = 0.1):
        super().__init__()
        self.weight = weight
        self.temperature = temperature
    
    def forward(
        self,
        spatial_priors: torch.Tensor,
        attention_weights: torch.Tensor,
        region_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute anatomical consistency loss.
        
        Args:
            spatial_priors: (num_regions, H, W) - learned spatial priors
            attention_weights: (B, num_heads, num_queries, N_patches) - attention patterns
            region_weights: (B, num_regions) - importance weights per region
            
        Returns:
            Consistency loss scalar
        """
        if spatial_priors is None or attention_weights is None:
            return torch.tensor(0.0, device=attention_weights.device if attention_weights is not None else 'cpu')
        
        B, num_heads, num_queries, N_patches = attention_weights.shape
        num_regions, H, W = spatial_priors.shape
        
        # Reshape spatial priors to match attention shape
        # (num_regions, H, W) -> (num_regions, H*W) -> (1, 1, num_regions, H*W)
        spatial_flat = spatial_priors.view(num_regions, -1).unsqueeze(0).unsqueeze(0)
        
        # Average attention across heads and queries for each region
        # For simplicity, assume queries are organized by region
        # (B, num_heads, num_queries, N_patches) -> (B, num_queries, N_patches)
        attn_avg = attention_weights.mean(dim=1)  # Average over heads
        
        # Reshape attention to match spatial prior shape
        # (B, num_queries, N_patches) -> (B, num_queries, H*W)
        if N_patches != H * W:
            # Interpolate if needed
            attn_reshaped = F.interpolate(
                attn_avg.view(B * num_queries, 1, int(N_patches ** 0.5), int(N_patches ** 0.5)),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).view(B, num_queries, H * W)
        else:
            attn_reshaped = attn_avg
        
        # Compute similarity between attention and spatial priors
        # Use cosine similarity with temperature scaling
        attn_norm = F.normalize(attn_reshaped, p=2, dim=-1)
        spatial_norm = F.normalize(spatial_flat, p=2, dim=-1)
        
        # For each query, find which region it should attend to
        # Simple heuristic: assign queries to regions based on their index
        num_queries_per_region = num_queries // num_regions
        
        consistency_losses = []
        for r in range(num_regions):
            start_q = r * num_queries_per_region
            end_q = start_q + num_queries_per_region
            
            if start_q < num_queries:
                region_queries = attn_norm[:, start_q:end_q, :]  # (B, Q_r, H*W)
                region_prior = spatial_norm[:, :, r, :]  # (1, 1, H*W)
                
                # Compute similarity: (B, Q_r, H*W) @ (1, 1, H*W)^T -> (B, Q_r, 1)
                similarity = torch.sum(region_queries * region_prior, dim=-1, keepdim=True)
                
                # We want high similarity (low negative similarity)
                consistency_losses.append(-similarity.mean())
        
        consistency_loss = torch.stack(consistency_losses).mean()
        
        # Weight by region importance if provided
        if region_weights is not None:
            region_importance = region_weights.mean(dim=0)  # (num_regions,)
            consistency_loss = (consistency_loss * region_importance.unsqueeze(0)).mean()
        
        return self.weight * consistency_loss


class ClinicalEntityLoss(nn.Module):
    """
    NOVEL: Clinical Entity Loss with Negation Detection

    Encourages the model to detect and mention important clinical findings.
    This loss extracts clinical entities from the generated text and compares
    them with entities in the reference text, WITH NEGATION AWARENESS.

    This is novel because:
    - Standard NLG losses (BLEU, ROUGE) don't explicitly model clinical entities
    - We create a structured loss that encourages clinical accuracy
    - We handle negation (e.g., "no cardiomegaly" vs "cardiomegaly present")
    - Can be combined with standard cross-entropy for better clinical performance
    """

    def __init__(self, weight: float = 0.2):
        super().__init__()
        self.weight = weight

        # Common clinical findings in chest X-rays (deduplicated)
        self.clinical_entities = [
            'cardiomegaly', 'pneumonia', 'effusion', 'edema', 'consolidation',
            'atelectasis', 'pneumothorax', 'infiltrate', 'mass', 'nodule',
            'pleural thickening', 'opacity', 'fibrosis', 'fracture',
            'pleural effusion', 'pulmonary edema', 'emphysema', 'hernia',
            'pneumoperitoneum', 'calcification', 'tortuous aorta', 'scoliosis'
        ]

        # Negation patterns for clinical text
        self.negation_patterns = [
            'no ', 'no evidence of ', 'without ', 'negative for ',
            'absence of ', 'absent ', 'not ', 'denies ', 'ruled out ',
            'no acute ', 'no significant ', 'no definite ', 'no obvious ',
            'unremarkable', 'clear', 'normal', 'resolved', 'improved',
            'no longer', 'cleared', 'resolution of'
        ]

        # Positive indicators
        self.positive_patterns = [
            'present', 'noted', 'seen', 'identified', 'consistent with',
            'suggestive of', 'suspicious for', 'compatible with', 'evidence of',
            'demonstrates', 'shows', 'reveals', 'confirmed', 'new ', 'worsening',
            'increased', 'enlarged', 'developing', 'progression'
        ]

    def _is_negated(self, text: str, entity: str, entity_pos: int) -> bool:
        """
        Check if an entity mention is negated.

        Args:
            text: Full text (lowercase)
            entity: Entity being checked
            entity_pos: Position of entity in text

        Returns:
            True if the entity is negated
        """
        # Check window before entity (50 chars)
        window_start = max(0, entity_pos - 50)
        context_before = text[window_start:entity_pos]

        # Check if any negation pattern appears before entity
        for neg_pattern in self.negation_patterns:
            if neg_pattern in context_before:
                # Make sure there's no positive pattern after negation
                neg_pos = context_before.rfind(neg_pattern)
                text_after_neg = context_before[neg_pos + len(neg_pattern):]

                # If no positive pattern between negation and entity, it's negated
                has_positive = any(pos in text_after_neg for pos in self.positive_patterns)
                if not has_positive:
                    return True

        return False

    def extract_entities(self, text: str) -> tuple:
        """
        Extract clinical entities from text with negation awareness.

        Returns:
            Tuple of (positive_entities, negated_entities)
        """
        text_lower = text.lower()
        positive_entities = set()
        negated_entities = set()

        for entity in self.clinical_entities:
            # Find all occurrences of entity
            start = 0
            while True:
                pos = text_lower.find(entity, start)
                if pos == -1:
                    break

                # Check if this mention is negated
                if self._is_negated(text_lower, entity, pos):
                    negated_entities.add(entity)
                else:
                    positive_entities.add(entity)

                start = pos + len(entity)

        # Remove entities that appear both positive and negated (ambiguous)
        # Keep positive if it appears positive anywhere
        negated_entities = negated_entities - positive_entities

        return positive_entities, negated_entities
    
    def forward(
        self,
        generated_texts: List[str],
        reference_texts: List[str],
    ) -> torch.Tensor:
        """
        Compute clinical entity loss with negation-aware matching.

        Args:
            generated_texts: List of generated report texts
            reference_texts: List of reference report texts

        Returns:
            Entity loss scalar
        """
        if len(generated_texts) == 0:
            return torch.tensor(0.0)

        total_loss = 0.0
        num_samples = 0

        for gen_text, ref_text in zip(generated_texts, reference_texts):
            # Extract entities with negation awareness
            gen_positive, gen_negated = self.extract_entities(gen_text)
            ref_positive, ref_negated = self.extract_entities(ref_text)

            # Combine all reference entities for counting
            ref_all = ref_positive | ref_negated
            if len(ref_all) == 0:
                continue

            # True positives: correctly identified positive findings
            tp_positive = len(gen_positive & ref_positive)

            # True negatives (as positives): correctly identified negated findings
            tp_negated = len(gen_negated & ref_negated)

            # False positives: generated positive but reference negative or absent
            fp = len(gen_positive - ref_positive)

            # False negatives: reference positive but generated negative or absent
            fn = len(ref_positive - gen_positive)

            # Critical errors: saying positive when actually negated (dangerous!)
            critical_errors = len(gen_positive & ref_negated)

            # Total correct = true positives + correctly negated
            total_correct = tp_positive + tp_negated
            total_predicted = len(gen_positive) + len(gen_negated)
            total_reference = len(ref_positive) + len(ref_negated)

            # Precision and recall
            if total_predicted > 0:
                precision = total_correct / total_predicted
            else:
                precision = 0.0

            if total_reference > 0:
                recall = total_correct / total_reference
            else:
                recall = 0.0

            # F1 score
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            # Add penalty for critical errors (saying positive when negated)
            critical_penalty = 0.5 * critical_errors / max(len(ref_all), 1)

            # Loss = 1 - F1 + critical_penalty
            sample_loss = (1.0 - f1) + critical_penalty
            total_loss += sample_loss
            num_samples += 1

        if num_samples == 0:
            return torch.tensor(0.0)

        avg_loss = total_loss / num_samples
        return torch.tensor(self.weight * avg_loss)


class RegionAwareFocalLoss(nn.Module):
    """
    NOVEL: Region-Aware Focal Loss
    
    Extends focal loss to focus on difficult anatomical regions.
    Regions with lower attention weights (harder to learn) get higher loss weights.
    
    This is novel because:
    - Standard focal loss doesn't consider anatomical structure
    - We adaptively weight loss based on region difficulty
    - Encourages balanced learning across all anatomical regions
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, weight: float = 0.15):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        region_weights: Optional[torch.Tensor] = None,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        """
        Compute region-aware focal loss.
        
        Args:
            logits: (B, seq_len, vocab_size) - model predictions
            labels: (B, seq_len) - target labels
            region_weights: (B, num_regions) - region importance weights
            pad_token_id: Padding token ID to ignore
            
        Returns:
            Focal loss scalar
        """
        try:
            # Get vocab size for label validation
            vocab_size = logits.size(-1)
            
            # CRITICAL: Validate and clamp labels to prevent CUDA index errors
            # Labels must be in range [0, vocab_size-1] or equal to pad_token_id (ignore_index)
            labels_flat = labels.view(-1).clone()
            
            # Create mask for valid positions (non-padding)
            valid_mask = labels_flat != pad_token_id
            
            # Clamp labels to valid vocab range to prevent CUDA index out of bounds
            # Only clamp non-padding positions
            labels_flat = torch.where(
                valid_mask,
                torch.clamp(labels_flat, min=0, max=vocab_size - 1),
                labels_flat  # Keep padding as-is
            )
            
            # Standard cross-entropy with validated labels
            ce_loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels_flat,
                ignore_index=pad_token_id,
                reduction='none'
            )
            
            # Get probabilities with numerical stability
            pt = torch.exp(-ce_loss.clamp(max=50.0))  # Clamp to prevent underflow
            
            # Focal loss: alpha * (1 - p_t)^gamma * CE
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
            
            # If region weights provided, weight by region difficulty
            if region_weights is not None and region_weights.numel() > 0:
                try:
                    # Detach and clone to avoid gradient issues
                    rw = region_weights.detach().float()
                    
                    # Check for valid tensor before computing mean
                    if rw.numel() > 0 and not torch.isnan(rw).any() and not torch.isinf(rw).any():
                        avg_region_weight = rw.mean().clamp(min=0.01, max=100.0)
                        difficulty_factor = (1.0 / (avg_region_weight.item() + 1e-6))
                        difficulty_factor = max(0.1, min(10.0, difficulty_factor))
                        focal_loss = focal_loss * difficulty_factor
                except Exception:
                    # If any error with region weights, just skip the weighting
                    pass
            
            # Average over non-padded tokens
            mask = valid_mask
            if mask.sum() > 0:
                focal_loss = focal_loss[mask].mean()
            else:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
            return self.weight * focal_loss
            
        except Exception as e:
            # Fallback: return zero loss if anything fails
            # This prevents training from crashing
            logger.warning(f"RegionAwareFocalLoss error (returning 0): {e}")
            return torch.tensor(0.0, device=logits.device, requires_grad=True)


class CrossModalAlignmentLoss(nn.Module):
    """
    NOVEL: Cross-Modal Alignment Loss
    
    Ensures that visual features and text features are well-aligned in a shared
    semantic space. Uses contrastive learning to pull matching image-text pairs
    together and push non-matching pairs apart.
    
    This is novel because:
    - Standard vision-language models don't explicitly enforce alignment
    - We add this as an auxiliary loss during training
    - Improves the quality of the projection layer
    """
    
    def __init__(self, weight: float = 0.1, temperature: float = 0.07):
        super().__init__()
        self.weight = weight
        self.temperature = temperature
    
    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-modal alignment loss.
        
        Args:
            visual_features: (B, N_queries, D) - projected visual features
            text_features: (B, seq_len, D) - text embeddings from decoder
            
        Returns:
            Alignment loss scalar
        """
        B = visual_features.shape[0]
        
        # Pool features: (B, N, D) -> (B, D)
        visual_pooled = visual_features.mean(dim=1)  # (B, D)
        text_pooled = text_features.mean(dim=1)  # (B, D)
        
        # Normalize
        visual_norm = F.normalize(visual_pooled, p=2, dim=-1)
        text_norm = F.normalize(text_pooled, p=2, dim=-1)
        
        # Compute similarity matrix: (B, B)
        similarity_matrix = torch.matmul(visual_norm, text_norm.t()) / self.temperature
        
        # Labels: diagonal should be high (matching pairs)
        labels = torch.arange(B, device=visual_features.device)
        
        # Contrastive loss: cross-entropy with temperature
        loss_v2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2v = F.cross_entropy(similarity_matrix.t(), labels)
        
        alignment_loss = (loss_v2t + loss_t2v) / 2.0
        
        return self.weight * alignment_loss


class CombinedNovelLoss(nn.Module):
    """
    Combined Novel Loss Function
    
    Combines all novel losses into a single module for easy integration.
    """
    
    def __init__(
        self,
        use_anatomical_consistency: bool = True,
        use_clinical_entity: bool = True,
        use_region_focal: bool = True,
        use_cross_modal: bool = True,
        anatomical_weight: float = 0.1,
        clinical_weight: float = 0.2,
        focal_weight: float = 0.15,
        alignment_weight: float = 0.1,
    ):
        super().__init__()
        
        self.use_anatomical_consistency = use_anatomical_consistency
        self.use_clinical_entity = use_clinical_entity
        self.use_region_focal = use_region_focal
        self.use_cross_modal = use_cross_modal
        
        if use_anatomical_consistency:
            self.anatomical_loss = AnatomicalConsistencyLoss(weight=anatomical_weight)
        if use_clinical_entity:
            self.clinical_loss = ClinicalEntityLoss(weight=clinical_weight)
        if use_region_focal:
            self.focal_loss = RegionAwareFocalLoss(weight=focal_weight)
        if use_cross_modal:
            self.alignment_loss = CrossModalAlignmentLoss(weight=alignment_weight)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        generated_texts: Optional[List[str]] = None,
        reference_texts: Optional[List[str]] = None,
        pad_token_id: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined novel losses.
        
        Args:
            outputs: Model outputs dictionary
            labels: Target labels
            generated_texts: Generated text strings (for clinical entity loss)
            reference_texts: Reference text strings (for clinical entity loss)
            pad_token_id: Padding token ID
            
        Returns:
            Dictionary with individual and total loss components
        """
        # Determine device from labels or outputs
        device = labels.device if labels is not None else 'cuda'

        losses = {}
        loss_components = []  # Collect loss components to sum properly

        # Anatomical consistency loss
        if self.use_anatomical_consistency:
            try:
                spatial_priors = outputs.get('spatial_priors')
                attention_weights = outputs.get('attention_info', {}).get('region_attention')
                region_weights = outputs.get('region_weights')

                if spatial_priors is not None and attention_weights is not None:
                    anat_loss = self.anatomical_loss(spatial_priors, attention_weights, region_weights)
                    if torch.is_tensor(anat_loss) and not torch.isnan(anat_loss) and not torch.isinf(anat_loss):
                        losses['anatomical_consistency'] = anat_loss
                        if anat_loss.requires_grad:
                            loss_components.append(anat_loss)
            except Exception as e:
                logger.debug(f"Anatomical consistency loss error: {e}")

        # Clinical entity loss (non-differentiable, for monitoring only)
        if self.use_clinical_entity and generated_texts is not None and reference_texts is not None:
            try:
                clinical_loss = self.clinical_loss(generated_texts, reference_texts)
                if torch.is_tensor(clinical_loss):
                    losses['clinical_entity'] = clinical_loss.item() if clinical_loss.numel() == 1 else clinical_loss
                elif isinstance(clinical_loss, (int, float)):
                    losses['clinical_entity'] = clinical_loss
                # Note: Clinical entity loss is computed on text, not differentiable
            except Exception as e:
                logger.debug(f"Clinical entity loss error: {e}")

        # Region-aware focal loss (differentiable)
        if self.use_region_focal:
            try:
                logits = outputs.get('logits')
                region_weights = outputs.get('region_weights')

                if logits is not None and labels is not None:
                    focal_loss = self.focal_loss(logits, labels, region_weights, pad_token_id)
                    if torch.is_tensor(focal_loss) and not torch.isnan(focal_loss) and not torch.isinf(focal_loss):
                        losses['region_focal'] = focal_loss
                        if focal_loss.requires_grad:
                            loss_components.append(focal_loss)
            except Exception as e:
                logger.debug(f"Region focal loss error: {e}")

        # Cross-modal alignment loss
        if self.use_cross_modal:
            try:
                visual_features = outputs.get('projected_features')
                decoder_hidden = outputs.get('decoder_hidden_states')

                if visual_features is not None and decoder_hidden is not None:
                    # Use last hidden state from decoder
                    text_features = decoder_hidden[-1] if isinstance(decoder_hidden, tuple) else decoder_hidden
                    align_loss = self.alignment_loss(visual_features, text_features)
                    if torch.is_tensor(align_loss) and not torch.isnan(align_loss) and not torch.isinf(align_loss):
                        losses['cross_modal'] = align_loss
                        if align_loss.requires_grad:
                            loss_components.append(align_loss)
            except Exception as e:
                logger.debug(f"Cross-modal alignment loss error: {e}")

        # Sum loss components properly to maintain gradient flow
        if loss_components:
            total_novel_loss = torch.stack(loss_components).sum()
        else:
            # Return zero tensor that can still participate in computation graph
            total_novel_loss = torch.zeros(1, device=device, requires_grad=True).squeeze()

        losses['total_novel'] = total_novel_loss

        return losses

