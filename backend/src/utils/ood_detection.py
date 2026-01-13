"""
Out-of-Distribution Detection Module for XR2Text

This module implements OOD detection to identify samples that the model
shouldn't process confidently - CRITICAL for clinical safety.

Novel Contributions:
1. Mahalanobis Distance - Detect samples far from training distribution
2. Energy-Based OOD - Use energy scores for detection
3. Feature Space Analysis - Analyze representation statistics
4. Ensemble OOD Detection - Combine multiple methods
5. Clinical Safety Flags - Actionable OOD alerts

This is NOVEL because:
- Medical AI must know when it's outside its competence
- OOD detection prevents overconfident errors
- Multiple detection methods provide robustness
- Essential for FDA/regulatory compliance

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
class OODResult:
    """Container for OOD detection results."""
    is_ood: bool  # Whether sample is out-of-distribution
    ood_score: float  # 0 = in-distribution, 1 = out-of-distribution
    confidence: float  # Confidence in the OOD prediction
    detection_method: str  # Which method triggered the detection
    recommendations: List[str]  # Suggested actions


class MahalanobisOODDetector(nn.Module):
    """
    NOVEL: Mahalanobis Distance-Based OOD Detection

    Measures how far a sample is from the training distribution
    using Mahalanobis distance in feature space.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_classes: int = 1,  # For report generation, single class
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Running statistics (updated during training)
        self.register_buffer('mean', torch.zeros(hidden_dim))
        self.register_buffer('covariance', torch.eye(hidden_dim))
        self.register_buffer('covariance_inv', torch.eye(hidden_dim))
        self.register_buffer('num_samples', torch.tensor(0))

        # Threshold (calibrated during validation)
        self.register_buffer('threshold', torch.tensor(100.0))

    def update_statistics(
        self,
        features: torch.Tensor,
    ) -> None:
        """
        Update running mean and covariance from training features.

        Args:
            features: Feature vectors (B, D)
        """
        batch_size = features.shape[0]
        current_n = self.num_samples.item()

        # Update mean
        batch_mean = features.mean(dim=0)
        new_n = current_n + batch_size

        if current_n == 0:
            self.mean = batch_mean
            self.covariance = torch.cov(features.t())
        else:
            # Incremental update
            delta = batch_mean - self.mean
            self.mean = self.mean + delta * batch_size / new_n

            # Update covariance (approximate)
            batch_cov = torch.cov(features.t())
            self.covariance = (
                current_n * self.covariance +
                batch_size * batch_cov +
                current_n * batch_size / new_n * torch.outer(delta, delta)
            ) / new_n

        self.num_samples = torch.tensor(new_n)

        # Update inverse (for efficiency, only periodically in practice)
        try:
            self.covariance_inv = torch.inverse(
                self.covariance + 1e-4 * torch.eye(self.hidden_dim, device=features.device)
            )
        except Exception:
            # Fall back to pseudo-inverse if singular
            self.covariance_inv = torch.pinverse(self.covariance)

    def compute_mahalanobis_distance(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Mahalanobis distance from training distribution.

        Args:
            features: Feature vectors (B, D)

        Returns:
            Mahalanobis distances (B,)
        """
        diff = features - self.mean.unsqueeze(0)  # (B, D)

        # Mahalanobis distance: sqrt((x-mu)^T @ Sigma^-1 @ (x-mu))
        left = torch.matmul(diff, self.covariance_inv)  # (B, D)
        mahal_sq = (left * diff).sum(dim=-1)  # (B,)
        mahal = torch.sqrt(mahal_sq.clamp(min=0))

        return mahal

    def forward(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect OOD samples.

        Args:
            features: Feature vectors (B, D)

        Returns:
            Tuple of (is_ood, ood_scores)
        """
        mahal_dist = self.compute_mahalanobis_distance(features)

        # Normalize to [0, 1] score
        # Higher distance = higher OOD score
        ood_scores = torch.sigmoid(mahal_dist / self.threshold - 1.0)

        is_ood = ood_scores > 0.5

        return is_ood, ood_scores


class EnergyOODDetector(nn.Module):
    """
    NOVEL: Energy-Based OOD Detection

    Uses the energy of model outputs (negative log-sum-exp of logits)
    to detect OOD samples. Lower energy = more confident = more in-distribution.
    """

    def __init__(
        self,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature

        # Calibrated threshold
        self.register_buffer('threshold', torch.tensor(0.0))

    def compute_energy(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute energy score from logits.

        Args:
            logits: Model output logits (B, vocab_size) or (B, seq_len, vocab_size)

        Returns:
            Energy scores (B,)
        """
        # If sequence output, pool over sequence
        if logits.dim() == 3:
            logits = logits.mean(dim=1)  # (B, vocab_size)

        # Energy = -T * log(sum(exp(logits/T)))
        energy = -self.temperature * torch.logsumexp(
            logits / self.temperature, dim=-1
        )

        return energy

    def forward(
        self,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect OOD samples using energy.

        Args:
            logits: Model logits

        Returns:
            Tuple of (is_ood, ood_scores)
        """
        energy = self.compute_energy(logits)

        # Higher energy = more OOD
        # Normalize to [0, 1] using sigmoid
        ood_scores = torch.sigmoid(energy - self.threshold)

        is_ood = ood_scores > 0.5

        return is_ood, ood_scores


class FeatureSpaceOODDetector(nn.Module):
    """
    NOVEL: Feature Space Analysis OOD Detection

    Analyzes properties of feature vectors to detect anomalies:
    - Unusual activation patterns
    - Low activation density
    - High feature variance
    """

    def __init__(
        self,
        hidden_dim: int = 768,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Running statistics for normalization
        self.register_buffer('activation_mean', torch.zeros(hidden_dim))
        self.register_buffer('activation_std', torch.ones(hidden_dim))
        self.register_buffer('sparsity_threshold', torch.tensor(0.1))

    def compute_activation_stats(
        self,
        features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute activation statistics for OOD detection.

        Args:
            features: Feature vectors (B, D) or (B, N, D)

        Returns:
            Dictionary of statistics
        """
        # Pool if needed
        if features.dim() == 3:
            features = features.mean(dim=1)

        # Compute statistics
        stats = {}

        # Mean activation
        stats['mean_activation'] = features.mean(dim=-1)

        # Activation variance
        stats['activation_variance'] = features.var(dim=-1)

        # Sparsity (fraction of near-zero activations)
        sparsity = (features.abs() < self.sparsity_threshold).float().mean(dim=-1)
        stats['sparsity'] = sparsity

        # L2 norm
        stats['l2_norm'] = features.norm(dim=-1)

        # Max activation
        stats['max_activation'] = features.max(dim=-1).values

        return stats

    def forward(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect OOD using feature space analysis.

        Args:
            features: Feature vectors

        Returns:
            Tuple of (is_ood, ood_scores)
        """
        stats = self.compute_activation_stats(features)

        # Combine statistics into OOD score
        # Unusual values in any statistic increase OOD score

        # Normalize statistics (using training statistics would be better)
        anomaly_scores = []

        # High variance is unusual
        var_score = torch.sigmoid(stats['activation_variance'] - 1.0)
        anomaly_scores.append(var_score)

        # Very high or low sparsity is unusual
        sparsity_score = (stats['sparsity'] - 0.3).abs()
        anomaly_scores.append(sparsity_score)

        # Very high or low L2 norm is unusual
        norm_score = torch.sigmoid((stats['l2_norm'] - 30.0).abs() / 10.0 - 1.0)
        anomaly_scores.append(norm_score)

        # Combine scores
        ood_scores = torch.stack(anomaly_scores, dim=-1).mean(dim=-1)

        is_ood = ood_scores > 0.5

        return is_ood, ood_scores


class EnsembleOODDetector(nn.Module):
    """
    NOVEL: Ensemble OOD Detection

    Combines multiple OOD detection methods for robust detection.
    Uses voting or weighted averaging to make final decision.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        use_mahalanobis: bool = True,
        use_energy: bool = True,
        use_feature_space: bool = True,
        voting_threshold: float = 0.5,
    ):
        super().__init__()

        self.detectors = nn.ModuleDict()
        self.weights = {}

        if use_mahalanobis:
            self.detectors['mahalanobis'] = MahalanobisOODDetector(hidden_dim)
            self.weights['mahalanobis'] = 0.4

        if use_energy:
            self.detectors['energy'] = EnergyOODDetector()
            self.weights['energy'] = 0.3

        if use_feature_space:
            self.detectors['feature_space'] = FeatureSpaceOODDetector(hidden_dim)
            self.weights['feature_space'] = 0.3

        self.voting_threshold = voting_threshold

        logger.info(
            f"EnsembleOODDetector initialized with methods: {list(self.detectors.keys())}"
        )

    def forward(
        self,
        features: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
    ) -> OODResult:
        """
        Detect OOD using ensemble of methods.

        Args:
            features: Feature vectors (B, D) or (B, N, D)
            logits: Optional model logits for energy-based detection

        Returns:
            OODResult with detection results
        """
        batch_size = features.shape[0] if features.dim() == 2 else features.shape[0]
        device = features.device

        all_scores = []
        all_detections = []
        detection_details = {}

        # Pool features if needed
        if features.dim() == 3:
            pooled_features = features.mean(dim=1)
        else:
            pooled_features = features

        # Run each detector
        for name, detector in self.detectors.items():
            if name == 'energy' and logits is not None:
                is_ood, scores = detector(logits)
            elif name in ['mahalanobis', 'feature_space']:
                is_ood, scores = detector(pooled_features)
            else:
                continue

            weight = self.weights.get(name, 1.0)
            all_scores.append(scores * weight)
            all_detections.append(is_ood)
            detection_details[name] = scores.mean().item()

        # Combine scores (weighted average)
        if all_scores:
            combined_score = torch.stack(all_scores, dim=-1).sum(dim=-1)
            combined_score = combined_score / sum(self.weights.values())
        else:
            combined_score = torch.zeros(batch_size, device=device)

        # Final decision
        is_ood = combined_score > self.voting_threshold
        is_ood_final = is_ood.any().item() if batch_size > 1 else is_ood.item()

        # Determine which method triggered detection
        if is_ood_final:
            detection_method = max(detection_details, key=detection_details.get)
        else:
            detection_method = "none"

        # Generate recommendations
        recommendations = self._generate_recommendations(
            is_ood_final, combined_score.mean().item(), detection_details
        )

        return OODResult(
            is_ood=is_ood_final,
            ood_score=combined_score.mean().item(),
            confidence=1.0 - abs(combined_score.mean().item() - 0.5) * 2,
            detection_method=detection_method,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        is_ood: bool,
        ood_score: float,
        detection_details: Dict[str, float],
    ) -> List[str]:
        """Generate recommendations based on OOD detection."""
        recommendations = []

        if is_ood:
            recommendations.append(
                "WARNING: This sample appears to be out-of-distribution. "
                "Model predictions may be unreliable."
            )

            if detection_details.get('mahalanobis', 0) > 0.5:
                recommendations.append(
                    "Sample features are statistically unusual compared to training data."
                )

            if detection_details.get('energy', 0) > 0.5:
                recommendations.append(
                    "Model output confidence is unusually low."
                )

            recommendations.append(
                "RECOMMENDATION: Have a radiologist review this case directly."
            )

        elif ood_score > 0.3:
            recommendations.append(
                "Note: This sample shows some unusual characteristics. "
                "Consider having findings verified."
            )

        return recommendations

    def update_statistics(
        self,
        features: torch.Tensor,
    ) -> None:
        """Update detector statistics from training data."""
        if 'mahalanobis' in self.detectors:
            if features.dim() == 3:
                features = features.mean(dim=1)
            self.detectors['mahalanobis'].update_statistics(features)


class ClinicalOODSafetyChecker:
    """
    NOVEL: Clinical Safety Checker using OOD Detection

    Provides safety checks for clinical deployment, including:
    - Image quality assessment
    - Unusual finding detection
    - Confidence thresholding
    - Automatic referral triggers
    """

    def __init__(
        self,
        ood_detector: EnsembleOODDetector,
        confidence_threshold: float = 0.7,
        referral_threshold: float = 0.4,
    ):
        self.ood_detector = ood_detector
        self.confidence_threshold = confidence_threshold
        self.referral_threshold = referral_threshold

    def check_safety(
        self,
        features: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        confidence_score: Optional[float] = None,
    ) -> Dict[str, any]:
        """
        Perform comprehensive safety check.

        Args:
            features: Model features
            logits: Model logits
            confidence_score: Optional external confidence score

        Returns:
            Safety check results
        """
        # Run OOD detection
        ood_result = self.ood_detector(features, logits)

        # Determine safety status
        is_safe = True
        safety_concerns = []
        action_required = "none"

        # Check OOD
        if ood_result.is_ood:
            is_safe = False
            safety_concerns.append("Out-of-distribution sample detected")
            action_required = "referral_required"

        # Check confidence
        if confidence_score is not None:
            if confidence_score < self.confidence_threshold:
                safety_concerns.append(f"Low confidence: {confidence_score:.1%}")
                if confidence_score < self.referral_threshold:
                    is_safe = False
                    action_required = "referral_required"
                else:
                    action_required = "review_recommended"

        # Build result
        result = {
            'is_safe': is_safe,
            'ood_result': ood_result,
            'safety_concerns': safety_concerns,
            'action_required': action_required,
            'recommendations': ood_result.recommendations,
        }

        return result


def build_ood_detector(config: Dict) -> EnsembleOODDetector:
    """Factory function to build OOD detector from config."""
    return EnsembleOODDetector(
        hidden_dim=config.get('language_dim', 768),
        use_mahalanobis=config.get('use_mahalanobis_ood', True),
        use_energy=config.get('use_energy_ood', True),
        use_feature_space=config.get('use_feature_space_ood', True),
        voting_threshold=config.get('ood_threshold', 0.5),
    )
