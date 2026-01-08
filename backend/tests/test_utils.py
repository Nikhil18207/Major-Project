"""
Test suite for utility modules.

Tests:
- Metrics computation
- Clinical validation
- Device utilities
- Loss functions
"""

import pytest
import torch

pytest.importorskip("torch")


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_bleu_score(self):
        """Test BLEU score computation."""
        from src.utils.metrics import compute_bleu

        predictions = ["The lungs are clear."]
        references = ["The lungs are clear without infiltrate."]

        scores = compute_bleu(predictions, references)

        assert "bleu_1" in scores
        assert "bleu_4" in scores
        assert 0 <= scores["bleu_1"] <= 1
        assert 0 <= scores["bleu_4"] <= 1

    def test_rouge_score(self):
        """Test ROUGE score computation."""
        from src.utils.metrics import compute_rouge

        predictions = ["The heart is normal in size."]
        references = ["The heart size is normal."]

        scores = compute_rouge(predictions, references)

        assert "rouge_1" in scores
        assert "rouge_l" in scores
        assert 0 <= scores["rouge_l"] <= 1

    def test_all_metrics(self):
        """Test computing all metrics together."""
        from src.utils.metrics import compute_all_metrics

        predictions = [
            "No acute cardiopulmonary abnormality.",
            "There is a small pleural effusion.",
        ]
        references = [
            "No acute cardiopulmonary process.",
            "Small left pleural effusion noted.",
        ]

        metrics = compute_all_metrics(predictions, references)

        assert "bleu_4" in metrics
        assert "rouge_l" in metrics
        assert all(0 <= v <= 1 for v in metrics.values() if isinstance(v, float))


class TestClinicalValidator:
    """Tests for clinical validation."""

    def test_entity_extraction(self):
        """Test clinical entity extraction."""
        from src.utils.clinical_validator import ClinicalValidator

        validator = ClinicalValidator()

        text = "There is pneumonia in the right lower lobe. No pleural effusion."
        entities = validator.extract_entities(text)

        assert "findings" in entities
        assert "locations" in entities
        assert "negations" in entities

    def test_negation_detection(self):
        """Test negation-aware entity extraction."""
        from src.utils.clinical_validator import ClinicalValidator

        validator = ClinicalValidator()

        # Positive finding
        text1 = "Pneumonia is present."
        entities1 = validator.extract_entities(text1)

        # Negated finding
        text2 = "No pneumonia is seen."
        entities2 = validator.extract_entities(text2)

        # The negated version should have the entity in negations
        assert len(entities2.get("negations", [])) >= len(entities1.get("negations", []))

    def test_clinical_accuracy(self):
        """Test clinical accuracy computation."""
        from src.utils.clinical_validator import ClinicalValidator

        validator = ClinicalValidator()

        predictions = ["Pneumonia in right lung. No effusion."]
        references = ["Right lower lobe pneumonia. No pleural effusion."]

        accuracy = validator.compute_clinical_accuracy(predictions, references)

        assert "entity_precision" in accuracy
        assert "entity_recall" in accuracy
        assert "entity_f1" in accuracy


class TestDeviceUtils:
    """Tests for device utilities."""

    def test_get_device(self):
        """Test device selection."""
        from src.utils.device import get_device

        device = get_device(prefer_cuda=False)
        assert device == torch.device("cpu")

    def test_gpu_memory_monitor(self):
        """Test GPU memory monitor context manager."""
        from src.utils.device import GPUMemoryMonitor

        with GPUMemoryMonitor("test"):
            _ = torch.randn(100, 100)

        # Should complete without error

    def test_temperature_function(self):
        """Test GPU temperature function."""
        from src.utils.device import get_gpu_temperature

        temp = get_gpu_temperature()

        # Should return int or None
        assert temp is None or isinstance(temp, int)

        if temp is not None:
            assert 0 <= temp <= 120  # Reasonable temperature range


class TestLossFunctions:
    """Tests for custom loss functions."""

    def test_anatomical_consistency_loss(self):
        """Test anatomical consistency loss."""
        from src.training.losses import AnatomicalConsistencyLoss

        loss_fn = AnatomicalConsistencyLoss()

        # Mock predictions and targets
        predictions = torch.randn(2, 100, 1000)  # [batch, seq, vocab]
        targets = torch.randint(0, 1000, (2, 100))  # [batch, seq]
        region_weights = torch.softmax(torch.randn(2, 7), dim=-1)

        loss = loss_fn(predictions, targets, region_weights)

        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)

    def test_clinical_entity_loss(self):
        """Test clinical entity loss."""
        from src.training.losses import ClinicalEntityLoss

        loss_fn = ClinicalEntityLoss()

        predictions = torch.randn(2, 50, 1000)
        targets = torch.randint(0, 1000, (2, 50))

        loss = loss_fn(predictions, targets)

        assert loss.ndim == 0
        assert not torch.isnan(loss)

    def test_combined_loss(self):
        """Test combined loss function."""
        from src.training.losses import CombinedLoss

        loss_fn = CombinedLoss(
            anatomical_weight=0.1,
            entity_weight=0.2,
            focal_weight=0.15,
        )

        predictions = torch.randn(2, 50, 1000)
        targets = torch.randint(0, 1000, (2, 50))
        region_weights = torch.softmax(torch.randn(2, 7), dim=-1)

        loss = loss_fn(predictions, targets, region_weights)

        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert loss > 0


class TestCurriculumLearning:
    """Tests for curriculum learning."""

    def test_curriculum_scheduler(self):
        """Test curriculum scheduler stages."""
        from src.training.curriculum import CurriculumScheduler

        scheduler = CurriculumScheduler(
            stages=[
                {"name": "normal", "epoch_range": [0, 5]},
                {"name": "single", "epoch_range": [5, 15]},
                {"name": "multi", "epoch_range": [15, 30]},
                {"name": "complex", "epoch_range": [30, 50]},
            ]
        )

        assert scheduler.get_stage(0) == "normal"
        assert scheduler.get_stage(5) == "single"
        assert scheduler.get_stage(20) == "multi"
        assert scheduler.get_stage(40) == "complex"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
