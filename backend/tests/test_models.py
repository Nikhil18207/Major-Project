"""
Test suite for XR2Text models.

Tests the core model components:
- SwinEncoder
- AnatomicalAttention (HAQT-ARR)
- BioBarTDecoder
- ProjectionLayer
- XR2TextModel (end-to-end)
"""

import pytest
import torch
import torch.nn as nn

# Skip if torch not available
pytest.importorskip("torch")


class TestSwinEncoder:
    """Tests for Swin Transformer encoder."""

    @pytest.fixture
    def encoder(self):
        from src.models.swin_encoder import SwinEncoder
        return SwinEncoder(pretrained=False)

    def test_encoder_output_shape(self, encoder):
        """Test encoder produces correct output shape."""
        batch_size = 2
        x = torch.randn(batch_size, 3, 384, 384)

        with torch.no_grad():
            features = encoder(x)

        # Swin Base outputs 1024-dim features at 12x12 for 384 input
        assert features.shape == (batch_size, 144, 1024), f"Expected (2, 144, 1024), got {features.shape}"

    def test_encoder_freeze_layers(self, encoder):
        """Test that layer freezing works."""
        encoder.freeze_layers(2)

        # Check that some parameters are frozen
        frozen_count = sum(1 for p in encoder.parameters() if not p.requires_grad)
        assert frozen_count > 0, "Expected some frozen parameters"


class TestAnatomicalAttention:
    """Tests for HAQT-ARR anatomical attention module."""

    @pytest.fixture
    def attention_module(self):
        from src.models.anatomical_attention import AnatomicalAttention
        return AnatomicalAttention(
            visual_dim=1024,
            language_dim=768,
            num_regions=7,
            num_global_queries=8,
            num_region_queries=4,
        )

    def test_attention_output_shape(self, attention_module):
        """Test attention module produces correct output shape."""
        batch_size = 2
        seq_len = 144  # 12x12
        visual_features = torch.randn(batch_size, seq_len, 1024)

        with torch.no_grad():
            output, region_weights = attention_module(visual_features)

        # Should have 8 global + 7*4 = 36 queries
        expected_queries = 8 + 7 * 4
        assert output.shape == (batch_size, expected_queries, 768), f"Expected (2, 36, 768), got {output.shape}"
        assert region_weights.shape == (batch_size, 7), f"Expected (2, 7), got {region_weights.shape}"

    def test_spatial_priors_shape(self, attention_module):
        """Test spatial prior generation."""
        priors = attention_module.get_spatial_priors()

        # Should have 7 regions, each 12x12
        assert priors.shape == (7, 12, 12), f"Expected (7, 12, 12), got {priors.shape}"

    def test_region_weights_sum_to_one(self, attention_module):
        """Test that region weights are normalized."""
        batch_size = 2
        seq_len = 144
        visual_features = torch.randn(batch_size, seq_len, 1024)

        with torch.no_grad():
            _, region_weights = attention_module(visual_features)

        # Weights should sum to approximately 1
        sums = region_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(batch_size), atol=0.01)


class TestProjectionLayer:
    """Tests for multimodal projection layer."""

    @pytest.fixture
    def projection(self):
        from src.models.projection_layer import MultimodalProjection
        return MultimodalProjection(
            visual_dim=1024,
            language_dim=768,
            num_queries=32,
        )

    def test_projection_output_shape(self, projection):
        """Test projection layer output shape."""
        batch_size = 2
        seq_len = 144
        visual_features = torch.randn(batch_size, seq_len, 1024)

        with torch.no_grad():
            output = projection(visual_features)

        assert output.shape == (batch_size, 32, 768), f"Expected (2, 32, 768), got {output.shape}"


class TestXR2TextModel:
    """End-to-end tests for XR2Text model."""

    @pytest.fixture
    def model(self):
        from src.models.xr2text import XR2TextModel, DEFAULT_CONFIG
        config = DEFAULT_CONFIG.copy()
        config['use_anatomical_attention'] = True
        return XR2TextModel.from_config(config)

    def test_model_forward(self, model):
        """Test model forward pass."""
        batch_size = 1
        images = torch.randn(batch_size, 3, 384, 384)

        # Create dummy labels
        labels = torch.randint(0, 1000, (batch_size, 50))

        with torch.no_grad():
            loss, logits = model(images, labels)

        assert loss is not None
        assert logits.shape[0] == batch_size

    def test_model_generate(self, model):
        """Test model generation."""
        batch_size = 1
        images = torch.randn(batch_size, 3, 384, 384)

        with torch.no_grad():
            _, texts, _ = model.generate(
                images,
                max_length=50,
                num_beams=2,
            )

        assert len(texts) == batch_size
        assert isinstance(texts[0], str)

    def test_model_attention_visualization(self, model):
        """Test attention visualization output."""
        batch_size = 1
        images = torch.randn(batch_size, 3, 384, 384)

        with torch.no_grad():
            attn_data = model.get_attention_visualization(images)

        assert 'anatomical_regions' in attn_data
        assert 'region_weights' in attn_data
        assert len(attn_data['anatomical_regions']) == 7


class TestMultiScaleFeatures:
    """Tests for multi-scale feature fusion."""

    def test_fusion_output_shape(self):
        """Test multi-scale fusion output."""
        from src.models.multiscale_features import MultiScaleFeatureFusion

        fusion = MultiScaleFeatureFusion(
            feature_dims=[256, 512, 1024],
            output_dim=768,
            fusion_method="attention",
        )

        features = [
            torch.randn(2, 144, 256),  # Scale 1
            torch.randn(2, 36, 512),   # Scale 2
            torch.randn(2, 9, 1024),   # Scale 3
        ]

        with torch.no_grad():
            output = fusion(features)

        assert output.shape == (2, 144, 768)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
