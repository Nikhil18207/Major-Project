"""
Pytest configuration and shared fixtures.

Provides common test fixtures and configuration for all test modules.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def device():
    """Get test device (CPU for testing)."""
    import torch
    return torch.device("cpu")


@pytest.fixture(scope="session")
def tokenizer():
    """Get shared tokenizer."""
    from transformers import BartTokenizer
    return BartTokenizer.from_pretrained("facebook/bart-base")


@pytest.fixture
def sample_image_tensor():
    """Create sample image tensor."""
    import torch
    return torch.randn(1, 3, 384, 384)


@pytest.fixture
def sample_batch():
    """Create sample training batch."""
    import torch
    return {
        "images": torch.randn(2, 3, 384, 384),
        "input_ids": torch.randint(0, 1000, (2, 50)),
        "attention_mask": torch.ones(2, 50),
        "labels": torch.randint(0, 1000, (2, 50)),
    }


# Configure pytest markers
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
