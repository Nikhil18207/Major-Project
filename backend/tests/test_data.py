"""
Test suite for data loading and transforms.

Tests:
- Dataset loading
- Image transforms
- DataLoader creation
- Collation
"""

import pytest
import torch
import numpy as np
from PIL import Image

pytest.importorskip("torch")


class TestTransforms:
    """Tests for image transforms."""

    def test_train_transforms(self):
        """Test training transforms."""
        from src.data.transforms import get_train_transforms, XRayTransform

        transform = XRayTransform(get_train_transforms(384))

        # Create dummy image
        image = Image.new('RGB', (512, 512), color='white')
        tensor = transform(image)

        assert tensor.shape == (3, 384, 384)
        assert tensor.dtype == torch.float32

    def test_val_transforms(self):
        """Test validation transforms."""
        from src.data.transforms import get_val_transforms, XRayTransform

        transform = XRayTransform(get_val_transforms(384))

        image = Image.new('RGB', (512, 512), color='white')
        tensor = transform(image)

        assert tensor.shape == (3, 384, 384)

    def test_transform_normalization(self):
        """Test that transforms apply normalization."""
        from src.data.transforms import get_val_transforms, XRayTransform

        transform = XRayTransform(get_val_transforms(384))

        # Create uniform gray image
        image = Image.new('RGB', (384, 384), color=(128, 128, 128))
        tensor = transform(image)

        # After ImageNet normalization, values should be around 0
        assert tensor.mean().abs() < 1.0


class TestDataset:
    """Tests for MIMIC-CXR dataset."""

    @pytest.fixture
    def tokenizer(self):
        from transformers import BartTokenizer
        return BartTokenizer.from_pretrained("facebook/bart-base")

    def test_dataset_creation(self, tokenizer):
        """Test dataset can be created."""
        from src.data.dataset import MIMICCXRDataset
        from src.data.transforms import get_val_transforms, XRayTransform

        transform = XRayTransform(get_val_transforms(384))

        # Create with small subset for testing
        dataset = MIMICCXRDataset(
            split="train",
            transform=transform,
            tokenizer=tokenizer,
            max_length=256,
            subset_size=10,
        )

        assert len(dataset) <= 10

    def test_dataset_item_format(self, tokenizer):
        """Test dataset returns correct item format."""
        from src.data.dataset import MIMICCXRDataset
        from src.data.transforms import get_val_transforms, XRayTransform

        transform = XRayTransform(get_val_transforms(384))

        dataset = MIMICCXRDataset(
            split="train",
            transform=transform,
            tokenizer=tokenizer,
            max_length=256,
            subset_size=5,
        )

        if len(dataset) > 0:
            item = dataset[0]

            assert 'image' in item
            assert 'input_ids' in item
            assert 'attention_mask' in item
            assert 'text' in item

            assert item['image'].shape == (3, 384, 384)


class TestCollator:
    """Tests for batch collation."""

    def test_collator(self):
        """Test collator pads batches correctly."""
        from src.data.dataset import MIMICCXRCollator

        collator = MIMICCXRCollator(pad_token_id=0)

        # Create mock batch
        batch = [
            {
                'image': torch.randn(3, 384, 384),
                'input_ids': torch.tensor([1, 2, 3, 4, 5]),
                'attention_mask': torch.tensor([1, 1, 1, 1, 1]),
                'text': 'Sample text 1',
            },
            {
                'image': torch.randn(3, 384, 384),
                'input_ids': torch.tensor([1, 2, 3]),
                'attention_mask': torch.tensor([1, 1, 1]),
                'text': 'Sample text 2',
            },
        ]

        collated = collator(batch)

        assert collated['images'].shape == (2, 3, 384, 384)
        assert collated['input_ids'].shape[0] == 2
        # Second item should be padded to match first
        assert collated['input_ids'].shape[1] == 5


class TestDataLoader:
    """Tests for DataLoader creation."""

    def test_dataloader_creation(self):
        """Test dataloaders can be created."""
        from transformers import BartTokenizer
        from src.data.dataloader import get_dataloaders

        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

        train_loader, val_loader, test_loader = get_dataloaders(
            tokenizer=tokenizer,
            batch_size=2,
            num_workers=0,
            image_size=384,
            max_length=256,
            train_subset=10,
            val_subset=5,
        )

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
