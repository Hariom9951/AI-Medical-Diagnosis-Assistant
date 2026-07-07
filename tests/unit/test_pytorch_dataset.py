"""Unit Tests for PyTorch Dataset & DataLoader Component.

Asserts lazy loading RAM caching, text and multimodal dataset tuple returns,
and safe dataloader argument configurations.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import albumentations as A
import pytest
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from src.components.pytorch_dataset import (
    MedicalImageDataset,
    MultimodalMedicalDataset,
    SymptomTextDataset,
    create_pytorch_dataloader,
)


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary folder directory."""
    return tmp_path


def test_image_dataset_returns_path_and_caching(temp_dir: Path) -> None:
    """Verifies MedicalImageDataset returns path string and caches loaded tensors in RAM."""
    # Setup mock image file
    img_path = temp_dir / "scan_01.png"
    img = Image.new("RGB", (100, 100), color=(0, 255, 0))
    img.save(img_path)

    # Compile Albumentations compose transform
    transform = A.Compose([A.Resize(height=224, width=224), ToTensorV2()])

    # 1. Test dataset with caching active
    dataset = MedicalImageDataset(
        image_paths=[img_path], labels=[1], transform=transform, cache_active=True
    )
    assert len(dataset) == 1
    assert dataset.cache == {}

    # Initial get retrieves from disk and writes cache
    image_tensor, label, path_str = dataset[0]
    assert label == 1
    assert path_str == str(img_path)
    assert image_tensor.shape == (3, 224, 224)
    assert dataset.cache is not None  # narrows Optional[Dict] -> Dict for type checker
    assert str(img_path) in dataset.cache
    assert torch.equal(dataset.cache[str(img_path)], image_tensor)

    # 2. Test dataset without caching
    nocache_dataset = MedicalImageDataset(
        image_paths=[img_path], labels=[2], transform=transform, cache_active=False
    )
    assert nocache_dataset.cache is None
    _, _, _ = nocache_dataset[0]


def test_text_dataset_tokenization() -> None:
    """Verifies SymptomTextDataset outputs tokenized input ids, attention masks, and label."""
    symptom_strings = ["itching, skin rash"]
    labels = [5]

    # Setup tokenizer mock
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[101, 102, 103, 104]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0]]),
    }

    dataset = SymptomTextDataset(
        symptom_strings=symptom_strings, labels=labels, tokenizer=mock_tokenizer, max_length=4
    )
    assert len(dataset) == 1

    input_ids, attention_mask, label = dataset[0]
    assert label == 5
    assert torch.equal(input_ids, torch.tensor([101, 102, 103, 104]))
    assert torch.equal(attention_mask, torch.tensor([1, 1, 1, 0]))


def test_multimodal_dataset_returns(temp_dir: Path) -> None:
    """Verifies combined MultimodalMedicalDataset paired modal inputs output conformant shapes."""
    # Setup mock image file
    img_path = temp_dir / "scan_multimodal.png"
    img = Image.new("RGB", (100, 100), color=(0, 0, 255))
    img.save(img_path)

    # Setup tokenizer mock
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[101, 102]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }

    # Initialize Multimodal dataset
    dataset = MultimodalMedicalDataset(
        image_paths=[img_path],
        symptom_texts=["cough, fever"],
        labels=[0],
        image_transform=A.Compose([A.Resize(height=224, width=224), ToTensorV2()]),
        tokenizer=mock_tokenizer,
        max_length=2,
        cache_active=True,
    )
    assert len(dataset) == 1

    image_tensor, input_ids, attention_mask, label = dataset[0]
    assert label == 0
    assert image_tensor.shape == (3, 224, 224)
    assert torch.equal(input_ids, torch.tensor([101, 102]))
    assert torch.equal(attention_mask, torch.tensor([1, 1]))


def test_dataloader_prefetch_handling() -> None:
    """Verifies prefetch_factor is set to None if num_workers is 0 to avoid PyTorch crashes."""

    # Create simple mock dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, idx: int) -> int:
            return idx

    dataset = DummyDataset()

    # Case A: num_workers = 0 (safest for Windows)
    loader_workers_0 = create_pytorch_dataloader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2,
    )
    # Loader parameters must be normalized to prevent runtime exceptions
    assert loader_workers_0.num_workers == 0
    assert loader_workers_0.prefetch_factor is None
    assert loader_workers_0.persistent_workers is False

    # Case B: num_workers > 0
    loader_workers_gt_0 = create_pytorch_dataloader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=3,
    )
    assert loader_workers_gt_0.num_workers == 2
    assert loader_workers_gt_0.prefetch_factor == 3
    assert loader_workers_gt_0.persistent_workers is True
