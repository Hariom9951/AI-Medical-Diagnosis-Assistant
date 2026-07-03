"""Unit Tests for Data Transformation Component.

Tests configuration parsing, pandas symptom text sanitization, custom PyTorch
Dataset indexing, tokenization maps, stratified splits, and DataLoader batch shapes.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
    MedicalImageDataset,
    SymptomTextDataset,
)
from src.utils.exceptions import AppValidationError


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary folder directory."""
    return tmp_path


@pytest.fixture
def mock_transformation_config_yaml(temp_dir: Path) -> Path:
    """Fixture creating a valid transformation config YAML file pointing to temp_dir."""
    config_data = {
        "validated_images_dir": str(temp_dir / "raw_images"),
        "validated_symptoms_csv": str(temp_dir / "dataset.csv"),
        "transformed_dir": str(temp_dir / "transformed"),
        "reports_dir": str(temp_dir / "reports"),
        "train_split": 0.70,
        "val_split": 0.15,
        "test_split": 0.15,
        "random_state": 42,
        "image_size": 224,
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
        "disease_mapping_file": str(temp_dir / "disease_mapping.json"),
        "batch_size": 2,
        "shuffle_train": True,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
    }
    config_path = temp_dir / "transformation_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)
    return config_path


def test_transformation_config_load_success(mock_transformation_config_yaml: Path, temp_dir: Path) -> None:
    """Verifies that DataTransformationConfig loads parameters correctly from YAML."""
    config = DataTransformationConfig.from_yaml(mock_transformation_config_yaml)
    assert config.validated_images_dir == temp_dir / "raw_images"
    assert config.validated_symptoms_csv == temp_dir / "dataset.csv"
    assert config.reports_dir == temp_dir / "reports"
    assert config.train_split == 0.70
    assert config.random_state == 42
    assert config.image_size == 224
    assert config.imagenet_mean == [0.485, 0.456, 0.406]
    assert config.batch_size == 2


def test_transformation_config_load_missing_keys(temp_dir: Path) -> None:
    """Verifies config loading raises AppValidationError if required keys are missing."""
    invalid_data = {
        "train_split": 0.70,
        # missing validated_images_dir and other folders
    }
    config_path = temp_dir / "invalid_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(invalid_data, f)

    with pytest.raises(AppValidationError) as excinfo:
        DataTransformationConfig.from_yaml(config_path)
    assert "Missing required transformation" in str(excinfo.value)


def test_symptom_preprocessing(mock_transformation_config_yaml: Path, temp_dir: Path) -> None:
    """Verifies pandas symptoms cleaning: lowercase, removes formatting, drops duplicates."""
    csv_path = temp_dir / "dataset.csv"
    data = {
        "Disease": ["Allergy", "GERD"],
        "Symptom_1": [" Sneezing  ", "cough"],
        "Symptom_2": ["itching", "Heartburn!!!"],
        "Symptom_3": ["sneezing", None]  # duplicate symptom sneezing for patient 1
    }
    pd.DataFrame(data).to_csv(csv_path, index=False)

    transformer = DataTransformation(mock_transformation_config_yaml)
    texts, labels, mapping = transformer.preprocess_symptoms_df()

    assert len(texts) == 2
    assert len(labels) == 2
    assert mapping["Allergy"] == 0
    assert mapping["GERD"] == 1

    # Check text formatting (lowercase, stripped punctuation, duplicates removed)
    assert texts[0] == "sneezing, itching"  # duplicates dropped, spaces stripped, lowercase
    assert texts[1] == "cough, heartburn"   # punctuation stripped

    # Verify label map JSON written
    mapping_json = temp_dir / "disease_mapping.json"
    assert mapping_json.exists()
    with open(mapping_json, "r") as f:
        loaded_map = json.load(f)
    assert loaded_map == {"Allergy": 0, "GERD": 1}


def test_medical_image_dataset(temp_dir: Path) -> None:
    """Verifies custom PyTorch MedicalImageDataset returns transformed image tensors and labels."""
    # Write a mock PNG image
    img_path = temp_dir / "mock_img.png"
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img.save(img_path)

    # Setup albumentations transformation
    transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    dataset = MedicalImageDataset(image_paths=[img_path], labels=[3], transform=transform)
    assert len(dataset) == 1

    # MedicalImageDataset now returns (tensor, label, path_str)
    image_tensor, label, path_str = dataset[0]
    assert label == 3
    assert path_str == str(img_path)
    assert isinstance(image_tensor, torch.Tensor)
    assert image_tensor.shape == (3, 224, 224)
    assert image_tensor.dtype == torch.float32


def test_symptom_text_dataset() -> None:
    """Verifies custom PyTorch SymptomTextDataset outputs tokenized input_ids and masks."""
    symptom_strings = ["itching, skin rash", "cough, fever"]
    labels = [0, 1]

    # Mock Hugging Face Tokenizer return values
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[101, 102, 103, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0]])
    }

    dataset = SymptomTextDataset(
        symptom_strings=symptom_strings,
        labels=labels,
        tokenizer=mock_tokenizer,
        max_length=4
    )
    assert len(dataset) == 2

    batch_input_ids, batch_mask, batch_label = dataset[0]
    # SymptomTextDataset now returns (input_ids, attention_mask, label) tuple
    assert batch_label == 0
    assert torch.equal(batch_input_ids, torch.tensor([101, 102, 103, 0]))
    assert torch.equal(batch_mask, torch.tensor([1, 1, 1, 0]))
    mock_tokenizer.assert_called_once_with(
        "itching, skin rash",
        padding="max_length",
        truncation=True,
        max_length=4,
        return_tensors="pt"
    )


@patch("transformers.DistilBertTokenizer.from_pretrained")
def test_create_dataloaders_success(
    mock_tokenizer_load: MagicMock, mock_transformation_config_yaml: Path, temp_dir: Path
) -> None:
    """Verifies splits logic and DataLoader batch collations."""
    # 1. Setup mock images folders structure
    images_dir = temp_dir / "raw_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    classes = ["COVID", "Normal"]
    for cls in classes:
        (images_dir / cls).mkdir(parents=True, exist_ok=True)
        # Create 10 images per class to satisfy train_test_split minimum sample constraints
        for i in range(10):
            img_path = images_dir / cls / f"img_{i}.png"
            img = Image.new("RGB", (50, 50), color=(0, 255, 0))
            img.save(img_path)

    # 2. Setup mock symptoms CSV dataset (20 records total)
    csv_path = temp_dir / "dataset.csv"
    data = {
        "Disease": ["Allergy"] * 10 + ["GERD"] * 10,
        "Symptom_1": ["itching"] * 20,
        "Symptom_2": ["cough"] * 20,
        "Symptom_3": ["fever"] * 20
    }
    pd.DataFrame(data).to_csv(csv_path, index=False)

    # Mock tokenizer object return value
    mock_tok_inst = MagicMock()
    mock_tok_inst.return_value = {
        "input_ids": torch.ones((1, 64), dtype=torch.long),
        "attention_mask": torch.ones((1, 64), dtype=torch.long)
    }
    mock_tokenizer_load.return_value = mock_tok_inst

    # Execute Transformation pipeline
    transformer = DataTransformation(mock_transformation_config_yaml)
    loaders = transformer.create_dataloaders()

    # Verify split size ratios (Train: 70% of 20 = 14, Val: 15% of 20 = 3, Test: 15% of 20 = 3)
    splits = loaders["splits"]
    assert len(splits["train_paths"]) == 14
    assert len(splits["val_paths"]) == 3
    assert len(splits["test_paths"]) == 3

    assert len(splits["train_texts"]) == 14
    assert len(splits["val_texts"]) == 3
    assert len(splits["test_texts"]) == 3

    # Image loader now returns (tensor, label, path_str) 3-tuple
    img_batch, img_labels, img_paths = next(iter(loaders["train_img_loader"]))
    assert img_batch.shape == (2, 3, 224, 224)  # batch size is 2, image dimensions targets are 224x224
    assert img_labels.shape == (2,)

    # Symptom loader now returns (input_ids, attention_mask, label) 3-tuple
    symp_ids, symp_mask, symp_labels = next(iter(loaders["train_symp_loader"]))
    assert symp_ids.shape == (2, 64)
    assert symp_mask.shape == (2, 64)
    assert symp_labels.shape == (2,)

    # Check report generation
    transformer.generate_transformation_report(loaders)
    report_file = temp_dir / "reports" / "Data_Transformation_Report.md"
    assert report_file.exists()
    
    report_content = report_file.read_text()
    assert "Training Images Count:** `14`" in report_content
    assert "Image Batch Tensor Dims:** `[2, 3, 224, 224]`" in report_content
