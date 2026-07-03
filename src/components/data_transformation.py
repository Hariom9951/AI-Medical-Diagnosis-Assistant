"""Data Transformation Component.

Prepares validated medical scans and symptom CSV datasets for PyTorch training.
Performs stratified data splits, applies training augmentations using Albumentations,
cleans and tokenizes symptom strings, and wraps datasets in custom PyTorch DataLoaders.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

import albumentations as A
import pandas as pd
import yaml
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer

from src.components.pytorch_dataset import (
    MedicalImageDataset,
    SymptomTextDataset,
    create_pytorch_dataloader,
)
from src.utils.exceptions import AppStorageError, AppValidationError
from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)


@dataclass(frozen=True)
class DataTransformationConfig:
    """Configuration options parsed from YAML for the data transformation pipeline.

    Attributes:
        validated_images_dir (Path): Input folder of audited class directories.
        validated_symptoms_csv (Path): Path to the cleaned symptom CSV.
        transformed_dir (Path): Target folder for storing split files/maps.
        reports_dir (Path): Output folder for compilation reports.
        train_split (float): Ratio of dataset used for training (0.70).
        val_split (float): Ratio of dataset used for validation (0.15).
        test_split (float): Ratio of dataset used for testing (0.15).
        random_state (int): Random state seed for stratified splits.
        image_size (int): Target square dimension for image resizing.
        imagenet_mean (List[float]): Red/Green/Blue ImageNet channel means.
        imagenet_std (List[float]): Red/Green/Blue ImageNet channel deviations.
        disease_mapping_file (Path): Output file for disease label integers map.
        batch_size (int): Number of batch samples to load.
        shuffle_train (bool): If True, shuffle training inputs inside DataLoader.
        num_workers (int): Number of parallel CPU workers for DataLoader (0 for Windows).
        pin_memory (bool): If True, pins RAM memory to GPU inside DataLoader.
        persistent_workers (bool): If True, retains worker handles across epochs.
    """

    validated_images_dir: Path
    validated_symptoms_csv: Path
    transformed_dir: Path
    reports_dir: Path
    train_split: float
    val_split: float
    test_split: float
    random_state: int
    image_size: int
    imagenet_mean: List[float]
    imagenet_std: List[float]
    disease_mapping_file: Path
    batch_size: int
    shuffle_train: bool
    num_workers: int
    pin_memory: bool
    persistent_workers: bool

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "DataTransformationConfig":
        """Loads configuration from a YAML file.

        Args:
            yaml_path (Path): Path to the configuration YAML file.

        Returns:
            DataTransformationConfig: Parsed configuration dataclass.

        Raises:
            AppValidationError: If configuration is missing or invalid.
        """
        if not yaml_path.exists():
            raise AppValidationError(
                message=f"Configuration file not found at {yaml_path}",
                details={"yaml_path": str(yaml_path)},
            )

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                config_dict: dict[str, Any] = yaml.safe_load(f) or {}

            required_keys: Final[List[str]] = [
                "validated_images_dir",
                "validated_symptoms_csv",
                "transformed_dir",
                "reports_dir",
                "train_split",
                "val_split",
                "test_split",
                "random_state",
                "image_size",
                "imagenet_mean",
                "imagenet_std",
                "disease_mapping_file",
                "batch_size",
                "shuffle_train",
                "num_workers",
                "pin_memory",
                "persistent_workers"
            ]
            missing_keys = [k for k in required_keys if k not in config_dict]
            if missing_keys:
                raise AppValidationError(
                    message=f"Missing required transformation configuration keys: {missing_keys}",
                    details={"missing_keys": missing_keys},
                )

            return cls(
                validated_images_dir=Path(config_dict["validated_images_dir"]),
                validated_symptoms_csv=Path(config_dict["validated_symptoms_csv"]),
                transformed_dir=Path(config_dict["transformed_dir"]),
                reports_dir=Path(config_dict["reports_dir"]),
                train_split=float(config_dict["train_split"]),
                val_split=float(config_dict["val_split"]),
                test_split=float(config_dict["test_split"]),
                random_state=int(config_dict["random_state"]),
                image_size=int(config_dict["image_size"]),
                imagenet_mean=list(config_dict["imagenet_mean"]),
                imagenet_std=list(config_dict["imagenet_std"]),
                disease_mapping_file=Path(config_dict["disease_mapping_file"]),
                batch_size=int(config_dict["batch_size"]),
                shuffle_train=bool(config_dict["shuffle_train"]),
                num_workers=int(config_dict["num_workers"]),
                pin_memory=bool(config_dict["pin_memory"]),
                persistent_workers=bool(config_dict["persistent_workers"]),
            )
        except yaml.YAMLError as e:
            raise AppValidationError(
                message=f"Failed to parse YAML config: {e}",
                details={"error": str(e)},
            )
        except (ValueError, TypeError) as e:
            raise AppValidationError(
                message=f"Invalid value type in transformation config file: {e}",
                details={"error": str(e)},
            )


# Dataset classes are imported from pytorch_dataset.py


class DataTransformation:
    """Orchestrates stratified dataset splits, augmentation builds, and dataloaders."""

    def __init__(self, config_path: Path) -> None:
        """Initializes the Data Transformation component.

        Args:
            config_path (Path): Path to the YAML configuration file.
        """
        self.config: Final[DataTransformationConfig] = DataTransformationConfig.from_yaml(config_path)

        # Create output directories
        self.config.transformed_dir.mkdir(parents=True, exist_ok=True)
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        self.config.disease_mapping_file.parent.mkdir(parents=True, exist_ok=True)

        # Build Albumentations transformation pipelines
        self.train_image_transform = A.Compose([
            A.Resize(height=self.config.image_size, width=self.config.image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=15, p=0.5),
            A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.CLAHE(p=0.2),
            A.GaussianBlur(p=0.2),
            A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(1, 8), hole_width_range=(1, 8), p=0.2),
            A.Normalize(mean=tuple(self.config.imagenet_mean), std=tuple(self.config.imagenet_std)),
            ToTensorV2()
        ])

        self.val_test_image_transform = A.Compose([
            A.Resize(height=self.config.image_size, width=self.config.image_size),
            A.Normalize(mean=tuple(self.config.imagenet_mean), std=tuple(self.config.imagenet_std)),
            ToTensorV2()
        ])

    def get_image_paths_and_labels(self) -> Tuple[List[Path], List[int], Dict[str, int]]:
        """Scans validated image directories and maps class names to integer labels.

        Returns:
            Tuple[List[Path], List[int], Dict[str, int]]: Lists of paths, labels, and mapping.
        """
        logger.info("Scanning image directories: %s", self.config.validated_images_dir)
        
        class_folders = [p for p in self.config.validated_images_dir.iterdir() if p.is_dir() and p.name != "masks"]
        if not class_folders:
            subdirs = [p for p in self.config.validated_images_dir.glob("**/") if p.is_dir()]
            for d in subdirs:
                if d.name in ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]:
                    class_folders = [p for p in d.parent.iterdir() if p.is_dir() and p.name != "masks"]
                    break

        class_folders = sorted(class_folders, key=lambda x: x.name)
        class_to_idx = {folder.name: idx for idx, folder in enumerate(class_folders)}

        image_paths: List[Path] = []
        labels: List[int] = []

        for folder in class_folders:
            idx = class_to_idx[folder.name]
            # Recursively find images
            images_subfolder = folder / "images"
            if not images_subfolder.exists():
                images_subfolder = folder
            
            scans = [p for p in images_subfolder.glob("**/*") if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
            for scan in scans:
                image_paths.append(scan)
                labels.append(idx)

        logger.info("Scanned %d images across %d classes.", len(image_paths), len(class_to_idx))
        return image_paths, labels, class_to_idx

    def preprocess_symptoms_df(self) -> Tuple[List[str], List[int], Dict[str, int]]:
        """Cleans symptoms tabular columns and maps disease names to indices.

        Returns:
            Tuple[List[str], List[int], Dict[str, int]]: Symptom strings, encoded labels, mapping dict.
        """
        logger.info("Loading validated symptom CSV: %s", self.config.validated_symptoms_csv)
        df = pd.read_csv(self.config.validated_symptoms_csv)

        disease_col = next((col for col in df.columns if col.lower() == "disease"), "Disease")
        symptom_cols = [c for c in df.columns if c != disease_col]

        # 1. Label encode disease categories
        unique_diseases = sorted(df[disease_col].unique())
        disease_to_idx = {disease: idx for idx, disease in enumerate(unique_diseases)}

        # Save label mappings
        try:
            with open(self.config.disease_mapping_file, "w", encoding="utf-8") as f:
                json.dump(disease_to_idx, f, indent=4)
            logger.info("Saved disease label mapping to: %s", self.config.disease_mapping_file)
        except Exception as e:
            raise AppStorageError(f"Failed to write mapping metadata file: {e}")

        symptom_strings: List[str] = []
        labels: List[int] = []

        # 2. Text preprocessing loop
        for _, row in df.iterrows():
            disease = row[disease_col]
            labels.append(disease_to_idx[disease])

            # Gather symptoms, lowercase, strip punctuation, remove duplicates
            patient_symptoms: List[str] = []
            for col in symptom_cols:
                val = row[col]
                if pd.isna(val):
                    continue
                # Cleaning step: lowercase, remove punctuation, strip spacing
                cleaned = str(val).strip().lower()
                cleaned = re.sub(r"[^\w\s_]", "", cleaned)  # remove formatting punctuation except spaces/underscores
                cleaned = cleaned.replace("_", " ")
                cleaned = re.sub(r"\s+", " ", cleaned).strip()
                
                if cleaned and cleaned not in ["none", "nan", ""] and cleaned not in patient_symptoms:
                    patient_symptoms.append(cleaned)

            # Join symptoms with comma separation to provide a clear narrative text representation
            symptom_text = ", ".join(patient_symptoms)
            symptom_strings.append(symptom_text)

        logger.info("Symptom tabular text preparation completed. Cleaned %d patient records.", len(symptom_strings))
        return symptom_strings, labels, disease_to_idx

    def split_data(self) -> Dict[str, Any]:
        """Performs stratified Train/Val/Test splits for both datasets.

        Returns:
            Dict[str, Any]: Dictionary containing split list segments.
        """
        # A. Split Images
        img_paths, img_labels, img_mapping = self.get_image_paths_and_labels()
        
        # Stratified Train & Temp splits (Train: 70%, Temp: 30%)
        temp_ratio = self.config.val_split + self.config.test_split
        train_paths, temp_paths, train_img_lbl, temp_img_lbl = train_test_split(
            img_paths,
            img_labels,
            test_size=temp_ratio,
            stratify=img_labels,
            random_state=self.config.random_state
        )

        # Split Temp into Val (15%) and Test (15%)
        val_test_ratio = self.config.val_split / temp_ratio
        val_paths, test_paths, val_img_lbl, test_img_lbl = train_test_split(
            temp_paths,
            temp_img_lbl,
            test_size=1.0 - val_test_ratio,
            stratify=temp_img_lbl,
            random_state=self.config.random_state
        )

        # B. Split Symptoms Tabular
        symp_texts, symp_labels, symp_mapping = self.preprocess_symptoms_df()
        
        # Stratified Train & Temp splits
        try:
            train_texts, temp_texts, train_symp_lbl, temp_symp_lbl = train_test_split(
                symp_texts,
                symp_labels,
                test_size=temp_ratio,
                stratify=symp_labels,
                random_state=self.config.random_state
            )
        except ValueError:
            logger.warning("Stratified split failed on symptom CSV due to low class member counts. Falling back to non-stratified split.")
            train_texts, temp_texts, train_symp_lbl, temp_symp_lbl = train_test_split(
                symp_texts,
                symp_labels,
                test_size=temp_ratio,
                random_state=self.config.random_state
            )

        # Split Temp into Val and Test
        try:
            val_texts, test_texts, val_symp_lbl, test_symp_lbl = train_test_split(
                temp_texts,
                temp_symp_lbl,
                test_size=1.0 - val_test_ratio,
                stratify=temp_symp_lbl,
                random_state=self.config.random_state
            )
        except ValueError:
            logger.warning("Stratified split failed on validation/testing symptoms. Falling back to non-stratified split.")
            val_texts, test_texts, val_symp_lbl, test_symp_lbl = train_test_split(
                temp_texts,
                temp_symp_lbl,
                test_size=1.0 - val_test_ratio,
                random_state=self.config.random_state
            )

        logger.info(
            "Stratified splits complete. Image Train/Val/Test: %d/%d/%d | Tabular Train/Val/Test: %d/%d/%d",
            len(train_paths), len(val_paths), len(test_paths),
            len(train_texts), len(val_texts), len(test_texts)
        )

        return {
            "train_paths": train_paths, "train_img_lbl": train_img_lbl,
            "val_paths": val_paths, "val_img_lbl": val_img_lbl,
            "test_paths": test_paths, "test_img_lbl": test_img_lbl,
            "train_texts": train_texts, "train_symp_lbl": train_symp_lbl,
            "val_texts": val_texts, "val_symp_lbl": val_symp_lbl,
            "test_texts": test_texts, "test_symp_lbl": test_symp_lbl,
            "image_mapping": img_mapping,
            "symptom_mapping": symp_mapping
        }

    def create_dataloaders(self) -> Dict[str, Any]:
        """Builds Custom PyTorch Datasets and wraps them in DataLoaders.

        Returns:
            Dict[str, Any]: Dict containing DataLoader objects and split information.
        """
        logger.info("Initializing datasets and PyTorch DataLoaders creation...")
        splits = self.split_data()

        # 1. Build PyTorch Datasets
        train_img_dataset = MedicalImageDataset(
            splits["train_paths"], splits["train_img_lbl"], transform=self.train_image_transform
        )
        val_img_dataset = MedicalImageDataset(
            splits["val_paths"], splits["val_img_lbl"], transform=self.val_test_image_transform
        )
        test_img_dataset = MedicalImageDataset(
            splits["test_paths"], splits["test_img_lbl"], transform=self.val_test_image_transform
        )

        # Lazy tokenizer instantiation (pre-downloaded/cached in HuggingFace home)
        try:
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        except Exception as e:
            logger.warning("DistilBertTokenizer offline load failed. Standard mocks active. Error: %s", e)
            tokenizer = None

        train_symp_dataset = SymptomTextDataset(
            splits["train_texts"], splits["train_symp_lbl"], tokenizer=tokenizer
        )
        val_symp_dataset = SymptomTextDataset(
            splits["val_texts"], splits["val_symp_lbl"], tokenizer=tokenizer
        )
        test_symp_dataset = SymptomTextDataset(
            splits["test_texts"], splits["test_symp_lbl"], tokenizer=tokenizer
        )

        # 2. Build PyTorch DataLoaders using the factory function
        train_img_loader = create_pytorch_dataloader(
            train_img_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle_train,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers
        )
        val_img_loader = create_pytorch_dataloader(
            val_img_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers
        )
        test_img_loader = create_pytorch_dataloader(
            test_img_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers
        )

        train_symp_loader = create_pytorch_dataloader(
            train_symp_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle_train,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers
        )
        val_symp_loader = create_pytorch_dataloader(
            val_symp_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers
        )
        test_symp_loader = create_pytorch_dataloader(
            test_symp_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers
        )

        logger.info("Successfully created all 6 PyTorch DataLoaders.")

        return {
            "train_img_loader": train_img_loader,
            "val_img_loader": val_img_loader,
            "test_img_loader": test_img_loader,
            "train_symp_loader": train_symp_loader,
            "val_symp_loader": val_symp_loader,
            "test_symp_loader": test_symp_loader,
            "splits": splits
        }

    def generate_transformation_report(self, loader_results: Dict[str, Any]) -> None:
        """Writes the compilation transformation statistics Markdown report.

        Args:
            loader_results (Dict[str, Any]): Outputs returned by create_dataloaders().
        """
        logger.info("Compiling transformation reports under: %s", self.config.reports_dir)
        splits = loader_results["splits"]

        # Fetch example dimensions from loaders
        # MedicalImageDataset yields (image_tensor, label, img_path) 3-tuple
        img_batch, img_labels, _ = next(iter(loader_results["train_img_loader"]))
        # SymptomTextDataset yields (input_ids, attention_mask, label) 3-tuple
        symp_ids, symp_mask, symp_labels = next(iter(loader_results["train_symp_loader"]))

        report_path = self.config.reports_dir / "Data_Transformation_Report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"""# Data Transformation Report

Presents split allocations, augmentations, and tensor profiles.

---

## 1. Dataset Split Allocations

The ingested verified datasets have been split into Train (70%), Val (15%), and Test (15%) allocations using stratified split properties:

### Image Dataset Splits
*   **Total Scans:** `{len(splits["train_paths"]) + len(splits["val_paths"]) + len(splits["test_paths"])}`
*   **Training Images Count:** `{len(splits["train_paths"])}`
*   **Validation Images Count:** `{len(splits["val_paths"])}`
*   **Testing Images Count:** `{len(splits["test_paths"])}`

### Tabular Symptoms Dataset Splits
*   **Total Records:** `{len(splits["train_texts"]) + len(splits["val_texts"]) + len(splits["test_texts"])}`
*   **Training Records Count:** `{len(splits["train_texts"])}`
*   **Validation Records Count:** `{len(splits["val_texts"])}`
*   **Testing Records Count:** `{len(splits["test_texts"])}`

---

## 2. PyTorch DataLoader Batch Profiles

Loaded batch shapes with batch size `{self.config.batch_size}`:

*   **Image Batch Tensor Dims:** `{list(img_batch.shape)}` (expected: `[batch_size, 3, 224, 224]`)
*   **Image Labels Tensor Dims:** `{list(img_labels.shape)}` (expected: `[batch_size]`)
*   **Symptom Text input_ids Dims:** `{list(symp_ids.shape)}` (expected: `[batch_size, max_length]`)
*   **Symptom Text attention_mask Dims:** `{list(symp_mask.shape)}` (expected: `[batch_size, max_length]`)
*   **Symptom Labels Dims:** `{list(symp_labels.shape)}` (expected: `[batch_size]`)

---

## 3. Preprocessing Properties

*   **Image Dimension Targets:** `{self.config.image_size} x {self.config.image_size} (RGB)`
*   **Normalizations Applied:** ImageNet channel means `{self.config.imagenet_mean}` and std deviations `{self.config.imagenet_std}`.
*   **Disease Category Index Mappings:** Saved at `{self.config.disease_mapping_file}` mapping `{len(splits["symptom_mapping"])}` diagnostic classes.
*   **Augmentations Active (Train-Only):** HorizontalFlip, VerticalFlip, Rotate, ShiftScaleRotate, RandomBrightnessContrast, CLAHE, GaussianBlur, CoarseDropout.
""")
        logger.info("Saved data transformation report successfully.")
