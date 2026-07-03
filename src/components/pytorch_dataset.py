"""PyTorch Datasets and DataLoaders Module.

Defines custom PyTorch Dataset representations for clinical images, symptoms text narratives,
and combined multimodal formats. Includes RAM-caching options and DataLoader factories.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer

from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)


class MedicalImageDataset(Dataset):
    """Custom PyTorch Dataset for clinical medical scans.

    Implements lazy PIL image reading and optional RAM-based caching to avoid disk I/O.
    """

    def __init__(
        self,
        image_paths: List[Path],
        labels: List[int],
        transform: Optional[A.Compose] = None,
        cache_active: bool = False
    ) -> None:
        """Initializes the medical image dataset.

        Args:
            image_paths (List[Path]): List of absolute image paths.
            labels (List[int]): Matching integer class labels.
            transform (Optional[A.Compose]): Albumentations augmentation pipeline.
            cache_active (bool): If True, caches transformed tensors in RAM memory.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        # RAM Caching system
        self.cache: Optional[Dict[str, torch.Tensor]] = {} if cache_active else None
        if cache_active:
            logger.info("RAM Caching activated for MedicalImageDataset with %d samples.", len(image_paths))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """Loads image, applies transforms, and returns image tensor, label, and path.

        Returns:
            Tuple[torch.Tensor, int, str]: The image tensor [3, H, W], label, and string path.
        """
        img_path = self.image_paths[idx]
        img_path_str = str(img_path)
        label = self.labels[idx]

        # 1. Fetch from cache if present
        if self.cache is not None and img_path_str in self.cache:
            return self.cache[img_path_str], label, img_path_str

        # 2. Lazy loading PIL image
        try:
            with Image.open(img_path) as pil_img:
                rgb_img = pil_img.convert("RGB")
                image_np = np.array(rgb_img)

            if self.transform:
                augmented = self.transform(image=image_np)
                image_tensor = augmented["image"]
            else:
                image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1) / 255.0

            # Write cache
            if self.cache is not None:
                self.cache[img_path_str] = image_tensor

            return image_tensor, label, img_path_str
        except Exception as e:
            logger.error("Failed to lazy load image %s: %s", img_path_str, e)
            fallback_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
            return fallback_tensor, label, img_path_str


class SymptomTextDataset(Dataset):
    """Custom PyTorch Dataset for clinical symptoms text narratives."""

    def __init__(
        self,
        symptom_strings: List[str],
        labels: List[int],
        tokenizer: Optional[DistilBertTokenizer] = None,
        max_length: int = 64
    ) -> None:
        """Initializes the symptoms text dataset.

        Args:
            symptom_strings (List[str]): Clean narrative text descriptions.
            labels (List[int]): Target encoded disease category label.
            tokenizer (Optional[DistilBertTokenizer]): HuggingFace tokenizer instance.
            max_length (int): Token sequence truncation/padding boundaries.
        """
        self.symptom_strings = symptom_strings
        self.labels = labels
        self.max_length = max_length

        if tokenizer is None:
            try:
                self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            except Exception as e:
                logger.warning("DistilBertTokenizer online load failed: %s. Using fallback token mock.", e)
                self.tokenizer = None
        else:
            self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.symptom_strings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Tokenizes symptom text and returns token ids, attention masks, and label.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, int]: input_ids [max_length], attention_mask [max_length], label.
        """
        text = self.symptom_strings[idx]
        label = self.labels[idx]

        if self.tokenizer is not None:
            try:
                encoded = self.tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                return (
                    encoded["input_ids"].squeeze(0),
                    encoded["attention_mask"].squeeze(0),
                    label
                )
            except Exception as e:
                logger.error("Tokenization error for text '%s': %s", text, e)

        # Fallback empty tensors
        return (
            torch.zeros((self.max_length,), dtype=torch.long),
            torch.zeros((self.max_length,), dtype=torch.long),
            label
        )


class MultimodalMedicalDataset(Dataset):
    """Combined Multimodal PyTorch Dataset aligning chest scans and symptom text narratives."""

    def __init__(
        self,
        image_paths: List[Path],
        symptom_texts: List[str],
        labels: List[int],
        image_transform: Optional[A.Compose] = None,
        tokenizer: Optional[DistilBertTokenizer] = None,
        max_length: int = 64,
        cache_active: bool = False
    ) -> None:
        """Initializes the multimodal dataset.

        Args:
            image_paths (List[Path]): Clinical image file paths.
            symptom_texts (List[str]): Matching narrative symptom strings.
            labels (List[int]): Aligned disease class integers.
            image_transform (Optional[A.Compose]): Albumentations augmentation pipeline.
            tokenizer (Optional[DistilBertTokenizer]): HuggingFace tokenizer instance.
            max_length (int): Tokenizer max padding sequence length.
            cache_active (bool): If True, caches image scans in RAM.
        """
        self.image_paths = image_paths
        self.symptom_texts = symptom_texts
        self.labels = labels
        self.image_transform = image_transform
        self.max_length = max_length

        if tokenizer is None:
            try:
                self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            except Exception as e:
                logger.warning("DistilBertTokenizer load failed: %s. Using fallback token mock.", e)
                self.tokenizer = None
        else:
            self.tokenizer = tokenizer

        # Image caching system
        self.cache: Optional[Dict[str, torch.Tensor]] = {} if cache_active else None

    def __len__(self) -> int:
        # Len is defined by images list (major modal)
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Returns paired image tensor, token ids, attention masks, and label.

        Uses modulo index math to handle unaligned dataset lengths safely.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
                - Image tensor [3, H, W]
                - input_ids [max_length]
                - attention_mask [max_length]
                - disease label
        """
        img_path = self.image_paths[idx]
        img_path_str = str(img_path)
        
        # Safe indexing bounds mapping for unaligned datasets
        text = self.symptom_texts[idx % len(self.symptom_texts)]
        label = self.labels[idx % len(self.labels)]

        # 1. Fetch/Compute image representation
        image_tensor: Optional[torch.Tensor] = None
        if self.cache is not None and img_path_str in self.cache:
            image_tensor = self.cache[img_path_str]
        else:
            try:
                with Image.open(img_path) as pil_img:
                    rgb_img = pil_img.convert("RGB")
                    image_np = np.array(rgb_img)

                if self.image_transform:
                    augmented = self.image_transform(image=image_np)
                    image_tensor = augmented["image"]
                else:
                    image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1) / 255.0

                if self.cache is not None:
                    self.cache[img_path_str] = image_tensor
            except Exception as e:
                logger.error("Failed to load image in multimodal dataset: %s", e)
                image_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)

        # 2. Tokenize text narrative
        input_ids = torch.zeros((self.max_length,), dtype=torch.long)
        attention_mask = torch.zeros((self.max_length,), dtype=torch.long)

        if self.tokenizer is not None:
            try:
                encoded = self.tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                input_ids = encoded["input_ids"].squeeze(0)
                attention_mask = encoded["attention_mask"].squeeze(0)
            except Exception as e:
                logger.error("Tokenization failed in multimodal dataset: %s", e)

        # image_tensor cannot be None because of fallback block
        return image_tensor, input_ids, attention_mask, label


def create_pytorch_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: Optional[int] = 2
) -> DataLoader:
    """Safely constructs a PyTorch DataLoader instance.

    Ensures prefetch_factor and persistent_workers configurations are disabled if
    num_workers == 0 to prevent PyTorch constructor crashes.

    Args:
        dataset (Dataset): Target PyTorch dataset.
        batch_size (int): Batch size dimensions.
        shuffle (bool): True if training inputs shuffle.
        num_workers (int): parallel worker cores count (0 for Windows).
        pin_memory (bool): If True, pin loaders memory space to GPU.
        persistent_workers (bool): If True, retain process loaders across epochs.
        prefetch_factor (Optional[int]): Batches prefetched by each worker.

    Returns:
        DataLoader: Instantiated PyTorch DataLoader.
    """
    safe_prefetch: Optional[int] = prefetch_factor
    safe_persistent: bool = persistent_workers

    # Standard PyTorch assertion protection
    if num_workers == 0:
        safe_prefetch = None
        safe_persistent = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=safe_persistent,
        prefetch_factor=safe_prefetch
    )
