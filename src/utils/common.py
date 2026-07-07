"""Common utility functions for the AI Medical Diagnosis Assistant."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from src.utils.exceptions import AppStorageError
from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)


def load_best_checkpoint(
    checkpoint_dir: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Scans the checkpoint directory, identifies the best checkpoint based on validation loss
    (primary) and validation accuracy (secondary, if available), and restores the model and
    optimizer states.

    Args:
        checkpoint_dir: Directory containing .pth or .pt checkpoints.
        model: PyTorch model to restore weights into.
        optimizer: PyTorch optimizer to restore state into (if available).
        device: Device to load the checkpoint onto. If None, resolves to cuda if available.

    Returns:
        Dict[str, Any]: The loaded checkpoint dictionary containing at least:
            - "epoch": The epoch number.
            - "metrics": The dictionary of validation and other training metrics.
            - "model_state_dict": The model state dict.
            - "optimizer_state_dict": The optimizer state dict (if available).

    Raises:
        AppStorageError: If the directory does not exist, has no checkpoints, or
                         no valid checkpoints can be loaded.
    """
    chk_dir = Path(checkpoint_dir)
    if not chk_dir.exists() or not chk_dir.is_dir():
        raise AppStorageError(
            message=f"Checkpoint directory does not exist or is not a directory: {checkpoint_dir}",
            details={"checkpoint_dir": str(checkpoint_dir)},
        )

    # Scan for checkpoint files (.pth and .pt)
    checkpoint_files = list(chk_dir.glob("*.pth")) + list(chk_dir.glob("*.pt"))
    if not checkpoint_files:
        raise AppStorageError(
            message=f"No checkpoints found in directory: {checkpoint_dir}",
            details={"checkpoint_dir": str(checkpoint_dir)},
        )

    # Resolve device mapping
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    best_checkpoint_path = None
    best_val_loss = float("inf")
    best_val_acc = -float("inf")
    best_checkpoint_data = None

    for ckpt_path in checkpoint_files:
        try:
            # Load metadata only or the entire checkpoint.
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            # Check if expected model_state_dict key exists
            if "model_state_dict" not in checkpoint:
                logger.warning(
                    "Skipping checkpoint %s: missing 'model_state_dict'",
                    ckpt_path.name,
                )
                continue

            metrics = checkpoint.get("metrics", {})
            val_loss = metrics.get("val_loss", float("inf"))
            val_acc = metrics.get("val_acc", None)

            # Determine if this checkpoint is the best
            # Primary criterion: lower val_loss is better.
            # Secondary criterion: if val_loss is equal, higher val_acc is better.
            is_better = False
            if val_loss < best_val_loss:
                is_better = True
            elif val_loss == best_val_loss:
                # Compare val_acc if both are not None
                if val_acc is not None:
                    current_best_acc = best_val_acc if best_val_acc != -float("inf") else -1.0
                    if val_acc > current_best_acc:
                        is_better = True

            if is_better or best_checkpoint_data is None:
                best_val_loss = val_loss
                best_val_acc = val_acc if val_acc is not None else -float("inf")
                best_checkpoint_path = ckpt_path
                best_checkpoint_data = checkpoint

        except Exception as e:
            logger.warning("Failed to parse checkpoint %s: %s", ckpt_path, e)
            continue

    if best_checkpoint_data is None or best_checkpoint_path is None:
        raise AppStorageError(
            message=f"No valid checkpoints could be loaded from: {checkpoint_dir}",
            details={"checkpoint_dir": str(checkpoint_dir)},
        )

    # Print stats as required
    filename = best_checkpoint_path.name
    epoch = best_checkpoint_data.get("epoch", -1)
    metrics_dict = best_checkpoint_data.get("metrics", {})
    val_loss = metrics_dict.get("val_loss", None)
    val_acc = metrics_dict.get("val_acc", None)

    print(f"checkpoint filename: {filename}")
    print(f"epoch number: {epoch}")
    print(f"validation loss: {val_loss}")
    print(f"validation accuracy: {val_acc}")

    # Load states
    model.load_state_dict(best_checkpoint_data["model_state_dict"])

    # Restore optimizer state if available and optimizer is provided
    if optimizer is not None:
        opt_state = best_checkpoint_data.get("optimizer_state_dict", None)
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
            logger.info("Optimizer state restored from checkpoint.")
        else:
            logger.warning(
                "Optimizer was passed, but no 'optimizer_state_dict' was found in checkpoint."
            )

    # Move model to target device if needed
    model.to(device)

    logger.info(
        "Successfully loaded best checkpoint %s from %s",
        filename,
        checkpoint_dir,
    )

    best_checkpoint_data["checkpoint_path"] = best_checkpoint_path
    return best_checkpoint_data


def download_if_needed(local_path: Union[str, Path], filename: str) -> Path:
    """Helper to check if a model file is missing or a 0-byte placeholder.
    If so, and we are running in a Hugging Face Space, download it from the HF Hub repository.
    Supports HF_MODEL_REPO_ID (model repo) or falls back to SPACE_ID (space repo).
    """
    path = Path(local_path)
    if not path.exists() or path.stat().st_size == 0:
        repo_id = os.getenv("HF_MODEL_REPO_ID") or os.getenv("SPACE_ID")
        if repo_id:
            repo_type = "model" if os.getenv("HF_MODEL_REPO_ID") else "space"
            logger.info(
                "Local file %s is missing or 0-byte placeholder. Running inside Space. "
                "Attempting auto-download from HF %s '%s'...",
                path,
                repo_type,
                repo_id,
            )
            try:
                from huggingface_hub import hf_hub_download

                # Remove the 0-byte placeholder file if present to prevent symlink or caching conflicts
                if path.exists():
                    try:
                        path.unlink()
                    except Exception as rm_err:
                        logger.debug("Could not remove 0-byte placeholder %s before download: %s", path, rm_err)

                path.parent.mkdir(parents=True, exist_ok=True)
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    filename=filename,
                    local_dir=".",
                    token=os.getenv("HF_TOKEN"),
                )
                logger.info("Successfully downloaded %s from HF to %s", filename, downloaded)
                return Path(downloaded)
            except Exception as e:
                logger.error(
                    "Auto-download of %s failed: %s. Proceeding with local file check.", filename, e
                )
        else:
            logger.warning(
                "Local file %s is missing or 0-byte placeholder, and neither SPACE_ID nor HF_MODEL_REPO_ID is set.",
                path,
            )
    return path
