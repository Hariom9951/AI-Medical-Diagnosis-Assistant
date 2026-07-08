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


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _is_lfs_pointer(path: Path) -> bool:
    """Checks if the file is a Git LFS pointer instead of a real file.
    Git LFS pointers are small text files containing 'version https://git-lfs'.
    """
    if not path.exists():
        return False
    try:
        if path.stat().st_size < 1024:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(100)
                if content.startswith("version https://git-lfs"):
                    return True
    except Exception as e:
        logger.debug("Error checking LFS pointer status of %s: %s", path, e)
    return False


def download_if_needed(local_path: Union[str, Path], filename: str) -> Path:
    """Helper to check if a model file is missing, a 0-byte placeholder, or a Git LFS pointer.
    If so, download it from the Hugging Face repository using huggingface_hub.
    Supports HF_MODEL_REPO_ID (model repo), SPACE_ID (space repo), or falls back to
    'Hariom51/ai-medical-diagnosis-assistant'.
    """
    path = Path(local_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    # Environment detection
    if os.getenv("SPACE_ID"):
        env = "Hugging Face Space"
    elif os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER"):
        env = "Docker Container"
    elif os.getenv("RENDER"):
        env = "Render"
    elif os.name == "nt":
        env = "Local Windows"
    else:
        env = "Local/Cloud VM"

    logger.info("Detected environment: %s", env)
    logger.info(
        "Checking file: %s (Exists: %s, Size: %s bytes)",
        path,
        path.exists(),
        path.stat().st_size if path.exists() else "N/A",
    )

    is_lfs = _is_lfs_pointer(path)
    if is_lfs:
        logger.info("File %s identified as a Git LFS pointer file.", path)

    if not path.exists() or path.stat().st_size == 0 or is_lfs:
        repo_id = (
            os.getenv("HF_MODEL_REPO_ID")
            or os.getenv("SPACE_ID")
            or "Hariom51/ai-medical-diagnosis-assistant"
        )
        repo_type = "model" if os.getenv("HF_MODEL_REPO_ID") else "space"
        logger.info(
            "Local file %s is missing, 0-byte, or LFS pointer. "
            "Attempting auto-download from HF %s '%s'...",
            path,
            repo_type,
            repo_id,
        )
        try:
            from huggingface_hub import hf_hub_download

            # Remove placeholder or LFS pointer file before download to avoid cache/symlink conflicts
            if path.exists():
                try:
                    path.unlink()
                    logger.info("Successfully removed placeholder/LFS pointer: %s", path)
                except Exception as rm_err:
                    logger.debug(
                        "Could not remove placeholder file %s before download: %s", path, rm_err
                    )

            path.parent.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                filename=filename,
                local_dir=str(PROJECT_ROOT),
                token=os.getenv("HF_TOKEN"),
            )
            downloaded_path = Path(downloaded)
            logger.info("Successfully downloaded %s from HF to %s (Size: %d bytes)", filename, downloaded_path, downloaded_path.stat().st_size if downloaded_path.exists() else 0)
            
            # If the downloaded file is 0 bytes (a dummy placeholder on HF), unlink it and don't write/copy it.
            if downloaded_path.exists() and downloaded_path.stat().st_size == 0:
                logger.warning("Downloaded file %s is a 0-byte placeholder on HF. Removing from local disk.", downloaded_path)
                try:
                    downloaded_path.unlink()
                except Exception:
                    pass
                if path.exists() and path.resolve() != downloaded_path.resolve():
                    try:
                        path.unlink()
                    except Exception:
                        pass
                return path

            # If the requested path is different from the downloaded path (e.g. temp dir in unit tests), copy it there
            if downloaded_path.resolve() != path.resolve():
                import shutil
                shutil.copy2(downloaded_path, path)
                logger.info("Copied downloaded file from %s to target path %s", downloaded_path, path)
            return path
        except Exception as e:
            logger.error(
                "Auto-download of %s failed: %s. Proceeding with local file check.", filename, e
            )
    else:
        logger.info("Local file %s is valid and ready.", path)
    return path
