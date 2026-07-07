"""Unit Tests for loading the best checkpoint automatically."""

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD

from src.utils.common import load_best_checkpoint
from src.utils.exceptions import AppStorageError


def test_load_best_checkpoint_by_loss(tmp_path):
    """Verifies that the checkpoint with the lowest validation loss is selected."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    model = nn.Linear(2, 2)
    original_weight = model.weight.clone()

    # Checkpoint 1: val_loss = 0.5, val_acc = 0.8
    ckpt1_path = ckpt_dir / "ckpt_1.pth"
    model.weight.data.fill_(1.0)
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "metrics": {"val_loss": 0.5, "val_acc": 0.8},
        },
        ckpt1_path,
    )

    # Checkpoint 2: val_loss = 0.3, val_acc = 0.9 (Best: lowest loss)
    ckpt2_path = ckpt_dir / "ckpt_2.pth"
    model.weight.data.fill_(2.0)
    torch.save(
        {
            "epoch": 2,
            "model_state_dict": model.state_dict(),
            "metrics": {"val_loss": 0.3, "val_acc": 0.9},
        },
        ckpt2_path,
    )

    # Checkpoint 3: val_loss = 0.7, val_acc = 0.95
    ckpt3_path = ckpt_dir / "ckpt_3.pth"
    model.weight.data.fill_(3.0)
    torch.save(
        {
            "epoch": 3,
            "model_state_dict": model.state_dict(),
            "metrics": {"val_loss": 0.7, "val_acc": 0.95},
        },
        ckpt3_path,
    )

    # Reset model weight
    model.weight.data.copy_(original_weight)

    # Load best checkpoint
    res = load_best_checkpoint(ckpt_dir, model, device="cpu")

    assert res["epoch"] == 2
    assert res["metrics"]["val_loss"] == 0.3
    assert res["metrics"]["val_acc"] == 0.9
    assert torch.all(model.weight == 2.0)


def test_load_best_checkpoint_tie_breaker(tmp_path):
    """Verifies that if validation losses are equal, validation accuracy is used as a tie-breaker."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    model = nn.Linear(2, 2)

    # Checkpoint 1: val_loss = 0.4, val_acc = 0.8
    model.weight.data.fill_(1.0)
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "metrics": {"val_loss": 0.4, "val_acc": 0.8},
        },
        ckpt_dir / "ckpt_1.pth",
    )

    # Checkpoint 2: val_loss = 0.4, val_acc = 0.9 (Best: higher accuracy)
    model.weight.data.fill_(2.0)
    torch.save(
        {
            "epoch": 2,
            "model_state_dict": model.state_dict(),
            "metrics": {"val_loss": 0.4, "val_acc": 0.9},
        },
        ckpt_dir / "ckpt_2.pth",
    )

    # Reset model weight
    model.weight.data.fill_(0.0)

    res = load_best_checkpoint(ckpt_dir, model, device="cpu")
    assert res["epoch"] == 2
    assert torch.all(model.weight == 2.0)


def test_load_best_checkpoint_restore_optimizer(tmp_path):
    """Verifies that the optimizer state is restored correctly."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    model = nn.Linear(2, 2)
    optimizer = SGD(model.parameters(), lr=0.1)

    # Run optimizer step to initialize state
    loss = model(torch.randn(1, 2)).sum()
    loss.backward()
    optimizer.step()

    # Save state
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": {"val_loss": 0.1, "val_acc": 0.99},
        },
        ckpt_dir / "ckpt_1.pth",
    )

    # Change optimizer learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = 0.5

    # Load best checkpoint
    load_best_checkpoint(ckpt_dir, model, optimizer, device="cpu")

    # Verify optimizer learning rate is restored back to 0.1
    for param_group in optimizer.param_groups:
        assert param_group["lr"] == 0.1


def test_load_best_checkpoint_no_checkpoint_error(tmp_path):
    """Verifies that proper error is raised when no checkpoints exist or folder is missing."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    model = nn.Linear(2, 2)

    # Empty directory
    with pytest.raises(AppStorageError) as exc_info:
        load_best_checkpoint(ckpt_dir, model, device="cpu")
    assert "No checkpoints found" in str(exc_info.value)

    # Missing directory
    with pytest.raises(AppStorageError) as exc_info:
        load_best_checkpoint(tmp_path / "nonexistent", model, device="cpu")
    assert "does not exist" in str(exc_info.value)


def test_load_best_checkpoint_printing(tmp_path, capsys):
    """Verifies that the stats are printed correctly to stdout."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    model = nn.Linear(2, 2)
    torch.save(
        {
            "epoch": 5,
            "model_state_dict": model.state_dict(),
            "metrics": {"val_loss": 0.25, "val_acc": 0.85},
        },
        ckpt_dir / "checkpoint_epoch_005.pth",
    )

    load_best_checkpoint(ckpt_dir, model, device="cpu")

    captured = capsys.readouterr()
    assert "checkpoint filename: checkpoint_epoch_005.pth" in captured.out
    assert "epoch number: 5" in captured.out
    assert "validation loss: 0.25" in captured.out
    assert "validation accuracy: 0.85" in captured.out


def test_load_best_checkpoint_invalid_checkpoint_skip(tmp_path):
    """Verifies that checkpoints missing 'model_state_dict' or otherwise invalid are skipped."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    model = nn.Linear(2, 2)

    # 1. Invalid checkpoint (missing model_state_dict)
    torch.save(
        {
            "epoch": 1,
            "metrics": {"val_loss": 0.1, "val_acc": 0.99},
        },
        ckpt_dir / "ckpt_invalid.pth",
    )

    # 2. Corrupt/empty file (will raise exception on load)
    with open(ckpt_dir / "ckpt_corrupt.pth", "w") as f:
        f.write("corrupt data")

    # 3. Valid checkpoint
    torch.save(
        {
            "epoch": 2,
            "model_state_dict": model.state_dict(),
            "metrics": {"val_loss": 0.5, "val_acc": 0.8},
        },
        ckpt_dir / "ckpt_valid.pth",
    )

    res = load_best_checkpoint(ckpt_dir, model, device="cpu")
    assert res["epoch"] == 2


def test_load_best_checkpoint_no_valid_checkpoint_error(tmp_path):
    """Verifies that an error is raised if all files in the checkpoint directory are invalid."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    model = nn.Linear(2, 2)

    # Only invalid checkpoint
    torch.save(
        {
            "epoch": 1,
            "metrics": {"val_loss": 0.1, "val_acc": 0.99},
        },
        ckpt_dir / "ckpt_invalid.pth",
    )

    with pytest.raises(AppStorageError) as exc_info:
        load_best_checkpoint(ckpt_dir, model, device="cpu")
    assert "No valid checkpoints could be loaded" in str(exc_info.value)


def test_load_best_checkpoint_missing_optimizer_state_in_checkpoint(tmp_path):
    """Verifies that a warning is logged but loading succeeds if optimizer is passed but not in checkpoint."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    model = nn.Linear(2, 2)
    optimizer = SGD(model.parameters(), lr=0.1)

    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "metrics": {"val_loss": 0.1, "val_acc": 0.99},
            # missing optimizer_state_dict
        },
        ckpt_dir / "ckpt_1.pth",
    )

    res = load_best_checkpoint(ckpt_dir, model, optimizer, device="cpu")
    assert res["epoch"] == 1
