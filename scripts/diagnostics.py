"""Diagnostics Script — Prints startup diagnostics for Docker and Hugging Face Spaces."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_lfs_pointer(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with open(path, "rb") as f:
            header = f.read(100)
            return b"version https://git-lfs" in header
    except Exception:
        return False


def main() -> None:
    print("=" * 70)
    print("  AI MEDICAL DIAGNOSIS ASSISTANT — STARTUP DIAGNOSTICS")
    print("=" * 70)

    # 1. Paths and Working Directory
    print(f"[DIR]  Current Working Directory: {os.getcwd()}")
    print(f"[DIR]  Project Root:              {PROJECT_ROOT}")
    print(f"[DIR]  sys.path:                  {sys.path}")

    # 2. Environment Variables
    env_vars = [
        "SPACE_ID",
        "HF_SPACE_ID",
        "TRANSFORMERS_OFFLINE",
        "HF_HOME",
        "PYTHONPATH",
        "PORT",
        "CUDA_VISIBLE_DEVICES",
    ]
    print("\n--- Environment Variables ---")
    for var in env_vars:
        print(f"  {var:<25} : {os.getenv(var, 'NOT SET')}")

    token = os.getenv("HF_TOKEN")
    if token:
        print(f"  {'HF_TOKEN':<25} : PRESENT (Length: {len(token)})")
    else:
        print(f"  {'HF_TOKEN':<25} : NOT SET / MISSING")

    # 3. File Verification List
    files_to_check = [
        ("Image Checkpoint (best)", "artifacts/checkpoints/best_model.pth"),
        ("Image Checkpoint (epoch 50)", "artifacts/checkpoints/checkpoint_epoch_050.pth"),
        ("NLP Checkpoint (best)", "artifacts/checkpoints_nlp/best_model.pt"),
        ("NLP Tokenizer JSON", "artifacts/checkpoints_nlp/tokenizer.json"),
        ("NLP Tokenizer Config", "artifacts/checkpoints_nlp/tokenizer_config.json"),
        ("NLP Special Tokens Map", "artifacts/checkpoints_nlp/special_tokens_map.json"),
        ("NLP Vocab", "artifacts/checkpoints_nlp/vocab.txt"),
        ("NLP Config JSON", "artifacts/checkpoints_nlp/config.json"),
        ("NLP Label Encoder", "artifacts/checkpoints_nlp/label_encoder.pkl"),
        ("NLP Temperature Scaler", "artifacts/checkpoints_nlp/temperature_scaler.json"),
        ("NLP Clinical Explanations", "artifacts/checkpoints_nlp/clinical_explanations.json"),
        ("NLP Model Metadata", "artifacts/checkpoints_nlp/model_metadata.json"),
        ("Disease Mapping 41", "data/processed/disease_mapping_41.json"),
        ("Disease Mapping Default", "data/processed/disease_mapping.json"),
    ]

    print("\n--- File Checks & Git LFS Verification ---")
    for name, rel_path in files_to_check:
        abs_path = PROJECT_ROOT / rel_path
        exists = abs_path.exists()
        size = abs_path.stat().st_size if exists else -1
        is_pointer = check_lfs_pointer(abs_path)

        status = "OK"
        if not exists:
            status = "MISSING"
        elif is_pointer:
            status = "LFS POINTER"
        elif size == 0:
            status = "0-BYTE DUMMY"

        print(f"  {name:<28} : {status:<12} | Size: {size:>10} bytes | Path: {rel_path}")

    print("=" * 70)


if __name__ == "__main__":
    main()
