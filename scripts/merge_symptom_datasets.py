"""Merge Symptom Datasets for BioBERT Fine-Tuning.

This script performs three tasks:
  1. Reads the existing deduplicated wide-format tabular CSV
     (data/processed/disease_symptom_cleaned.csv) and converts each row
     of keyword symptom columns into a natural-language clinical sentence:
       "itching, skin rash" -> "The patient presents with itching and skin rash."

  2. Downloads the Symptom2Disease dataset from Kaggle
     (niyarrbarman/symptom2disease) into data/downloads/ (skipped if already
     present).  Falls back gracefully when Kaggle credentials are unavailable.

  3. Normalises disease names in Symptom2Disease against the canonical 41-label
     mapping in data/processed/disease_mapping_41.json using case-insensitive
     fuzzy matching, drops unmatched rows with a warning, then concatenates both
     sources into data/processed/biobert_merged_dataset.csv with columns:
       text      - natural-language symptom description
       label     - integer class index (from disease_mapping_41.json)
       disease   - canonical disease name string
       source    - "dataset1" or "symptom2disease"

Usage
-----
  # Full run (download + merge):
  python scripts/merge_symptom_datasets.py

  # Skip Kaggle download (use already-downloaded CSV):
  python scripts/merge_symptom_datasets.py --skip-download

  # Dry-run: print stats without writing output CSV:
  python scripts/merge_symptom_datasets.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Path constants (all relative to the repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

CLEANED_CSV = REPO_ROOT / "data" / "processed" / "disease_symptom_cleaned.csv"
DISEASE_MAPPING_JSON = REPO_ROOT / "data" / "processed" / "disease_mapping_41.json"
DOWNLOAD_DIR = REPO_ROOT / "data" / "downloads"
S2D_ZIP = DOWNLOAD_DIR / "symptom2disease.zip"
S2D_CSV_GLOB = "*.csv"  # Symptom2Disease archive contains one CSV
OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "biobert_merged_dataset.csv"

KAGGLE_SLUG = "niyarrbarman/symptom2disease"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_disease_mapping(path: Path) -> Dict[str, int]:
    """Load {disease_name: int_label} from JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Disease mapping not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        mapping: Dict[str, int] = json.load(f)
    print(f"[mapping] Loaded {len(mapping)} canonical disease labels from {path.name}")
    return mapping


def _build_normaliser(mapping: Dict[str, int]) -> Dict[str, str]:
    """Build lowercase-key -> canonical-name lookup for fuzzy matching."""
    return {k.strip().lower(): k for k in mapping}


def _normalise_disease_name(
    raw: str,
    normaliser: Dict[str, str],
) -> Optional[str]:
    """Return canonical name if raw matches (case-insensitive), else None."""
    key = raw.strip().lower()
    if key in normaliser:
        return normaliser[key]

    # Try removing trailing/leading spaces that some entries have
    key_stripped = key.strip()
    if key_stripped in normaliser:
        return normaliser[key_stripped]

    # Try partial match: canonical name starts with raw or raw starts with canonical
    for canon_key, canon_name in normaliser.items():
        if key_stripped.startswith(canon_key) or canon_key.startswith(key_stripped):
            return canon_name

    return None


def _clean_symptom_token(token: str) -> str:
    """Normalise a single symptom token from the wide-format CSV."""
    cleaned = token.strip().lower()
    cleaned = re.sub(r"[^\w\s_]", "", cleaned)
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _tokens_to_sentence(tokens: List[str]) -> str:
    """Convert a list of symptom tokens into a natural-language sentence.

    Examples
    --------
    ["itching"] -> "The patient presents with itching."
    ["itching", "skin rash"] -> "The patient presents with itching and skin rash."
    ["a", "b", "c"] -> "The patient presents with a, b, and c."
    """
    if not tokens:
        return "The patient reports no specific symptoms."
    if len(tokens) == 1:
        return f"The patient presents with {tokens[0]}."
    if len(tokens) == 2:
        return f"The patient presents with {tokens[0]} and {tokens[1]}."
    # Oxford comma for 3+
    head = ", ".join(tokens[:-1])
    return f"The patient presents with {head}, and {tokens[-1]}."


# ---------------------------------------------------------------------------
# Dataset 1: Tabular → natural-language sentences
# ---------------------------------------------------------------------------


def convert_tabular_to_sentences(
    csv_path: Path,
    disease_mapping: Dict[str, int],
    normaliser: Dict[str, str],
) -> pd.DataFrame:
    """Convert wide-format symptom CSV rows into natural-language sentences.

    Args:
        csv_path: Path to disease_symptom_cleaned.csv.
        disease_mapping: Canonical {disease: int} mapping.
        normaliser: Lowercase-key normaliser dict.

    Returns:
        DataFrame with columns [text, label, disease, source].
    """
    print(f"\n[dataset1] Loading tabular CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[dataset1] Raw rows: {len(df)}")

    disease_col = next((c for c in df.columns if c.strip().lower() == "disease"), "Disease")
    symptom_cols = [c for c in df.columns if c != disease_col]

    rows = []
    skipped = 0

    for _, row in df.iterrows():
        raw_disease = str(row[disease_col]).strip()
        canonical = _normalise_disease_name(raw_disease, normaliser)
        if canonical is None:
            skipped += 1
            continue

        label = disease_mapping[canonical]

        # Gather and clean symptom tokens
        tokens: List[str] = []
        seen: set = set()
        for col in symptom_cols:
            val = str(row[col]).strip() if pd.notna(row[col]) else ""
            if not val or val.lower() in ("", "nan", "none"):
                continue
            cleaned = _clean_symptom_token(val)
            if cleaned and cleaned not in seen:
                tokens.append(cleaned)
                seen.add(cleaned)

        sentence = _tokens_to_sentence(tokens)
        rows.append(
            {
                "text": sentence,
                "label": label,
                "disease": canonical,
                "source": "dataset1",
            }
        )

    result = pd.DataFrame(rows)
    print(f"[dataset1] Converted rows: {len(result)}  |  Skipped (unmatched): {skipped}")
    return result


# ---------------------------------------------------------------------------
# Dataset 2: Download + parse Symptom2Disease
# ---------------------------------------------------------------------------


def download_symptom2disease(download_dir: Path, zip_path: Path) -> None:
    """Download Symptom2Disease from Kaggle API into download_dir."""
    try:
        import kaggle  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("kaggle package not installed. Run: pip install kaggle")

    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dataset2] Downloading '{KAGGLE_SLUG}' from Kaggle...")
    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            KAGGLE_SLUG,
            path=str(download_dir),
            quiet=False,
            unzip=False,
        )
        print(f"[dataset2] Download complete. Archive: {zip_path}")
    except Exception as e:
        raise RuntimeError(f"Kaggle download failed: {e}") from e


def extract_symptom2disease(zip_path: Path, extract_dir: Path) -> Path:
    """Extract the Symptom2Disease zip and return path to the CSV inside."""
    print(f"[dataset2] Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise FileNotFoundError("No CSV found inside Symptom2Disease archive.")
        zf.extractall(extract_dir)
    csv_path = extract_dir / csv_names[0]
    print(f"[dataset2] Extracted CSV: {csv_path}")
    return csv_path


def find_symptom2disease_csv(download_dir: Path) -> Optional[Path]:
    """Look for an already-extracted Symptom2Disease CSV in download_dir."""
    candidates = list(download_dir.glob("*.csv"))
    return candidates[0] if candidates else None


def load_symptom2disease(
    download_dir: Path,
    zip_path: Path,
    disease_mapping: Dict[str, int],
    normaliser: Dict[str, str],
    skip_download: bool = False,
) -> Tuple[pd.DataFrame, int]:
    """Load Symptom2Disease, normalise labels, return (DataFrame, n_dropped).

    Args:
        download_dir: Directory for downloads/extraction.
        zip_path: Expected path of downloaded zip file.
        disease_mapping: Canonical label mapping.
        normaliser: Lowercase normaliser dict.
        skip_download: If True, skip Kaggle download (use existing file).

    Returns:
        Tuple of (processed DataFrame, number of rows dropped due to label mismatch).
    """
    # 1. Locate or download CSV
    existing_csv = find_symptom2disease_csv(download_dir)

    if existing_csv and existing_csv.exists():
        print(f"[dataset2] Found existing CSV: {existing_csv}")
        s2d_csv = existing_csv
    elif skip_download:
        print(
            "[dataset2] --skip-download requested and no local CSV found. Skipping Symptom2Disease."
        )
        return pd.DataFrame(columns=["text", "label", "disease", "source"]), 0
    else:
        if not zip_path.exists():
            download_symptom2disease(download_dir, zip_path)
        s2d_csv = extract_symptom2disease(zip_path, download_dir)

    print(f"\n[dataset2] Loading: {s2d_csv}")
    raw = pd.read_csv(s2d_csv)
    print(f"[dataset2] Raw rows: {len(raw)}  |  columns: {list(raw.columns)}")

    # Identify label and text columns
    label_col = next((c for c in raw.columns if c.strip().lower() in ("label", "disease")), None)
    text_col = next(
        (c for c in raw.columns if c.strip().lower() in ("text", "description", "symptom")), None
    )
    if label_col is None or text_col is None:
        raise ValueError(
            f"Could not identify label/text columns in Symptom2Disease CSV. "
            f"Found columns: {list(raw.columns)}"
        )
    print(f"[dataset2] Detected columns — label: '{label_col}', text: '{text_col}'")

    rows = []
    dropped = 0
    drop_details: Dict[str, int] = {}

    for _, row in raw.iterrows():
        raw_disease = str(row[label_col]).strip()
        canonical = _normalise_disease_name(raw_disease, normaliser)
        if canonical is None:
            dropped += 1
            drop_details[raw_disease] = drop_details.get(raw_disease, 0) + 1
            continue

        label = disease_mapping[canonical]
        text = str(row[text_col]).strip()
        if not text:
            dropped += 1
            continue

        rows.append(
            {
                "text": text,
                "label": label,
                "disease": canonical,
                "source": "symptom2disease",
            }
        )

    if drop_details:
        print(f"[dataset2] WARNING: Dropped {dropped} rows due to unmatched disease names:")
        for name, count in sorted(drop_details.items(), key=lambda x: -x[1]):
            print(f"           '{name}': {count} rows")

    result = pd.DataFrame(rows)
    print(f"[dataset2] Processed rows: {len(result)}  |  Dropped: {dropped}")
    return result, dropped


# ---------------------------------------------------------------------------
# Class distribution report
# ---------------------------------------------------------------------------


def print_distribution_report(
    df: pd.DataFrame,
    disease_mapping: Dict[str, int],
) -> None:
    """Print a per-class sample count table."""
    idx_to_disease = {v: k for k, v in disease_mapping.items()}
    counts = df.groupby("label").size().reset_index(name="count")
    source_counts = df.groupby(["label", "source"]).size().unstack(fill_value=0)

    print("\n" + "=" * 70)
    print("MERGED DATASET — CLASS DISTRIBUTION REPORT")
    print("=" * 70)
    print(f"{'Label':>5}  {'Disease':<45}  {'Total':>6}  {'DS1':>5}  {'S2D':>5}")
    print("-" * 70)

    total_ds1 = 0
    total_s2d = 0
    for _, cnt_row in counts.iterrows():
        lbl = int(cnt_row["label"])
        disease = idx_to_disease.get(lbl, f"Unknown({lbl})")
        n_total = int(cnt_row["count"])
        n_ds1 = int(source_counts.get("dataset1", pd.Series(dtype=int)).get(lbl, 0))
        n_s2d = int(source_counts.get("symptom2disease", pd.Series(dtype=int)).get(lbl, 0))
        total_ds1 += n_ds1
        total_s2d += n_s2d
        print(f"{lbl:>5}  {disease:<45}  {n_total:>6}  {n_ds1:>5}  {n_s2d:>5}")

    print("-" * 70)
    print(f"{'TOTAL':>5}  {'':45}  {len(df):>6}  {total_ds1:>5}  {total_s2d:>5}")
    print("=" * 70)

    # Diseases covered only by Dataset 1 (not in Symptom2Disease)
    ds1_only = [
        idx_to_disease[lbl]
        for lbl in sorted(disease_mapping.values())
        if (source_counts.get("symptom2disease", pd.Series(dtype=int)).get(lbl, 0) == 0)
        and (source_counts.get("dataset1", pd.Series(dtype=int)).get(lbl, 0) > 0)
    ]
    if ds1_only:
        print(
            f"\n[note] {len(ds1_only)} diseases covered by Dataset 1 only (no Symptom2Disease samples):"
        )
        for d in ds1_only:
            print(f"         • {d}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge tabular symptom dataset with Symptom2Disease for BioBERT fine-tuning."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Kaggle download; use already-downloaded/extracted CSV if present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without writing the output CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_CSV),
        help=f"Output CSV path (default: {OUTPUT_CSV})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    print("=" * 70)
    print("BioBERT Dataset Merge Script")
    print("=" * 70)

    # 1. Load disease mapping
    disease_mapping = _load_disease_mapping(DISEASE_MAPPING_JSON)
    normaliser = _build_normaliser(disease_mapping)

    # 2. Convert tabular Dataset 1
    df_ds1 = convert_tabular_to_sentences(CLEANED_CSV, disease_mapping, normaliser)

    # 3. Load Symptom2Disease (Dataset 2)
    df_s2d, s2d_dropped = load_symptom2disease(
        download_dir=DOWNLOAD_DIR,
        zip_path=S2D_ZIP,
        disease_mapping=disease_mapping,
        normaliser=normaliser,
        skip_download=args.skip_download,
    )

    # 4. Merge
    df_merged = pd.concat([df_ds1, df_s2d], ignore_index=True)
    df_merged = df_merged.dropna(subset=["text", "label"])
    df_merged["label"] = df_merged["label"].astype(int)
    df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n[merge] Total merged rows: {len(df_merged)}")
    print(f"[merge] Dataset 1 rows:     {len(df_ds1)}")
    print(f"[merge] Symptom2Disease rows: {len(df_s2d)}")
    print(f"[merge] S2D rows dropped (label mismatch): {s2d_dropped}")

    # 5. Distribution report
    print_distribution_report(df_merged, disease_mapping)

    # 6. Write output
    if args.dry_run:
        print("[dry-run] Output CSV NOT written (--dry-run flag set).")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_merged.to_csv(output_path, index=False, encoding="utf-8")
        print(f"[output] Merged dataset written to: {output_path}")
        print(f"[output] Shape: {df_merged.shape}")

    print("\n[done] Merge script completed successfully.")


if __name__ == "__main__":
    main()
