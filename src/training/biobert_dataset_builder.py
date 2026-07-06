"""BioBERT Dataset Builder — Phase 2.1.

Handles the complete dataset preparation pipeline for fine-tuning BioBERT on
medical symptom classification.  Specifically:

  1. Downloads the Symptom2Disease dataset from Kaggle (skips if already present).
  2. Loads the existing 41-disease project dataset (disease_symptom_cleaned.csv).
  3. Validates both datasets: structure, disease label consistency, missing values,
     and duplicate rows.
  4. Converts Dataset 1's wide-format keyword rows into natural-language clinical
     sentences aligned with BioBERT's pre-training corpus style.
  5. Normalises Symptom2Disease disease labels against the canonical 41-label
     mapping from disease_mapping_41.json using case-insensitive matching.
  6. Merges both sources and writes intermediate artifacts to data/merged/.
  7. Generates a comprehensive Markdown statistics report.

Directory outputs
-----------------
  data/raw/symptom2disease/          — raw extracted Symptom2Disease CSV
  data/interim/dataset1_sentences.csv — Dataset 1 after keyword→sentence conversion
  data/interim/dataset2_normalised.csv — Dataset 2 after label normalisation
  data/merged/biobert_merged_dataset.csv — final merged dataset for BioBERT
  data/merged/dataset_statistics_report.md — human-readable statistics report
  data/merged/plots/                 — distribution and analysis charts

Usage
-----
  # Run the full pipeline:
  python -m src.training.biobert_dataset_builder

  # Programmatic usage (importable):
  from src.training.biobert_dataset_builder import BioBERTDatasetBuilder
  builder = BioBERTDatasetBuilder()
  builder.run()
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import class_weight as sk_class_weight
import numpy as np

from src.utils.exceptions import AppStorageError, AppValidationError
from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Kaggle dataset identifier for Symptom2Disease
_S2D_KAGGLE_SLUG: Final[str] = "niyarrbarman/symptom2disease"
_S2D_DATASET_NAME: Final[str] = "symptom2disease"

# Curated alias table: maps known Symptom2Disease label variants to the exact
# canonical names used in disease_mapping_41.json.  Entries here take priority
# over the generic fuzzy matching in normalise_disease_label().
_KNOWN_ALIASES: Final[Dict[str, str]] = {
    # Symptom2Disease spelling          : canonical name in disease_mapping_41.json
    "dimorphic hemorrhoids"             : "Dimorphic hemmorhoids(piles)",
    "dimorphic haemorrhoids"            : "Dimorphic hemmorhoids(piles)",
    "dimorphic hemorrhoids(piles)"      : "Dimorphic hemmorhoids(piles)",
    "gastroesophageal reflux disease"   : "GERD",
    "gastro-esophageal reflux disease"  : "GERD",
    "gastro esophageal reflux disease"  : "GERD",
    "peptic ulcer disease"              : "Peptic ulcer diseae",
    "peptic ulcer"                      : "Peptic ulcer diseae",
    "cervical spondylosis"              : "Cervical spondylosis",
    "chicken pox"                       : "Chicken pox",
    "urinary tract infection"           : "Urinary tract infection",
    "varicose veins"                    : "Varicose veins",
    "hepatitis a"                       : "hepatitis A",
    "hepatitis b"                       : "Hepatitis B",
    "hepatitis c"                       : "Hepatitis C",
    "hepatitis d"                       : "Hepatitis D",
    "hepatitis e"                       : "Hepatitis E",
    "heart attack"                      : "Heart attack",
    "paralysis (brain hemorrhage)"       : "Paralysis (brain hemorrhage)",
    "paralysis brain hemorrhage"         : "Paralysis (brain hemorrhage)",
    "(vertigo) paroymsal positional vertigo" : "(vertigo) Paroymsal  Positional Vertigo",
    "vertigo"                           : "(vertigo) Paroymsal  Positional Vertigo",
}

# File paths relative to the project root (resolved at build time)
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_DATA_ROOT: Final[Path] = _REPO_ROOT / "data"

# Input paths
_CLEANED_CSV: Final[Path] = _DATA_ROOT / "processed" / "disease_symptom_cleaned.csv"
_DISEASE_MAPPING_JSON: Final[Path] = _DATA_ROOT / "processed" / "disease_mapping_41.json"

# Output directory layout
_DOWNLOAD_DIR: Final[Path] = _DATA_ROOT / "downloads"
_RAW_S2D_DIR: Final[Path] = _DATA_ROOT / "raw" / "symptom2disease"
_INTERIM_DIR: Final[Path] = _DATA_ROOT / "interim"
_MERGED_DIR: Final[Path] = _DATA_ROOT / "merged"
_PLOTS_DIR: Final[Path] = _MERGED_DIR / "plots"

# Output file paths
_INTERIM_DS1: Final[Path] = _INTERIM_DIR / "dataset1_sentences.csv"
_INTERIM_DS2: Final[Path] = _INTERIM_DIR / "dataset2_normalised.csv"
_MERGED_CSV: Final[Path] = _MERGED_DIR / "biobert_merged_dataset.csv"
_REPORT_PATH: Final[Path] = _MERGED_DIR / "dataset_statistics_report.md"


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetStats:
    """Container for per-dataset validation statistics.

    Attributes:
        name (str): Dataset identifier label.
        source_path (str): Path or slug the data was loaded from.
        total_rows (int): Number of rows loaded before any filtering.
        valid_rows (int): Number of rows retained after validation.
        dropped_rows (int): Number of rows dropped during validation.
        num_diseases (int): Count of unique disease labels after normalisation.
        num_missing_cells (int): Total cells with null/NaN values.
        num_duplicate_rows (int): Number of duplicate rows detected.
        disease_distribution (Dict[str, int]): Per-disease sample counts.
        drop_reasons (Dict[str, int]): Counts grouped by drop reason.
        text_length_stats (Dict[str, float]): Min/max/mean token lengths.
    """

    name: str
    source_path: str
    total_rows: int = 0
    valid_rows: int = 0
    dropped_rows: int = 0
    num_diseases: int = 0
    num_missing_cells: int = 0
    num_duplicate_rows: int = 0
    disease_distribution: Dict[str, int] = field(default_factory=dict)
    drop_reasons: Dict[str, int] = field(default_factory=dict)
    text_length_stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class MergedDatasetStats:
    """Container for the merged dataset statistics.

    Attributes:
        total_rows (int): Total rows in the merged dataset.
        num_diseases (int): Total unique disease labels in merge.
        dataset1_rows (int): Rows contributed by Dataset 1.
        dataset2_rows (int): Rows contributed by Dataset 2 (Symptom2Disease).
        diseases_only_in_ds1 (List[str]): Diseases with no Symptom2Disease samples.
        diseases_in_both (List[str]): Diseases covered by both sources.
        per_class_counts (Dict[str, int]): Final per-disease row counts.
    """

    total_rows: int = 0
    num_diseases: int = 0
    dataset1_rows: int = 0
    dataset2_rows: int = 0
    diseases_only_in_ds1: List[str] = field(default_factory=list)
    diseases_in_both: List[str] = field(default_factory=list)
    per_class_counts: Dict[str, int] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Text Preprocessing Helpers
# ─────────────────────────────────────────────────────────────────────────────

def clean_symptom_token(raw_token: str) -> str:
    """Normalises a single symptom token extracted from a wide-format CSV cell.

    Applies the same cleaning logic used in SymptomDataPreprocessor so that
    converted sentences are consistent with inference-time preprocessing.

    Args:
        raw_token (str): Raw cell value (e.g. " nodal_skin_eruptions ").

    Returns:
        str: Cleaned, human-readable symptom phrase (e.g. "nodal skin eruptions").
    """
    token = raw_token.strip().lower()
    token = re.sub(r"[^\w\s_]", "", token)   # remove special chars except _ and space
    token = token.replace("_", " ")
    token = re.sub(r"\s+", " ", token).strip()
    return token


def tokens_to_clinical_sentence(tokens: List[str]) -> str:
    """Converts a list of symptom tokens into a natural-language clinical sentence.

    Produces sentences aligned with BioBERT's PubMed pre-training corpus style.

    Examples
    --------
    >>> tokens_to_clinical_sentence([])
    'The patient reports no specific symptoms.'
    >>> tokens_to_clinical_sentence(["fever"])
    'The patient presents with fever.'
    >>> tokens_to_clinical_sentence(["fever", "cough"])
    'The patient presents with fever and cough.'
    >>> tokens_to_clinical_sentence(["fever", "cough", "headache"])
    'The patient presents with fever, cough, and headache.'

    Args:
        tokens (List[str]): Cleaned symptom token strings.

    Returns:
        str: Natural-language clinical sentence.
    """
    if not tokens:
        return "The patient reports no specific symptoms."
    if len(tokens) == 1:
        return f"The patient presents with {tokens[0]}."
    if len(tokens) == 2:
        return f"The patient presents with {tokens[0]} and {tokens[1]}."
    # Oxford-comma form for 3 or more symptoms
    symptom_body = ", ".join(tokens[:-1])
    return f"The patient presents with {symptom_body}, and {tokens[-1]}."


# ─────────────────────────────────────────────────────────────────────────────
# Label Normalisation Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_label_normaliser(mapping: Dict[str, int]) -> Dict[str, str]:
    """Builds a lowercase-key → canonical-name lookup for fuzzy label matching.

    Args:
        mapping (Dict[str, int]): Canonical {disease_name: label_index} mapping.

    Returns:
        Dict[str, str]: {lowercase_name: canonical_name} for case-insensitive lookup.
    """
    return {name.strip().lower(): name for name in mapping}


def normalise_disease_label(
    raw_name: str,
    normaliser: Dict[str, str],
) -> Optional[str]:
    """Returns the canonical disease name from the 41-label mapping, or None.

    Resolution order:
      1. Curated alias table (_KNOWN_ALIASES) — highest priority.
      2. Exact case-insensitive match against the canonical mapping.
      3. Prefix-based partial match as a last-resort fallback.

    Args:
        raw_name (str): Disease name string from the source dataset.
        normaliser (Dict[str, str]): Lowercase -> canonical lookup built by
            build_label_normaliser().

    Returns:
        Optional[str]: Canonical disease name, or None if unmatched.
    """
    key = raw_name.strip().lower()

    # 1. Curated alias lookup (highest-priority, catches known Symptom2Disease variants)
    if key in _KNOWN_ALIASES:
        return _KNOWN_ALIASES[key]

    # 2. Exact case-insensitive match against the canonical mapping
    if key in normaliser:
        return normaliser[key]

    # 3. Prefix-based partial match (handles trailing spaces, minor punctuation)
    for canon_key, canon_name in normaliser.items():
        if key.startswith(canon_key) or canon_key.startswith(key):
            return canon_name

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Kaggle Download Helper
# ─────────────────────────────────────────────────────────────────────────────

def download_symptom2disease(
    download_dir: Path,
    raw_output_dir: Path,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
) -> Path:
    """Downloads and extracts the Symptom2Disease dataset from Kaggle.

    Follows the project's established pattern (exponential backoff retries,
    idempotency check, Zip Slip security validation).  Skips the download
    entirely if the extracted directory already contains a CSV file.

    Args:
        download_dir (Path): Directory to save the downloaded .zip archive.
        raw_output_dir (Path): Directory to extract the CSV contents into.
        max_retries (int): Maximum Kaggle API retry attempts.
        backoff_factor (float): Multiplier for exponential backoff delay.

    Returns:
        Path: Path to the extracted Symptom2Disease CSV file.

    Raises:
        AppStorageError: If Kaggle credentials are missing or the download
            fails after all retries.
        AppValidationError: If the extracted archive contains no CSV files
            or zip-slip traversal paths are detected.
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = download_dir / f"{_S2D_DATASET_NAME}.zip"

    # ── Idempotency: check if CSV already extracted ───────────────────────
    existing_csvs = list(raw_output_dir.glob("*.csv"))
    if existing_csvs:
        logger.info(
            "Symptom2Disease CSV already present at %s — skipping download.",
            existing_csvs[0],
        )
        return existing_csvs[0]

    # ── Import Kaggle API ─────────────────────────────────────────────────
    try:
        import kaggle  # type: ignore[import-untyped]
    except ImportError as e:
        raise AppStorageError(
            message="kaggle package is not installed. Run: pip install kaggle",
            details={"error": str(e)},
        )

    # ── Authenticate ──────────────────────────────────────────────────────
    try:
        kaggle.api.authenticate()
        logger.info("Kaggle API authenticated successfully.")
    except Exception as e:
        raise AppStorageError(
            message=(
                "Kaggle API authentication failed. Ensure ~/.kaggle/kaggle.json "
                "is correctly configured with your API credentials."
            ),
            details={"error": str(e)},
        )

    # ── Download with exponential backoff ─────────────────────────────────
    last_exception: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Downloading Symptom2Disease from Kaggle (attempt %d/%d)...",
                attempt, max_retries,
            )
            kaggle.api.dataset_download_files(
                dataset=_S2D_KAGGLE_SLUG,
                path=str(download_dir),
                quiet=False,
                unzip=False,
            )
            if not zip_path.exists():
                raise FileNotFoundError(
                    f"Expected archive not found after download: {zip_path}"
                )
            logger.info("Download completed: %s", zip_path.name)
            break
        except Exception as exc:
            logger.warning("Download attempt %d failed: %s", attempt, exc)
            last_exception = exc
            if zip_path.exists():
                try:
                    zip_path.unlink()
                except OSError:
                    pass
            if attempt < max_retries:
                sleep_time = backoff_factor ** attempt
                logger.info("Retrying in %.1fs...", sleep_time)
                time.sleep(sleep_time)
    else:
        raise AppStorageError(
            message=(
                f"Failed to download {_S2D_KAGGLE_SLUG} from Kaggle after "
                f"{max_retries} attempts."
            ),
            details={"last_error": str(last_exception)},
        )

    # ── Extract with Zip Slip security validation ─────────────────────────
    logger.info("Extracting %s to %s ...", zip_path.name, raw_output_dir)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                abs_target = Path(os.path.abspath(raw_output_dir / member.filename))
                abs_root = Path(os.path.abspath(raw_output_dir))
                if not abs_target.is_relative_to(abs_root):
                    raise AppValidationError(
                        message=(
                            f"Zip Slip traversal detected in member: {member.filename}"
                        ),
                        details={"member": member.filename},
                    )
            zf.extractall(raw_output_dir)
    except AppValidationError:
        raise
    except Exception as e:
        raise AppStorageError(
            message=f"Failed to extract archive {zip_path}: {e}",
            details={"zip_path": str(zip_path), "error": str(e)},
        )

    extracted_csvs = list(raw_output_dir.glob("*.csv"))
    if not extracted_csvs:
        raise AppValidationError(
            message="No CSV file found inside the Symptom2Disease archive.",
            details={"raw_output_dir": str(raw_output_dir)},
        )

    logger.info("Extraction complete. CSV found at: %s", extracted_csvs[0])
    return extracted_csvs[0]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Loading and Validation Functions
# ─────────────────────────────────────────────────────────────────────────────

def load_disease_mapping(mapping_path: Path) -> Dict[str, int]:
    """Loads the canonical 41-label disease mapping from JSON.

    Args:
        mapping_path (Path): Path to disease_mapping_41.json.

    Returns:
        Dict[str, int]: {disease_name: integer_label} mapping.

    Raises:
        AppStorageError: If the file cannot be read or parsed.
    """
    if not mapping_path.exists():
        raise AppStorageError(
            message=f"Disease mapping file not found: {mapping_path}",
            details={"path": str(mapping_path)},
        )
    try:
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping: Dict[str, int] = json.load(f)
        logger.info(
            "Loaded %d canonical disease labels from %s",
            len(mapping), mapping_path.name,
        )
        return mapping
    except Exception as e:
        raise AppStorageError(
            message=f"Failed to parse disease mapping JSON: {e}",
            details={"path": str(mapping_path), "error": str(e)},
        )


def load_and_validate_dataset1(
    csv_path: Path,
    disease_mapping: Dict[str, int],
) -> Tuple[pd.DataFrame, DatasetStats]:
    """Loads the existing 41-disease project dataset and computes validation stats.

    Reads the wide-format CSV (Disease + Symptom_1 … Symptom_17) without
    modifying it.  Returns both the raw DataFrame and a populated DatasetStats.

    Args:
        csv_path (Path): Path to disease_symptom_cleaned.csv.
        disease_mapping (Dict[str, int]): Canonical 41-label mapping.

    Returns:
        Tuple[pd.DataFrame, DatasetStats]:
            - Raw, unmodified DataFrame.
            - Populated validation statistics.

    Raises:
        AppStorageError: If the CSV cannot be read.
        AppValidationError: If required columns are missing.
    """
    if not csv_path.exists():
        raise AppStorageError(
            message=f"Dataset 1 CSV not found: {csv_path}",
            details={"path": str(csv_path)},
        )

    logger.info("Loading Dataset 1 (project dataset): %s", csv_path)
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise AppStorageError(
            message=f"Failed to read Dataset 1 CSV: {e}",
            details={"path": str(csv_path), "error": str(e)},
        )

    stats = DatasetStats(name="Dataset 1 (Project — 41 Diseases)", source_path=str(csv_path))
    stats.total_rows = len(df)

    # ── Column validation ─────────────────────────────────────────────────
    disease_col = next(
        (c for c in df.columns if c.strip().lower() == "disease"), None
    )
    if disease_col is None:
        raise AppValidationError(
            message="Dataset 1 CSV is missing the 'Disease' column.",
            details={"columns_found": list(df.columns)},
        )

    symptom_cols = [c for c in df.columns if c.strip().lower() != "disease"]

    # ── Missing value analysis ─────────────────────────────────────────────
    stats.num_missing_cells = int(df.isnull().sum().sum())

    # ── Duplicate row analysis ────────────────────────────────────────────
    stats.num_duplicate_rows = int(df.duplicated().sum())

    # ── Disease label consistency ─────────────────────────────────────────
    normaliser = build_label_normaliser(disease_mapping)
    raw_disease_values = df[disease_col].dropna().astype(str).str.strip().unique().tolist()
    unmatched_labels: List[str] = []
    for raw_label in raw_disease_values:
        if normalise_disease_label(raw_label, normaliser) is None:
            unmatched_labels.append(raw_label)

    if unmatched_labels:
        logger.warning(
            "Dataset 1 contains %d disease names not found in canonical mapping: %s",
            len(unmatched_labels), unmatched_labels,
        )
    else:
        logger.info("Dataset 1 — all disease labels are consistent with canonical mapping.")

    # ── Class distribution ────────────────────────────────────────────────
    disease_series = df[disease_col].dropna().astype(str).str.strip()
    stats.disease_distribution = disease_series.value_counts().to_dict()
    stats.num_diseases = len(stats.disease_distribution)
    stats.valid_rows = stats.total_rows
    stats.dropped_rows = 0
    stats.drop_reasons = {"unmatched_label": len(unmatched_labels)} if unmatched_labels else {}

    # ── Text length analysis (on joined symptom strings) ──────────────────
    symptom_lengths: List[int] = []
    for _, row in df.iterrows():
        tokens: List[str] = []
        for col in symptom_cols:
            val = str(row[col]).strip() if pd.notna(row[col]) else ""
            if val and val.lower() not in ("", "nan", "none"):
                tokens.append(clean_symptom_token(val))
        sentence = tokens_to_clinical_sentence(tokens)
        symptom_lengths.append(len(sentence.split()))

    if symptom_lengths:
        stats.text_length_stats = {
            "min_words": float(min(symptom_lengths)),
            "max_words": float(max(symptom_lengths)),
            "mean_words": float(round(sum(symptom_lengths) / len(symptom_lengths), 2)),
        }

    logger.info(
        "Dataset 1 validation complete — %d rows, %d diseases, %d duplicates, %d missing cells.",
        stats.total_rows, stats.num_diseases, stats.num_duplicate_rows, stats.num_missing_cells,
    )
    return df, stats


def load_and_validate_dataset2(
    csv_path: Path,
    disease_mapping: Dict[str, int],
) -> Tuple[pd.DataFrame, DatasetStats]:
    """Loads and validates the Symptom2Disease CSV (Dataset 2).

    Detects label and text columns automatically, then validates disease name
    consistency against the 41-label canonical mapping.  Returns the raw,
    unmodified DataFrame alongside a populated DatasetStats.

    Args:
        csv_path (Path): Path to the extracted Symptom2Disease CSV.
        disease_mapping (Dict[str, int]): Canonical 41-label mapping.

    Returns:
        Tuple[pd.DataFrame, DatasetStats]:
            - Raw, unmodified DataFrame.
            - Populated validation statistics.

    Raises:
        AppStorageError: If the CSV cannot be read.
        AppValidationError: If label or text columns cannot be detected.
    """
    if not csv_path.exists():
        raise AppStorageError(
            message=f"Symptom2Disease CSV not found: {csv_path}",
            details={"path": str(csv_path)},
        )

    logger.info("Loading Dataset 2 (Symptom2Disease): %s", csv_path)
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise AppStorageError(
            message=f"Failed to read Symptom2Disease CSV: {e}",
            details={"path": str(csv_path), "error": str(e)},
        )

    stats = DatasetStats(
        name="Dataset 2 (Symptom2Disease — 24 Diseases)",
        source_path=str(csv_path),
    )
    stats.total_rows = len(df)

    # ── Column auto-detection ─────────────────────────────────────────────
    label_col = next(
        (c for c in df.columns if c.strip().lower() in ("label", "disease")), None
    )
    text_col = next(
        (c for c in df.columns if c.strip().lower() in ("text", "description", "symptom")), None
    )
    if label_col is None:
        raise AppValidationError(
            message="Could not find a label/disease column in Symptom2Disease CSV.",
            details={"columns_found": list(df.columns)},
        )
    if text_col is None:
        raise AppValidationError(
            message="Could not find a text/description column in Symptom2Disease CSV.",
            details={"columns_found": list(df.columns)},
        )
    logger.info(
        "Dataset 2 column mapping — label: '%s', text: '%s'", label_col, text_col
    )

    # ── Missing value analysis ─────────────────────────────────────────────
    stats.num_missing_cells = int(df[[label_col, text_col]].isnull().sum().sum())

    # ── Duplicate row analysis ────────────────────────────────────────────
    stats.num_duplicate_rows = int(df.duplicated().sum())

    # ── Disease label consistency ─────────────────────────────────────────
    normaliser = build_label_normaliser(disease_mapping)
    raw_labels = df[label_col].dropna().astype(str).str.strip().unique().tolist()
    matched_count = 0
    unmatched_labels: Dict[str, int] = {}

    for raw_label in raw_labels:
        if normalise_disease_label(raw_label, normaliser) is not None:
            matched_count += 1
        else:
            unmatched_labels[raw_label] = int(
                (df[label_col].astype(str).str.strip() == raw_label).sum()
            )

    if unmatched_labels:
        logger.warning(
            "Dataset 2 — %d disease names could not be matched to canonical labels "
            "(rows will be dropped during preprocessing): %s",
            len(unmatched_labels), list(unmatched_labels.keys()),
        )
        stats.drop_reasons = unmatched_labels
        stats.dropped_rows = sum(unmatched_labels.values())
    else:
        logger.info("Dataset 2 — all %d disease labels match the canonical mapping.", matched_count)

    # ── Class distribution (raw labels) ───────────────────────────────────
    disease_series = df[label_col].dropna().astype(str).str.strip()
    stats.disease_distribution = disease_series.value_counts().to_dict()
    stats.num_diseases = len(stats.disease_distribution)
    stats.valid_rows = stats.total_rows - stats.dropped_rows

    # ── Text length analysis ───────────────────────────────────────────────
    text_lengths = df[text_col].dropna().astype(str).apply(lambda t: len(t.split()))
    if not text_lengths.empty:
        stats.text_length_stats = {
            "min_words": float(text_lengths.min()),
            "max_words": float(text_lengths.max()),
            "mean_words": float(round(text_lengths.mean(), 2)),
        }

    logger.info(
        "Dataset 2 validation complete — %d rows, %d diseases, %d duplicates, "
        "%d missing cells, %d rows to drop.",
        stats.total_rows, stats.num_diseases, stats.num_duplicate_rows,
        stats.num_missing_cells, stats.dropped_rows,
    )
    return df, stats


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing Pipeline Functions
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_dataset1(
    df: pd.DataFrame,
    disease_mapping: Dict[str, int],
) -> pd.DataFrame:
    """Converts Dataset 1's wide-format keyword rows into natural-language sentences.

    The original DataFrame is NOT modified — a new DataFrame is returned.
    The output has columns: text, label, disease, source.

    Keyword-to-sentence conversion:
        "itching, skin_rash, nodal_skin_eruptions" →
        "The patient presents with itching, skin rash, and nodal skin eruptions."

    Args:
        df (pd.DataFrame): Raw Dataset 1 DataFrame (wide-format tabular).
        disease_mapping (Dict[str, int]): Canonical {disease_name: int} mapping.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with columns [text, label, disease, source].
    """
    logger.info("Preprocessing Dataset 1: converting keyword rows to clinical sentences...")
    normaliser = build_label_normaliser(disease_mapping)

    disease_col = next(
        (c for c in df.columns if c.strip().lower() == "disease"), "Disease"
    )
    symptom_cols = [c for c in df.columns if c.strip().lower() != "disease"]

    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        raw_disease = str(row[disease_col]).strip()
        canonical = normalise_disease_label(raw_disease, normaliser)
        if canonical is None:
            logger.debug("Skipping row — unmatched disease label: '%s'", raw_disease)
            continue

        label = disease_mapping[canonical]

        # Gather, clean, and deduplicate symptom tokens
        tokens: List[str] = []
        seen: set = set()
        for col in symptom_cols:
            cell_val = row[col]
            if pd.isna(cell_val):
                continue
            raw_val = str(cell_val).strip()
            if not raw_val or raw_val.lower() in ("", "nan", "none"):
                continue
            cleaned = clean_symptom_token(raw_val)
            if cleaned and cleaned not in seen:
                tokens.append(cleaned)
                seen.add(cleaned)

        sentence = tokens_to_clinical_sentence(tokens)
        records.append({
            "text": sentence,
            "label": label,
            "disease": canonical,
            "source": "dataset1",
        })

    result = pd.DataFrame(records)
    logger.info(
        "Dataset 1 preprocessing complete — %d rows produced from %d input rows.",
        len(result), len(df),
    )
    return result


def preprocess_dataset2(
    df: pd.DataFrame,
    disease_mapping: Dict[str, int],
) -> pd.DataFrame:
    """Normalises Symptom2Disease labels and prepares for merging.

    The original DataFrame is NOT modified — a new DataFrame is returned.
    Rows whose disease label cannot be matched to the canonical 41-label mapping
    are dropped with a warning log.

    Args:
        df (pd.DataFrame): Raw Symptom2Disease DataFrame.
        disease_mapping (Dict[str, int]): Canonical {disease_name: int} mapping.

    Returns:
        pd.DataFrame: Normalised DataFrame with columns [text, label, disease, source].
    """
    logger.info("Preprocessing Dataset 2: normalising disease labels...")
    normaliser = build_label_normaliser(disease_mapping)

    label_col = next(
        (c for c in df.columns if c.strip().lower() in ("label", "disease")), None
    )
    text_col = next(
        (c for c in df.columns if c.strip().lower() in ("text", "description", "symptom")), None
    )

    records: List[Dict[str, Any]] = []
    dropped = 0
    drop_names: Dict[str, int] = {}

    for _, row in df.iterrows():
        raw_disease = str(row[label_col]).strip() if label_col else ""
        text = str(row[text_col]).strip() if text_col else ""

        if not text:
            dropped += 1
            continue

        canonical = normalise_disease_label(raw_disease, normaliser)
        if canonical is None:
            dropped += 1
            drop_names[raw_disease] = drop_names.get(raw_disease, 0) + 1
            continue

        label = disease_mapping[canonical]
        records.append({
            "text": text,
            "label": label,
            "disease": canonical,
            "source": "symptom2disease",
        })

    if drop_names:
        logger.warning(
            "Dataset 2 preprocessing — dropped %d rows for unmatched labels: %s",
            dropped, drop_names,
        )

    result = pd.DataFrame(records)
    logger.info(
        "Dataset 2 preprocessing complete — %d rows retained, %d dropped.",
        len(result), dropped,
    )
    return result


def merge_datasets(
    df1_processed: pd.DataFrame,
    df2_processed: pd.DataFrame,
    disease_mapping: Dict[str, int],
    random_state: int = 42,
) -> Tuple[pd.DataFrame, MergedDatasetStats]:
    """Concatenates preprocessed Dataset 1 and Dataset 2 into the merged training set.

    Shuffles the result and computes per-class coverage statistics.

    Args:
        df1_processed (pd.DataFrame): Preprocessed Dataset 1 (text/label/disease/source).
        df2_processed (pd.DataFrame): Preprocessed Dataset 2 (text/label/disease/source).
        disease_mapping (Dict[str, int]): Canonical 41-label mapping.
        random_state (int): Shuffle seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, MergedDatasetStats]:
            - Merged, shuffled DataFrame.
            - Populated MergedDatasetStats.
    """
    logger.info("Merging Dataset 1 and Dataset 2...")

    merged = pd.concat([df1_processed, df2_processed], ignore_index=True)
    merged = merged.dropna(subset=["text", "label"])
    merged["label"] = merged["label"].astype(int)
    merged = merged.sample(frac=1, random_state=random_state).reset_index(drop=True)

    idx_to_disease = {v: k for k, v in disease_mapping.items()}
    stats = MergedDatasetStats()
    stats.total_rows = len(merged)
    stats.dataset1_rows = len(df1_processed)
    stats.dataset2_rows = len(df2_processed)

    # Per-class counts
    for label_idx, count in merged.groupby("label").size().items():
        disease_name = idx_to_disease.get(int(label_idx), f"Unknown({label_idx})")
        stats.per_class_counts[disease_name] = int(count)

    stats.num_diseases = len(stats.per_class_counts)

    # Coverage analysis
    ds1_diseases = set(df1_processed["disease"].unique()) if not df1_processed.empty else set()
    ds2_diseases = set(df2_processed["disease"].unique()) if not df2_processed.empty else set()

    stats.diseases_in_both = sorted(ds1_diseases & ds2_diseases)
    stats.diseases_only_in_ds1 = sorted(ds1_diseases - ds2_diseases)

    logger.info(
        "Merge complete — %d total rows | %d diseases | "
        "%d in both sources | %d only in Dataset 1",
        stats.total_rows, stats.num_diseases,
        len(stats.diseases_in_both), len(stats.diseases_only_in_ds1),
    )
    return merged, stats


# ─────────────────────────────────────────────────────────────────────────────
# Plotting Functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(
    merged_df: pd.DataFrame,
    plots_dir: Path,
) -> None:
    """Generates and saves a stacked bar chart of per-class sample counts by source.

    Args:
        merged_df (pd.DataFrame): Merged DataFrame with [disease, source] columns.
        plots_dir (Path): Directory to save the plot PNG.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    pivot = (
        merged_df.groupby(["disease", "source"])
        .size()
        .unstack(fill_value=0)
        .sort_values(by=merged_df["disease"].value_counts().index.tolist()[0]
                     if "dataset1" not in merged_df.get("source", pd.Series()).unique()
                     else "dataset1",
                     ascending=False)
    )
    # Sort by total count descending
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(16, 8))
    colors = {"dataset1": "#4e79a7", "symptom2disease": "#f28e2b"}
    bottom = pd.Series([0] * len(pivot), index=pivot.index)

    for source in ["dataset1", "symptom2disease"]:
        if source in pivot.columns:
            ax.bar(
                range(len(pivot)),
                pivot[source],
                bottom=bottom.values,
                label=source.replace("_", " ").title(),
                color=colors.get(source, "#76b7b2"),
                edgecolor="white",
                linewidth=0.5,
            )
            bottom = bottom + pivot[source]

    ax.set_xticks(range(len(pivot)))
    ax.set_xticklabels(pivot.index, rotation=55, ha="right", fontsize=8)
    ax.set_xlabel("Disease", fontsize=11)
    ax.set_ylabel("Sample Count", fontsize=11)
    ax.set_title("BioBERT Merged Dataset — Per-Class Sample Distribution", fontsize=13, fontweight="bold", pad=14)
    ax.legend(title="Source", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    save_path = plots_dir / "class_distribution.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Class distribution plot saved to %s", save_path)


def plot_text_length_distribution(
    merged_df: pd.DataFrame,
    plots_dir: Path,
) -> None:
    """Generates and saves a histogram of text (word count) lengths by source.

    Args:
        merged_df (pd.DataFrame): Merged DataFrame with [text, source] columns.
        plots_dir (Path): Directory to save the plot PNG.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    merged_df = merged_df.copy()
    merged_df["word_count"] = merged_df["text"].astype(str).apply(
        lambda t: len(t.split())
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"dataset1": "#4e79a7", "symptom2disease": "#f28e2b"}

    for source, grp in merged_df.groupby("source"):
        ax.hist(
            grp["word_count"],
            bins=30,
            alpha=0.65,
            label=source.replace("_", " ").title(),
            color=colors.get(source, "#76b7b2"),
            edgecolor="white",
            linewidth=0.4,
        )

    ax.set_xlabel("Word Count per Sample", fontsize=11)
    ax.set_ylabel("Number of Samples", fontsize=11)
    ax.set_title("Text Length Distribution by Source", fontsize=13, fontweight="bold", pad=14)
    ax.legend(title="Source", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    save_path = plots_dir / "text_length_distribution.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Text length distribution plot saved to %s", save_path)


def plot_source_split(merged_df: pd.DataFrame, plots_dir: Path) -> None:
    """Generates a pie chart showing the Dataset 1 vs Symptom2Disease sample split.

    Args:
        merged_df (pd.DataFrame): Merged DataFrame with a [source] column.
        plots_dir (Path): Directory to save the plot PNG.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    source_counts = merged_df["source"].value_counts()
    colors_list = ["#4e79a7", "#f28e2b"]
    labels = [s.replace("_", " ").title() for s in source_counts.index]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        source_counts.values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors_list[:len(source_counts)],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(12)
    ax.set_title("Sample Contribution by Source Dataset", fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    save_path = plots_dir / "source_split.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Source split pie chart saved to %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Report Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_statistics_report(
    stats_ds1: DatasetStats,
    stats_ds2: DatasetStats,
    merged_stats: MergedDatasetStats,
    disease_mapping: Dict[str, int],
    output_path: Path,
) -> None:
    """Writes a comprehensive Markdown dataset statistics report to disk.

    Args:
        stats_ds1 (DatasetStats): Validation statistics for Dataset 1.
        stats_ds2 (DatasetStats): Validation statistics for Dataset 2.
        merged_stats (MergedDatasetStats): Merge summary statistics.
        disease_mapping (Dict[str, int]): Canonical label mapping.
        output_path (Path): Target path for the Markdown report.

    Raises:
        AppStorageError: If the file cannot be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Generating dataset statistics report at %s ...", output_path)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    idx_to_disease = {v: k for k, v in disease_mapping.items()}

    # ── Build class distribution table rows ───────────────────────────────
    distribution_rows: List[str] = []
    for lbl in sorted(disease_mapping.values()):
        disease = idx_to_disease.get(lbl, f"Unknown({lbl})")
        total = merged_stats.per_class_counts.get(disease, 0)
        distribution_rows.append(f"| {lbl} | {disease} | {total} |")

    distribution_table = "\n".join(distribution_rows)

    # ── Build drop reason tables ───────────────────────────────────────────
    ds2_drop_table = "\n".join(
        f"| {name} | {count} |"
        for name, count in sorted(stats_ds2.drop_reasons.items(), key=lambda x: -x[1])
    ) or "| — | 0 |"

    # ── Diseases covered only by DS1 ──────────────────────────────────────
    ds1_only_list = "\n".join(
        f"- {d}" for d in merged_stats.diseases_only_in_ds1
    ) or "*(none — full overlap)*"

    ds_both_list = "\n".join(
        f"- {d}" for d in merged_stats.diseases_in_both
    ) or "*(none)*"

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"""# BioBERT Dataset Statistics Report

*Generated: {timestamp}*

---

## 1. Overview

This report documents the complete dataset preparation pipeline for fine-tuning
BioBERT on the AI Medical Diagnosis Assistant symptom classification task.
Two datasets are combined to create a merged training set that covers all **41
disease labels** in natural-language sentence format.

| Property | Value |
|---|---|
| **Total Merged Samples** | {merged_stats.total_rows} |
| **Number of Diseases (Labels)** | {merged_stats.num_diseases} |
| **Dataset 1 Contribution** | {merged_stats.dataset1_rows} rows |
| **Dataset 2 Contribution** | {merged_stats.dataset2_rows} rows |
| **Output CSV** | `data/merged/biobert_merged_dataset.csv` |

---

## 2. Dataset 1 — Project Dataset (41 Diseases)

**Source**: `{stats_ds1.source_path}`

| Metric | Value |
|---|---|
| **Total Rows** | {stats_ds1.total_rows} |
| **Unique Diseases** | {stats_ds1.num_diseases} |
| **Missing Cells (total)** | {stats_ds1.num_missing_cells} |
| **Duplicate Rows** | {stats_ds1.num_duplicate_rows} |
| **Text Min Words** | {stats_ds1.text_length_stats.get('min_words', 'N/A')} |
| **Text Max Words** | {stats_ds1.text_length_stats.get('max_words', 'N/A')} |
| **Text Mean Words** | {stats_ds1.text_length_stats.get('mean_words', 'N/A')} |

### Input Format → Output Format
```
Wide-format tabular (Disease + Symptom_1 … Symptom_17 columns):
  "itching, skin_rash, nodal_skin_eruptions"

↓  tokens_to_clinical_sentence()

Natural-language clinical sentence:
  "The patient presents with itching, skin rash, and nodal skin eruptions."
```

### Disease Class Distribution (Dataset 1)

| Disease | Sample Count |
|---|---|
{"".join(f"| {d} | {c} |{chr(10)}" for d, c in sorted(stats_ds1.disease_distribution.items(), key=lambda x: -x[1]))}

---

## 3. Dataset 2 — Symptom2Disease (24 Diseases)

**Source**: `{stats_ds2.source_path}`

| Metric | Value |
|---|---|
| **Total Rows** | {stats_ds2.total_rows} |
| **Unique Diseases** | {stats_ds2.num_diseases} |
| **Missing Cells** | {stats_ds2.num_missing_cells} |
| **Duplicate Rows** | {stats_ds2.num_duplicate_rows} |
| **Rows Dropped (label mismatch)** | {stats_ds2.dropped_rows} |
| **Text Min Words** | {stats_ds2.text_length_stats.get('min_words', 'N/A')} |
| **Text Max Words** | {stats_ds2.text_length_stats.get('max_words', 'N/A')} |
| **Text Mean Words** | {stats_ds2.text_length_stats.get('mean_words', 'N/A')} |

### Dropped Labels (not in canonical 41-label mapping)

| Disease Name | Rows Dropped |
|---|---|
{ds2_drop_table}

---

## 4. Merged Dataset — Class Distribution

**Total: {merged_stats.total_rows} samples across {merged_stats.num_diseases} diseases**

| Label | Disease | Total Samples |
|---|---|---|
{distribution_table}

### Coverage Analysis

**Diseases covered by BOTH sources ({len(merged_stats.diseases_in_both)}):**
{ds_both_list}

**Diseases covered by Dataset 1 ONLY ({len(merged_stats.diseases_only_in_ds1)}):**
{ds1_only_list}

---

## 5. Preprocessing Notes

- **Dataset 1** keyword columns were converted to clinical sentences using
  `tokens_to_clinical_sentence()`.  Original CSV was NOT modified.
- **Dataset 2** disease labels were normalised against `disease_mapping_41.json`
  using case-insensitive matching.  Unmatched rows were dropped.
- Both datasets were concatenated and shuffled with `random_state=42`.
- Intermediate outputs saved to `data/interim/` for auditability.

---

## 6. Output Artifacts

| Artifact | Path |
|---|---|
| Merged training CSV | `data/merged/biobert_merged_dataset.csv` |
| Interim Dataset 1 sentences | `data/interim/dataset1_sentences.csv` |
| Interim Dataset 2 normalised | `data/interim/dataset2_normalised.csv` |
| This report | `data/merged/dataset_statistics_report.md` |
| Class distribution plot | `data/merged/plots/class_distribution.png` |
| Text length plot | `data/merged/plots/text_length_distribution.png` |
| Source split pie chart | `data/merged/plots/source_split.png` |

---

*Next step: BioBERT fine-tuning using `configs/biobert_training_config.yaml`.*
""")
    except Exception as e:
        raise AppStorageError(
            message=f"Failed to write dataset statistics report: {e}",
            details={"output_path": str(output_path), "error": str(e)},
        )

    logger.info("Dataset statistics report written successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestrator Class
# ─────────────────────────────────────────────────────────────────────────────

class BioBERTDatasetBuilder:
    """Orchestrates the complete BioBERT dataset preparation pipeline.

    Responsibilities:
      1. Download Symptom2Disease from Kaggle (idempotent).
      2. Load and validate both datasets — read-only, no mutations.
      3. Convert Dataset 1 keywords to clinical sentences.
      4. Normalise Dataset 2 labels against the canonical mapping.
      5. Merge, shuffle, and save the final training CSV.
      6. Save intermediate outputs to data/interim/.
      7. Generate distribution plots and a statistics report.

    Attributes:
        download_dir (Path): Directory for Kaggle archive downloads.
        raw_s2d_dir (Path): Directory for extracted Symptom2Disease raw data.
        interim_dir (Path): Directory for intermediate processed CSVs.
        merged_dir (Path): Directory for the final merged output.
        plots_dir (Path): Directory for generated charts.
        cleaned_csv (Path): Path to the existing project symptom CSV.
        disease_mapping_path (Path): Path to the 41-label mapping JSON.
    """

    def __init__(
        self,
        cleaned_csv: Path = _CLEANED_CSV,
        disease_mapping_path: Path = _DISEASE_MAPPING_JSON,
        download_dir: Path = _DOWNLOAD_DIR,
        raw_s2d_dir: Path = _RAW_S2D_DIR,
        interim_dir: Path = _INTERIM_DIR,
        merged_dir: Path = _MERGED_DIR,
        plots_dir: Path = _PLOTS_DIR,
    ) -> None:
        """Initialises directory structures without downloading or loading data.

        Args:
            cleaned_csv (Path): Path to disease_symptom_cleaned.csv.
            disease_mapping_path (Path): Path to disease_mapping_41.json.
            download_dir (Path): Kaggle download target directory.
            raw_s2d_dir (Path): Symptom2Disease extraction directory.
            interim_dir (Path): Directory for interim CSV outputs.
            merged_dir (Path): Directory for the merged dataset output.
            plots_dir (Path): Directory for plots.
        """
        self.cleaned_csv = cleaned_csv
        self.disease_mapping_path = disease_mapping_path
        self.download_dir = download_dir
        self.raw_s2d_dir = raw_s2d_dir
        self.interim_dir = interim_dir
        self.merged_dir = merged_dir
        self.plots_dir = plots_dir

        # Create output directories
        for directory in [download_dir, raw_s2d_dir, interim_dir, merged_dir, plots_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        logger.info("BioBERTDatasetBuilder initialised.")
        logger.info("  Cleaned CSV      : %s", self.cleaned_csv)
        logger.info("  Disease mapping  : %s", self.disease_mapping_path)
        logger.info("  Download dir     : %s", self.download_dir)
        logger.info("  Interim dir      : %s", self.interim_dir)
        logger.info("  Merged dir       : %s", self.merged_dir)

    def step1_download_symptom2disease(self) -> Path:
        """Step 1: Downloads and extracts the Symptom2Disease dataset.

        Returns:
            Path: Path to the extracted Symptom2Disease CSV.
        """
        logger.info("=" * 60)
        logger.info("Step 1: Downloading Symptom2Disease from Kaggle")
        logger.info("=" * 60)
        s2d_csv = download_symptom2disease(
            download_dir=self.download_dir,
            raw_output_dir=self.raw_s2d_dir,
        )
        logger.info("Step 1 complete — Symptom2Disease CSV: %s", s2d_csv)
        return s2d_csv

    def step2_load_and_validate(
        self,
        s2d_csv_path: Path,
        disease_mapping: Dict[str, int],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, DatasetStats, DatasetStats]:
        """Step 2: Loads and validates both datasets without modifying them.

        Args:
            s2d_csv_path (Path): Path to Symptom2Disease CSV.
            disease_mapping (Dict[str, int]): Canonical label mapping.

        Returns:
            Tuple of (df1_raw, df2_raw, stats_ds1, stats_ds2).
        """
        logger.info("=" * 60)
        logger.info("Step 2: Loading and Validating Datasets")
        logger.info("=" * 60)
        df1_raw, stats_ds1 = load_and_validate_dataset1(self.cleaned_csv, disease_mapping)
        df2_raw, stats_ds2 = load_and_validate_dataset2(s2d_csv_path, disease_mapping)

        # ── Print console summary ─────────────────────────────────────────
        self._print_validation_summary(stats_ds1)
        self._print_validation_summary(stats_ds2)

        logger.info("Step 2 complete.")
        return df1_raw, df2_raw, stats_ds1, stats_ds2

    def step3_preprocess(
        self,
        df1_raw: pd.DataFrame,
        df2_raw: pd.DataFrame,
        disease_mapping: Dict[str, int],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Step 3: Applies preprocessing pipelines without mutating source DataFrames.

        Args:
            df1_raw (pd.DataFrame): Raw Dataset 1.
            df2_raw (pd.DataFrame): Raw Dataset 2.
            disease_mapping (Dict[str, int]): Canonical label mapping.

        Returns:
            Tuple of (df1_processed, df2_processed).
        """
        logger.info("=" * 60)
        logger.info("Step 3: Preprocessing Datasets")
        logger.info("=" * 60)
        df1_processed = preprocess_dataset1(df1_raw, disease_mapping)
        df2_processed = preprocess_dataset2(df2_raw, disease_mapping)

        # Save interim outputs
        df1_processed.to_csv(_INTERIM_DS1, index=False, encoding="utf-8")
        df2_processed.to_csv(_INTERIM_DS2, index=False, encoding="utf-8")
        logger.info("Interim Dataset 1 saved to: %s", _INTERIM_DS1)
        logger.info("Interim Dataset 2 saved to: %s", _INTERIM_DS2)

        logger.info("Step 3 complete.")
        return df1_processed, df2_processed

    def step4_merge_and_save(
        self,
        df1_processed: pd.DataFrame,
        df2_processed: pd.DataFrame,
        disease_mapping: Dict[str, int],
    ) -> Tuple[pd.DataFrame, MergedDatasetStats]:
        """Step 4: Merges both preprocessed datasets and writes the output CSV.

        Args:
            df1_processed (pd.DataFrame): Preprocessed Dataset 1.
            df2_processed (pd.DataFrame): Preprocessed Dataset 2.
            disease_mapping (Dict[str, int]): Canonical label mapping.

        Returns:
            Tuple of (merged_df, merged_stats).
        """
        logger.info("=" * 60)
        logger.info("Step 4: Merging Datasets and Saving Output")
        logger.info("=" * 60)
        merged_df, merged_stats = merge_datasets(df1_processed, df2_processed, disease_mapping)

        merged_df.to_csv(_MERGED_CSV, index=False, encoding="utf-8")
        logger.info("Merged dataset saved to: %s", _MERGED_CSV)
        logger.info("  Shape: %s", merged_df.shape)
        logger.info("  Columns: %s", list(merged_df.columns))

        logger.info("Step 4 complete.")
        return merged_df, merged_stats

    def step5_generate_plots_and_report(
        self,
        merged_df: pd.DataFrame,
        stats_ds1: DatasetStats,
        stats_ds2: DatasetStats,
        merged_stats: MergedDatasetStats,
        disease_mapping: Dict[str, int],
    ) -> None:
        """Step 5: Generates all distribution plots and the statistics report.

        Args:
            merged_df (pd.DataFrame): Final merged DataFrame.
            stats_ds1 (DatasetStats): Dataset 1 validation statistics.
            stats_ds2 (DatasetStats): Dataset 2 validation statistics.
            merged_stats (MergedDatasetStats): Merge summary statistics.
            disease_mapping (Dict[str, int]): Canonical label mapping.
        """
        logger.info("=" * 60)
        logger.info("Step 5: Generating Plots and Statistics Report")
        logger.info("=" * 60)
        plot_class_distribution(merged_df, self.plots_dir)
        plot_text_length_distribution(merged_df, self.plots_dir)
        plot_source_split(merged_df, self.plots_dir)
        generate_statistics_report(
            stats_ds1=stats_ds1,
            stats_ds2=stats_ds2,
            merged_stats=merged_stats,
            disease_mapping=disease_mapping,
            output_path=_REPORT_PATH,
        )
        logger.info("Step 5 complete.")

    def _print_validation_summary(self, stats: DatasetStats) -> None:
        """Prints a formatted console validation summary for a dataset.

        Uses ASCII-only characters to ensure compatibility with Windows
        cp1252 console encoding.

        Args:
            stats (DatasetStats): Validation statistics to display.
        """
        separator = "-" * 58  # ASCII only — avoids cp1252 UnicodeEncodeError on Windows
        logger.info(separator)
        logger.info("  Dataset   : %s", stats.name)
        logger.info("  Source    : %s", stats.source_path)
        logger.info("  Total rows: %d", stats.total_rows)
        logger.info("  Diseases  : %d", stats.num_diseases)
        logger.info("  Missing   : %d cells", stats.num_missing_cells)
        logger.info("  Duplicates: %d rows", stats.num_duplicate_rows)
        if stats.text_length_stats:
            logger.info(
                "  Text len  : min=%s | max=%s | mean=%s words",
                stats.text_length_stats.get("min_words"),
                stats.text_length_stats.get("max_words"),
                stats.text_length_stats.get("mean_words"),
            )
        logger.info(separator)

    def run(self) -> Dict[str, Any]:
        """Executes the complete dataset preparation pipeline sequentially.

        Steps:
            1. Download Symptom2Disease from Kaggle (idempotent).
            2. Load and validate both datasets (read-only).
            3. Preprocess: keyword→sentence (DS1), label normalisation (DS2).
            4. Merge and save to data/merged/.
            5. Generate plots and statistics report.

        Returns:
            Dict[str, Any]: Summary with output paths and statistics.

        Raises:
            AppStorageError: If any file I/O operation fails.
            AppValidationError: If dataset structure validation fails.
        """
        logger.info("=" * 60)
        logger.info("BioBERT Dataset Builder — Phase 2.1")
        logger.info("=" * 60)

        # Load disease mapping (required for all subsequent steps)
        disease_mapping = load_disease_mapping(self.disease_mapping_path)

        # Step 1: Download
        s2d_csv = self.step1_download_symptom2disease()

        # Step 2: Load and validate
        df1_raw, df2_raw, stats_ds1, stats_ds2 = self.step2_load_and_validate(
            s2d_csv_path=s2d_csv,
            disease_mapping=disease_mapping,
        )

        # Step 3: Preprocess (non-destructive)
        df1_processed, df2_processed = self.step3_preprocess(
            df1_raw, df2_raw, disease_mapping
        )

        # Step 4: Merge and save
        merged_df, merged_stats = self.step4_merge_and_save(
            df1_processed, df2_processed, disease_mapping
        )

        # Step 5: Plots and report
        self.step5_generate_plots_and_report(
            merged_df, stats_ds1, stats_ds2, merged_stats, disease_mapping
        )

        # ── Final summary ─────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("BioBERT Dataset Builder — COMPLETED SUCCESSFULLY")
        logger.info("  Total merged rows : %d", merged_stats.total_rows)
        logger.info("  Diseases covered  : %d / 41", merged_stats.num_diseases)
        logger.info("  Output CSV        : %s", _MERGED_CSV)
        logger.info("  Report            : %s", _REPORT_PATH)
        logger.info("=" * 60)

        return {
            "merged_csv": str(_MERGED_CSV),
            "interim_ds1_csv": str(_INTERIM_DS1),
            "interim_ds2_csv": str(_INTERIM_DS2),
            "report_path": str(_REPORT_PATH),
            "plots_dir": str(self.plots_dir),
            "total_rows": merged_stats.total_rows,
            "num_diseases": merged_stats.num_diseases,
            "dataset1_rows": merged_stats.dataset1_rows,
            "dataset2_rows": merged_stats.dataset2_rows,
            "diseases_only_in_ds1": merged_stats.diseases_only_in_ds1,
            "diseases_in_both": merged_stats.diseases_in_both,
            "stats_ds1": stats_ds1,
            "stats_ds2": stats_ds2,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entrypoint for the BioBERT dataset builder."""
    import sys

    logger.info("Launching BioBERT Dataset Builder via CLI...")
    try:
        builder = BioBERTDatasetBuilder()
        result = builder.run()
        print("\n" + "=" * 60)
        print("Dataset preparation complete.")
        print(f"  Merged CSV   : {result['merged_csv']}")
        print(f"  Total rows   : {result['total_rows']}")
        print(f"  Diseases     : {result['num_diseases']} / 41")
        print(f"  Report       : {result['report_path']}")
        print("=" * 60)
        sys.exit(0)
    except (AppStorageError, AppValidationError) as e:
        logger.error("Dataset builder failed: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error during dataset preparation: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
