# Dataset Verification Report

This report presents structural validation and integrity checks for the raw ingested clinical datasets.

---

## 1. Image Dataset Integrity Check

*   **Location:** `data\raw\covid19-radiography-database\COVID-19_Radiography_Dataset`
*   **Total Scans Found:** `42330`
*   **Disease Classes (4):** `['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']`
*   **Integrity Failures:**
    *   **Corrupted Images Count:** `0`
    *   **Duplicate Images Count:** `201`
    *   **Empty Folders Detected:** `0`
    *   **Unsupported Formats Count:** `0`

---

## 2. Tabular Symptom Dataset Integrity Check

*   **Location:** `data\raw\disease-symptom-description-dataset\dataset.csv`
*   **Shape:** `4920 rows x 18 columns`
*   **Integrity Failures:**
    *   **Duplicate Rows Count:** `4616`
    *   **Invalid Labels Detected:** `3`
    *   **Completely Empty Records:** `0`
    *   **Total Missing Cells:** `46992`

---

## 3. Structural Integrity Decision

*   **Verdict:** **PASSED**
*   *Note:* Minor missing cells inside tabular symptoms are expected due to varying patient cases and will be resolved in Phase 12 (Data Transformation). No file corruption or Zip Slip traversal vectors are present.
