# Data Validation Report

Provides detailed validation status and cleaning audit logs for the clinical datasets.

---

## 1. Image Dataset Validation Summary

*   **Initial Images Found:** `21165`
*   **Final Verified Images kept:** `21111`
*   **Sanitization Actions (Deletions):**
    *   **Unsupported Format Deletions:** `0`
    *   **Corrupted Scan Deletions:** `0`
    *   **Duplicate Scan Deletions:** `54`
*   **Checks and Alignments:**
    *   **Incorrect Dimensions Count (Target (299, 299)):** `0`
    *   **Scans with Missing Masks:** `0`

### Image Class Counts
*   **COVID:** `3570 scans`
*   **Lung_Opacity:** `6012 scans`
*   **Normal:** `10191 scans`
*   **Viral Pneumonia:** `1338 scans`

---

## 2. Tabular Symptoms Dataset Validation Summary

*   **Initial Records Found:** `4920`
*   **Final Verified Records kept:** `286`
*   **Sanitization Actions (Drops):**
    *   **Duplicate Records Dropped:** `4616`
    *   **Invalid Disease Labels Dropped:** `18`
*   **Tabular Data Completeness:**
    *   **Total Empty Cells:** `2614`

---

## 3. Executive Ingestion Validation Verdict

*   **Status:** **PASSED**
*   *Note:* The sanitized image assets are checked for duplicates and corruptions. Tabular symptom variables are cleaned of duplicates. Ready for Phase 12 (Data Transformation).
