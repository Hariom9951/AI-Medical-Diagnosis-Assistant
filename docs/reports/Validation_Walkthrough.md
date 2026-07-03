# Data Validation Ingest Walkthrough

This walkthrough presents the implementation and execution metrics of the **Data Validation** module.

---

## 1. File Changes and Architectural Structure

The module has been implemented with clean architecture patterns:

*   **Config Decoupling:** [configs/validation_config.yaml](file:///c:/Users/HP/OneDrive/Desktop/AI-Medical-Diagnosis-Assistant/configs/validation_config.yaml) stores folders, expected dims (299x299), and formats.
*   **Validation Component:** [src/components/data_validation.py](file:///c:/Users/HP/OneDrive/Desktop/AI-Medical-Diagnosis-Assistant/src/components/data_validation.py) handles in-place scan deletes, mask verification, column checks, and drops duplicate/invalid disease CSV rows.
*   **Unit Test Suite:** [tests/unit/test_data_validation.py](file:///c:/Users/HP/OneDrive/Desktop/AI-Medical-Diagnosis-Assistant/tests/unit/test_data_validation.py) checks config validation, file mock cleans, mask checks, and dataframe operations.
*   **Execution Script:** `scratch/run_validation.py` executes the pipeline.

---

## 2. Validation Execution Diagnostics

Running the pipeline on the ingested Kaggle datasets yields the following verification blocks:

*   **✔ Folder structure:** Validated expected class folders.
*   **✔ Total verified healthy scans remaining:** 21,111 scans (54 duplicates deleted in-place to save disk footprint).
    *   *COVID:* 3,570 scans
    *   *Lung_Opacity:* 6,012 scans
    *   *Normal:* 10,191 scans
    *   *Viral Pneumonia:* 1,338 scans
*   **✔ Deletion Metrics:**
    *   *Corrupted scans deleted:* 0
    *   *Duplicate scans deleted:* 54
    *   *Unsupported formats deleted:* 0
*   **✔ Dimension & Mask Alignment:**
    *   *Incorrect dimensions count (Target 299x299):* 0 (all raw scans conform)
    *   *Missing corresponding masks:* 0 (each image matches a mask target)
*   **✔ Tabular CSV validation summary:**
    *   *Initial rows:* 4,920 records
    *   *Final records kept:* 286 records
    *   *Duplicate records dropped:* 4,616
    *   *Invalid disease labels dropped:* 18 (dropped empty/numerical values)
    *   *Empty cell counts:* 2,614 nulls (expected and ready for Phase 12 imputation)

---

## 3. Validation Report Outputs

All validation reports are saved inside `docs/reports/`:
*   **Verification Report:** [Validation_Report.md](file:///c:/Users/HP/OneDrive/Desktop/AI-Medical-Diagnosis-Assistant/docs/reports/Validation_Report.md)
*   **Summary JSON:** [validation_summary.json](file:///c:/Users/HP/OneDrive/Desktop/AI-Medical-Diagnosis-Assistant/docs/reports/validation_summary.json)
*   **Statistics CSV:** [validation_statistics.csv](file:///c:/Users/HP/OneDrive/Desktop/AI-Medical-Diagnosis-Assistant/docs/reports/validation_statistics.csv)
*   **Clean Symptoms CSV:** [disease_symptom_cleaned.csv](file:///c:/Users/HP/OneDrive/Desktop/AI-Medical-Diagnosis-Assistant/data/processed/disease_symptom_cleaned.csv)

---

## 4. Testing and Static Verification Results

### A. pytest Execution
Run the pytest suite to verify validation mocks and deletions:
```bash
venv\Scripts\pytest.exe -v
```
```
collected 39 items

tests\unit\test_config.py .....                                          [ 12%]
tests\unit\test_data_ingestion.py ..............                         [ 48%]
tests\unit\test_data_validation.py ......                                [ 64%]
tests\unit\test_dataset_verification_eda.py .......                      [ 82%]
tests\unit\test_exceptions.py ....                                       [ 92%]
tests\unit\test_logger.py ...                                            [100%]

============================= 39 passed in 7.16s ==============================
```

### B. mypy Static Type Checking
Confirm typing compliance:
```bash
mypy --python-executable venv\Scripts\python.exe src/components/data_validation.py tests/unit/test_data_validation.py
```
```
Success: no issues found in 2 source files
```

---

## 5. Proposed Git Commit Message

```
feat: implement Data Validation module

- Create src/components/data_validation.py for image scans and tabular CSV cleaning
- Define targets inside configs/validation_config.yaml
- Remove 54 duplicate scans in-place from data/raw/
- Verify dimension profiles and mask allocations
- Drop duplicate entries and invalid diagnostic labels from CSV
- Save cleaned tabular dataset to data/processed/disease_symptom_cleaned.csv
- Save Validation_Report.md, validation_summary.json, and validation_statistics.csv to docs/reports/
- Write test suite under tests/unit/test_data_validation.py
```
