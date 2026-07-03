# Verification & EDA Walkthrough

This walkthrough presents the implementation and execution metrics of the **Dataset Verification & Exploratory Data Analysis (EDA)** module.

---

## 1. File Changes and Architectural Structure

The module has been implemented with clean architecture interfaces and is strictly decoupled from other pipeline segments:

*   **Config Decoupling:** [configs/eda_config.yaml](file:///c:/Users/HP/OneDrive/Desktop/AI-Medical-Diagnosis-Assistant/configs/eda_config.yaml) stores inputs and report targets.
*   **Verification Component:** [src/components/dataset_verification_eda.py](file:///c:/Users/HP/OneDrive/Desktop/AI-Medical-Diagnosis-Assistant/src/components/dataset_verification_eda.py) contains image audits, file hashing, tabular profiling, and report compiling logic.
*   **Unit Test Suite:** [tests/unit/test_dataset_verification_eda.py](file:///c:/Users/HP/OneDrive/Desktop/AI-Medical-Diagnosis-Assistant/tests/unit/test_dataset_verification_eda.py) checks validations, directories setup, and file mocks.
*   **Execution Script:** `scratch/run_eda.py` wraps the pipeline execution.

---

## 2. Ingested Datasets Diagnostics Metrics

Running the pipeline on the raw Kaggle datasets yields the following diagnostics block:

*   **✔ Dataset Structure:** Verified raw directories exist and match layouts.
*   **✔ Disease Classes (4):** `COVID`, `Normal`, `Lung_Opacity`, `Viral Pneumonia`
*   **✔ Total Images:** 42,330 scans
    *   *COVID:* 7,232 images
    *   *Normal:* 20,384 images
    *   *Lung_Opacity:* 12,024 images
    *   *Viral Pneumonia:* 2,690 images
*   **✔ Corrupted Images Count:** 0
*   **✔ Duplicate Images Count:** 201 duplicate scans found.
*   **✔ Tabular CSV shape:** 4,920 rows x 18 columns
    *   *Duplicate records:* 4,616 (mostly repeating symptom combinations)
    *   *Completely empty records:* 0
    *   *Invalid labels:* 3
*   **✔ Missing Values per column:**
    *   *Symptom_1 to Symptom_3:* 0 nulls
    *   *Symptom_4:* 348 nulls
    *   *Symptom_5:* 1,206 nulls
    *   *Symptom_6:* 1,986 nulls
    *   *Symptom_7 to Symptom_17:* gradual increase in null cells (max 4,848 nulls on Symptom_17)

---

## 3. Generated Visualizations Map

Matplotlib plots have been exported to the directory: `artifacts/eda/`
*   [image_class_distribution.png](file:///c:/Users/HP/OneDrive/Desktop/AI-Medical-Diagnosis-Assistant/artifacts/eda/image_class_distribution.png): Visualizes image scan balance.
*   [image_resolutions_histogram.png](file:///c:/Users/HP/OneDrive/Desktop/AI-Medical-Diagnosis-Assistant/artifacts/eda/image_resolutions_histogram.png): Displays image width and height distributions.
*   [symptom_missingness_heatmap.png](file:///c:/Users/HP/OneDrive/Desktop/AI-Medical-Diagnosis-Assistant/artifacts/eda/symptom_missingness_heatmap.png): Shows the missing value matrix.
*   [disease_frequency.png](file:///c:/Users/HP/OneDrive/Desktop/AI-Medical-Diagnosis-Assistant/artifacts/eda/disease_frequency.png): Top  diagnostic frequency.
*   [symptom_frequency.png](file:///c:/Users/HP/OneDrive/Desktop/AI-Medical-Diagnosis-Assistant/artifacts/eda/symptom_frequency.png): Most common patient-reported symptoms.
*   [samples/](file:///c:/Users/HP/OneDrive/Desktop/AI-Medical-Diagnosis-Assistant/artifacts/eda/samples): Class folders containing 5 random scans copied as verification candidates.

---

## 4. Report Output Locations

The pipeline compiled and saved three detailed reports in: `docs/reports/`
*   **Integrity Report:** [Dataset_Verification_Report.md](file:///c:/Users/HP/OneDrive/Desktop/AI-Medical-Diagnosis-Assistant/docs/reports/Dataset_Verification_Report.md)
*   **EDA Statistics Report:** [EDA_Report.md](file:///c:/Users/HP/OneDrive/Desktop/AI-Medical-Diagnosis-Assistant/docs/reports/EDA_Report.md)
*   **Executive Summary Report:** [Summary_Report.md](file:///c:/Users/HP/OneDrive/Desktop/AI-Medical-Diagnosis-Assistant/docs/reports/Summary_Report.md)

---

## 5. Testing and Validation Runs

### A. pytest Output
Run pytest to verify the full project test suite:
```bash
venv\Scripts\pytest.exe -v
```
```
collected 33 items

tests\unit\test_config.py .....                                          [ 15%]
tests\unit\test_data_ingestion.py ..............                         [ 57%]
tests\unit\test_dataset_verification_eda.py .......                      [ 78%]
tests\unit\test_exceptions.py ....                                       [ 90%]
tests\unit\test_logger.py ...                                            [100%]

============================= 33 passed in 5.82s ==============================
```

### B. mypy Static Checking
Run mypy to check verification component types:
```bash
mypy --python-executable venv\Scripts\python.exe src/components/dataset_verification_eda.py tests/unit/test_dataset_verification_eda.py
```
```
Success: no issues found in 2 source files
```

---

## 6. Proposed Git Commit Message

```
feat: implement Dataset Verification & Exploratory Data Analysis (EDA) module

- Create src/components/dataset_verification_eda.py containing image integrity audits and CSV profiles
- Configure target paths in configs/eda_config.yaml
- Generate missing value heatmaps, class distributions, and frequency bar charts in artifacts/eda/
- Save Markdown verification, EDA, and summary reports under docs/reports/
- Write unit tests under tests/unit/test_dataset_verification_eda.py with mock dataset fixtures
```
