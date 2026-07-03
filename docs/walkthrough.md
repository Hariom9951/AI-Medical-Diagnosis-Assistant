# Ingestion & Verification Walkthrough - Kaggle Datasets

This walkthrough presents the results of integrating and verifying real Kaggle datasets (images and symptoms) for **Phase 10: Data Ingestion** in the AI Medical Diagnosis Assistant.

---

## 1. Verified Ingestion Pipeline Actions

1.  **Code Decontamination (Dummy logic removed):** All previous dummy and synthetic dataset generation logic (using PIL image creation or fake metadata generation) has been permanently removed from the codebase.
2.  **Modular Kaggle Downloader:** Re-implemented `DataIngestion` using the Kaggle Python SDK. It handles API authentication, downloads compressed `.zip` folders with backoff retries, performs Zip Slip security checks, and extracts them into raw targets.
3.  **Path Configurations:** decopled directories and dataset slugs in [configs/ingestion_config.yaml](file:///c:/Users/HP/OneDrive/Desktop/AI-Medical-Diagnosis-Assistant/configs/ingestion_config.yaml):
    - `image_dataset_slug`: `"tawsifurrahman/covid19-radiography-database"`
    - `symptom_dataset_slug`: `"itachi9604/disease-symptom-description-dataset"`
    - `download_dir`: `"data/downloads"`
    - `extract_dir`: `"data/raw"`

---

## 2. Ingested Dataset Folder Structure

The pipeline successfully downloaded and extracted the raw files.
*   **Downloads Location:** `C:\Users\HP\OneDrive\Desktop\AI-Medical-Diagnosis-Assistant\data\downloads`
*   **Extracted Location:** `C:\Users\HP\OneDrive\Desktop\AI-Medical-Diagnosis-Assistant\data\raw`

```
data/raw/
├── covid19-radiography-database/
│   └── COVID-19_Radiography_Dataset/
│       ├── COVID/
│       │   ├── images/
│       │   └── masks/
│       ├── Lung_Opacity/
│       │   ├── images/
│       │   └── masks/
│       ├── Normal/
│       │   ├── images/
│       │   └── masks/
│       ├── Viral Pneumonia/
│       │   ├── images/
│       │   └── masks/
│       ├── COVID.metadata.xlsx
│       ├── Lung_Opacity.metadata.xlsx
│       ├── Normal.metadata.xlsx
│       └── README.md.txt
└── disease-symptom-description-dataset/
    ├── dataset.csv
    ├── Symptom-severity.csv
    ├── symptom_Description.csv
    └── symptom_precaution.csv
```

---

## 3. Dataset Sizes & General Metrics

*   **Compressed Download Volume:** 778.26 MB
*   **Extracted Raw File Volume:** 770.08 MB
*   **Disease Classes (4):** `COVID`, `Normal`, `Lung_Opacity`, `Viral Pneumonia`
*   **Total Medical Scans:** 42,330 images
*   **Class Distribution:**
    - `Normal`: 20,384 images
    - `Lung_Opacity`: 12,024 images
    - `COVID`: 7,232 images
    - `Viral Pneumonia`: 2,690 images
*   **Total Symptom CSV Records:** 4,920 records

---

## 4. Visual Scan Verification (5 Random Chest X-rays)

Below is an interactive carousel displaying 5 random scans from the extracted dataset:

````carousel
![Sample 1: Lung_Opacity](file:///C:/Users/HP/.gemini/antigravity-ide/brain/0c3677b0-f75a-4c38-b47a-ca7718886bec/scratch/kaggle_sample_0_Lung_Opacity-4390.png)
<!-- slide -->
![Sample 2: Lung_Opacity](file:///C:/Users/HP/.gemini/antigravity-ide/brain/0c3677b0-f75a-4c38-b47a-ca7718886bec/scratch/sample_1_PATIENT_010.png)
<!-- slide -->
![Sample 3: Lung_Opacity](file:///C:/Users/HP/.gemini/antigravity-ide/brain/0c3677b0-f75a-4c38-b47a-ca7718886bec/scratch/kaggle_sample_2_Lung_Opacity-3114.png)
<!-- slide -->
![Sample 4: Lung_Opacity Mask](file:///C:/Users/HP/.gemini/antigravity-ide/brain/0c3677b0-f75a-4c38-b47a-ca7718886bec/scratch/kaggle_sample_3_Lung_Opacity-2994.png)
<!-- slide -->
![Sample 5: Normal Mask](file:///C:/Users/HP/.gemini/antigravity-ide/brain/0c3677b0-f75a-4c38-b47a-ca7718886bec/scratch/kaggle_sample_4_Normal-9149.png)
````

---

## 5. CSV Metadata Profiling (`dataset.csv`)

### A. Columns, Types & Missing Values

*   **Total Records:** 4,920 rows
*   **Column Schema (18):** `Disease` (str), `Symptom_1` to `Symptom_17` (str)
*   **Missing Value Counts:** 
    - `Disease` to `Symptom_3`: 0 missing values
    - `Symptom_4`: 348 missing values
    - `Symptom_5`: 1,206 missing values
    - `Symptom_6`: 1,986 missing values
    - `Symptom_7` to `Symptom_17`: Gradual missing values (up to 4,848 missing values for `Symptom_17`). This reflects real-world clinical entries where patients have varying numbers of symptoms.

### B. First 10 Rows Preview

| Disease | Symptom_1 | Symptom_2 | Symptom_3 | Symptom_4 | Symptom_5 | Symptom_6 | Symptom_7 | ... |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Fungal infection | itching | skin_rash | nodal_skin_eruptions | dischromic_patches | *Null* | *Null* | *Null* | ... |
| Fungal infection | skin_rash | nodal_skin_eruptions | dischromic_patches | *Null* | *Null* | *Null* | *Null* | ... |
| Fungal infection | itching | nodal_skin_eruptions | dischromic_patches | *Null* | *Null* | *Null* | *Null* | ... |
| Fungal infection | itching | skin_rash | dischromic_patches | *Null* | *Null* | *Null* | *Null* | ... |
| Fungal infection | itching | skin_rash | nodal_skin_eruptions | *Null* | *Null* | *Null* | *Null* | ... |
| Fungal infection | skin_rash | nodal_skin_eruptions | dischromic_patches | *Null* | *Null* | *Null* | *Null* | ... |
| Fungal infection | itching | nodal_skin_eruptions | dischromic_patches | *Null* | *Null* | *Null* | *Null* | ... |
| Fungal infection | itching | skin_rash | dischromic_patches | *Null* | *Null* | *Null* | *Null* | ... |
| Fungal infection | itching | skin_rash | nodal_skin_eruptions | *Null* | *Null* | *Null* | *Null* | ... |
| Fungal infection | itching | skin_rash | nodal_skin_eruptions | dischromic_patches | *Null* | *Null* | *Null* | ... |

---

## 6. Testing & Static Verification Results

### A. pytest Execution
Run the pytest suite to verify Kaggle API mocks and extract routines:
```bash
venv\Scripts\pytest.exe tests/unit/test_data_ingestion.py -v
```
```
collected 14 items
tests\unit\test_data_ingestion.py ..............                         [100%]
============================= 14 passed in 1.43s = [100%]
```

### B. mypy Static Type Validation
Confirm typing compliance:
```bash
mypy --python-executable venv\Scripts\python.exe src/components/data_ingestion.py
```
```
Success: no issues found in 1 source file
```

---

## 7. Git Commit Message

```
feat: integrate Kaggle API for real dataset ingestion

- Replace synthetic dummy dataset logic with official Kaggle API downloading
- Set up target datasets: tawsifurrahman/covid19-radiography-database and itachi9604/disease-symptom-description-dataset
- Configure storage locations to downloads/ and raw/ folders in configs/ingestion_config.yaml
- Update unit tests with Kaggle API class patches to check retries, auth errors, and Zip Slip preventions
- Run and record dataset size, class distributions, and CSV profiling validations
```
