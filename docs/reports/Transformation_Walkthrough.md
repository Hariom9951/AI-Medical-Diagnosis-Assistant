# Data Transformation Walkthrough (Phase 12)

This walkthrough documents the implementation, testing, and output profile of the **Data Transformation** module.

---

## 1. Preprocessing and Augmentations Overview

The pipeline implements the following operations:

*   **Scans Transformation:** validated images are resized to `224x224`, converted to RGB, normalized via ImageNet mean/std, and converted to PyTorch tensors. Training data uses **Albumentations** augmentations (Horizontal/Vertical Flips, Rotate, Affine transforms, Brightness/Contrast, CLAHE, Gaussian Blur, and Coarse Dropout). Validation and test splits are resized and normalized without augmentations.
*   **Symptom Text Preprocessing:** Tabular clinical records are converted to lowercase, have punctuation stripped, and have duplicate symptoms removed. The clean lists are joined with comma separation to yield string descriptions (e.g. `"sneezing, itching"`). These strings are prepared for tokenization using the HuggingFace `DistilBertTokenizer`.
*   **Stratified Splits:** Maintains class balance ratios:
    *   **Images dataset:** splits into Train (70%), Val (15%), and Test (15%) stratified by class.
    *   **Symptoms dataset:** splits into Train (70%), Val (15%), and Test (15%). Due to extremely low sample counts for some disease labels (some classes having only 1 member), the code implements a try-except fallback block to gracefully use standard non-stratified splitting if stratification raises a `ValueError`, preventing runtime crashes.
*   **PyTorch Dataloaders:** Instantiates custom PyTorch datasets and wraps them in dataloaders configured from YAML.

---

## 2. Transformation Pipeline Diagnostics

Running `scratch/run_transformation.py` against the validated datasets returns the following metrics:

```
==================================================
DATA TRANSFORMATION PIPELINE RUN COMPLETED
==================================================
✔ Number of train images: 14777
✔ Number of validation images: 3167
✔ Number of test images: 3167
✔ Number of train symptom records: 200
✔ Number of validation symptom records: 43
✔ Number of test symptom records: 43

✔ Batch shape (Images): [32, 3, 224, 224]
✔ Tensor shape (Single Image): [3, 224, 224]
✔ Batch shape (input_ids): [32, 64]
✔ Batch shape (attention_mask): [32, 64]
✔ Batch shape (labels): [32]

✔ Report locations:
    - Transformation Report: docs/reports/Data_Transformation_Report.md
    - Preprocessing mappings: data/processed/disease_mapping.json
```

---

## 3. Visual Example of Transformed Scan

Here is a visual inspection of a sample chest scan image after undergoing the transformation pipeline (Resize to `224x224`, ImageNet de-normalization, and channel conversion):

![Example Transformed Scan](file:///C:/Users/HP/.gemini/antigravity-ide/brain/0c3677b0-f75a-4c38-b47a-ca7718886bec/scratch/sample_transformed_image.png)

---

## 4. Test Suite Execution Logs

All automated unit tests run successfully. The test suite verifies dataset resizing, symptom lowercasing/cleaning, label index mappings, tokenizer returns, and dataloader batch dimensions.

```bash
venv\Scripts\pytest.exe -v
```
```
collected 45 items

tests\unit\test_config.py .....                                          [ 11%]
tests\unit\test_data_ingestion.py ..............                         [ 42%]
tests\unit\test_data_transformation.py ./././././.                       [ 55%]
tests\unit\test_data_validation.py ......                                [ 68%]
tests\unit\test_dataset_verification_eda.py .......                      [ 84%]
tests\unit\test_exceptions.py ....                                       [ 93%]
tests\unit\test_logger.py ...                                            [100%]

============================= 45 passed in 16.35s =============================
```

---

## 5. File Explanation and Structure

The data transformation module contains these files:

1.  **[configs/transformation_config.yaml](file:///c:/Users/HP/OneDrive/Desktop/AI-Medical-Diagnosis-Assistant/configs/transformation_config.yaml):** Stores directories, split ratios, image sizing, and PyTorch dataloader attributes.
2.  **[src/components/data_transformation.py](file:///c:/Users/HP/OneDrive/Desktop/AI-Medical-Diagnosis-Assistant/src/components/data_transformation.py):** Implements preprocessing rules, `MedicalImageDataset` and `SymptomTextDataset` custom classes, stratified split algorithms, and report generators.
3.  **[tests/unit/test_data_transformation.py](file:///c:/Users/HP/OneDrive/Desktop/AI-Medical-Diagnosis-Assistant/tests/unit/test_data_transformation.py):** Contains unit tests asserting dataset sizes, preprocessing checks, mock tokenization outputs, and loader shapes.
4.  **`docs/reports/Data_Transformation_Report.md`:** Standardized project document summarizing dataset statistics, batch shapes, and parameters.
