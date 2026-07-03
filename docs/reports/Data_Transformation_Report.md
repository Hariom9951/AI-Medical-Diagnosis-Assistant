# Data Transformation Report

Presents split allocations, augmentations, and tensor profiles.

---

## 1. Dataset Split Allocations

The ingested verified datasets have been split into Train (70%), Val (15%), and Test (15%) allocations using stratified split properties:

### Image Dataset Splits
*   **Total Scans:** `21111`
*   **Training Images Count:** `14777`
*   **Validation Images Count:** `3167`
*   **Testing Images Count:** `3167`

### Tabular Symptoms Dataset Splits
*   **Total Records:** `286`
*   **Training Records Count:** `200`
*   **Validation Records Count:** `43`
*   **Testing Records Count:** `43`

---

## 2. PyTorch DataLoader Batch Profiles

Loaded batch shapes with batch size `32`:

*   **Image Batch Tensor Dims:** `[32, 3, 224, 224]` (expected: `[batch_size, 3, 224, 224]`)
*   **Image Labels Tensor Dims:** `[32]` (expected: `[batch_size]`)
*   **Symptom Text input_ids Dims:** `[32, 64]` (expected: `[batch_size, max_length]`)
*   **Symptom Text attention_mask Dims:** `[32, 64]` (expected: `[batch_size, max_length]`)
*   **Symptom Labels Dims:** `[32]` (expected: `[batch_size]`)

---

## 3. Preprocessing Properties

*   **Image Dimension Targets:** `224 x 224 (RGB)`
*   **Normalizations Applied:** ImageNet channel means `[0.485, 0.456, 0.406]` and std deviations `[0.229, 0.224, 0.225]`.
*   **Disease Category Index Mappings:** Saved at `data\processed\disease_mapping.json` mapping `38` diagnostic classes.
*   **Augmentations Active (Train-Only):** HorizontalFlip, VerticalFlip, Rotate, ShiftScaleRotate, RandomBrightnessContrast, CLAHE, GaussianBlur, CoarseDropout.
