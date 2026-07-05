# Model Evaluation Report — Phase 15

Presents evaluation metrics, per-class performance matrix tables, and support distributions.

---

## 1. Overall Performance Summary
*   **Total Test Accuracy:** `0.700000`
*   **Top-1 Accuracy:** `0.700000`
*   **Macro ROC-AUC:** `0.819094`
*   **Macro F1-Score:** `0.574095`
*   **Weighted F1-Score:** `0.680528`

---

## 2. Per-class Support Table

| Diagnostic Disease Class | Precision | Recall | F1-Score | Support Count |
| :--- | :---: | :---: | :---: | :---: |
| **COVID** | `1.0000` | `0.3333` | `0.5000` | `3` |
| **Lung_Opacity** | `0.6087` | `0.8750` | `0.7179` | `16` |
| **Normal** | `0.7600` | `0.7308` | `0.7451` | `26` |
| **Viral Pneumonia** | `1.0000` | `0.2000` | `0.3333` | `5` |

---

## 3. Visual Performance Artifacts
*   **Confusion Matrix:** Refer to `confusion_matrix.png`
*   **Normalized Matrix:** Refer to `confusion_matrix_normalized.png`
*   **ROC Curve (OVR):** Refer to `roc_curves.png`
*   **Precision-Recall Curve:** Refer to `precision_recall_curves.png`
