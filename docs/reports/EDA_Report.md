# Exploratory Data Analysis (EDA) Report

Provides statistics on features distributions and patterns.

---

## 1. Clinical Scan Image Distributions

*   **Average File Size:** `18.60 KB`
*   **Image Resolutions (Widths x Heights):**
    *   Min Width: `256 px` | Max Width: `299 px`
    *   Min Height: `256 px` | Max Height: `299 px`
*   **Aspect Ratio (Width/Height) Mean:** `1.00`

---

## 2. Class Balance Distributions

*   **COVID:** `7232 scans`
*   **Lung_Opacity:** `12024 scans`
*   **Normal:** `20384 scans`
*   **Viral Pneumonia:** `2690 scans`

---

## 3. Top Tabular Symptom Frequencies

*   **fatigue:** `1932 occurrences`
*   **vomiting:** `1914 occurrences`
*   **high fever:** `1362 occurrences`
*   **loss of appetite:** `1152 occurrences`
*   **nausea:** `1146 occurrences`
*   **headache:** `1134 occurrences`
*   **abdominal pain:** `1032 occurrences`
*   **yellowish skin:** `912 occurrences`
*   **yellowing of eyes:** `816 occurrences`
*   **chills:** `798 occurrences`

---

## 4. Visualizations Map
All plots are successfully generated and saved to:
*   Class Balance Plot: `image_class_distribution.png`
*   Resolution Distributions: `image_resolutions_histogram.png`
*   Symptom Missingness Matrix: `symptom_missingness_heatmap.png`
*   Diagnostic Frequencies: `disease_frequency.png`
*   Symptom Occurrences: `symptom_frequency.png`
