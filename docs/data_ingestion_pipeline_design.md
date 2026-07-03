# Data Ingestion Pipeline Design Document

This document outlines the architectural blueprint and operational workflow for a production-grade **Multimodal Data Ingestion Pipeline** for the AI Medical Diagnosis Assistant. The design prioritizes modularity, robustness, data validation, and strict reproducibility using MLOps best practices (DVC and remote versioning).

---

## 1. Architectural Decisions & SOLID Alignment

To build a maintainable pipeline that scales, we separate concerns using **SOLID** principles:

```
                  ┌───────────────────────┐
                  │ Ingestion Orchestrator│
                  └───────────┬───────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  IDownloader    │  │   IExtractor    │  │   IValidator    │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  HTTPDownloader │  │  ZipExtractor   │  │ SchemaValidator │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

*   **Single Responsibility Principle (SRP):** 
    *   `Downloader` manages network retrieval and checksum checks.
    *   `Extractor` manages archive decompression.
    *   `Validator` checks data integrity, format correctness, and column schemas.
    *   `IngestionOrchestrator` coordinates the workflow sequence.
*   **Open/Closed Principle (OCP):** Downloaders and Extractors implement abstract interfaces. Support for new protocols (e.g., S3, SFTP, DICOM servers) is added by implementing the interface without modifying the core orchestrator.
*   **Dependency Inversion Principle (DIP):** The orchestrator depends on abstractions (`IDownloader`, `IExtractor`, `IValidator`), not on concrete implementations, enabling easy unit-testing using mocks.

---

## 2. Directory Structure

A standardized workspace keeps raw files isolated from validated data:

```
AI-Medical-Diagnosis-Assistant/
├── configs/
│   ├── ingestion_config.yaml   # URLs, local directories, chunk sizes, retries
│   └── validation_schema.yaml  # Data schemas, expected columns, image sizes
├── data/                       # Tracked by DVC (ignored by Git)
│   ├── external/               # Raw compressed archives (.zip, .tar.gz)
│   ├── raw/                    # Extracted raw files
│   └── verified/               # Validated datasets ready for model training
├── docs/
│   └── data_ingestion_pipeline_design.md
└── src/
    └── components/
        ├── data_ingestion.py   # Code containing Downloader, Extractor
        └── data_validation.py  # Code containing Validator and Schema check
```

---

## 3. Data Ingestion Stages

### Stage 1: Downloading
*   **Trigger:** Executed via configuration directive (providing a dataset URL or API endpoint).
*   **Mechanism:**
    *   Fetches configuration parameters (`url`, `destination`, `expected_hash`, `chunk_size`, `max_retries`) from `configs/ingestion_config.yaml`.
    *   Uses a chunked streaming mechanism rather than loading the entire file into RAM, preventing Out of Memory (OOM) errors during large chest X-ray or CT scans downloads.
    *   Implements an exponential backoff retry mechanism (e.g., retrying up to 5 times with 2s, 4s, 8s delays) to survive network instability.
    *   Validates the integrity of the downloaded file immediately via MD5/SHA-256 hash comparison. If the hash mismatch occurs, it removes the corrupted file and raises a validation exception.

### Stage 2: Extracting
*   **Trigger:** Successfully verified download of the raw archive.
*   **Mechanism:**
    *   Discovers the file extension (`.zip`, `.tar.gz`, `.tar`) and delegates to the appropriate extractor.
    *   Performs extraction within a temporary buffer directory.
    *   Extracts contents to `data/raw/` in a multi-threaded manner to utilize multi-core server processors.
    *   Implements automatic cleanup: if extraction fails (due to corrupted files or disk fullness), the orchestrator catches the exception, deletes any partially extracted folders to prevent dirty states, and logs the traceback.

### Stage 3: Data Validation
*   **Trigger:** Successful extraction of files to `data/raw/`.
*   **Mechanism:**
    *   **Schema Checking:** Verifies tabular clinical notes/metadata. Validates that the metadata file (e.g., `metadata.csv`) exists, contains the required columns (e.g., `image_path`, `symptoms`, `label`), and conforms to data types (e.g., `age` is numeric, `itching` is boolean).
    *   **Structural Checking:** Counts the total number of images and matches them to the metadata records. Verifies image resolutions and file formats (e.g., checking that images are valid PNGs or JPEGs and not corrupt header bytes).
    *   **Outlier & Value Range Checks:** Flags records containing out-of-range clinical values (e.g., negative ages, invalid diagnostic labels) or missing labels.
*   **Lifecycle Outcome:**
    *   If validation **passes**, files are safely copied or symlinked to `data/verified/` and the pipeline proceeds.
    *   If validation **fails**, the validator produces a detailed error report (`validation_report.json`), alerts the engineering/clinical team, and terminates the pipeline to protect downstream model training.

---

## 4. Dataset Versioning using DVC

To guarantee reproducibility and avoid bloating the Git repository with gigabytes of binary image data, we use **Data Version Control (DVC)**.

### A. Initialization
DVC is initialized at the root of the workspace. This sets up internal tracking structures and an automatic `.gitignore` to prevent data files from being tracked by Git.
```bash
dvc init
```

### B. Tracking Data
Instead of tracking actual images, Git only tracks lightweight pointer files (`.dvc`) generated when we add data directories to DVC:
```bash
dvc add data/raw
dvc add data/verified
```
This moves the actual directories to the local DVC cache (`.dvc/cache`) and creates:
1. `data/raw.dvc` (a YAML pointer containing the directory's MD5 checksum)
2. `data/verified.dvc`
3. Excludes `data/raw` and `data/verified` in the project's `.gitignore` file.

These `.dvc` files are committed directly to Git:
```bash
git add data/raw.dvc data/verified.dvc data/.gitignore
git commit -m "chore: track raw and verified datasets with DVC"
```

### C. DVC Pipeline Stages (`dvc.yaml`)
To automate execution, we define the ingestion workflow as a pipeline in `dvc.yaml`. This ensures that if the source dataset or validation rules change, DVC automatically detects what needs to be rerun:

```yaml
stages:
  ingest:
    cmd: python src/components/data_ingestion.py
    deps:
      - src/components/data_ingestion.py
      - configs/ingestion_config.yaml
    outs:
      - data/raw
      
  validate:
    cmd: python src/components/data_validation.py
    deps:
      - src/components/data_validation.py
      - data/raw
      - configs/validation_schema.yaml
    outs:
      - data/verified
```

---

## 5. Google Drive Remote Storage Configuration

To synchronize datasets across team members and remote training servers, Google Drive acts as the remote storage repository.

### A. Configuration Steps

1.  **Create shared Google Drive Folder:** A dedicated folder is created in Google Drive, and its unique ID (the alphanumeric string in the folder URL) is extracted.
2.  **Add DVC Remote:** Add the remote configuration to the local project:
    ```bash
    dvc remote add -d gdrive_remote gdrive://<GOOGLE_DRIVE_FOLDER_ID>
    ```
    *The `-d` flag sets it as the default remote.*

3.  **Authentication:**
    *   **Interactive Dev Environment:** When running `dvc push` or `dvc pull` for the first time, DVC opens a browser window prompting the developer to sign in with their Google account and authorize DVC to access the folder.
    *   **CI/CD or Headless Servers:** For automated environments (like GitHub Actions or Kubernetes training pods), we configure DVC using a Google Service Account key:
        ```bash
        dvc remote modify gdrive_remote gdrive_service_account_json_path_key /path/to/service_account.json
        ```

4.  **Data Synchronization:**
    *   **Uploading Data:** To share newly ingested/validated datasets:
        ```bash
        dvc push
        ```
    *   **Downloading Data:** When checking out the code on a clean system, run:
        ```bash
        git pull
        dvc pull
        ```
        DVC resolves the `.dvc` files and fetches the exact, immutable images and datasets matched to that Git commit.
