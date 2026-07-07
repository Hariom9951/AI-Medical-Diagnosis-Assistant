#!/usr/bin/env python3
"""Project Template Generator.

This script automates the creation of a production-grade folder structure
and placeholder files for the AI Medical Diagnosis Assistant project.
It adheres to SOLID principles and Clean Architecture structure.
"""

import logging
from pathlib import Path
from typing import Final, List

# Configure structured logging to output to console with timestamps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)


class ProjectTemplateGenerator:
    """Handles the creation of the application directory structure and placeholder files."""

    def __init__(self, root_dir: str | Path = ".") -> None:
        """Initializes the template generator with a root directory.

        Args:
            root_dir (str | Path): The root path of the project workspace.
        """
        self.root: Final[Path] = Path(root_dir).resolve()

        # Define project directory layout & initial empty placeholders
        self.files_to_create: Final[List[Path]] = [
            # Core source package initialization
            self.root / "src" / "__init__.py",
            # Application components (modular pipeline components)
            self.root / "src" / "components" / "__init__.py",
            self.root / "src" / "components" / "data_ingestion.py",
            self.root / "src" / "components" / "data_validation.py",
            self.root / "src" / "components" / "data_transformation.py",
            self.root / "src" / "components" / "model_trainer.py",
            self.root / "src" / "components" / "model_evaluation.py",
            # Orchestration pipelines
            self.root / "src" / "pipeline" / "__init__.py",
            self.root / "src" / "pipeline" / "training_pipeline.py",
            self.root / "src" / "pipeline" / "prediction_pipeline.py",
            # Entities & configuration models
            self.root / "src" / "entity" / "__init__.py",
            self.root / "src" / "entity" / "config_entity.py",
            self.root / "src" / "entity" / "artifact_entity.py",
            # Configuration interfaces
            self.root / "src" / "config" / "__init__.py",
            self.root / "src" / "config" / "configuration.py",
            # Constants and utilities
            self.root / "src" / "constants" / "__init__.py",
            self.root / "src" / "utils" / "__init__.py",
            self.root / "src" / "utils" / "common.py",
            # ML training and evaluation routines
            self.root / "src" / "training" / "__init__.py",
            self.root / "src" / "training" / "train.py",
            # Inference services
            self.root / "src" / "inference" / "__init__.py",
            self.root / "src" / "inference" / "predict.py",
            # API presentation layer (FastAPI)
            self.root / "src" / "api" / "__init__.py",
            self.root / "src" / "api" / "main.py",
            self.root / "src" / "api" / "router.py",
            # UI presentation layer (Streamlit)
            self.root / "src" / "frontend" / "__init__.py",
            self.root / "src" / "frontend" / "app.py",
            # Drift, observability and performance monitoring
            self.root / "src" / "monitoring" / "__init__.py",
            self.root / "src" / "monitoring" / "drift.py",
            # Configurations (YAML, JSON)
            self.root / "configs" / "app_config.yaml",
            self.root / "configs" / "model_config.yaml",
            self.root / "configs" / "training_config.yaml",
            self.root / "configs" / "logging_config.json",
            # Testing suites
            self.root / "tests" / "__init__.py",
            self.root / "tests" / "unit" / "__init__.py",
            self.root / "tests" / "integration" / "__init__.py",
            # Artifacts, logs, models, documentation
            self.root / "artifacts" / ".gitkeep",
            self.root / "logs" / ".gitkeep",
            self.root / "models" / ".gitkeep",
            self.root / "docs" / ".gitkeep",
            self.root / "scripts" / ".gitkeep",
            self.root / "notebooks" / ".gitkeep",
            # Root orchestration metadata files
            self.root / "requirements.txt",
            self.root / "requirements-dev.txt",
            self.root / "pyproject.toml",
            self.root / "dvc.yaml",
            self.root / "params.yaml",
            self.root / ".env.example",
            self.root / ".gitignore",
            self.root / "README.md",
        ]

    def create_structure(self) -> None:
        """Iterates through and safely initializes the directory layout and files.

        This method is completely idempotent: it creates missing directories and
        empty files, but does not overwrite or modify existing non-empty files.
        """
        logger.info("Initializing workspace structure under: %s", self.root)

        for filepath in self.files_to_create:
            try:
                # Ensure the parent directory structure exists
                parent_dir: Path = filepath.parent
                if not parent_dir.exists():
                    parent_dir.mkdir(parents=True, exist_ok=True)
                    logger.info("Created directory path: %s", parent_dir.relative_to(self.root))

                # Handle idempotent file instantiation
                if not filepath.exists() or filepath.stat().st_size == 0:
                    filepath.touch(exist_ok=True)
                    logger.info("Initialized empty file: %s", filepath.relative_to(self.root))
                else:
                    logger.debug(
                        "Skipping existing non-empty file: %s", filepath.relative_to(self.root)
                    )

            except (PermissionError, OSError) as e:
                logger.error("Failed to create file or directory at %s. Error: %s", filepath, e)
                raise

        logger.info("Workspace initialization successfully completed.")


if __name__ == "__main__":
    # Instantiate and execute generator
    generator = ProjectTemplateGenerator()
    generator.create_structure()
