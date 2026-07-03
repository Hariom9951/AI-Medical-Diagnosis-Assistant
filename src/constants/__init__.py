"""System Constants.

Defines global static constants, primarily filesystem paths to config templates
to decouple storage references from code logic.
"""

from pathlib import Path
from typing import Final

CONFIG_DIR: Final[Path] = Path("configs")

APP_CONFIG_PATH: Final[Path] = CONFIG_DIR / "app_config.yaml"
MODEL_CONFIG_PATH: Final[Path] = CONFIG_DIR / "model_config.yaml"
TRAINING_CONFIG_PATH: Final[Path] = CONFIG_DIR / "training_config.yaml"
LOGGING_CONFIG_PATH: Final[Path] = CONFIG_DIR / "logging_config.json"
