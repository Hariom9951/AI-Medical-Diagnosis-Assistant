"""Configuration Manager.

Orchestrates loading, parsing, environment variable merging, and validation of application,
model, and training configurations. Adheres to clean architecture boundaries.
"""

import json
import logging
import logging.config
import os
from pathlib import Path
from typing import Any, Final
import yaml
from dotenv import load_dotenv

from src.constants import (
    APP_CONFIG_PATH,
    MODEL_CONFIG_PATH,
    TRAINING_CONFIG_PATH,
    LOGGING_CONFIG_PATH,
)
from src.entity.config_entity import AppConfig, ModelConfig, TrainingConfig

# Load environment secrets at bootstrap stage
load_dotenv()


class ConfigurationManager:
    """Manages system configuration loading, validation, and logging bootstrap."""

    def __init__(
        self,
        app_config_path: Path = APP_CONFIG_PATH,
        model_config_path: Path = MODEL_CONFIG_PATH,
        training_config_path: Path = TRAINING_CONFIG_PATH,
        logging_config_path: Path = LOGGING_CONFIG_PATH,
    ) -> None:
        """Initializes the ConfigurationManager and loads config structures.

        Args:
            app_config_path (Path): Path to application YAML configuration.
            model_config_path (Path): Path to model architecture YAML configuration.
            training_config_path (Path): Path to training setup YAML configuration.
            logging_config_path (Path): Path to logging setup JSON configuration.
        """
        self.app_path: Final[Path] = app_config_path
        self.model_path: Final[Path] = model_config_path
        self.training_path: Final[Path] = training_config_path
        self.logging_path: Final[Path] = logging_config_path

        # Setup runtime directory constraints
        Path("logs").mkdir(exist_ok=True)
        Path("artifacts").mkdir(exist_ok=True)

        # 1. Initialize Logging system
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        # 2. Parse Raw Configurations
        self.raw_app_config: Final[dict[str, Any]] = self._load_yaml(self.app_path)
        self.raw_model_config: Final[dict[str, Any]] = self._load_yaml(self.model_path)
        self.raw_training_config: Final[dict[str, Any]] = self._load_yaml(self.training_path)

        self.logger.info("Configuration Manager successfully initialized.")

    def _setup_logging(self) -> None:
        """Configures logging handlers from the logging config file.

        Falls back to standard console logging if JSON format configuration is missing or broken.
        """
        if self.logging_path.exists():
            try:
                with open(self.logging_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                logging.config.dictConfig(config)
            except Exception as e:
                # Basic fallback configuration to keep application from crashing
                logging.basicConfig(level=logging.INFO)
                logging.warning("Failed to parse logging JSON configuration. Error: %s", e)
        else:
            logging.basicConfig(level=logging.INFO)
            logging.warning("Logging config file not found at %s. Initialized basic configuration.", self.logging_path)

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        """Loads a YAML configuration file.

        Args:
            path (Path): Location of the target YAML.

        Returns:
            dict[str, Any]: Parsed configuration values.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If the YAML parsing fails.
        """
        if not path.exists():
            error_msg = f"Configuration file not found at path: {path}"
            if hasattr(self, "logger"):
                self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
                if content is None:
                    return {}
                return dict(content)
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse YAML configuration at {path}. Error: {e}"
            if hasattr(self, "logger"):
                self.logger.error(error_msg)
            raise ValueError(error_msg)

    def get_app_config(self) -> AppConfig:
        """Processes and returns the validated AppConfig object.

        Automatically overrides parameters if corresponding environment variables are present:
        - DB_PASSWORD overrides database.password
        - DB_HOST overrides database.host
        - DB_PORT overrides database.port
        - DB_USERNAME overrides database.username
        - DB_NAME overrides database.database_name

        Returns:
            AppConfig: Validated Pydantic AppConfig model.
        """
        # Work on a copy of the dictionary to prevent mutating raw loaded configurations
        config_data = dict(self.raw_app_config)

        # Ensure sub-keys exist before overriding to prevent KeyErrors
        if "database" not in config_data:
            config_data["database"] = {}
        
        # Override database parameters if declared in environment variables
        db_env_mappings = {
            "DB_HOST": "host",
            "DB_PORT": "port",
            "DB_USERNAME": "username",
            "DB_PASSWORD": "password",
            "DB_NAME": "database_name",
        }
        
        for env_var, config_key in db_env_mappings.items():
            val = os.getenv(env_var)
            if val is not None:
                if config_key == "port":
                    try:
                        config_data["database"][config_key] = int(val)
                    except ValueError:
                        self.logger.warning("Environment override DB_PORT is not a valid integer: %s", val)
                else:
                    config_data["database"][config_key] = val

        return AppConfig(**config_data)

    def get_model_config(self) -> ModelConfig:
        """Processes and returns the validated ModelConfig object.

        Returns:
            ModelConfig: Validated Pydantic ModelConfig model.
        """
        return ModelConfig(**self.raw_model_config)

    def get_training_config(self) -> TrainingConfig:
        """Processes and returns the validated TrainingConfig object.

        Returns:
            TrainingConfig: Validated Pydantic TrainingConfig model.
        """
        return TrainingConfig(**self.raw_training_config)
