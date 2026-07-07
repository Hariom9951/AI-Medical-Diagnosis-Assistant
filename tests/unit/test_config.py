"""Unit Tests for Configuration Management.

Tests schema validation, environment overrides, error handling, and file loading logic.
"""

import os
from pathlib import Path
import pytest
from pydantic import ValidationError

from src.config.configuration import ConfigurationManager
from src.entity.config_entity import AppConfig, ModelConfig, TrainingConfig


@pytest.fixture
def temp_config_dir(tmp_path: Path) -> Path:
    """Fixture to create temporary configuration templates for isolated test assertions.

    Args:
        tmp_path (Path): pytest default temporary directory fixture.

    Returns:
        Path: Path to the temporary config directory.
    """
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()

    # 1. Write App Config
    app_yaml = """
api:
  host: "127.0.0.1"
  port: 8000
  prefix: "/api/v1"
database:
  host: "localhost"
  port: 5432
  username: "db_user"
  password: "db_password"
  database_name: "test_db"
storage:
  bucket_name: "test-bucket"
  endpoint_url: null
  region_name: "us-east-1"
"""
    (configs_dir / "app_config.yaml").write_text(app_yaml)

    # 2. Write Model Config
    model_yaml = """
vision_model_name: "vit_base_patch16_224"
vision_img_size: 224
text_model_name: "clinical_bert"
embedding_dim: 768
fusion_hidden_dim: 128
num_classes: 2
target_gradcam_layer: "layer4"
"""
    (configs_dir / "model_config.yaml").write_text(model_yaml)

    # 3. Write Training Config
    training_yaml = """
epochs: 5
batch_size: 16
learning_rate: 0.001
checkpoint_dir: "artifacts/checkpoints"
mlflow_tracking_uri: "http://localhost:5000"
mlflow_experiment_name: "test-run"
"""
    (configs_dir / "training_config.yaml").write_text(training_yaml)

    # 4. Write Logging Config
    logging_json = """
{
  "version": 1,
  "disable_existing_loggers": false,
  "root": {
    "handlers": [],
    "level": "INFO"
  }
}
"""
    (configs_dir / "logging_config.json").write_text(logging_json)

    return configs_dir


def test_configuration_manager_initialization(temp_config_dir: Path) -> None:
    """Asserts that all YAML assets load and validate successfully."""
    config_manager = ConfigurationManager(
        app_config_path=temp_config_dir / "app_config.yaml",
        model_config_path=temp_config_dir / "model_config.yaml",
        training_config_path=temp_config_dir / "training_config.yaml",
        logging_config_path=temp_config_dir / "logging_config.json",
    )

    app_config = config_manager.get_app_config()
    model_config = config_manager.get_model_config()
    training_config = config_manager.get_training_config()

    assert isinstance(app_config, AppConfig)
    assert isinstance(model_config, ModelConfig)
    assert isinstance(training_config, TrainingConfig)

    # Validate specific attributes
    assert app_config.api.host == "127.0.0.1"
    assert app_config.api.port == 8000
    assert model_config.num_classes == 2
    assert training_config.epochs == 5


def test_configuration_manager_missing_file_raises_error(temp_config_dir: Path) -> None:
    """Asserts that missing files raise a FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        ConfigurationManager(
            app_config_path=temp_config_dir / "non_existent.yaml",
            model_config_path=temp_config_dir / "model_config.yaml",
            training_config_path=temp_config_dir / "training_config.yaml",
            logging_config_path=temp_config_dir / "logging_config.json",
        )


def test_configuration_manager_invalid_yaml_raises_error(temp_config_dir: Path) -> None:
    """Asserts that malformed YAML files raise a ValueError."""
    corrupted_yaml = temp_config_dir / "corrupted_app_config.yaml"
    corrupted_yaml.write_text("{invalid: yaml: formatting: [")

    with pytest.raises(ValueError, match="Failed to parse YAML"):
        ConfigurationManager(
            app_config_path=corrupted_yaml,
            model_config_path=temp_config_dir / "model_config.yaml",
            training_config_path=temp_config_dir / "training_config.yaml",
            logging_config_path=temp_config_dir / "logging_config.json",
        )


def test_configuration_manager_invalid_pydantic_schema_raises_error(temp_config_dir: Path) -> None:
    """Asserts that parameters violating Pydantic validators raise a ValidationError."""
    invalid_app_yaml = """
api:
  host: "127.0.0.1"
  port: -50  # Port must be > 0
  prefix: "/api/v1"
database:
  host: "localhost"
  port: 5432
  username: "db_user"
  password: "db_password"
  database_name: "test_db"
storage:
  bucket_name: "test-bucket"
  endpoint_url: null
  region_name: "us-east-1"
"""
    invalid_app_path = temp_config_dir / "invalid_app_config.yaml"
    invalid_app_path.write_text(invalid_app_yaml)

    config_manager = ConfigurationManager(
        app_config_path=invalid_app_path,
        model_config_path=temp_config_dir / "model_config.yaml",
        training_config_path=temp_config_dir / "training_config.yaml",
        logging_config_path=temp_config_dir / "logging_config.json",
    )

    with pytest.raises(ValidationError):
        config_manager.get_app_config()


def test_configuration_manager_env_override(
    temp_config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asserts that environment variable overrides function correctly."""
    # Setup environment variables using monkeypatch
    monkeypatch.setenv("DB_HOST", "production-db-url.com")
    monkeypatch.setenv("DB_PORT", "9999")
    monkeypatch.setenv("DB_PASSWORD", "super_secret_override")

    config_manager = ConfigurationManager(
        app_config_path=temp_config_dir / "app_config.yaml",
        model_config_path=temp_config_dir / "model_config.yaml",
        training_config_path=temp_config_dir / "training_config.yaml",
        logging_config_path=temp_config_dir / "logging_config.json",
    )

    app_config = config_manager.get_app_config()
    assert app_config.database.host == "production-db-url.com"
    assert app_config.database.port == 9999
    assert app_config.database.password == "super_secret_override"
