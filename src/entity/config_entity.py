"""Configuration Entities.

Defines Pydantic models for type-safe parameter validation of application configuration
structures. These entities correspond to the domain entity layer in clean architecture.
"""

from pathlib import Path
from pydantic import BaseModel, Field


class APIConfig(BaseModel):
    """Configuration schema for the FastAPI server gateway."""

    host: str = Field(..., description="The hostname or IP address of the FastAPI server")
    port: int = Field(..., gt=0, lt=65536, description="The port to bind the FastAPI server")
    prefix: str = Field("/api/v1", description="Global prefix route for all endpoints")


class DatabaseConfig(BaseModel):
    """Configuration schema for PostgreSQL database connections."""

    host: str = Field(..., description="Database server address")
    port: int = Field(5432, gt=0, lt=65536, description="PostgreSQL communication port")
    username: str = Field(..., description="Database access user")
    password: str = Field(..., description="Database user credentials")
    database_name: str = Field(..., description="Target database schema name")


class S3StorageConfig(BaseModel):
    """Configuration schema for AWS S3 / Object storage paths."""

    bucket_name: str = Field(..., description="Cloud bucket target identifier")
    endpoint_url: str | None = Field(
        None, description="Optional custom endpoint URL for local MinIO/mock servers"
    )
    region_name: str = Field("us-east-1", description="Cloud bucket geography zone")


class ModelConfig(BaseModel):
    """Configuration schema for deep learning architectures and explainability."""

    vision_model_name: str = Field(
        "vit_base_patch16_224", description="Vision backbone transformer architecture key"
    )
    vision_img_size: int = Field(
        224, gt=0, description="Square input resolution dimension (e.g. 224x224)"
    )
    text_model_name: str = Field(
        "emilyalsentzer/Bio_ClinicalBERT", description="Clinical text processing encoder name"
    )
    embedding_dim: int = Field(
        768, gt=0, description="Embedding vector size output by text encoder"
    )
    fusion_hidden_dim: int = Field(
        256, gt=0, description="Dense projection dimension size for fused modalities"
    )
    num_classes: int = Field(5, gt=0, description="Number of target diagnosis categories")
    target_gradcam_layer: str = Field(
        "blocks.11.norm1", description="Target backbone layer for visual explainability extraction"
    )


class TrainingConfig(BaseModel):
    """Configuration schema for model optimization workflows."""

    epochs: int = Field(20, gt=0, description="Number of complete training epochs")
    batch_size: int = Field(32, gt=0, description="Batch sizing size for dataloaders")
    learning_rate: float = Field(
        1e-4, gt=0.0, lt=1.0, description="Learning optimizer speed constant"
    )
    checkpoint_dir: Path = Field(
        ..., description="Local directory target to save training weight runs"
    )
    mlflow_tracking_uri: str = Field(..., description="MLflow logging server connection string")
    mlflow_experiment_name: str = Field(..., description="MLflow project namespace identifier")


class AppConfig(BaseModel):
    """Unified application configuration encapsulating structural sub-schemas."""

    api: APIConfig
    database: DatabaseConfig
    storage: S3StorageConfig
