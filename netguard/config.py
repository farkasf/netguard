"""Central configuration. All tunables live here (pydantic-settings).

Every value can be overridden by an environment variable prefixed with
``NETGUARD_`` (e.g. ``NETGUARD_API_PORT=9000``) or via a ``.env`` file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root — used for default data/model locations.
REPO_ROOT = Path(__file__).resolve().parent.parent


class GBMConfig(BaseSettings):
    """Hyperparameters for the from-scratch Gradient Boosting Machine."""

    model_config = SettingsConfigDict(env_prefix="NETGUARD_GBM_")

    n_estimators: int = 100
    learning_rate: float = 0.3
    max_depth: int = 4
    min_samples_leaf: int = 5
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    min_split_gain: float = 0.0
    subsample: float = 1.0
    colsample: float = 1.0
    # Early stopping (used only when a validation split is supplied).
    early_stopping_rounds: int = 0  # 0 disables early stopping
    random_state: int = 42


class Settings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(
        env_prefix="NETGUARD_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Paths -----------------------------------------------------------
    db_path: Path = REPO_ROOT / "data" / "netguard.db"
    models_dir: Path = REPO_ROOT / "models"
    schema_path: Path = REPO_ROOT / "netguard" / "store" / "schema.sql"

    # --- Capture / flow assembly ----------------------------------------
    iface: str = "eth0"
    inactive_timeout: float = 15.0  # seconds of silence closes a flow
    active_timeout: float = 120.0  # max lifetime of a long-lived flow
    sweep_interval: float = 5.0  # how often runner flushes expired flows

    # --- Scoring ---------------------------------------------------------
    benign_label: str = "BENIGN"
    low_confidence_threshold: float = 0.5  # below this, flag even if benign

    # --- Store -----------------------------------------------------------
    flows_table_cap: int = 5000  # rolling flows kept for the UI

    # --- API -------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # --- Retraining ------------------------------------------------------
    retrain_cron: str = "0 3 * * *"  # daily at 03:00
    retrain_enabled: bool = False  # APScheduler off by default in the API

    # --- ML --------------------------------------------------------------
    gbm: GBMConfig = Field(default_factory=GBMConfig)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
