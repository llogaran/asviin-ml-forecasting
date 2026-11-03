"""Runtime configuration, loaded from environment variables.

Every value can be overridden with an environment variable so the same image can
run in dev, CI and production without code changes. Secrets are never persisted
in the repository: see ``.env.example``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _path_from_env(name: str, default: Path) -> Path:
    """Resolve a path env var relative to the repository root."""
    raw = os.getenv(name)
    if not raw:
        return default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings."""

    env: str = field(default_factory=lambda: os.getenv("MLFC_ENV", "dev"))
    data_dir: Path = field(
        default_factory=lambda: _path_from_env("MLFC_DATA_DIR", PROJECT_ROOT / "data")
    )
    artifacts_dir: Path = field(
        default_factory=lambda: _path_from_env("MLFC_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")
    )
    model_name: str = field(default_factory=lambda: os.getenv("MLFC_MODEL_NAME", "champion"))
    random_seed: int = field(default_factory=lambda: int(os.getenv("MLFC_RANDOM_SEED", "42")))
    api_key: str | None = field(default_factory=lambda: os.getenv("MLFC_API_KEY") or None)
    openai_api_key: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY") or None
    )
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            origin.strip()
            for origin in os.getenv("MLFC_CORS_ORIGINS", "*").split(",")
            if origin.strip()
        )
    )


def get_settings() -> Settings:
    """Return a fresh :class:`Settings` instance (handy for tests/overrides)."""
    return Settings()
