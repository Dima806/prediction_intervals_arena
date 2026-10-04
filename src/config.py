"""Typed configuration loaded from ``config/settings.yaml``.

Everything tunable (nominal coverage, seeds, data sizes) lives in the YAML so the
notebooks and the ``src`` library never hard-code it. Any field can be overridden
with an environment variable prefixed ``PIA_``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SETTINGS_YAML = _PROJECT_ROOT / "config" / "settings.yaml"


class Settings(BaseSettings):
    """Project-wide settings, read from YAML with environment-variable overrides."""

    model_config = SettingsConfigDict(
        yaml_file=_SETTINGS_YAML,
        env_prefix="PIA_",
        extra="ignore",
    )

    nominal_coverage: float = 0.90
    seed: int = 42
    n_samples: int = 5000
    test_fraction: float = 0.30
    calibration_fraction: float = 0.25
    n_bootstrap: int = 100
    n_bins: int = 10
    base_scale: float = 2.0

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Precedence: explicit init args > environment variables > settings.yaml.
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


@lru_cache
def get_settings() -> Settings:
    """Return the cached project settings."""
    return Settings()
