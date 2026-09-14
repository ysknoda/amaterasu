"""Configuration loading: YAML -> validated dataclass tree.

``load_config`` is the single entry point used by the CLI and by tests. It runs three
passes and reports every problem from all three in one exception:

1. YAML parsing;
2. type/choice validation against :mod:`amaterasu.config.schema`, including unknown-key
   detection with spelling suggestions;
3. cross-field semantic checks in :mod:`amaterasu.config.defaults`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from ._loader import ConfigError, ErrorLog, build_dataclass, to_plain_dict
from ._yaml import safe_load
from .defaults import resolve_and_validate
from .overrides import apply_overrides, parse_override
from .schema import Config

__all__ = [
    "Config",
    "ConfigError",
    "config_to_yaml",
    "load_config",
    "load_config_mapping",
    "parse_override",
]


def load_config_mapping(
    data: dict[str, Any], *, overrides: list[str] | None = None, source: str = "input"
) -> Config:
    """Build a validated :class:`Config` from an already-parsed mapping."""
    if not isinstance(data, dict):
        raise ConfigError([f"expected a mapping of settings, got {type(data).__name__}"], source)
    if overrides:
        data = apply_overrides(data, overrides)

    log = ErrorLog()
    config = build_dataclass(Config, data, "", log)
    # Run the semantic pass even when the structural pass found problems, so one --dry-run
    # surfaces every mistake. Fields that failed to build are left as None, which the
    # semantic checks may not expect; if one of them raises, the structural errors already
    # explain why, so the crash is swallowed rather than replacing a useful report with a
    # traceback.
    try:
        resolve_and_validate(config, log)
    except (TypeError, AttributeError, ValueError):
        if not log.problems:
            raise
    log.raise_if_any(source)
    return config


def load_config(path: str | os.PathLike, *, overrides: list[str] | None = None) -> Config:
    """Load, override and validate a calculation input file."""
    path = Path(path)
    try:
        data = safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError([f"invalid YAML: {exc}"], str(path)) from exc
    if data is None:
        raise ConfigError(["file is empty"], str(path))
    return load_config_mapping(data, overrides=overrides, source=str(path))


def config_to_yaml(config: Config) -> str:
    """Serialise a config, with every default expanded, for ``resolved_input.yaml``."""
    return yaml.safe_dump(to_plain_dict(config), sort_keys=False, default_flow_style=False)
