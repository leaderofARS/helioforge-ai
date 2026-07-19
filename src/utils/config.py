from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "configs" / "config.yaml"


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path_value = Path(str(value))
    if path_value.is_absolute():
        return path_value
    return base_dir / path_value


def _resolve_mapping(mapping: dict[str, Any], base_dir: Path, resolve_all_strings: bool = False) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            resolved[key] = _resolve_mapping(value, base_dir, resolve_all_strings)
        elif isinstance(value, str):
            if resolve_all_strings:
                resolved[key] = str(_resolve_path(value, base_dir))
            else:
                resolved[key] = value
        else:
            resolved[key] = value
    return resolved


def load_config() -> dict[str, Any]:
    with _CONFIG_PATH.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    project_root = _resolve_path(raw_config.get("paths", {}).get("project_root", "."), _REPO_ROOT)

    paths = _resolve_mapping(raw_config.get("paths", {}), project_root, resolve_all_strings=True)
    files = _resolve_mapping(raw_config.get("files", {}), project_root, resolve_all_strings=True)
    exports = _resolve_mapping(raw_config.get("exports", {}), project_root, resolve_all_strings=False)

    for export_section in exports.values():
        if isinstance(export_section, dict):
            directory = export_section.get("directory")
            if isinstance(directory, str):
                export_section["directory"] = str(_resolve_path(directory, project_root))

    return {
        "project": raw_config.get("project", {}),
        "paths": paths,
        "exports": exports,
        "files": files,
        "logging": raw_config.get("logging", {}),
    }


CONFIG = load_config()
PROJECT_ROOT = Path(CONFIG["paths"]["project_root"])


def get_path(*keys: str) -> Path:
    current: Any = CONFIG["paths"]
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Unknown config path: {'/'.join(keys)}")
        current = current[key]
    return Path(current)
