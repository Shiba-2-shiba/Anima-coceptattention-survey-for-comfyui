from __future__ import annotations

import os
from pathlib import Path

from .config import DEFAULT_JSONL_RELATIVE_PATH


def comfy_output_dir() -> Path:
    try:
        import folder_paths  # type: ignore
    except Exception:
        return Path.cwd()

    get_output_directory = getattr(folder_paths, "get_output_directory", None)
    if callable(get_output_directory):
        try:
            return Path(get_output_directory())
        except Exception:
            return Path.cwd()
    output_directory = getattr(folder_paths, "output_directory", None)
    if output_directory:
        return Path(output_directory)
    return Path.cwd()


def resolve_comfy_output_path(
    value: str | None,
    *,
    default_relative: str | None = None,
    base_dir: str | Path | None = None,
) -> str | None:
    text = "" if value is None else str(value).strip().strip("\"'")
    if not text:
        if default_relative is None:
            return None
        text = default_relative

    expanded = os.path.expandvars(os.path.expanduser(text))
    path = Path(expanded)
    if not path.is_absolute():
        base = Path(base_dir) if base_dir is not None else comfy_output_dir()
        path = base / path
    return str(path)


def resolve_comfy_jsonl_path(
    value: str | None,
    *,
    default_relative: str | None = DEFAULT_JSONL_RELATIVE_PATH,
    base_dir: str | Path | None = None,
) -> str | None:
    if value is not None and not str(value).strip():
        return None
    path_text = resolve_comfy_output_path(value, default_relative=default_relative, base_dir=base_dir)
    if path_text is None:
        return None
    path = Path(path_text)
    original = "" if value is None else str(value).strip().strip("\"'")
    if original.endswith(("/", "\\")) or path.exists() and path.is_dir() or path.suffix.lower() != ".jsonl":
        path = path / "survey.jsonl"
    return str(path)
