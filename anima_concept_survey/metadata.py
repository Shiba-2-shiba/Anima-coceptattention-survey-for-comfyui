from __future__ import annotations

from typing import Any

import torch


def safe_metadata_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if torch.is_tensor(value):
        return {"tensor_shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, tuple | list):
        if all(item is None or isinstance(item, str | int | float | bool) for item in value):
            return list(value)
        return {"type": type(value).__name__, "len": len(value)}
    if isinstance(value, dict):
        return {"type": "dict", "count": len(value), "keys": [str(key) for key in list(value.keys())[:16]]}
    return str(value)


def discover_transformer_metadata(transformer_options: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    keys = ("block", "block_index", "transformer_index", "module_path", "patches_replace")
    metadata = {key: safe_metadata_value(transformer_options[key]) for key in keys if key in transformer_options}
    block = metadata.get("block")
    if isinstance(block, list) and block:
        block_id = ":".join(str(part) for part in block)
    elif "block_index" in metadata:
        block_id = str(metadata["block_index"])
    elif "transformer_index" in metadata:
        block_id = str(metadata["transformer_index"])
    elif "module_path" in metadata:
        block_id = str(metadata["module_path"])
    else:
        block_id = "unknown"
    return block_id, metadata


def is_anima_like_model(model: Any) -> bool:
    inner = getattr(model, "model", model)
    diffusion_model = getattr(inner, "diffusion_model", inner)
    if diffusion_model.__class__.__name__ == "Anima":
        return True
    return hasattr(diffusion_model, "llm_adapter") and hasattr(diffusion_model, "blocks")
