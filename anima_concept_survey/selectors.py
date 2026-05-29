from __future__ import annotations

import math

import torch


def parse_call_index_scope(spec: str) -> set[int] | None:
    return _parse_index_scope(spec, label="call")


def parse_step_index_scope(spec: str) -> set[int] | None:
    return _parse_index_scope(spec, label="step")


def _parse_index_scope(spec: str, *, label: str) -> set[int] | None:
    normalized = str(spec).strip().lower()
    if normalized == "all":
        return None
    if not normalized:
        raise ValueError(f"{label} index scope must be 'all' or a comma-separated list of non-negative integers/ranges")

    indices: set[int] = set()
    for raw_part in normalized.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError(f"Invalid {label} index scope entry in {spec!r}")
        if "-" in part:
            start_text, _, end_text = part.partition("-")
            if not start_text or not end_text:
                raise ValueError(f"Invalid {label} index range: {part!r}")
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError as exc:
                raise ValueError(f"Invalid {label} index range: {part!r}") from exc
            if start < 0 or end < 0:
                raise ValueError(f"{label} index scope entries must be non-negative")
            if end < start:
                raise ValueError(f"Invalid descending {label} index range: {part!r}")
            indices.update(range(start, end + 1))
        else:
            try:
                index = int(part)
            except ValueError as exc:
                raise ValueError(f"Invalid {label} index scope entry: {part!r}") from exc
            if index < 0:
                raise ValueError(f"{label} index scope entries must be non-negative")
            indices.add(index)
    return indices


def infer_square_spatial_shape(query_len: int) -> tuple[int, int] | None:
    side = math.isqrt(int(query_len))
    if side * side != query_len:
        return None
    return side, side


def estimate_logits_mib(batch: int, heads: int, query_len: int, text_len: int, bytes_per: int = 4) -> float:
    return batch * heads * query_len * text_len * bytes_per / (1024**2)


def shape_key(q: torch.Tensor, k: torch.Tensor) -> str:
    return f"q{int(q.shape[-2])},k{int(k.shape[-2])},h{int(q.shape[1])},d{int(q.shape[-1])}"
