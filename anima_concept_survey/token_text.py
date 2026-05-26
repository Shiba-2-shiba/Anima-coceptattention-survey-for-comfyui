from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class TokenTextEntry:
    token_index: int
    token_id: int | None
    token_text: str
    token_source: str
    weight: float | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "token_index": self.token_index,
            "token_id": self.token_id,
            "token_text": self.token_text,
            "token_source": self.token_source,
            "weight": self.weight,
        }


def build_token_text_map(clip: Any, prompt_text: str, max_length: int | None = None) -> dict[int, dict[str, Any]]:
    if clip is None or not prompt_text:
        return {}
    tokenize = getattr(clip, "tokenize", None)
    if not callable(tokenize):
        return {}

    try:
        tokenized = tokenize(prompt_text)
    except Exception:
        return {}

    entries = flatten_tokenized(tokenized)
    if max_length is not None:
        entries = entries[:max_length]
    decoders = list(_candidate_decoders(clip))

    result: dict[int, dict[str, Any]] = {}
    for entry in entries:
        token_text = _decode_token(entry.token_id, decoders)
        if not token_text:
            token_text = _fallback_token_text(entry.token_id)
        mapped = TokenTextEntry(
            token_index=entry.token_index,
            token_id=entry.token_id,
            token_text=token_text,
            token_source=entry.token_source,
            weight=entry.weight,
        )
        result[mapped.token_index] = mapped.to_record()
    return result


@dataclass(frozen=True)
class _FlatToken:
    token_index: int
    token_source: str
    token_id: int | None
    weight: float | None


def flatten_tokenized(tokenized: Any) -> list[_FlatToken]:
    entries: list[_FlatToken] = []
    if isinstance(tokenized, dict):
        for source, value in tokenized.items():
            _append_stream(entries, str(source), value)
    else:
        _append_stream(entries, "tokens", tokenized)
    return entries


def _append_stream(entries: list[_FlatToken], source: str, value: Any) -> None:
    for item in _iter_leaf_tokens(value):
        token_id, weight = _parse_token_item(item)
        entries.append(_FlatToken(
            token_index=len(entries),
            token_source=source,
            token_id=token_id,
            weight=weight,
        ))


def _iter_leaf_tokens(value: Any) -> Iterable[Any]:
    if torch.is_tensor(value):
        for item in value.detach().cpu().flatten().tolist():
            yield item
        return
    if isinstance(value, tuple):
        if value and _looks_like_token_tuple(value):
            yield value
            return
        for item in value:
            yield from _iter_leaf_tokens(item)
        return
    if isinstance(value, list):
        if value and all(_is_scalar_token(part) for part in value) and not _looks_like_batch(value):
            for item in value:
                yield item
            return
        for item in value:
            yield from _iter_leaf_tokens(item)
        return
    yield value


def _looks_like_batch(value: list[Any]) -> bool:
    return any(isinstance(item, list | tuple | dict) or torch.is_tensor(item) for item in value)


def _looks_like_token_tuple(value: tuple[Any, ...]) -> bool:
    if not value:
        return False
    first = value[0]
    return _is_scalar_token(first) and len(value) <= 4


def _is_scalar_token(value: Any) -> bool:
    return isinstance(value, int | float | str)


def _parse_token_item(item: Any) -> tuple[int | None, float | None]:
    weight = None
    raw_token = item
    if isinstance(item, tuple | list):
        raw_token = item[0] if item else None
        if len(item) > 1 and isinstance(item[1], int | float):
            weight = float(item[1])
    if torch.is_tensor(raw_token):
        raw_token = raw_token.detach().cpu().flatten()[0].item() if raw_token.numel() else None
    try:
        return int(raw_token), weight
    except (TypeError, ValueError):
        return None, weight


def _candidate_decoders(clip: Any) -> Iterable[Any]:
    seen: set[int] = set()
    stack = [clip]
    for attr in ("tokenizer", "cond_stage_model", "cond_stage_model.tokenizer", "patcher.model.tokenizer"):
        current = clip
        for part in attr.split("."):
            current = getattr(current, part, None)
            if current is None:
                break
        if current is not None:
            stack.append(current)

    while stack:
        obj = stack.pop(0)
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        for name in ("convert_ids_to_tokens", "decode", "batch_decode", "id_to_token", "untokenize"):
            fn = getattr(obj, name, None)
            if callable(fn):
                yield fn
        for name in ("tokenizer", "tokenizer_l", "tokenizer_g", "t5xxl", "clip_l", "clip_g"):
            child = getattr(obj, name, None)
            if child is not None and id(child) not in seen:
                stack.append(child)


def _decode_token(token_id: int | None, decoders: list[Any]) -> str:
    if token_id is None:
        return ""
    for decoder in decoders:
        for payload in (token_id, [token_id]):
            try:
                decoded = decoder(payload)
            except Exception:
                continue
            text = _decoded_to_text(decoded)
            if text:
                return text
    return ""


def _decoded_to_text(decoded: Any) -> str:
    if decoded is None:
        return ""
    if isinstance(decoded, str):
        return decoded
    if isinstance(decoded, bytes):
        return decoded.decode("utf-8", errors="replace")
    if isinstance(decoded, list | tuple):
        parts = [_decoded_to_text(item) for item in decoded]
        parts = [part for part in parts if part]
        return " ".join(parts)
    return str(decoded)


def _fallback_token_text(token_id: int | None) -> str:
    if token_id is None:
        return ""
    return f"<token:{token_id}>"
