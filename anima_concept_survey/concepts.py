from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class ConceptTermSpec:
    raw: str
    term: str
    normalized: str
    source_filter: str | None = None


@dataclass(frozen=True)
class ConceptTokenMatch:
    term: str
    normalized_term: str
    token_source: str
    token_indices: tuple[int, ...]
    source_token_indices: tuple[int, ...]
    token_texts: tuple[str, ...]
    token_ids: tuple[int | None, ...]
    ignored_token_indices: tuple[int, ...] = ()
    occurrence_index: int = 0
    match_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConceptMatchReport:
    matches: tuple[ConceptTokenMatch, ...]
    unmatched_terms: tuple[ConceptTermSpec, ...]
    ambiguous_terms: tuple[ConceptTermSpec, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _TokenRecord:
    token_index: int
    source_token_index: int
    token_id: int | None
    token_text: str
    token_source: str


def parse_concept_terms(spec: str) -> list[ConceptTermSpec]:
    terms: list[ConceptTermSpec] = []
    seen: set[tuple[str | None, str]] = set()
    for raw_part in re.split(r"[\n;,]+", str(spec or "")):
        raw = raw_part.strip()
        if not raw:
            continue
        source_filter, term = _split_source_prefix(raw)
        normalized = normalize_concept_text(term)
        if not normalized:
            continue
        key = (source_filter, normalized)
        if key in seen:
            continue
        seen.add(key)
        terms.append(ConceptTermSpec(raw=raw, term=term, normalized=normalized, source_filter=source_filter))
    return terms


def normalize_concept_text(value: str) -> str:
    if re.fullmatch(r"\s*<token:\d+>\s*", str(value or "")):
        return ""
    value = str(value or "").replace("▁", " ").replace("Ġ", " ")
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def build_concept_token_matches(
    concept_terms: str,
    token_text_map: dict[int, dict[str, Any]],
    *,
    allow_ambiguous: bool = False,
) -> ConceptMatchReport:
    terms = parse_concept_terms(concept_terms)
    if not terms or not token_text_map:
        return ConceptMatchReport(matches=(), unmatched_terms=tuple(terms), ambiguous_terms=(), warnings=())

    by_source = _tokens_by_source(token_text_map)
    matches: list[ConceptTokenMatch] = []
    unmatched: list[ConceptTermSpec] = []
    ambiguous: list[ConceptTermSpec] = []
    warnings: list[str] = []

    for term in terms:
        candidate_sources = [term.source_filter] if term.source_filter is not None else sorted(by_source)
        source_matches: list[ConceptTokenMatch] = []
        for source in candidate_sources:
            source_tokens = by_source.get(source or "")
            if not source_tokens:
                continue
            source_matches.extend(_find_source_matches(term, source or "", source_tokens))

        matched_sources = {match.token_source for match in source_matches}
        if not source_matches:
            unmatched.append(term)
            continue
        if term.source_filter is None and len(matched_sources) > 1 and not allow_ambiguous:
            ambiguous.append(term)
            warning = f"ambiguous_across_sources:{term.term}"
            if warning not in warnings:
                warnings.append(warning)
            continue
        matches.extend(source_matches)
        for match in source_matches:
            for warning in match.match_warnings:
                if warning not in warnings:
                    warnings.append(warning)

    return ConceptMatchReport(
        matches=tuple(matches),
        unmatched_terms=tuple(unmatched),
        ambiguous_terms=tuple(ambiguous),
        warnings=tuple(warnings),
    )


def _split_source_prefix(raw: str) -> tuple[str | None, str]:
    source, sep, term = raw.partition(":")
    if sep and source.strip() and term.strip():
        return source.strip(), term.strip()
    return None, raw.strip()


def _tokens_by_source(token_text_map: dict[int, dict[str, Any]]) -> dict[str, list[_TokenRecord]]:
    by_source: dict[str, list[_TokenRecord]] = {}
    source_counts: dict[str, int] = {}
    for token_index in sorted(token_text_map):
        meta = token_text_map[token_index]
        source = str(meta.get("token_source") or "")
        fallback_source_index = source_counts.get(source, 0)
        source_counts[source] = fallback_source_index + 1
        record = _TokenRecord(
            token_index=int(meta.get("token_index", token_index)),
            source_token_index=int(meta.get("source_token_index", fallback_source_index)),
            token_id=_optional_int(meta.get("token_id")),
            token_text=str(meta.get("token_text") or ""),
            token_source=source,
        )
        by_source.setdefault(source, []).append(record)
    return by_source


def _find_source_matches(term: ConceptTermSpec, source: str, tokens: list[_TokenRecord]) -> list[ConceptTokenMatch]:
    found: list[ConceptTokenMatch] = []
    occurrence_index = 0
    for start in range(len(tokens)):
        token_indices: list[int] = []
        source_token_indices: list[int] = []
        token_texts: list[str] = []
        token_ids: list[int | None] = []
        ignored_token_indices: list[int] = []
        combined = ""
        consumed_meaningful = False

        for token in tokens[start:]:
            piece = normalize_concept_text(token.token_text)
            if not piece:
                if consumed_meaningful and term.normalized.startswith(combined):
                    ignored_token_indices.append(token.token_index)
                    continue
                if not consumed_meaningful:
                    continue
                break

            next_combined = combined + piece
            if not term.normalized.startswith(next_combined):
                break

            combined = next_combined
            consumed_meaningful = True
            token_indices.append(token.token_index)
            source_token_indices.append(token.source_token_index)
            token_texts.append(token.token_text)
            token_ids.append(token.token_id)

            if combined == term.normalized:
                match_warnings = ()
                if ignored_token_indices:
                    match_warnings = ("punctuation_tokens_ignored_inside_match",)
                found.append(ConceptTokenMatch(
                    term=term.term,
                    normalized_term=term.normalized,
                    token_source=source,
                    token_indices=tuple(token_indices),
                    source_token_indices=tuple(source_token_indices),
                    token_texts=tuple(token_texts),
                    token_ids=tuple(token_ids),
                    ignored_token_indices=tuple(ignored_token_indices),
                    occurrence_index=occurrence_index,
                    match_warnings=match_warnings,
                ))
                occurrence_index += 1
                break
    return found


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
