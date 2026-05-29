from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any, Iterable


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
    concept_uid: str = ""


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

        source_matches = list(dedupe_concept_matches(source_matches))
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
        matches=dedupe_concept_matches(matches),
        unmatched_terms=tuple(unmatched),
        ambiguous_terms=tuple(ambiguous),
        warnings=tuple(warnings),
    )


def dedupe_concept_matches(matches: Iterable[ConceptTokenMatch]) -> tuple[ConceptTokenMatch, ...]:
    unique: dict[tuple[str, str, tuple[int, ...], tuple[int, ...]], ConceptTokenMatch] = {}
    for match in matches:
        key = (
            match.normalized_term,
            match.token_source,
            tuple(match.token_indices),
            tuple(match.source_token_indices),
        )
        existing = unique.get(key)
        if existing is None:
            unique[key] = match
            continue
        unique[key] = replace(
            existing,
            ignored_token_indices=_merge_int_tuple(existing.ignored_token_indices, match.ignored_token_indices),
            match_warnings=_merge_str_tuple(existing.match_warnings, match.match_warnings),
        )

    counts: dict[tuple[str, str], int] = {}
    deduped = sorted(unique.values(), key=_match_sort_key)
    rows: list[ConceptTokenMatch] = []
    for match in deduped:
        occurrence_key = (match.normalized_term, match.token_source)
        occurrence_index = counts.get(occurrence_key, 0)
        counts[occurrence_key] = occurrence_index + 1
        rows.append(replace(
            match,
            occurrence_index=occurrence_index,
            concept_uid=make_concept_uid(
                normalized_term=match.normalized_term,
                token_source=match.token_source,
                occurrence_index=occurrence_index,
                token_indices=match.token_indices,
            ),
        ))
    return tuple(rows)


def make_concept_uid(
    *,
    normalized_term: str,
    token_source: str,
    occurrence_index: int,
    token_indices: tuple[int, ...],
) -> str:
    term = _slug_text(normalized_term) or "concept"
    source = _slug_text(token_source) or "source"
    span = _token_span_slug(token_indices)
    return f"{term}__{source}__occ{occurrence_index}__tok{span}"


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


def _match_sort_key(match: ConceptTokenMatch) -> tuple[int, int, str, str, tuple[int, ...], tuple[int, ...]]:
    first_token = match.token_indices[0] if match.token_indices else 10**12
    return (
        first_token,
        len(match.token_indices),
        match.normalized_term,
        match.token_source,
        match.token_indices,
        match.source_token_indices,
    )


def _merge_int_tuple(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(set(left).union(right)))


def _merge_str_tuple(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    merged: list[str] = []
    for value in (*left, *right):
        if value in seen:
            continue
        seen.add(value)
        merged.append(value)
    return tuple(merged)


def _slug_text(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return slug.strip("._-")


def _token_span_slug(token_indices: tuple[int, ...]) -> str:
    if not token_indices:
        return "none"
    if len(token_indices) == 1:
        return f"{token_indices[0]:03d}"
    return f"{token_indices[0]:03d}-{token_indices[-1]:03d}"
