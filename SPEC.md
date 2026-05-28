# SPEC.md — Codex Refactor Specification

Project: Anima ConceptAttention Survey for ComfyUI  
Focus: `concept_terms` heatmap correctness, maintainability, and safe refactoring  
Target reader: Codex or a developer applying incremental patches to the existing repository

---

## 0. Primary Objective

Refactor the current MVP without changing its observe-only behavior, while making `concept_terms` phrase heatmaps reliable, auditable, and testable.

The most important user-facing output is:

```text
heatmaps/concepts/aggregate/aggregate_<branch>_concept_<term>_preview.png
heatmaps/concepts/aggregate/manifest.json
JSONL attention_observation[].concept_scores
```

A correct `concept_terms` heatmap means:

1. The requested human phrase is matched to the intended prompt-token key positions.
2. The matched token positions are source-scoped and never silently joined across tokenizer streams.
3. The concept attention map is computed as summed attention probability mass over the matched token positions.
4. Raw `.npy` values and manifest statistics are based on the un-normalized latent-grid attention values.
5. PNG normalization is display-only and does not alter score calculations.
6. If the phrase cannot be matched or alignment is ambiguous, the system records an explicit diagnostic instead of silently producing a misleading heatmap.

---

## 1. Non-Negotiable Invariants

### 1.1 Observe-only invariant

The survey node must never edit model attention.

For every attention call:

```python
return original_func(*args, **kwargs)
```

must remain the final returned value. The observer may inspect tensors, write JSONL, and save files, but it must not modify `q`, `k`, `v`, `kwargs`, or the returned attention output.

Acceptance test:

- Existing exact passthrough test still passes.
- Add a test where the original attention backend returns a sentinel tensor; the override must return exactly that tensor.

### 1.2 Fail-safe invariant

When the observer cannot prove the concept heatmap is meaningful, it must not create a misleading concept heatmap.

Use one of these explicit outcomes:

- `attention_observation` with valid `concept_scores`
- `concept_unmatched` diagnostic event
- `concept_alignment_warning` diagnostic event
- `attention_fallback` for unsupported attention shapes/signatures

### 1.3 Backward-compatible public node invariant

Do not remove existing public inputs:

- `model`
- `clip`
- `mode`
- `capture_level`
- `prompt_text`
- `concept_terms`
- `target_call_indices`
- `diagnostic_call_indices`
- `branch_mode`
- `max_tokens`
- `max_steps`
- `jsonl_path`
- `save_heatmaps`
- `heatmap_dir`
- `heatmap_output`
- `max_logits_mib`
- `fail_mode`

Changing defaults requires an explicit update to this spec and README.

### 1.4 Schema compatibility invariant

Keep JSONL `schema_version = 1` unless a breaking schema change is introduced.

Add optional fields rather than removing or renaming existing fields.

---

## 2. Current MVP Summary

The repository currently implements:

- ComfyUI V3 node `Anima Concept Survey Model Patch`
- `optimized_attention_override` observer
- JSONL records for attention observations, skipped calls, fallback calls, and run summary
- token text recovery from connected `CLIP` and `prompt_text`
- top-token heatmaps
- `concept_terms` phrase heatmaps
- concept aggregate heatmaps
- report scripts and CSV/Markdown summary outputs
- synthetic unit tests for phrase matching and concept heatmap export

The current risk is not that the feature is absent. The historical risk was that correctness depended on a monolithic `survey_attention.py` implementation and a weak assumption that flattened `clip.tokenize(prompt_text)` order is identical to runtime attention key order.

The refactor now makes that assumption explicit, testable, and diagnosable through source-scoped matching, `source_token_index`, JSONL diagnostics, and runtime `text_len` guards. Manual ComfyUI validation is still required for real Anima workflows.

---

## 3. Definition of Correct `concept_terms` Behavior

### 3.1 Input parsing

`concept_terms` accepts one or more terms separated by newlines, semicolons, or commas.

Examples:

```text
big breasts
red hair; blue eyes
qwen3_06b:big breasts
```

Required parsing behavior:

- Trim surrounding whitespace.
- Remove empty entries.
- Deduplicate by normalized `(source_filter, normalized_term)` pair.
- Preserve the original display text in records and filenames.
- Support optional source prefix syntax:

```text
<token_source>:<term>
```

Examples:

```text
qwen3_06b:big breasts
clip_l:red hair
```

If no source prefix is supplied, the matcher searches each token source independently and chooses only unambiguous matches.

### 3.2 Token map semantics

Token text records must include enough information to audit matching.

Required token record fields:

```json
{
  "token_index": 0,
  "source_token_index": 0,
  "token_id": 123,
  "token_text": "big",
  "token_source": "qwen3_06b",
  "weight": 1.0
}
```

Definitions:

- `token_index`: global flattened prompt-token index, intended to correspond to runtime attention key index.
- `source_token_index`: token index within a tokenizer stream.
- `token_source`: tokenizer/encoder stream name, such as `qwen3_06b`, `t5xxl`, `clip_l`, `l`, or `tokens`.
- `token_text`: decoded token text or fallback placeholder.

If `source_token_index` is not available from the existing token flattening logic, assign it deterministically while flattening each source stream.

### 3.3 Source-scoped phrase matching

A concept phrase must match a contiguous or explicitly tolerated sequence within one token source.

The matcher must never combine tokens from different `token_source` values to form one concept.

For this token map:

```json
[
  {"token_index": 0, "source_token_index": 0, "token_text": "big", "token_source": "qwen"},
  {"token_index": 1, "source_token_index": 1, "token_text": " breasts", "token_source": "qwen"},
  {"token_index": 2, "source_token_index": 0, "token_text": "big", "token_source": "clip_l"}
]
```

`big breasts` may match indices `[0, 1]`; it must not match `[2, 1]`.

### 3.4 Normalization

Use a single shared normalization function for both terms and token text.

Normalization should:

- Lowercase.
- Treat common tokenizer word-start markers as spaces: `▁`, `Ġ`.
- Remove whitespace for phrase comparison.
- Remove punctuation for backward compatibility, but record ignored punctuation tokens when they appear inside a match.
- Preserve original `token_texts` in emitted records.

Recommended normalization:

```python
normalize_for_match(" breasts") -> "breasts"
normalize_for_match("big breasts") -> "bigbreasts"
normalize_for_match("▁big") -> "big"
normalize_for_match(",") -> ""
```

Important diagnostic rule:

- If punctuation-only or special tokens are skipped inside a match, include `ignored_token_indices` and `match_warnings` in the concept match record.

### 3.5 Duplicate occurrences

If the same concept phrase appears more than once in the same source, do not silently choose the first match.

Required behavior:

- Emit one match object per occurrence.
- Each occurrence has `occurrence_index` starting at 0.
- The aggregate concept heatmap for the term may sum all occurrences, but the manifest must list the individual occurrence token indices.

Example:

Prompt tokens:

```text
big breasts, small breasts
```

`concept_terms=breasts` should produce two occurrence records if both occurrences match.

### 3.6 Ambiguity policy

If no source prefix is specified and the same term matches multiple token sources, the matcher must mark the result as ambiguous.

Default behavior:

- Do not create a concept heatmap for ambiguous matches.
- Emit `concept_alignment_warning` or `concept_unmatched` with enough detail to explain the ambiguity.

Optional later behavior:

- Add a UI input for ambiguity handling.
- For now, use explicit source prefix to resolve ambiguity.

### 3.7 Runtime attention-key alignment

The core risk is token index alignment.

The system must verify the following before emitting concept heatmaps:

```text
max(matched token_index) < text_len from attention call
```

If this fails:

- Do not emit the concept score for that call.
- Emit diagnostic reason `concept_token_index_out_of_range`.

Also record these fields in run summary or diagnostics:

```json
{
  "token_text_count": 512,
  "attention_text_len_seen": [512],
  "concept_match_count": 1,
  "concept_unmatched_terms": [],
  "concept_ambiguous_terms": []
}
```

If `token_text_count != text_len`, this is not automatically fatal because some decoders may not expose every runtime token. However, concept matches are valid only if all matched token indices are in range.

### 3.8 Heatmap math

Given selected cross-attention probabilities:

```text
attention_probs shape = [selected_batch, heads, image_query, text_key]
```

For one concept match with token indices `I`:

```python
concept_mass = attention_probs.index_select(dim=-1, indices=I).sum(dim=-1)
concept_heatmap = concept_mass.mean(dim=(0, 1)).reshape(spatial)
score_mean = concept_mass.mean()
score_max = concept_mass.max()
score_entropy = normalized_entropy(concept_heatmap.flatten())
```

This is the intended MVP behavior.

Interpretation:

- The heatmap is the probability mass assigned to the phrase tokens at each latent image query position.
- It is not a final-image segmentation mask.
- It is not a separate ConceptAttention concept-vector stream.
- It is phrase-token attention over the already-tokenized `prompt_text`.

### 3.9 Branch handling

Concept heatmaps must respect `branch_mode` exactly like token heatmaps.

Expected behavior:

- `branch_mode=both`: produce positive and negative branch records when `cond_or_uncond` exposes both.
- `positive_only`: produce only positive branch records.
- `negative_only`: produce only negative branch records.
- If branch layout is unavailable and branch-specific mode is requested, fallback/diagnostic must explain this.

### 3.10 Output files

For `capture_level=heatmap`, `save_heatmaps=true`, and `heatmap_output=concepts_only`, concept output must be saved under:

```text
<heatmap_dir>/concepts/
<heatmap_dir>/concepts/aggregate/
```

Per-call concept files:

```text
concepts/step{step:03d}_call{call:03d}_{branch}_concept_{safe_term}.npy
concepts/step{step:03d}_call{call:03d}_{branch}_concept_{safe_term}.png
concepts/step{step:03d}_call{call:03d}_{branch}_concept_{safe_term}_preview.png
concepts/manifest.json
```

Aggregate concept files:

```text
concepts/aggregate/aggregate_{branch}_concept_{safe_term}.npy
concepts/aggregate/aggregate_{branch}_concept_{safe_term}.png
concepts/aggregate/aggregate_{branch}_concept_{safe_term}_preview.png
concepts/aggregate/manifest.json
```

Raw `.npy` values must be the unnormalized concept heatmap. PNGs may be normalized for display.

### 3.11 Manifest requirements

`concepts/manifest.json` rows must include:

```json
{
  "term": "big breasts",
  "normalized_term": "bigbreasts",
  "branch": "positive",
  "step_index": 0,
  "num_steps": 24,
  "eligible_call_index": 7,
  "spatial": [64, 64],
  "token_indices": [0, 1],
  "source_token_indices": [0, 1],
  "token_texts": ["big", " breasts"],
  "token_sources": ["qwen3_06b", "qwen3_06b"],
  "ignored_token_indices": [],
  "occurrence_index": 0,
  "score_mean": 0.01,
  "score_max": 0.08,
  "score_entropy": 0.75,
  "heatmap_mean": 0.01,
  "heatmap_max": 0.08,
  "heatmap_std": 0.02,
  "heatmap_max_over_mean": 8.0,
  "npy": "...",
  "png": "...",
  "preview_png": "..."
}
```

Aggregate manifest rows must include:

```json
{
  "term": "big breasts",
  "branch": "positive",
  "observation_count": 48,
  "match_count": 1,
  "matches": [
    {
      "occurrence_index": 0,
      "token_indices": [0, 1],
      "token_texts": ["big", " breasts"],
      "token_sources": ["qwen3_06b", "qwen3_06b"]
    }
  ],
  "score_mean": 0.01,
  "score_max": 0.08,
  "heatmap_mean": 0.01,
  "heatmap_max": 0.08,
  "heatmap_std": 0.02,
  "heatmap_max_over_mean": 8.0,
  "npy": "...",
  "png": "...",
  "preview_png": "..."
}
```

---

## 4. Refactored Architecture

The monolithic `survey_attention.py` has been split by responsibility. `survey_attention.py` remains as a backward-compatible re-export layer for existing imports; new implementation work should target the responsibility-specific modules below.

Target package layout:

```text
anima_concept_survey/
  __init__.py
  config.py
  paths.py
  progress.py
  branches.py
  selectors.py
  metadata.py
  token_text.py
  concepts.py
  scoring.py
  heatmaps.py
  records.py
  writer.py
  override.py
  survey_attention.py  # compatibility re-export surface
  reporting.py
scripts/
  parse_survey_log.py
  summarize_survey.py
nodes.py
```

### 4.1 `config.py`

Owns:

- constants: modes, capture levels, branch modes, fail modes, heatmap outputs
- `SurveyConfig`
- validation

No torch-heavy logic.

### 4.2 `paths.py`

Owns:

- ComfyUI output directory detection
- relative/absolute path resolution
- default JSONL and heatmap paths

### 4.3 `progress.py`

Owns:

- `ProgressInfo`
- `progress_from_sigmas()`

### 4.4 `branches.py`

Owns:

- `selected_branch_indices()`
- `branch_index_groups()`

### 4.5 `selectors.py`

Owns:

- call-index parsing
- square spatial inference
- logits memory estimate
- simple shape keys

### 4.6 `metadata.py`

Owns:

- model detection
- transformer metadata discovery
- safe metadata serialization

### 4.7 `token_text.py`

Owns:

- token flattening
- token text map construction
- source-local indexing
- decoder discovery

Must be extended to emit `source_token_index`.

### 4.8 `concepts.py`

Owns all `concept_terms` parsing and matching.

Required public API:

```python
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
```

Required functions:

```python
def parse_concept_terms(spec: str) -> list[ConceptTermSpec]: ...
def normalize_concept_text(value: str) -> str: ...
def build_concept_token_matches(
    concept_terms: str,
    token_text_map: dict[int, dict[str, Any]],
    *,
    allow_ambiguous: bool = False,
) -> ConceptMatchReport: ...
```

### 4.9 `scoring.py`

Owns:

- normalized entropy
- top-token scores
- concept score calculation

Required function:

```python
def concept_scores_from_attention(
    attention_probs: torch.Tensor,
    spatial: tuple[int, int],
    matches: Sequence[ConceptTokenMatch],
) -> list[dict[str, Any]]: ...
```

### 4.10 `heatmaps.py`

Owns:

- token heatmap extraction
- concept heatmap extraction
- raw `.npy` and display PNG writing
- aggregate accumulators
- manifest writing

Do not put phrase matching in this module.

### 4.11 `records.py`

Owns:

- JSONL record construction
- public filtering of private fields such as `_heatmap`
- schema helpers

### 4.12 `writer.py`

Owns:

- JSONL appending
- optional buffered writing
- safe directory creation

### 4.13 `override.py`

Owns only runtime orchestration:

1. Receive attention override call.
2. Validate shape/gates.
3. Compute observer attention probabilities under `torch.no_grad()`.
4. Ask scoring/heatmap/records modules to produce outputs.
5. Return original attention backend output exactly.

`override.py` is the runtime owner. It is smaller than the former monolithic implementation because matching, scoring, heatmap output, record construction, path handling, and writing live in separate modules.

---

## 5. JSONL Diagnostics for Concept Matching

Add diagnostic events so users can debug why no concept heatmap appeared.

### 5.1 `concept_match_summary`

Emit once during observer initialization or first observation.

```json
{
  "schema_version": 1,
  "event": "concept_match_summary",
  "run_id": "survey-...",
  "prompt_text": "big breasts,",
  "terms": ["big breasts"],
  "matches": [
    {
      "term": "big breasts",
      "token_source": "qwen3_06b",
      "token_indices": [0, 1],
      "token_texts": ["big", " breasts"],
      "occurrence_index": 0
    }
  ],
  "unmatched_terms": [],
  "ambiguous_terms": [],
  "warnings": []
}
```

### 5.2 `concept_unmatched`

Emit for unmatched terms.

```json
{
  "schema_version": 1,
  "event": "concept_unmatched",
  "run_id": "survey-...",
  "term": "big breasts",
  "normalized_term": "bigbreasts",
  "available_sources": ["qwen3_06b"],
  "reason": "no_contiguous_token_match"
}
```

### 5.3 `concept_alignment_warning`

Emit when a match exists but confidence is reduced.

Reasons:

- `ambiguous_across_sources`
- `concept_token_index_out_of_range`
- `token_text_count_mismatch_attention_text_len`
- `punctuation_tokens_ignored_inside_match`
- `decoded_token_text_fallback_used`

---

## 6. Testing Requirements

All tests must be automated and runnable with:

```bash
python -m pytest -q
```

### 6.1 Concept parser tests

Required cases:

- newline-separated terms
- semicolon-separated terms
- comma-separated terms
- deduped normalized terms
- source-prefixed terms
- empty terms ignored

### 6.2 Concept matching tests

Required cases:

- `big` + ` breasts` matches `big breasts`
- tokenizer marker `▁big` + `▁breasts` matches `big breasts`
- `Ġbig` + `Ġbreasts` matches `big breasts`
- no cross-source matching
- explicit source prefix chooses the requested source
- ambiguous cross-source match emits ambiguity
- duplicate occurrences produce multiple occurrence records
- punctuation-only token inside match is recorded as ignored
- fallback token text such as `<token:123>` does not accidentally match human terms

### 6.3 Concept heatmap math tests

Use small deterministic tensors.

Required tests:

- concept heatmap equals sum of individual matched token heatmaps
- concept score mean equals mean probability mass over matched tokens
- concept score max equals max probability mass over matched tokens
- concept entropy is computed from spatial concept heatmap
- out-of-range token index prevents concept score for that call
- raw `.npy` values equal the expected unnormalized heatmap
- manifest heatmap stats equal `.npy` stats exactly within tolerance

### 6.4 Integration tests

Required tests:

- `heatmap_output=concepts_only` writes concept files and does not write token top-k files.
- `heatmap_output=tokens_only` writes token files and does not write concept files.
- `heatmap_output=tokens_and_concepts` writes both.
- positive/negative branch separation is preserved.
- `concept_scores` appear in JSONL `attention_observation` records when matched.
- unmatched concept terms produce diagnostic JSONL events.
- observe mode returns original attention output exactly.

### 6.5 Reporting tests

Add concept-aware report tests:

Expected new outputs:

```text
survey_by_concept.csv
```

Minimum fields:

```text
term,branch,eligible_call_index,observation_count,score_mean,score_max,score_entropy,token_indices,token_texts,token_sources
```

`recommended_lora_targets.csv` may continue to work from token scores, but should not break when token scores are absent and only concept scores exist.

---

## 7. Manual Validation Protocol

Use this fixed ComfyUI protocol after automated tests pass.

### 7.1 Baseline invariance

1. Restart ComfyUI.
2. Run fixed seed/prompt/sampler without the survey node.
3. Save output image.
4. Run the same fixed seed/prompt/sampler with survey node:

```text
mode=observe
capture_level=heatmap
save_heatmaps=true
heatmap_output=concepts_only
prompt_text=<exact generation prompt>
concept_terms=big breasts
```

5. Confirm generated image is unchanged.

### 7.2 Concept output check

Confirm:

```text
heatmaps/concepts/aggregate/aggregate_positive_concept_big_breasts_preview.png
heatmaps/concepts/aggregate/manifest.json
```

exist.

Confirm JSONL contains:

- `concept_match_summary`
- `attention_observation` records with `concept_scores`
- no unexpected `concept_unmatched` for `big breasts`

### 7.3 Alignment check

Inspect JSONL and confirm:

```text
text_len >= max(token_indices) + 1
token_texts correspond to the intended phrase
token_sources are not mixed
```

### 7.4 Report check

Run:

```bash
python scripts/summarize_survey.py <survey.jsonl> --out-dir <report_dir> --top-k 32
```

Confirm:

- report generation succeeds
- `survey_by_concept.csv` exists
- concept rows include `big breasts`

---

## 8. Refactor Strategy

Use small, reversible patches.

Recommended sequence:

1. Add characterization tests against current behavior.
2. Extract pure helpers from `survey_attention.py` without changing behavior.
3. Add new concept data classes and tests.
4. Replace old concept group logic with new match report logic.
5. Add diagnostics for unmatched/ambiguous concepts.
6. Move heatmap writing and aggregation behind a small API.
7. Simplify `AnimaConceptSurveyAttentionOverride`.
8. Extend reporting to concept scores.
9. Run automated tests after every patch.

Avoid large rewrites that change architecture and behavior simultaneously.

---

## 9. Acceptance Criteria

The refactor is complete when all of the following are true:

- `python -m pytest -q` passes.
- Existing public node inputs remain available.
- Existing token heatmap behavior still works.
- `concept_terms=big breasts` maps to the intended prompt tokens in unit tests and manual ComfyUI validation.
- Concept heatmap raw values equal summed attention probability over matched token indices.
- Concept aggregate manifest stats match saved `.npy` values.
- Unmatched and ambiguous concept terms are visible in JSONL diagnostics.
- Observe mode output remains bitwise identical to the original backend return in tests.
- `survey_by_concept.csv` is generated from JSONL concept scores.
- README, SPEC, PROGRESS, and TASKS describe the final behavior.
