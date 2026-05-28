# TASKS.md — Codex Refactor Tasks

Project: Anima ConceptAttention Survey for ComfyUI  
Focus: `concept_terms` heatmap correctness, diagnostics, and safe modular refactor

---

## Task Execution Rules for Codex

1. Work in small patches.
2. Run tests after each meaningful patch:

```bash
python -m pytest -q
```

3. Do not remove public node inputs.
4. Do not change observe-only output behavior.
5. Add tests before changing behavior.
6. Prefer pure functions and dataclasses for concept matching.
7. Keep JSONL `schema_version = 1`; add optional fields only.
8. Update this task file when a task is completed.

---

## P0 — Protect Current Behavior

### P0-T01 Add characterization tests for existing concept behavior

Priority: highest  
Status: completed

Objective:

Protect current `concept_terms` behavior before refactoring.

Add or extend tests to cover:

- `concept_terms="big breasts"` maps `big` + ` breasts` to token indices `[0, 1]`.
- `concept_scores` appear in `attention_observation` when `capture_level=heatmap` and concept terms match.
- `heatmap_output=concepts_only` writes concept files under `concepts/` and `concepts/aggregate/`.
- `heatmap_output=concepts_only` does not write root top-token heatmap files.
- aggregate concept manifest stats equal saved `.npy` stats.
- observe output remains exact passthrough.

Expected files:

- `tests/test_survey_attention.py` or new `tests/test_concepts_integration.py`

Acceptance:

- Tests fail only if current behavior is actually missing or broken.
- `python -m pytest -q` passes after test-compatible fixes.

---

### P0-T02 Add deterministic concept heatmap math test

Priority: highest  
Status: completed

Objective:

Prove that concept heatmap equals summed attention probability mass over matched tokens.

Use tiny tensors, for example:

```text
batch=1
heads=1
image_query=4
text_key=3
spatial=(2, 2)
concept token indices=[0, 2]
```

Compute expected value manually:

```python
expected = attention_probs[..., [0, 2]].sum(dim=-1).mean(dim=(0, 1)).reshape(2, 2)
```

Acceptance:

- Raw concept heatmap equals expected tensor within tolerance.
- `score_mean` equals `concept_mass.mean()`.
- `score_max` equals `concept_mass.max()`.
- saved `.npy` equals expected heatmap.

---

## P1 — Extract Pure Infrastructure Modules

### P1-T01 Extract `config.py`

Priority: high  
Status: completed

Move from `survey_attention.py`:

- `MODES`
- `CAPTURE_LEVELS`
- `BRANCH_MODES`
- `FAIL_MODES`
- `HEATMAP_OUTPUTS`
- default output path constants
- `SurveyConfig`

Update imports:

- `anima_concept_survey/__init__.py`
- `nodes.py`
- tests

Acceptance:

- No behavior change.
- `python -m pytest -q` passes.

---

### P1-T02 Extract `paths.py`

Priority: high  
Status: completed

Move:

- `comfy_output_dir()`
- `resolve_comfy_output_path()`
- `resolve_comfy_jsonl_path()`

Acceptance:

- Existing path resolution tests pass.
- `nodes.py` imports from the package public API or new module.

---

### P1-T03 Extract `progress.py`

Priority: high  
Status: completed

Move:

- `ProgressInfo`
- `progress_from_sigmas()`

Acceptance:

- Existing progress tests pass.

---

### P1-T04 Extract `branches.py`

Priority: high  
Status: completed

Move:

- `selected_branch_indices()`
- `branch_index_groups()`

Acceptance:

- Existing branch selection tests pass.
- New branch concept heatmap tests pass.

---

### P1-T05 Extract `selectors.py`

Priority: high  
Status: completed

Move:

- `parse_call_index_scope()`
- `infer_square_spatial_shape()`
- `estimate_logits_mib()`
- `shape_key()`

Acceptance:

- Existing parser and spatial tests pass.

---

### P1-T06 Extract `metadata.py`

Priority: medium  
Status: completed

Move:

- `_safe_metadata_value()`
- `discover_transformer_metadata()`
- `is_anima_like_model()`

Acceptance:

- Existing import behavior remains stable.
- Add a small unit test for Anima-like model detection if practical.

---

## P2 — Token Text Alignment Improvements

### P2-T01 Add `source_token_index` to token flattening

Priority: highest for concept correctness  
Status: completed

Objective:

Make concept matching auditable within each token source.

Update `token_text.py`:

- Add `source_token_index` to `TokenTextEntry`.
- Add `source_token_index` to `_FlatToken`.
- When flattening each source stream, assign source-local indices starting at 0.
- Keep existing global `token_index` behavior unchanged.

Acceptance:

- Existing token text tests pass after expected-field updates.
- New test confirms stream-local indices:

```text
source a: token_index 0,1 / source_token_index 0,1
source b: token_index 2 / source_token_index 0
```

---

### P2-T02 Record token map diagnostics in run summary

Priority: medium  
Status: completed

Objective:

Make token alignment inspectable from JSONL.

Update run summary to include:

- `token_text_count`
- `token_text_map`
- `token_sources`
- `concept_match_count`
- `concept_unmatched_terms`
- `concept_ambiguous_terms`

Acceptance:

- Existing run summary test still passes or is updated.
- JSONL contains source-local indices.

---

## P3 — Concept Matching Module

### P3-T01 Create `concepts.py` dataclasses and parser

Priority: highest  
Status: completed

Create:

```python
ConceptTermSpec
ConceptTokenMatch
ConceptMatchReport
parse_concept_terms()
normalize_concept_text()
```

Parser requirements:

- newline, semicolon, comma separation
- source prefix `<source>:<term>`
- normalized deduplication
- original display text preserved

Tests:

- parse multiline terms
- parse source-prefixed terms
- dedupe normalized duplicates
- ignore empty entries

Acceptance:

- New parser tests pass.

---

### P3-T02 Implement source-scoped concept matching

Priority: highest  
Status: completed

Create:

```python
build_concept_token_matches(concept_terms, token_text_map, allow_ambiguous=False)
```

Rules:

- Match only within one `token_source`.
- Never join tokens across sources.
- Use `normalize_concept_text()` for term and token pieces.
- Return all duplicate occurrences.
- If the same unsourced term matches multiple sources, mark ambiguous unless `allow_ambiguous=True`.
- Preserve actual `token_texts` and token IDs.

Tests:

- `big` + ` breasts` matches `big breasts`.
- `▁big` + `▁breasts` matches.
- `Ġbig` + `Ġbreasts` matches.
- no cross-source match.
- source prefix selects correct stream.
- duplicate occurrences produce multiple matches.
- ambiguous cross-source match is reported.

Acceptance:

- All concept matching tests pass.

---

### P3-T03 Add punctuation/special token diagnostics

Priority: high  
Status: completed

Objective:

Avoid silent false confidence when punctuation is skipped inside a match.

Behavior:

- If normalized token text is empty and skipped inside a match, record it in `ignored_token_indices`.
- Add `match_warnings=("punctuation_tokens_ignored_inside_match",)`.

Tests:

- A token sequence `big`, `,`, ` breasts` can match `big breasts` only with ignored index recorded.
- The emitted match includes the warning.

Acceptance:

- Match remains backward-compatible but auditable.

---

### P3-T04 Replace old `ConceptTokenGroup` usage

Priority: high  
Status: completed

Replace in the override:

- old `build_concept_token_groups()`
- old `ConceptTokenGroup`

with:

- `build_concept_token_matches()`
- `ConceptMatchReport`
- `ConceptTokenMatch`

Keep backward-compatible package exports if tests or users import old names:

- Option A: keep `build_concept_token_groups` as a wrapper returning the old shape for now.
- Option B: update tests and exports, but note breaking API in README.

Preferred: Option A for compatibility.

Acceptance:

- Existing concept phrase tests pass.
- New concept match tests pass.

---

## P4 — Concept Heatmap Scoring and Output

### P4-T01 Extract `scoring.py`

Priority: high  
Status: completed

Move:

- normalized entropy
- token score calculation
- concept score calculation

Required API:

```python
token_scores_from_attention(attention_probs, max_tokens)
concept_scores_from_attention(attention_probs, spatial, matches)
```

Acceptance:

- Existing token score tests pass.
- Deterministic concept heatmap math tests pass.

---

### P4-T02 Add out-of-range concept token guard

Priority: highest  
Status: completed

Objective:

Prevent invalid heatmaps if matched token indices exceed runtime attention text length.

Behavior:

- For each attention observation, check every concept match.
- If any `token_index >= text_len`, skip that match for the call.
- Emit `concept_alignment_warning` with reason `concept_token_index_out_of_range`.

Tests:

- token map match uses index 7 with `text_len=5`.
- no concept score emitted for that match.
- diagnostic event is written.

Acceptance:

- No index error.
- No misleading heatmap.

---

### P4-T03 Extract `heatmaps.py`

Priority: high  
Status: completed

Move:

- `_heatmap_for_token()`
- `_safe_filename_text()`
- `_colorize_heatmap()`
- `_heatmap_stats()`
- `HeatmapAccumulator`
- `ConceptHeatmapAccumulator`
- heatmap save functions
- aggregate save functions
- manifest writer

Do not move concept matching into this module.

Acceptance:

- Existing heatmap export tests pass.
- New concept heatmap math tests pass.

---

### P4-T04 Ensure `concepts_only` suppresses token heatmap files

Priority: high  
Status: completed

Objective:

For phrase inspection, output should not be polluted by unrelated top-token heatmaps.

Test:

- `heatmap_output=concepts_only`
- `max_tokens=2`
- `concept_terms` matched
- assert root heatmap directory has no `step*_token*.npy/png`
- assert `concepts/` files exist

Acceptance:

- Test passes.

---

## P5 — JSONL Records and Writer

### P5-T01 Extract `records.py`

Priority: medium  
Status: completed

Move record-building logic out of the override.

Create helpers for:

- attention observation records
- skipped records
- fallback records
- run summary records
- concept diagnostic records
- public concept score filtering

Acceptance:

- Existing JSONL tests pass.
- New concept diagnostic tests pass.

---

### P5-T02 Add concept diagnostic events

Priority: highest  
Status: completed

Implement JSONL events:

- `concept_match_summary`
- `concept_unmatched`
- `concept_alignment_warning`

Required behavior:

- Emit match summary once per run.
- Emit unmatched terms once per run.
- Emit alignment warning per relevant runtime call or summarized if repeated.

Tests:

- unmatched term emits `concept_unmatched`.
- ambiguous term emits `concept_alignment_warning` or appears in match summary ambiguity list.
- out-of-range match emits `concept_alignment_warning`.

Acceptance:

- Diagnostics make it clear why no heatmap was produced.

---

### P5-T03 Extract `writer.py`

Priority: medium  
Status: completed

Create `JsonlWriter` or equivalent.

Requirements:

- safe directory creation
- append mode
- compact JSON line output
- no-op behavior when path is empty
- optional logger mirror

Acceptance:

- Existing JSONL writing tests pass.

---

## P6 — Simplify Override

### P6-T01 Move runtime observer to `override.py`

Priority: high  
Status: completed

Objective:

Make `AnimaConceptSurveyAttentionOverride` orchestrate, not own all logic.

After refactor, `override.py` should mainly:

- parse/extract attention args
- apply fallback gates
- compute observer attention probabilities under `torch.no_grad()`
- call scoring/heatmap/record helpers
- return original backend result exactly

Acceptance:

- Class behavior unchanged from user perspective.
- Tests pass.
- File is significantly smaller and easier to read.

---

### P6-T02 Add `torch.no_grad()` around observer computation

Priority: medium  
Status: completed

Objective:

Avoid unnecessary autograd graph creation.

Behavior:

```python
with torch.no_grad():
    self._observe(...)
```

Acceptance:

- Tests pass.
- Observe output remains exact passthrough.

---

## P7 — Reporting Improvements

### P7-T01 Add concept summary aggregation

Priority: high  
Status: completed

Update `reporting.py`:

- Read `concept_scores` from attention observation records.
- Aggregate by `(term, branch, eligible_call_index)`.
- Include score mean, max, entropy, observation count, token indices, token texts, token sources.

Add output:

```text
survey_by_concept.csv
```

Acceptance:

- New reporting test passes.
- Existing report outputs still generated.

---

### P7-T02 Make reports robust when `token_scores` are empty

Priority: high  
Status: completed

Scenario:

`capture_level=heatmap` and `heatmap_output=concepts_only` may focus on concepts rather than top tokens.

Behavior:

- `survey_summary.json` still reports observation count.
- `recommended_lora_targets.csv` does not crash if token scores are empty.
- Markdown report includes concept section when concept scores exist.

Acceptance:

- Test with records containing only `concept_scores` passes.

---

## P8 — Public API and Docs

### P8-T01 Update package exports

Priority: medium  
Status: completed

Update `anima_concept_survey/__init__.py` to export stable APIs from new modules.

Keep compatibility for commonly imported names:

- `SurveyConfig`
- `AnimaConceptSurveyAttentionOverride`
- `parse_call_index_scope`
- `infer_square_spatial_shape`
- `progress_from_sigmas`
- `selected_branch_indices`
- `parse_concept_terms`
- `build_concept_token_groups` compatibility wrapper if retained
- `build_token_text_map`
- `flatten_tokenized`

Acceptance:

- Existing tests import successfully.
- ComfyUI `nodes.py` imports successfully.

---

### P8-T02 Update README

Priority: medium  
Status: completed

Document:

- source-prefixed `concept_terms`
- concept diagnostics
- interpretation of concept heatmaps
- ambiguity behavior
- `survey_by_concept.csv`
- manual validation protocol

Acceptance:

- README aligns with implementation.

---

### P8-T03 Update SPEC/PROGRESS/TASKS after implementation

Priority: medium  
Status: completed

After refactor, update:

- SPEC.md with final module structure and behavior
- PROGRESS.md with completed milestones
- TASKS.md with checked-off tasks

Acceptance:

- Docs reflect actual code.

---

## P9 — Manual Validation

### P9-T01 Fixed-seed observe invariance test in ComfyUI

Priority: highest after automated tests  
Status: not started

Steps:

1. Run baseline without survey node.
2. Run observe mode with survey node.
3. Same seed, prompt, sampler, model, dimensions.
4. Compare output image.

Acceptance:

- Image output is unchanged.

---

### P9-T02 Fresh `concept_terms` validation in ComfyUI

Priority: highest after automated tests  
Status: not started

Node settings:

```text
mode=observe
capture_level=heatmap
save_heatmaps=true
heatmap_output=concepts_only
prompt_text=<exact actual generation prompt>
concept_terms=big breasts
heatmap_dir=anima_concept_survey/heatmaps_big_breasts
```

Acceptance:

- `concepts/aggregate/aggregate_positive_concept_big_breasts_preview.png` exists.
- aggregate manifest includes `term=big breasts`.
- token indices and token texts correspond to the intended phrase.
- JSONL includes `concept_match_summary`.
- JSONL includes `attention_observation.concept_scores`.
- no unexpected concept ambiguity.

---

### P9-T03 Report script validation

Priority: high  
Status: not started

Run:

```bash
python scripts/summarize_survey.py <survey.jsonl> --out-dir <report_dir> --top-k 32
```

Acceptance:

- Existing report files are produced.
- `survey_by_concept.csv` is produced.
- Markdown report includes concept rows.

---

## Later / Optional

### L-T01 Move heavy PNG rendering to postprocess

Status: not started

Rationale:

Runtime PNG/NPY writes during sampling may be heavy. After correctness is proven, consider moving preview rendering and aggregate generation into a report/postprocess script.

---

### L-T02 Rectangular latent layouts

Status: not started

Current MVP supports square image query grids only. Rectangular support requires runtime spatial metadata or user-supplied width/height.

---

### L-T03 ConceptAttention-style concept vector stream

Status: not started

Current `concept_terms` sums already-tokenized prompt-token attention. A later phase may encode separate concept prompts and compare concept/image vectors, but this is out of scope for this refactor.

---

## Completion Checklist

The Codex refactor is done when:

- [x] All P0 tasks are complete.
- [x] All P1 extraction tasks are complete.
- [x] `source_token_index` is implemented.
- [x] `concepts.py` exists and is tested.
- [x] duplicate/ambiguous/source-prefixed concept matching works.
- [x] concept heatmap math is proven by deterministic tests.
- [x] out-of-range concept token indices are guarded.
- [x] concept diagnostics are emitted.
- [x] `override.py` is smaller and orchestration-focused.
- [x] `survey_by_concept.csv` is generated.
- [x] README explains concept heatmap interpretation and diagnostics.
- [x] `python -m pytest -q` passes.
- [ ] manual ComfyUI fixed-seed observe run does not change generated image.
- [ ] manual ComfyUI `concept_terms=big breasts` run creates the expected aggregate concept preview.
