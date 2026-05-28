# PROGRESS.md — Codex Refactor Progress

Project: Anima ConceptAttention Survey for ComfyUI  
Focus: `concept_terms` heatmap correctness and modular refactor

---

## Current Status

Status: automated refactor is complete through documentation alignment. The highest-risk `concept_terms` matching path now has source-scoped matching, source-local token indices, explicit diagnostics, out-of-range guards, concept scoring tests, concept-aware reporting, and aligned README/SPEC/TASKS documentation.

The current repository already has:

- ComfyUI V3 node registration.
- `Anima Concept Survey Model Patch`.
- observe-only attention override.
- JSONL attention observation records.
- token heatmap export.
- aggregate token heatmaps.
- `concept_terms` input.
- phrase/concept heatmaps under `heatmaps/concepts`.
- aggregate concept heatmaps under `heatmaps/concepts/aggregate`.
- manifest statistics for saved heatmaps.
- report scripts.
- synthetic tests for concept phrase matching and concept heatmap export.

Main remaining unresolved issue:

- The system still relies on flattened `clip.tokenize(prompt_text)` global token indices matching runtime attention key order. This assumption is now auditable and guarded by runtime `text_len` checks, but still needs manual ComfyUI validation with actual Anima workflows.

---

## Decisions Carried Forward

### Existing MVP decisions

- Use ComfyUI `optimized_attention_override` for observation.
- Keep mode observe-only.
- Do not port full ConceptAttention concept-vector stream yet.
- Interpret MVP concept heatmaps as summed prompt-token cross-attention, not final-image segmentation.
- Keep JSONL output even in heatmap mode because image files alone lose token/call/step/branch metadata.
- Save phrase heatmaps under `heatmaps/concepts`.
- Use `heatmap_output=concepts_only` for phrase inspection without unrelated top-token heatmaps.

### New refactor decisions

- Treat `concept_terms` correctness as the highest priority.
- Split monolithic `survey_attention.py` into small modules.
- Add concept matching data classes.
- Add source-local token indexing.
- Add concept diagnostics for unmatched, ambiguous, and out-of-range terms.
- Do not silently create concept heatmaps for ambiguous source matches.
- Add `survey_by_concept.csv` to reporting.
- Keep JSONL schema version 1 and add optional fields rather than breaking existing records.

---

## Evidence Already Collected

### Runtime shape evidence

Known Anima/Cosmos cross-attention shape from manual observation:

```text
q=(2, 16, 4096, 128)
k=(2, 16, 512, 128)
v=(2, 16, 512, 128)
spatial=(64, 64)
```

This indicates image latent queries attending over 512 text key positions.

### Token text evidence

Manual run with:

```text
prompt_text="big breasts,"
```

produced decoded token records similar to:

```text
token_index=0, token_text="big", token_source="qwen3_06b"
token_index=1, token_text=" breasts", token_source="qwen3_06b"
```

This validates the expected `big breasts` unit-test fixture, but more diagnostics are needed for other CLIP/tokenizer variants.

### Test evidence

Existing tests include:

- call-index parser
- square spatial inference
- ComfyUI path resolution
- progress from sigmas
- branch selection
- observe-mode passthrough
- unsupported shape fallback
- token heatmap export
- concept phrase matching
- concept phrase heatmap export
- report aggregation

---

## Known Risks

### R1. Token index alignment risk

`token_index` from `clip.tokenize(prompt_text)` may not always match runtime `text_key` order.

Mitigation:

- Store `source_token_index`.
- Record token map in run summary.
- Verify matched token indices are within runtime `text_len`.
- Emit diagnostics on mismatch.
- Add manual validation using actual Anima JSONL.

### R2. Cross-source false match risk

The same phrase may decode from multiple tokenizer streams.

Mitigation:

- Match per source.
- Do not join across sources.
- Treat multi-source matches as ambiguous unless user supplies `source:term`.

### R3. Punctuation normalization risk

Current normalization can skip punctuation-only tokens, which may unintentionally match across punctuation.

Mitigation:

- Keep backward-compatible normalization.
- Record ignored punctuation token indices and warnings.
- Add tests.

### R4. Duplicate phrase risk

A concept may appear multiple times in the prompt.

Mitigation:

- Emit occurrence-indexed matches.
- Add tests for duplicate occurrences.

### R5. Heavy runtime I/O risk

Writing PNG/NPY during sampling may slow generation.

Mitigation:

- Keep current behavior for MVP.
- Move expensive rendering to postprocess later if needed.
- Ensure `concepts_only` avoids top-token heatmap noise.

### R6. Finalize lifecycle risk

`finalize()` may not always be called in real ComfyUI runs.

Mitigation:

- Write per-call concept heatmaps immediately as now.
- Ensure reports can work from attention observations without run summary.
- Later: design a safer lifecycle hook or report-time aggregation.

---

## Milestone Status

### M0. Planning Documents

Status: completed

- [x] Create Codex-focused SPEC.md.
- [x] Create Codex-focused PROGRESS.md.
- [x] Create Codex-focused TASKS.md.

### M1. Characterization Tests

Status: completed

Goal: preserve current behavior before moving code.

- [x] Add tests for current concept heatmap math.
- [x] Add tests for `concepts_only` output behavior.
- [x] Add tests for no cross-source matching.
- [x] Add tests for unmatched concept diagnostics after implementation.

### M2. Module Extraction Without Behavior Change

Status: completed

Goal: split pure helpers from `survey_attention.py` while keeping tests green.

Planned modules:

- `config.py` — completed.
- `paths.py` — completed.
- `progress.py` — completed.
- `branches.py` — completed.
- `selectors.py` — completed.
- `metadata.py` — completed.
- `concepts.py` — completed.
- `scoring.py` — completed.
- `heatmaps.py` — completed.
- `records.py` — completed.
- `writer.py` — completed.
- `override.py` — completed.

### M3. Concept Matching Rewrite

Status: completed

Goal: replace current `ConceptTokenGroup` logic with auditable match records.

Required additions:

- `ConceptTermSpec`
- `ConceptTokenMatch`
- `ConceptMatchReport`
- [x] source prefix parsing
- [x] source-local token indices
- [x] duplicate occurrence support
- [x] ambiguity diagnostics
- [x] ignored punctuation diagnostics

### M4. Concept Heatmap Correctness

Status: completed

Goal: prove concept heatmaps equal summed matched-token attention probability mass.

Required work:

- [x] Extract concept scoring into `scoring.py`.
- [x] Test deterministic tiny tensors.
- [x] Test `.npy` raw values.
- [x] Test manifest statistics.
- [x] Test branch separation.

### M5. JSONL Diagnostics

Status: completed

Goal: make missing or ambiguous concept heatmaps explainable.

Required events:

- [x] `concept_match_summary`
- [x] `concept_unmatched`
- [x] `concept_alignment_warning`

### M6. Reporting Extension

Status: completed

Goal: make report scripts useful for phrase-level analysis.

Implemented output:

```text
survey_by_concept.csv
```

Implemented updates:

- [x] include concept score summaries in `survey_summary.json`
- [x] include concept section in Markdown report
- [x] do not break reports when `token_scores` are empty

### M7. Manual ComfyUI Validation

Status: pending

Manual validation checklist:

- [ ] Restart ComfyUI after refactor.
- [ ] Run fixed seed baseline without survey node.
- [ ] Run fixed seed observe mode with survey node.
- [ ] Confirm output image does not change.
- [ ] Use `capture_level=heatmap`.
- [ ] Use `save_heatmaps=true`.
- [ ] Use `heatmap_output=concepts_only`.
- [ ] Use exact `prompt_text` from generation prompt.
- [ ] Use `concept_terms=big breasts`.
- [ ] Confirm aggregate concept preview exists.
- [ ] Confirm JSONL has `concept_match_summary`.
- [ ] Confirm JSONL has `attention_observation.concept_scores`.
- [ ] Confirm report script creates `survey_by_concept.csv`.

---

## Current Next Step for Codex

Continue with manual ComfyUI validation when a runtime is available.

Automated tests are green after module extraction and documentation alignment. Manual ComfyUI validation is still pending.

---

## Progress Log Template

Append future work here using this format:

```text
### YYYY-MM-DD — <short title>

Changed:
- ...

Tests:
- command: python -m pytest -q
- result: ...

Notes:
- ...

Next:
- ...
```

---

## Change Log

### 2026-05-28 — Codex refactor plan created

Changed:

- Added Codex-focused planning docs for concept heatmap correctness.
- Defined stricter meaning of correct `concept_terms` heatmaps.
- Defined module split plan.
- Defined diagnostics for unmatched and ambiguous concept terms.
- Defined concept-aware report requirements.

Tests:

- Not run. Documentation-only update.

Next:

- Add characterization tests before refactoring implementation.

### 2026-05-28 — P0-P5/P7 concept correctness pass

Changed:

- Added characterization tests for exact passthrough, `concepts_only` token-file suppression, deterministic concept heatmap math, unmatched diagnostics, and out-of-range concept guards.
- Extracted pure infrastructure modules: `config.py`, `paths.py`, `progress.py`, `branches.py`, `selectors.py`, and `metadata.py`.
- Added `source_token_index` to token flattening and token text records.
- Added `concepts.py` with source-prefixed parsing, source-scoped matching, duplicate occurrence records, ambiguity handling, fallback-token exclusion, and ignored punctuation diagnostics.
- Replaced runtime concept scoring inputs with `ConceptMatchReport` / `ConceptTokenMatch` while keeping `build_concept_token_groups` as a compatibility wrapper.
- Added `scoring.py` for token and concept score calculation.
- Added JSONL `concept_match_summary`, `concept_unmatched`, and `concept_alignment_warning` events.
- Added concept report aggregation and `survey_by_concept.csv`; reports tolerate empty `token_scores`.

Tests:

- command: python -m pytest -q
- result: 28 passed, 7 subtests passed

Notes:

- Runtime token-key alignment is now diagnosable and guarded by `text_len`, but still needs manual ComfyUI validation.
- `heatmaps.py`, `records.py`, `writer.py`, and `override.py` extraction remain.

Next:

- Extract heatmap writing and record/writer helpers without changing behavior.

### 2026-05-28 — Heatmap/record/writer/override extraction

Changed:

- Added `heatmaps.py` with heatmap accumulators, PNG/NPY writing, manifest writing, and aggregate save logic.
- Added `records.py` helpers for skipped/fallback records, concept diagnostics, public concept filtering, and run summaries.
- Added `writer.py` with `JsonlWriter`.
- Moved the runtime observer implementation to `override.py`.
- Kept `survey_attention.py` as a compatibility export surface for existing imports.
- Wrapped observer-side computation in `torch.no_grad()`.

Tests:

- command: python -m pytest -q
- result: 28 passed, 7 subtests passed

Notes:

- `survey_attention.py` is now a re-export layer; new implementation work should target the responsibility-specific modules.
- Manual ComfyUI validation remains pending.

Next:

- Update README/SPEC to match the final module layout and run manual validation when a ComfyUI runtime is available.

### 2026-05-28 — Documentation alignment and branch coverage

Changed:

- Added an automated concept heatmap branch-separation test for positive and negative branches.
- Updated README with source-prefixed `concept_terms`, diagnostics, concept heatmap interpretation, `survey_by_concept.csv`, package layout, and manual validation protocol.
- Updated SPEC to describe the implemented module split and `survey_attention.py` compatibility surface.
- Updated TASKS/PROGRESS completion states.

Tests:

- command: python -m pytest -q
- result: 29 passed, 7 subtests passed

Notes:

- Automated refactor work is complete against the local test suite.
- Manual ComfyUI validation remains the main remaining acceptance item.

Next:

- Run fixed-seed ComfyUI observe invariance and fresh `concept_terms=big breasts` validation.
