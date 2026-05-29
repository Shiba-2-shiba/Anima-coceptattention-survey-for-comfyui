# PROGRESS.md — Dev Branch Follow-up Refactor Progress

Project: Anima ConceptAttention Survey for ComfyUI
Target branch: `dev`
Focus: make concept survey outputs reliable, run-isolated, and hard to misinterpret

---

## Current Status

Status: automated implementation is complete through P6 documentation alignment. Manual ComfyUI validation remains.

The current `dev` branch has completed the main modular refactor and the follow-up automated refactor for safer concept survey interpretation:

- duplicate same-span concept matches are removed;
- append-only JSONL reports are run-filtered by default;
- concept-specific target recommendations are generated;
- preview normalization and near-uniform/weak-focus warnings are reported;
- positive and negative branch interpretation is separated;
- two run IDs can be compared from JSONL alone.

---

## Evidence from ComfyUI Trials

### Trial 1: concept-only positive branch

Observed configuration:

```text
run_id=survey-21d49acdd90
branch_mode=positive_only
concept_term=big breasts
token_source=qwen3_06b
token_indices=[27, 28]
text_len=512
attention_observation=560
concept_score_rows=1120
token_score_rows=0
```

Findings:

- Qwen token matching worked.
- No unmatched concept terms were reported.
- No concept warnings were reported.
- The run is a useful baseline for concept-only behavior.
- Calls `9`, `10`, and `13` looked like first LoRA/slider target candidates.
- The same token span `[27, 28]` was duplicated as occurrence `0` and occurrence `1`, doubling concept row counts and aggregate observation counts.

### Trial 2: both branches with token heatmaps

Observed configuration:

```text
run_id=survey-21e5470fe90
branch_mode=both
concept_term=big breasts
token_source=qwen3_06b
token_indices=[24, 25]
positive_observations=560
negative_observations=560
token_score_rows=17920
concept_score_rows=2240
```

Findings:

- Qwen concept matching still worked.
- Positive concept attention was weaker and more diffuse than Trial 1.
- Negative concept heatmap was near-uniform.
- Token heatmaps added a large amount of visual noise for single-concept QA.
- Per-image min-max preview normalization made weak/near-uniform negative maps look visually louder than the raw values justify.

---

## Decisions Carried Forward

- Keep observe-only behavior.
- Keep JSONL `schema_version=1` and add optional fields only.
- Keep `concept_terms` as summed prompt-token attention, not a separate concept-vector stream.
- Treat `.npy` and manifest/JSONL stats as ground truth.
- Keep min-max preview PNGs for shape inspection, but warn that they are not comparable by absolute strength.
- Recommend `branch_mode=positive_only` and `heatmap_output=concepts_only` for initial single-concept localization checks.
- Reports should be run-isolated by default.
- Concept target recommendations should be separate from token target recommendations.

---

## Known Risks

### R1. Duplicate concept matches distort counts

The same token span can currently be emitted more than once. This inflates `concept_score_rows` and aggregate observation counts.

Mitigation plan:

- Deduplicate by normalized term, token source, global token span, and source-local span.
- Preserve true repeated phrases only when token spans differ.

### R2. Run mixing distorts reports

Append-only JSONL files can contain multiple runs.

Mitigation plan:

- Add `--run-id`, `--latest-run`, and `--list-runs`.
- Refuse silent multi-run aggregation by default.

### R3. Preview PNGs overstate weak signals

Min-max normalization is useful for seeing shape but misleading for strength comparisons.

Mitigation plan:

- Add preview normalization metadata.
- Add near-uniform warnings.
- Promote raw heatmap stats in reports.

### R4. Negative branch maps can be misread

Negative maps may be near-uniform but still look visually colored.

Mitigation plan:

- Separate positive targets from negative diagnostics.
- Add branch delta CSV.
- Penalize near-uniform negative maps in concept target ranking.

### R5. Concept-specific ranking is missing

Token-based target recommendations do not help concept-only runs.

Mitigation plan:

- Add `recommended_concept_targets.csv` using concept score and focus metrics.

### R6. Runtime attention key alignment remains an audit concern

The Qwen trials indicate current index behavior works for that workflow, but multi-encoder workflows may need a stronger resolver.

Mitigation plan:

- Preserve and report source-local indices.
- Add optional `attention_key_indices`, `alignment_strategy`, and `alignment_confidence` when feasible.
- Keep full resolver as later work unless new validation reveals a mismatch.

---

## Milestone Status

### M0. Planning Documents

Status: completed

- [x] Create follow-up `SPEC.md`.
- [x] Create follow-up `TASKS.md`.
- [x] Create follow-up `PROGRESS.md`.

### M1. Duplicate Concept Fixes

Status: completed

Goal:

- Remove same-span duplicate concept matches.
- Preserve true repeated phrases.
- Add stable concept identity.
- Prevent concept heatmap filename/aggregate/report collisions.

Planned:

- [x] Add duplicate same-span regression tests.
- [x] Add repeated phrase preservation tests.
- [x] Implement match dedupe.
- [x] Add `concept_uid`.
- [x] Update heatmap accumulator keys and filenames.
- [x] Update report grouping keys.

### M2. Run-Filtered Reporting

Status: completed

Goal:

- Make append-only JSONL safe to analyze.

Planned:

- [x] Add run discovery utilities.
- [x] Add `--list-runs`.
- [x] Add `--run-id`.
- [x] Add `--latest-run`.
- [x] Add `--allow-mixed-runs`.
- [x] Generate `survey_runs.csv`.
- [x] Add run metadata to JSON/Markdown reports.

### M3. Concept Metrics and Recommendations

Status: completed

Goal:

- Make concept-only runs useful for LoRA/slider target selection.

Planned:

- [x] Add heatmap focus metrics to JSONL concept scores.
- [x] Add uniform baseline metrics.
- [x] Extend `survey_by_concept.csv`.
- [x] Add `recommended_concept_targets.csv`.
- [x] Add concept target Markdown section.

### M4. Preview Safety

Status: completed

Goal:

- Prevent min-max preview PNGs from being mistaken for absolute-strength maps.

Planned:

- [x] Add `preview_normalization` to manifests.
- [x] Add near-uniform warnings.
- [x] Document interpretation in README.
- [ ] Optionally add fixed-scale preview postprocess later.

### M5. Branch Guidance and Delta

Status: completed

Goal:

- Make positive/negative branch behavior clear.

Planned:

- [x] Add `survey_branch_concept_delta.csv`.
- [x] Add positive concept target section.
- [x] Add negative concept diagnostic section.
- [x] Add branch interpretation labels.

### M6. Run Comparison

Status: completed

Goal:

- Make Trial 1 vs Trial 2 style analysis scriptable.

Planned:

- [x] Add `scripts/compare_survey_runs.py`.
- [x] Add `compare_summary.json`.
- [x] Add concept delta CSVs.
- [x] Add Markdown comparison report.

### M7. Documentation Alignment

Status: completed

Goal:

- Align docs with new reporting behavior.

Planned:

- [x] Update README.
- [x] Update project `SPEC.md`.
- [x] Update project `TASKS.md`.
- [x] Update project `PROGRESS.md`.

### M8. Manual Validation

Status: not started

Goal:

- Confirm fixes in ComfyUI.

Planned:

- [ ] Fresh positive-only concept run.
- [ ] Confirm no duplicate same-span occurrence.
- [ ] Confirm aggregate counts are not doubled.
- [ ] Confirm run filtering on append-only JSONL.
- [ ] Confirm concept recommendation output.
- [ ] Confirm near-uniform warnings for both-branch run.

---

## Current Next Step for Codex

Start with M8 / P7:

1. Run a fresh positive-only concept validation in ComfyUI.
2. Validate run-filtered reporting on an append-only JSONL containing two runs.
3. Validate concept recommendations and near-uniform warnings on a both-branch run.

P0-P6 are now in place; manual ComfyUI validation remains.

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

### 2026-05-29 — P6 documentation alignment completed

Changed:

- Updated README with recommended concept-only workflow settings, `max_tokens` behavior, run filtering examples, and run comparison examples.
- Updated SPEC completion status to show automated P0-P6 completion and remaining manual validation.
- Marked P6 documentation tasks complete in TASKS and PROGRESS.

Tests:

- command: python -m pytest -q
- result: 43 passed, 7 subtests passed

Next:

- Start M8 / P7 manual ComfyUI validation.

### 2026-05-29 — P5 run comparison reporting completed

Changed:

- Added `scripts/compare_survey_runs.py` for JSONL-only comparison of two run IDs.
- Added comparison outputs: `compare_summary.json`, `concept_score_delta.csv`, `concept_call_delta.csv`, `branch_delta.csv`, `prompt_token_span_delta.csv`, and `compare_report.md`.
- Added comparison logic for concept score/focus deltas, call-level deltas, branch deltas, and prompt/token span differences.
- Added CLI smoke test for comparing a positive-only stronger run against a both-branch weaker/near-uniform run.

Tests:

- command: python -m pytest tests\test_reporting.py -q
- result: 13 passed
- command: python -m pytest -q
- result: 43 passed, 7 subtests passed

Next:

- Start M7 / P6 documentation alignment.

### 2026-05-29 — P4 branch delta reporting completed

Changed:

- Added `survey_branch_concept_delta.csv` comparing positive and negative concept rows by call and concept identity.
- Added branch interpretation labels: `positive-localized`, `negative-uniform`, `both-diffuse`, and `branch-ambiguous`.
- Split Markdown into Positive Concept Targets, Negative Concept Diagnostics, and Branch Delta sections.
- Updated README branch guidance for `positive_only`, `both`, and negative-branch preview interpretation.

Tests:

- command: python -m pytest tests\test_reporting.py -q
- result: 12 passed
- command: python -m pytest -q
- result: 42 passed, 7 subtests passed

Next:

- Start M6 / P5 run comparison reporting.

### 2026-05-29 — P3 preview safety warnings completed

Changed:

- Added `preview_normalization=per_file_minmax` to token, concept, and aggregate heatmap manifests.
- Added near-uniform and weak-focus preview warnings to report summaries and Markdown reports.
- Updated README guidance to explain per-file min-max preview normalization and raw-stat interpretation.

Tests:

- command: python -m pytest tests\test_reporting.py tests\test_survey_attention.py -q
- result: 27 passed, 5 subtests passed
- command: python -m pytest -q
- result: 41 passed, 7 subtests passed

Next:

- Start M5 / P4 branch concept delta reporting and branch-specific Markdown sections.

### 2026-05-29 — P2 concept metrics and recommendations completed

Changed:

- Added concept heatmap stats, uniform baseline, mean-over-uniform, attention key indices, and near-uniform flags to JSONL concept scores.
- Extended `survey_by_concept.csv` with concept identity, source-local indices, heatmap focus metrics, uniform baseline, and near-uniform status.
- Added `recommended_concept_targets.csv` with the specified heuristic ranking formula.
- Added a Recommended Concept Targets section to Markdown reports.

Tests:

- command: python -m pytest tests\test_reporting.py tests\test_survey_attention.py -q
- result: 26 passed, 5 subtests passed
- command: python -m pytest -q
- result: 40 passed, 7 subtests passed

Next:

- Start M4 / P3 preview normalization metadata and near-uniform warnings.

### 2026-05-29 — P1 run-filtered reporting completed

Changed:

- Added `RunInfo`, run discovery, latest-run detection, and run-id filtering helpers.
- Updated `summarize_survey.py` with `--list-runs`, `--run-id`, `--latest-run`, and `--allow-mixed-runs`.
- Made multi-run JSONL summarization fail by default unless a run filter or explicit mixed-run mode is supplied.
- Added `survey_runs.csv` plus selected/available run metadata in JSON and Markdown reports.

Tests:

- command: python -m pytest tests\test_reporting.py -q
- result: 9 passed
- command: python -m pytest -q
- result: 39 passed, 7 subtests passed

Next:

- Start M3 / P2 concept heatmap metrics and recommendations.

### 2026-05-29 — P0 concept identity and dedupe completed

Changed:

- Added regression tests for duplicate same-span concept matches and repeated phrase preservation.
- Added concept match dedupe keyed by normalized term, token source, global token span, and source-local token span.
- Added stable `concept_uid` to concept matches, JSONL concept scores, concept diagnostics, heatmap manifests, aggregate manifests, and report rows.
- Updated concept heatmap filenames, aggregate keys, and report grouping to keep distinct concept identities separate.

Tests:

- command: python -m pytest -q
- result: 34 passed, 7 subtests passed

Next:

- Start M2 / P1 run discovery and run-filtered reporting.

### 2026-05-28 — Follow-up refactor plan created

Changed:

- Added follow-up planning docs based on ComfyUI trial evidence.
- Prioritized duplicate concept match fix.
- Defined run-filtered reporting behavior.
- Defined concept-specific target recommendation output.
- Defined preview normalization warnings.
- Defined branch delta and run comparison outputs.

Tests:

- Not run. Documentation-only update.

Next:

- Add duplicate same-span concept match regression tests.
