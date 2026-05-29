# SPEC.md — Dev Branch Follow-up Refactor Specification

Project: Anima ConceptAttention Survey for ComfyUI
Branch target: current `dev` branch after the modular refactor
Primary goal: make `concept_terms` survey outputs reliable, run-isolated, and hard to misinterpret

Implementation status as of 2026-05-29:

- Automated implementation is complete for duplicate concept dedupe, stable concept identity, run-filtered reporting, concept target ranking, preview warnings, branch delta reporting, and JSONL-only run comparison.
- Automated validation passes with `python -m pytest -q`.
- Fresh ComfyUI manual validation is still pending and remains the final validation step.

---

## 1. Background

The current `dev` branch has moved beyond the first MVP. It already supports:

- observe-only Anima/Cosmos cross-attention inspection through `optimized_attention_override`;
- modularized survey implementation;
- source-scoped `concept_terms` matching;
- phrase/concept heatmaps under `heatmaps/concepts`;
- concept match diagnostics;
- concept-aware reporting output such as `survey_by_concept.csv`.

Two local ComfyUI trials were run against a Qwen-based Anima workflow. Both trials resolved `big breasts` against the `qwen3_06b` stream and emitted concept heatmaps.

Trial 1 was a clean concept-only positive-branch run:

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

Trial 1 produced useful concept-target candidates. Calls `9`, `10`, and `13` were the first candidates for LoRA/slider target exploration.

Trial 2 used both branches and saved token heatmaps as well as concept heatmaps:

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

Trial 2 demonstrated that token heatmaps are noisy for single-concept QA, and that negative-branch preview PNGs can look visually strong even when raw heatmap values are nearly uniform.

This follow-up refactor focuses on making the generated data safer to interpret, not on producing more heatmaps.

---

## 2. Non-negotiable Constraints

### 2.1 Observe-only behavior

The survey node must never change generated output.

- `AnimaConceptSurveyAttentionOverride.__call__()` must continue to return exactly the original attention backend output.
- Any observer failure must fallback to the original backend unless `fail_mode=raise`.
- Existing observe passthrough tests must continue to pass.

### 2.2 JSONL compatibility

Keep `schema_version = 1`.

Allowed:

- add optional fields;
- add optional JSONL events;
- add new CSV/Markdown outputs.

Avoid:

- deleting existing fields;
- changing existing field meanings;
- requiring old logs to contain new fields.

### 2.3 Public node stability

Do not remove public node inputs.

Allowed:

- add advanced inputs only if clearly useful;
- improve tooltips;
- change documentation defaults/recommendations.

### 2.4 Raw arrays are ground truth

Preview PNGs are display aids only.

- `.npy` heatmaps and manifest/JSONL stats are the truth for comparisons.
- Per-image min-max preview normalization must be documented and reported as potentially misleading.

### 2.5 Concept heatmap definition remains unchanged

For matched concept token indices, the concept heatmap is:

```python
concept_mass = attention_probs[..., attention_key_indices].sum(dim=-1)
concept_heatmap = concept_mass.mean(dim=(0, 1)).reshape(spatial)
```

This is prompt-token cross-attention mass. It is not a final-image segmentation mask and not a separate ConceptAttention concept-vector stream.

---

## 3. Problems Addressed by This Refactor

### Problem A: duplicate concept occurrence records

Observed issue:

- Trial 1 emitted the same Qwen token span `[27, 28]` as occurrence `0` and occurrence `1`.
- The duplicated rows had the same token span and same scores.
- This doubled concept row counts and aggregate observation counts.

Implemented behavior:

- Deduplicate matches that resolve to the exact same token span.
- Preserve true repeated phrase occurrences only when they map to distinct token spans.

### Problem B: JSONL run mixing

`survey.jsonl` is append-only. If multiple runs are written to the same file, the current report can aggregate all runs together.

Implemented behavior:

- Add run discovery and run filtering.
- Do not silently mix multiple `run_id`s by default.

### Problem C: concept-specific target ranking is missing

`recommended_lora_targets.csv` is token-score oriented. It is not useful for concept-only runs where `token_scores` may be empty.

Implemented behavior:

- Add `recommended_concept_targets.csv`.
- Rank by concept score and heatmap focus metrics.

### Problem D: preview PNGs can overstate weak signals

Current previews normalize each heatmap independently by min/max. Near-uniform negative maps can therefore look visually dramatic.

Implemented behavior:

- Record preview normalization metadata.
- Add near-uniform warnings in manifests/reports.
- Prefer absolute stats for interpretation.

### Problem E: branch interpretation is under-specified

Negative branch maps can be useful diagnostics, but are often not meaningful localization maps.

Implemented behavior:

- Separate positive/negative concept sections in reports.
- Add branch-delta outputs.
- Recommend `positive_only` for single-concept localization runs.

### Problem F: run comparison is manual

Trial 1 vs Trial 2 differences are useful but currently require manual analysis.

Implemented behavior:

- Add comparison script or report mode for two run IDs.

---

## 4. Required Data Model Additions

### 4.1 Concept match dedupe key

Concept matching must deduplicate identical spans using a key equivalent to:

```python
ConceptSpanKey = tuple[
    str,          # normalized_term
    str,          # token_source
    tuple[int],   # token_indices
    tuple[int],   # source_token_indices
]
```

Optional additions to the key are acceptable if stable:

- `token_ids`
- `source_filter`

Do not dedupe distinct token spans. Repeated text in the prompt should still produce separate matches if it resolves to different token indices.

### 4.2 Stable concept identity

Every concept score and concept heatmap should expose a stable identity.

Required public fields:

```json
{
  "concept_uid": "big_breasts__qwen3_06b__occ0__tok027-028",
  "term": "big breasts",
  "normalized_term": "bigbreasts",
  "token_source": "qwen3_06b",
  "occurrence_index": 0,
  "token_indices": [27, 28],
  "source_token_indices": [27, 28],
  "token_texts": [" big", " breasts"],
  "token_ids": [123, 456]
}
```

`concept_uid` does not need to be exactly this string, but it must be:

- stable within a run;
- filename-safe or convertible to a filename-safe slug;
- unique across same-term duplicate/source-specific matches;
- included in JSONL concept scores, heatmap manifests, aggregate manifests, and report rows.

### 4.3 Attention key indices

The current Qwen trials show that using token indices works for the tested workflow. However, keep the design ready for multi-source encoders.

Add optional fields when feasible:

```json
{
  "attention_key_indices": [27, 28],
  "alignment_strategy": "global_token_index",
  "alignment_confidence": "observed_qwen_text_len_512"
}
```

Minimum acceptable behavior for this follow-up refactor:

- preserve `source_token_indices`;
- use current `token_indices` behavior unless a safer resolver is implemented;
- include enough metadata to audit the mapping.

Preferred future resolver:

```text
if text_len == len(token_text_map):
    use global token_indices
elif text_len == count(tokens in token_source):
    use source_token_indices
elif known source offsets can explain text_len:
    use source_offset + source_token_indices
else:
    emit concept_alignment_warning and skip heatmap
```

### 4.4 Concept heatmap statistics in JSONL

For concept targets, the report should not have to read PNG/NPY manifests. Add heatmap stats directly to `attention_observation.concept_scores`.

Required fields:

```json
{
  "heatmap_mean": 0.0027,
  "heatmap_max": 0.0164,
  "heatmap_std": 0.0008,
  "heatmap_max_over_mean": 2.41,
  "uniform_baseline": 0.00390625,
  "score_mean_over_uniform": 0.69,
  "near_uniform": false
}
```

Notes:

- `heatmap_mean` should usually equal or nearly equal `score_mean`.
- `heatmap_max` is max over the averaged spatial heatmap.
- `score_max` remains max over raw concept attention mass before batch/head averaging.
- `uniform_baseline = len(attention_key_indices or token_indices) / text_len`.

---

## 5. Heatmap Output Requirements

### 5.1 Concept file names

Concept heatmap file names must avoid collisions for duplicate/source-specific terms.

Recommended stem:

```text
step{step:03d}_call{call:03d}_{branch}_concept_{source}_{term}_occ{occ}_tok{span}
```

Example:

```text
step004_call009_positive_concept_qwen3_06b_big_breasts_occ0_tok027-028.npy
```

Aggregate example:

```text
aggregate_positive_concept_qwen3_06b_big_breasts_occ0_tok027-028_preview.png
```

### 5.2 Concept accumulator key

Do not aggregate concepts by `(branch, term)` only.

Required key should include at least:

```python
(
    branch,
    concept_uid,
)
```

or equivalent:

```python
(
    branch,
    normalized_term,
    token_source,
    occurrence_index,
    tuple(token_indices),
)
```

### 5.3 Manifest fields

Each concept manifest row should include:

```json
{
  "concept_uid": "...",
  "term": "big breasts",
  "normalized_term": "bigbreasts",
  "token_source": "qwen3_06b",
  "occurrence_index": 0,
  "token_indices": [27, 28],
  "source_token_indices": [27, 28],
  "attention_key_indices": [27, 28],
  "token_texts": [" big", " breasts"],
  "branch": "positive",
  "eligible_call_index": 9,
  "step_index": 4,
  "preview_normalization": "per_file_minmax",
  "heatmap_mean": 0.0027,
  "heatmap_max": 0.0164,
  "heatmap_std": 0.0008,
  "heatmap_max_over_mean": 2.41,
  "uniform_baseline": 0.00390625,
  "score_mean_over_uniform": 0.69,
  "near_uniform": false
}
```

---

## 6. Reporting Requirements

### 6.1 Run discovery and filtering

Add utility functions:

```python
list_runs(records) -> list[RunInfo]
filter_records_by_run_id(records, run_id) -> list[dict]
latest_run_id(records) -> str | None
```

`RunInfo` should include:

```json
{
  "run_id": "survey-...",
  "first_record_index": 0,
  "last_record_index": 1234,
  "events": {"attention_observation": 560},
  "prompt_text": "...",
  "branches": ["positive"],
  "concept_terms": ["big breasts"],
  "observation_count": 560
}
```

### 6.2 CLI behavior

Update `scripts/summarize_survey.py`:

```powershell
python scripts\summarize_survey.py survey.jsonl --out-dir report --run-id survey-21d49acdd90
python scripts\summarize_survey.py survey.jsonl --out-dir report --latest-run
python scripts\summarize_survey.py survey.jsonl --list-runs
python scripts\summarize_survey.py survey.jsonl --out-dir report --allow-mixed-runs
```

Default behavior:

- If exactly one `run_id` exists, summarize it.
- If multiple `run_id`s exist and no run filter is supplied, fail with a clear message listing available run IDs.
- `--allow-mixed-runs` is required to intentionally aggregate multiple runs.

### 6.3 Report outputs

Existing outputs remain:

```text
survey_summary.json
survey_by_call.csv
survey_by_step.csv
survey_by_branch.csv
survey_by_token.csv
survey_by_concept.csv
recommended_lora_targets.csv
survey_report.md
```

Add:

```text
survey_runs.csv
recommended_concept_targets.csv
survey_branch_concept_delta.csv
```

Optional later:

```text
survey_preview_warnings.csv
```

### 6.4 Concept target ranking

`recommended_concept_targets.csv` should rank concept-specific call targets.

Required columns:

```text
rank
rank_score
term
concept_uid
token_source
occurrence_index
branch
eligible_call_index
observation_count
score_mean
score_max
score_entropy
heatmap_mean
heatmap_max
heatmap_std
heatmap_max_over_mean
uniform_baseline
score_mean_over_uniform
near_uniform
target_hint
```

Initial ranking formula:

```python
uniform_baseline = token_count / text_len
mean_over_uniform = score_mean / uniform_baseline if uniform_baseline else None
focus_factor = max(0.0, heatmap_max_over_mean - 1.0)
focus_factor = min(focus_factor, 4.0)
near_uniform_penalty = 0.25 if heatmap_max_over_mean < 1.05 else 1.0
branch_factor = 1.0 if branch == "positive" else 0.5
rank_score = mean_over_uniform * (1.0 + focus_factor) * near_uniform_penalty * branch_factor
```

This formula is intentionally simple. It should be documented as a heuristic, not a scientific metric.

### 6.5 Near-uniform warning

A concept heatmap is near-uniform when:

```python
heatmap_max_over_mean < 1.05
```

Weak focus warning:

```python
1.05 <= heatmap_max_over_mean < 1.15
```

Markdown report should include warnings such as:

```text
Warning: negative / big breasts / call 13 is near-uniform.
Preview PNG is min-max normalized and may overstate localization.
Use raw .npy and heatmap_max_over_mean for interpretation.
```

### 6.6 Branch delta output

Add `survey_branch_concept_delta.csv` grouped by:

```text
term
concept_uid or term/token_source/occurrence/token span
eligible_call_index
```

Columns:

```text
term
concept_uid
eligible_call_index
positive_score_mean
negative_score_mean
positive_heatmap_max_over_mean
negative_heatmap_max_over_mean
pos_minus_neg_mean
pos_focus_minus_neg_focus
interpretation
```

Interpretation rules:

```text
positive-localized: positive focus >= 1.15 and positive mean > negative mean
negative-uniform: negative focus < 1.05
both-diffuse: both focus < 1.15
branch-ambiguous: otherwise
```

---

## 7. Comparison Report Requirements

Add a new script:

```powershell
python scripts\compare_survey_runs.py survey.jsonl ^
  --run-a survey-21d49acdd90 ^
  --run-b survey-21e5470fe90 ^
  --out-dir compare_big_breasts
```

Outputs:

```text
compare_summary.json
concept_score_delta.csv
concept_call_delta.csv
branch_delta.csv
prompt_token_span_delta.csv
compare_report.md
```

Minimum useful comparison:

- concept mean delta;
- concept focus delta;
- top call changes;
- branch mode differences;
- token span/source differences;
- prompt summary differences.

The script should work from JSONL alone. It does not need to inspect image files.

---

## 8. Documentation Requirements

Update README with:

- recommended concept-only workflow;
- `max_tokens` explanation: it controls top-token heatmap output only and does not sharpen concept heatmaps;
- warning that preview PNGs use per-file min-max normalization;
- instruction to inspect `manifest.json`, `.npy`, and `heatmap_max_over_mean` before trusting preview colors;
- run filtering examples;
- `recommended_concept_targets.csv` interpretation;
- branch guidance: use `positive_only` first for concept localization.

Recommended workflow text:

```text
For one short concept such as "big breasts":

mode=observe
capture_level=heatmap
branch_mode=positive_only
save_heatmaps=true
heatmap_output=concepts_only
prompt_text=<exact generation prompt>
concept_terms=big breasts
jsonl_path=<one file per run, or use --run-id later>
heatmap_dir=<one directory per run>
```

---

## 9. Testing Requirements

Every behavior change must be covered by tests.

Required tests:

1. Duplicate match dedupe:
   - leading punctuation/special token should not create duplicate same-span matches.
2. Repeated phrase preservation:
   - `big breasts, big breasts` should produce two matches only if token spans differ.
3. Concept identity:
   - same term in different sources or occurrences must have different `concept_uid`.
4. Concept heatmap file names:
   - same term but different occurrence/source does not overwrite files.
5. Concept accumulator key:
   - aggregate manifests keep distinct rows for distinct concept identities.
6. Run discovery:
   - detect two run IDs in one JSONL.
7. Report default safety:
   - multiple run IDs without filter raises or returns an explicit error.
8. `--run-id`:
   - summarizes only selected run.
9. `--latest-run`:
   - summarizes last run by file order.
10. Concept JSONL stats:
   - concept score includes `heatmap_max_over_mean`, `uniform_baseline`, and `near_uniform`.
11. Concept target recommendations:
   - concept-only records produce non-empty `recommended_concept_targets`.
12. Near-uniform warnings:
   - heatmap with max-over-mean near 1.0 is flagged.
13. Branch delta:
   - positive/negative concept records produce a delta row.
14. CLI smoke tests:
   - `summarize_survey.py --list-runs` works.
   - `compare_survey_runs.py` writes output files.

All tests must pass with:

```bash
python -m pytest -q
```

---

## 10. Manual Validation Requirements

After implementation, validate in ComfyUI with a clean run.

### 10.1 Concept-only positive run

Settings:

```text
mode=observe
capture_level=heatmap
branch_mode=positive_only
save_heatmaps=true
heatmap_output=concepts_only
prompt_text=<exact actual generation prompt>
concept_terms=big breasts
jsonl_path=anima_concept_survey/logs/big_breasts_positive.jsonl
heatmap_dir=anima_concept_survey/heatmaps_big_breasts_positive
```

Expected:

- no duplicate same-span concept matches;
- `concept_match_summary` has one match for one phrase occurrence;
- aggregate concept manifest has correct observation count;
- `recommended_concept_targets.csv` contains calls comparable to Trial 1 candidates;
- report warns only if focus is weak or near-uniform.

### 10.2 Multi-run JSONL validation

Append two runs to one JSONL.

Expected:

- `--list-runs` shows both runs;
- no-filter summary fails or warns clearly;
- `--run-id` isolates one run;
- `--latest-run` isolates the last run.

### 10.3 Both-branch validation

Settings:

```text
branch_mode=both
heatmap_output=concepts_only
```

Expected:

- positive and negative concept sections are separate;
- near-uniform negative maps are flagged;
- branch delta CSV is generated.

---

## 11. Out of Scope for This Follow-up

Do not implement these in this pass unless explicitly requested:

- separate ConceptAttention concept-vector stream;
- final image segmentation or overlay node;
- LoRA training;
- model merge automation;
- rectangular latent layouts;
- video/Cosmos layouts;
- composing with other `optimized_attention_override` patches.

---

## 12. Completion Criteria

Automated completion status:

- [x] duplicate same-span concept matches are eliminated;
- [x] true repeated phrase occurrences remain supported;
- [x] concept heatmap file/aggregate/report identities cannot collide;
- [x] `summarize_survey.py` supports run filtering and protects against accidental run mixing;
- [x] `recommended_concept_targets.csv` exists and works for concept-only records;
- [x] near-uniform preview warnings appear in reports;
- [x] branch delta output exists;
- [x] README explains concept-only workflow, preview normalization, and run filtering;
- [x] all automated tests pass;
- [ ] at least one fresh ComfyUI concept-only run validates the fixes.
