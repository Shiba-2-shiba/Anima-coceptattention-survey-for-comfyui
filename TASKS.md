# TASKS.md — Dev Branch Follow-up Refactor Tasks

Project: Anima ConceptAttention Survey for ComfyUI
Target branch: `dev`
Focus: duplicate concept fixes, run-filtered reporting, concept target ranking, safer previews

---

## Execution Rules for Codex

1. Work in small patches.
2. Add or update tests before changing behavior.
3. Run after each meaningful patch:

```bash
python -m pytest -q
```

4. Do not remove existing public node inputs.
5. Preserve observe-only exact passthrough behavior.
6. Keep JSONL `schema_version = 1`; add optional fields only.
7. Keep `survey_attention.py` as a compatibility export layer if the current dev branch already has it.
8. Do not implement a separate ConceptAttention concept-vector stream in this pass.
9. Treat `.npy` and manifest/JSONL stats as ground truth; preview PNGs are display aids.
10. Update this task file and `PROGRESS.md` when each task is completed.

---

## P0 — Fix Duplicate Concept Matches and Concept Identity

### P0-T01 Add regression tests for duplicate same-span concept matches

Priority: highest
Status: completed

Objective:

Reproduce the observed issue where the same Qwen token span is emitted as multiple occurrences.

Add tests in `tests/test_concepts.py` or equivalent:

- token sequence contains an empty-normalized token before the phrase;
- `concept_terms="big breasts"`;
- only one match is returned for the same token span;
- `occurrence_index == 0`.

Example fixture:

```python
token_text_map = {
    0: {"token_index": 0, "source_token_index": 0, "token_text": "<s>", "token_source": "qwen3_06b"},
    1: {"token_index": 1, "source_token_index": 1, "token_text": " big", "token_source": "qwen3_06b"},
    2: {"token_index": 2, "source_token_index": 2, "token_text": " breasts", "token_source": "qwen3_06b"},
}
```

Acceptance:

- The test fails on the duplicate bug and passes after dedupe.
- True repeated phrases with distinct token spans are not blocked.

---

### P0-T02 Preserve true repeated phrase occurrences

Priority: highest
Status: completed

Objective:

Ensure dedupe does not remove legitimate repeated prompt phrases.

Test:

```text
prompt tokens: " big", " breasts", ",", " big", " breasts"
concept_terms: big breasts
expected token spans: [0,1] and [3,4]
expected occurrence_index: 0 and 1
```

Acceptance:

- Two matches are returned.
- The two matches have distinct token spans.
- Same-span duplicates are still removed.

---

### P0-T03 Implement concept match dedupe

Priority: highest
Status: completed

Implementation target:

- `anima_concept_survey/concepts.py`

Requirements:

- Deduplicate by normalized term, token source, global token span, and source-local token span.
- Keep stable ordering by first token index.
- Recompute `occurrence_index` after dedupe per `(normalized_term, token_source)`.
- Preserve match warnings and ignored token indices.
- Do not silently drop distinct spans.

Suggested helper:

```python
def dedupe_concept_matches(matches: Iterable[ConceptTokenMatch]) -> tuple[ConceptTokenMatch, ...]:
    ...
```

Acceptance:

- P0-T01 and P0-T02 tests pass.
- Existing ambiguous/source-prefixed tests still pass.

---

### P0-T04 Add stable `concept_uid`

Priority: highest
Status: completed

Implementation targets:

- `concepts.py`
- `scoring.py`
- `records.py`
- `heatmaps.py`
- `reporting.py`

Requirements:

Each concept match/score should expose a stable identity that distinguishes:

- same term in different token sources;
- same term repeated at different token spans;
- same term with different occurrence index.

Recommended fields:

```json
{
  "concept_uid": "big_breasts__qwen3_06b__occ0__tok027-028",
  "normalized_term": "bigbreasts",
  "token_source": "qwen3_06b",
  "occurrence_index": 0
}
```

Acceptance:

- `attention_observation.concept_scores[]` includes `concept_uid`.
- concept heatmap manifests include `concept_uid`.
- report rows include `concept_uid`.
- same term/different occurrence has different UID.

---

### P0-T05 Prevent concept heatmap filename and aggregate collisions

Priority: highest
Status: completed

Implementation target:

- `anima_concept_survey/heatmaps.py`

Requirements:

- Concept heatmap file stems must include source, occurrence, and token span or `concept_uid`.
- Concept accumulator key must not be `(branch, term)` only.
- Aggregate concept manifests must keep separate rows for distinct concept identities.

Recommended key:

```python
(branch, concept_uid)
```

Acceptance:

- Two concept matches with the same term but different token spans write two distinct files.
- Aggregate manifest has two distinct rows.
- No file is overwritten.

---

## P1 — Run Discovery and Run-Filtered Reporting

### P1-T01 Add run discovery utilities

Priority: highest
Status: completed

Implementation target:

- `anima_concept_survey/reporting.py`

Add:

```python
@dataclass(frozen=True)
class RunInfo:
    run_id: str
    first_record_index: int
    last_record_index: int
    observation_count: int
    events: dict[str, int]
    prompt_text: str | None
    branches: tuple[str, ...]
    concept_terms: tuple[str, ...]

list_runs(records) -> list[RunInfo]
latest_run_id(records) -> str | None
filter_records_by_run_id(records, run_id) -> list[dict]
```

Acceptance:

- Tests detect two run IDs in one JSONL-like fixture.
- `latest_run_id` returns the run that appears last by file order.

---

### P1-T02 Add `--list-runs`, `--run-id`, `--latest-run`, and `--allow-mixed-runs`

Priority: highest
Status: completed

Implementation target:

- `scripts/summarize_survey.py`

Required CLI:

```bash
python scripts/summarize_survey.py survey.jsonl --list-runs
python scripts/summarize_survey.py survey.jsonl --out-dir report --run-id survey-...
python scripts/summarize_survey.py survey.jsonl --out-dir report --latest-run
python scripts/summarize_survey.py survey.jsonl --out-dir report --allow-mixed-runs
```

Default behavior:

- If one run exists, summarize it.
- If multiple runs exist and no filter is supplied, fail with clear instructions.
- `--allow-mixed-runs` intentionally preserves old behavior.

Acceptance:

- CLI tests or subprocess smoke tests pass.
- Multiple-run fixture without filter does not silently aggregate.

---

### P1-T03 Add `survey_runs.csv`

Priority: high
Status: completed

Implementation target:

- `reporting.py`

Requirements:

- Write `survey_runs.csv` in report output.
- Include run id, observation count, branch list, prompt summary, concept terms, and event counts.

Acceptance:

- `write_summary_outputs()` produces `survey_runs.csv` when run metadata is available.
- Output is useful even for single-run reports.

---

### P1-T04 Add run metadata to report JSON and Markdown

Priority: high
Status: completed

Requirements:

`survey_summary.json` should include:

```json
{
  "selected_run_id": "survey-...",
  "run_count_in_input": 2,
  "mixed_runs": false,
  "available_runs": [...]
}
```

Markdown should include:

```text
Run ID: survey-...
Prompt summary: ...
Branches: positive
Concept terms: big breasts
```

Acceptance:

- Generated Markdown makes run selection obvious.

---

## P2 — Concept Heatmap Metrics in JSONL and Reports

### P2-T01 Add heatmap stats to concept scores in JSONL

Priority: highest
Status: completed

Implementation target:

- `scoring.py` or wherever concept scores are built

Add fields to every public concept score:

```text
heatmap_mean
heatmap_max
heatmap_std
heatmap_max_over_mean
uniform_baseline
score_mean_over_uniform
near_uniform
```

Rules:

- `uniform_baseline = len(attention_key_indices or token_indices) / text_len`.
- `near_uniform = heatmap_max_over_mean < 1.05`.
- Handle missing or zero values safely.

Acceptance:

- Deterministic tensor tests assert these fields.
- Existing heatmap manifest stats remain correct.

---

### P2-T02 Extend `survey_by_concept.csv`

Priority: high
Status: completed

Implementation target:

- `reporting.py`

Add columns:

```text
concept_uid
normalized_term
token_source
occurrence_index
source_token_indices
heatmap_mean
heatmap_max
heatmap_std
heatmap_max_over_mean
uniform_baseline
score_mean_over_uniform
near_uniform
```

Group by:

```text
eligible_call_index
branch
concept_uid
```

not just `(call, branch, term)`.

Acceptance:

- Same term/different occurrence remains separate in report.
- Concept-only records aggregate correctly.

---

### P2-T03 Add `recommended_concept_targets.csv`

Priority: highest
Status: completed

Implementation target:

- `reporting.py`

Create new report output:

```text
recommended_concept_targets.csv
```

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
focus_factor = max(0.0, heatmap_max_over_mean - 1.0)
focus_factor = min(focus_factor, 4.0)
near_uniform_penalty = 0.25 if heatmap_max_over_mean < 1.05 else 1.0
branch_factor = 1.0 if branch == "positive" else 0.5
rank_score = score_mean_over_uniform * (1.0 + focus_factor) * near_uniform_penalty * branch_factor
```

Acceptance:

- Concept-only JSONL fixture produces non-empty recommended concept targets.
- Near-uniform negative maps rank lower than focused positive maps.

---

### P2-T04 Add concept target section to Markdown

Priority: high
Status: completed

Requirements:

Markdown report should include:

```text
## Recommended Concept Targets
```

Table columns:

```text
rank | term | call | branch | rank_score | mean | mean/uniform | focus | warning
```

Acceptance:

- Trial-like concept-only fixture produces a readable concept target section.

---

## P3 — Preview Safety and Near-Uniform Warnings

### P3-T01 Record preview normalization metadata

Priority: medium
Status: completed

Implementation target:

- `heatmaps.py`

Add to manifest rows:

```json
{
  "preview_normalization": "per_file_minmax"
}
```

Acceptance:

- Token and concept manifests include preview normalization mode.

---

### P3-T02 Add near-uniform report warnings

Priority: high
Status: completed

Implementation target:

- `reporting.py`

Rules:

```python
near_uniform = heatmap_max_over_mean < 1.05
weak_focus = 1.05 <= heatmap_max_over_mean < 1.15
```

Markdown should warn:

```text
Preview PNG is min-max normalized and may overstate localization.
```

Acceptance:

- Negative near-uniform fixture emits a warning.
- Positive focused fixture does not emit the same warning.

---

### P3-T03 Optional fixed-scale preview mode

Priority: low
Status: deferred (optional)

Implement only after P0-P2 are complete.

Possible approaches:

- postprocess-only fixed scale previews;
- optional advanced node setting;
- report script rendering from `.npy` files.

Acceptance if implemented:

- Existing min-max previews remain available.
- Fixed-scale previews are clearly named.

---

## P4 — Branch Guidance and Branch Delta Reports

### P4-T01 Add `survey_branch_concept_delta.csv`

Priority: high
Status: completed

Implementation target:

- `reporting.py`

Group positive and negative rows by:

```text
eligible_call_index
concept_uid or term/source/occurrence/token span
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

Acceptance:

- Both-branch fixture creates branch delta rows.
- Negative near-uniform maps are identifiable.

---

### P4-T02 Separate positive and negative sections in Markdown

Priority: medium
Status: completed

Requirements:

Markdown should avoid mixing positive and negative concept target interpretation.

Add sections:

```text
## Positive Concept Targets
## Negative Concept Diagnostics
## Branch Delta
```

Acceptance:

- Report is easier to interpret for both-branch runs.

---

### P4-T03 Update docs/tooltips for branch guidance

Priority: medium
Status: completed

Requirements:

Document:

- `positive_only` is recommended first for concept localization;
- `both` is useful for diagnostics;
- negative branch heatmaps may be near-uniform and should not be judged by preview color alone.

Acceptance:

- README includes this guidance.

---

## P5 — Compare Survey Runs

### P5-T01 Add `scripts/compare_survey_runs.py`

Priority: medium
Status: completed

CLI:

```bash
python scripts/compare_survey_runs.py survey.jsonl \
  --run-a survey-21d49acdd90 \
  --run-b survey-21e5470fe90 \
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

Acceptance:

- Works from JSONL only.
- Does not require image files.
- Test fixture comparing two runs produces delta rows.

---

### P5-T02 Add comparison report tests

Priority: medium
Status: completed

Test fixture:

- Run A: positive-only, concept mean stronger.
- Run B: both-branch, positive weaker, negative near-uniform.

Acceptance:

- Delta report ranks Run A as stronger for the concept.
- Prompt/token span differences are shown.

---

## P6 — README, SPEC, PROGRESS, TASKS Alignment

### P6-T01 Update README concept-only workflow

Priority: high
Status: completed

Add recommended settings:

```text
mode=observe
capture_level=heatmap
branch_mode=positive_only
save_heatmaps=true
heatmap_output=concepts_only
prompt_text=<exact prompt>
concept_terms=<phrase>
```

Also document:

- `max_tokens` does not change concept heatmap quality;
- use `.npy` and manifest stats as ground truth;
- use run-specific JSONL/heatmap directories or `--run-id`.

Acceptance:

- README matches new CLI and report outputs.

---

### P6-T02 Update project docs after implementation

Priority: medium
Status: completed

Update:

- `SPEC.md`
- `PROGRESS.md`
- `TASKS.md`

Acceptance:

- Docs describe implemented behavior, not planned behavior.
- Completed tasks are marked completed.

---

## P7 — Manual ComfyUI Validation

### P7-T01 Validate duplicate fix with concept-only positive run

Priority: highest after automated tests
Status: not started

Settings:

```text
mode=observe
capture_level=heatmap
branch_mode=positive_only
save_heatmaps=true
heatmap_output=concepts_only
prompt_text=<exact actual generation prompt>
concept_terms=big breasts
```

Acceptance:

- `concept_match_summary` has one match if phrase appears once.
- Same-span duplicate occurrence does not appear.
- Aggregate observation count is not doubled.
- Concept heatmap files include concept UID/source/token span.

---

### P7-T02 Validate run-filtered reporting

Priority: high
Status: not started

Steps:

1. Append two runs to one JSONL.
2. Run `--list-runs`.
3. Run `--run-id` for each run.
4. Run `--latest-run`.
5. Run without filter.

Acceptance:

- Both runs are listed.
- Each filtered report contains only selected run observations.
- No-filter report does not silently mix runs.

---

### P7-T03 Validate concept target recommendation

Priority: high
Status: not started

Acceptance:

- `recommended_concept_targets.csv` is generated.
- For a Trial 1-like run, calls similar to `9`, `10`, and `13` are visible near the top.
- Near-uniform negative maps are penalized in both-branch runs.

---

## Later / Optional

### L-T01 Add robust attention-key alignment resolver

Status: not started

Rationale:

The tested Qwen workflow appears to align token indices with runtime text keys, but multi-encoder configurations may need a resolver that chooses global vs source-local key indices.

---

### L-T02 Move heavy PNG rendering to postprocess

Status: not started

Rationale:

Runtime PNG writing can slow sampling. After report safety improves, consider writing only JSONL and `.npy` during sampling, then rendering previews in postprocess.

---

### L-T03 Add final-image overlay node

Status: not started

Rationale:

Useful for visualization, but outside the current correctness-focused refactor.

---

## Completion Checklist

The follow-up refactor is done when:

- [x] duplicate same-span concept matches are fixed;
- [x] true repeated phrase occurrences are preserved;
- [x] `concept_uid` exists in JSONL, manifests, and reports;
- [x] concept heatmap filenames and aggregate keys cannot collide;
- [x] `summarize_survey.py` supports run filtering;
- [x] multiple-run JSONL files are not silently mixed by default;
- [x] `recommended_concept_targets.csv` exists;
- [x] concept scores include heatmap focus and uniform-baseline metrics;
- [x] near-uniform warnings appear in Markdown reports;
- [x] branch delta CSV exists;
- [x] README explains concept-only workflow and preview normalization;
- [x] `python -m pytest -q` passes;
- [ ] fresh ComfyUI validation confirms no duplicate concept count inflation.
