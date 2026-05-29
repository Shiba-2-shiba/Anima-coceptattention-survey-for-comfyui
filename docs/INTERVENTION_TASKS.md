# INTERVENTION_TASKS.md — Separate Intervention Node Refactor Tasks

Project: Anima ConceptAttention Survey for ComfyUI  
Target branch: `dev`  
Primary decision: add a separate intervention node; do not turn the survey node into an editing node.

---

## Execution Rules for Codex

1. Work in small patches.
2. Add or update tests before changing behavior.
3. Run after each meaningful patch:

```bash
python -m pytest -q
```

4. Do not remove existing public inputs from `AnimaConceptSurveyModelPatch`.
5. Preserve survey observe-only exact passthrough behavior.
6. Keep survey JSONL `schema_version = 1`; intervention events may also use `schema_version = 1`.
7. The new intervention node must be visibly separate from the survey node.
8. In fallback mode, unsupported intervention calls must return original backend output.
9. Do not claim that attention/intervention maps are final-image segmentation masks.
10. Use safe no-op defaults for intervention controls: `strength=1.0`, `logit_bias=0.0`, and `mode=shadow`.
11. Treat fixed-seed prompt-pair image differences as required evidence before LoRA/merge interpretation.
12. Update `INTERVENTION_PROGRESS.md` whenever a task is completed.

---

## P0 — Planning Documents

### P0-T01 Add intervention planning documents

Priority: highest  
Status: completed by this document set

Create:

```text
INTERVENTION_SPEC.md
INTERVENTION_TASKS.md
INTERVENTION_PROGRESS.md
```

Acceptance:

- The docs specify separate-node implementation.
- The docs explicitly preserve observe-only survey behavior.
- The docs define testing and manual validation rules.

---

## P1 — Protect Existing Survey Behavior

### P1-T01 Add or confirm survey passthrough test

Priority: highest  
Status: completed

Objective:

Before adding intervention, pin the current guarantee that the survey node does not edit generation.

Test target:

- mock `original_func` returns a known tensor;
- survey override observes if possible;
- returned tensor is exactly the `original_func` output.

Acceptance:

- Test fails if survey override returns its own computed attention output.
- Test passes before intervention implementation begins.

---

### P1-T02 Confirm survey node schema remains stable

Priority: high  
Status: completed

Objective:

Ensure adding the intervention node does not remove or rename survey inputs.

Acceptance:

- Existing `AnimaConceptSurveyModelPatch` remains importable.
- Existing schema still includes current survey inputs such as `mode`, `capture_level`, `prompt_text`, `concept_terms`, `branch_mode`, `heatmap_output`, and `fail_mode`.

---

### P1-T03 Keep existing tests green

Priority: highest  
Status: completed

Run:

```bash
python -m pytest -q
```

Acceptance:

- Baseline test suite passes before intervention patches.
- Record the test result in `INTERVENTION_PROGRESS.md`.

---

## P2 — Shared Math and Intervention Primitives

### P2-T01 Add attention math helper tests

Priority: highest  
Status: completed

Create tests for pure tensor helpers before integrating a node.

Required cases:

- compute logits from `q`, `k`, and scale;
- softmax probabilities sum to one over text keys;
- output from probabilities and `v` has expected shape;
- helper functions work on CPU with deterministic tensors.

Acceptance:

- Tests do not require ComfyUI runtime.

---

### P2-T02 Implement `token_attention_scale` primitive

Priority: highest  
Status: completed

Function requirement:

```python
apply_token_attention_scale(probs, token_indices, scale, eps=1e-12) -> torch.Tensor
```

Behavior:

- `scale=1.0` is no-op within tolerance;
- `scale=0.0` suppresses selected token probability mass;
- output distribution remains normalized over text keys;
- invalid / empty token list returns unchanged probabilities or is skipped by caller.

Acceptance:

- Unit tests cover no-op, suppression, boost, and normalization.

---

### P2-T03 Implement `attention_logit_bias` primitive

Priority: highest  
Status: completed

Function requirement:

```python
apply_attention_logit_bias(logits, token_indices, bias) -> torch.Tensor
```

Behavior:

- `bias=0.0` is no-op within tolerance after softmax;
- negative bias reduces selected token probability mass;
- positive bias increases selected token probability mass;
- non-target token logits are unchanged.

Acceptance:

- Unit tests compare target mass before/after bias.
- Public UI and records use `attention_logit_bias`; `token_logit_bias` may be accepted only as a documented compatibility alias.

---

### P2-T04 Avoid destabilizing survey math

Priority: high  
Status: completed

Objective:

If attention math is extracted from survey code, ensure existing survey scores remain unchanged in tests.

Acceptance:

- Existing scoring/reporting tests pass.
- New helper tests pass.

---

## P3 — Intervention Config and Records

### P3-T01 Add `InterventionConfig`

Priority: highest  
Status: completed

Implementation target:

```text
anima_concept_survey/intervention_config.py
```

Fields:

```python
mode: str = "shadow"                 # off | shadow | intervene
intervention_kind: str = "attention_logit_bias"
prompt_text: str = ""
intervention_terms: str = ""
target_call_indices: str = "all"
target_step_indices: str = "all"
branch_mode: str = "positive_only"
strength: float = 1.0
logit_bias: float = 0.0
max_steps: int = 0
jsonl_path: str | None = None
max_logits_mib: float = 1024.0
fail_mode: str = "fallback"
token_text_map: dict[int, dict[str, Any]] = field(default_factory=dict)
```

Acceptance:

- Validation rejects unsupported modes/kinds/branch modes/fail modes.
- Validation parses call and step scopes.
- Numeric ranges are checked.

---

### P3-T02 Add step scope parser

Priority: medium  
Status: completed

Current code has call scope parsing. Reuse it or add equivalent step parsing.

Acceptance:

- `all`, `0`, `0,3,5`, `4-10` work.
- negative and descending ranges fail.

---

### P3-T03 Add intervention record builders

Priority: high  
Status: completed

Implementation target:

```text
anima_concept_survey/intervention_records.py
```

Add helpers:

```python
intervention_plan_record(...)
intervention_observation_record(...)
intervention_skipped_record(...)
intervention_fallback_record(...)
intervention_summary_record(...)
```

Acceptance:

- Records are JSON-serializable.
- Required event names are present.
- Concept match fields include `concept_uid`, token indices, token texts, and token source.

---

## P4 — Add Separate Intervention Node

### P4-T01 Add `AnimaConceptInterventionModelPatch`

Priority: highest  
Status: completed

Implementation target:

```text
nodes.py or a new nodes_intervention.py imported by nodes.py
```

Requirements:

- Add a new class; do not mutate `AnimaConceptSurveyModelPatch` into an intervention node.
- Use display name `Anima Concept Intervention Model Patch`.
- Use category `model_patches/anima`.
- Tooltip/description must warn that generation is modified in `intervene` mode.

Acceptance:

- New node imports without ComfyUI test failures.
- Existing survey node still imports.

---

### P4-T02 Update extension node list

Priority: highest  
Status: completed

Current extension returns one node. Update it to return both:

```python
return [AnimaConceptSurveyModelPatch, AnimaConceptInterventionModelPatch]
```

Acceptance:

- Tests verify both node classes are returned or importable.
- Survey node remains first or at least still available.

---

### P4-T03 Prevent override stacking conflict

Priority: high  
Status: completed

Current survey node refuses to combine with an existing `optimized_attention_override`. Intervention node should do the same for now.

Acceptance:

- If model already has `optimized_attention_override`, intervention node raises a clear error.
- Do not attempt composition in this pass.

---

## P5 — Intervention Override: Shadow Mode

### P5-T01 Implement override skeleton

Priority: highest  
Status: completed

Implementation target:

```text
anima_concept_survey/intervention.py
```

Class:

```python
class AnimaConceptInterventionAttentionOverride:
    ...
```

Requirements:

- Resolve token text map from `clip` and `prompt_text`.
- Build concept match report from `intervention_terms`.
- Emit `intervention_plan`.
- Implement fallback/skipped record handling.
- In `shadow` mode, always return original backend output.

Acceptance:

- Unit test confirms `shadow` returns original output.
- JSONL writer emits plan and observation/skipped/fallback records in a temporary file.

---

### P5-T02 Implement eligibility checks

Priority: highest  
Status: completed

Use survey-like checks:

- valid q/k/v signature;
- rank 4 tensors;
- no unsupported mask;
- `skip_reshape=True`;
- cross-attention only;
- square spatial shape if heatmap/delta map is requested;
- known branch layout for branch-specific modes;
- memory guard.

Acceptance:

- Tests cover unsupported mask, no `skip_reshape`, self-attention, and VRAM guard fallback.

---

### P5-T03 Implement scoped observation stats

Priority: high  
Status: completed

For scoped calls, compute:

```text
original_score_mean
modified_score_mean
score_delta
mean_abs_prob_delta
max_abs_prob_delta
output_delta_norm
```

In `shadow` mode, modified stats are computed but original output is returned.

Acceptance:

- Tests assert deltas are zero for no-op settings.
- Tests assert deltas are non-zero for suppression settings.

---

## P6 — Intervention Override: Intervene Mode

### P6-T01 Implement modified output reconstruction

Priority: highest  
Status: completed

Objective:

Return modified attention output for supported cross-attention calls.

Requirements:

- `strength=1.0` or `bias=0.0` must match original backend output within tolerance in mock tests.
- Unselected branch/call output must remain unchanged.
- Output dtype/device should match expected original behavior.

Acceptance:

- Mock attention tests pass.
- Unsupported layout falls back rather than returning malformed output.

---

### P6-T02 Apply intervention only to selected branches

Priority: highest  
Status: completed

Cases:

```text
branch_mode=positive_only
branch_mode=negative_only
branch_mode=both
```

Acceptance:

- Tests verify selected branch target mass changes.
- Tests verify unselected branch target mass/output remains unchanged.

---

### P6-T03 Apply target call and step filters

Priority: high  
Status: completed

Acceptance:

- With `target_call_indices=10`, only eligible call 10 changes.
- With `target_step_indices=5-10`, only those steps change.
- Skipped records identify out-of-scope calls/steps.

---

### P6-T04 Add robust no-op parity test

Priority: highest  
Status: completed

Test:

```text
original backend output
intervene, token_attention_scale, strength=1.0
intervene, attention_logit_bias, bias=0.0
```

Acceptance:

- Outputs match within agreed tolerance.
- This test is required before manual image validation.

---

## P7 — Intervention and Image Difference Reporting

### P7-T01 Add `scripts/summarize_intervention.py`

Priority: medium  
Status: completed

CLI:

```bash
python scripts/summarize_intervention.py intervention.jsonl --out-dir intervention_report
```

Outputs:

```text
intervention_summary.json
intervention_by_call.csv
intervention_by_concept.csv
intervention_report.md
```

Acceptance:

- Works from JSONL alone.
- Empty/no-op logs produce a clear report.

---

### P7-T02 Add intervention report tests

Priority: medium  
Status: completed

Fixture:

- one plan record;
- several observation records;
- one fallback record.

Acceptance:

- CSV and Markdown files are written.
- Top calls are sorted by absolute `score_delta` or `output_delta_norm`.

---

### P7-T03 Add `scripts/compare_intervention_images.py`

Priority: highest  
Status: completed

Rationale:

The fixed-seed `big breasts` present vs removed comparison showed the expected breast-region change even though the observe-only attention heatmap was not breast-localized. Image-difference reporting is therefore required evidence, not optional polish.

CLI:

```bash
python scripts/compare_intervention_images.py baseline.png variant.png --out-dir diff_report
```

Outputs:

```text
diff_abs.png
diff_summary.json
diff_report.md
```

Acceptance:

- Computes deterministic per-pixel absolute difference for same-size images.
- Fails clearly on size mismatch.
- Writes a visual diff PNG and JSON summary.
- Summary includes mean, max, p95 absolute delta, and changed pixel ratio.

---

### P7-T04 Add manual ROI bbox scoring

Priority: high  
Status: completed

CLI extension:

```bash
python scripts/compare_intervention_images.py baseline.png variant.png --out-dir diff_report --roi-bbox x,y,w,h
```

Acceptance:

- Computes `roi_delta_over_whole_delta` when ROI is supplied.
- Records the ROI bbox in `diff_summary.json`.
- Does not claim the ROI is an automatic semantic segmentation mask.

---

## P8 — Improve Survey-to-Intervention Bridge

### P8-T01 Rename or document `recommended_concept_targets` as attention candidates

Priority: high  
Status: completed

Objective:

Avoid implying that survey ranking is already a causal LoRA target ranking.

Acceptance:

- README/report text calls it an attention-only candidate ranking.
- Documentation says intervention target selection should combine survey ranking with fixed-seed prompt/image differences.
- No behavior change required.

---

### P8-T02 Add call-by-call aggregate concept heatmaps

Priority: medium  
Status: completed

Rationale:

Current aggregate concept heatmap averages all calls and steps, which can obscure call-specific behavior. Intervention target selection needs call-level visual summaries.

Suggested output:

```text
heatmaps/concepts/aggregate_by_call/call010_positive_concept_*.npy
heatmaps/concepts/aggregate_by_call/call010_positive_concept_*_preview.png
heatmaps/concepts/aggregate_by_call/manifest.json
```

Acceptance:

- Report top concept calls can be visually inspected by call.
- Existing aggregate output remains available.

---

### P8-T03 Preserve module metadata and document unknown block state

Priority: medium  
Status: completed

Objective:

Record module/block metadata in intervention logs and continue improving survey logs.

Acceptance:

- Logs include `block` and `metadata` fields.
- If block remains `unknown`, report says so explicitly.

---

## P9 — Documentation

### P9-T01 Update README with two-node workflow

Priority: high  
Status: completed

Add sections:

```text
Survey node: observe-only candidate discovery
Intervention node: experimental generation-changing causal probe
```

Acceptance:

- Users understand which node changes output.
- Fixed-seed comparison is required for intervention interpretation.

---

### P9-T02 Add intervention workflow examples

Priority: high  
Status: completed

Example:

```text
1. Run survey with positive_only / concepts_only.
2. Summarize survey and pick top call candidates.
3. Run fixed-seed prompt-pair comparison, e.g. big breasts present vs removed.
4. Generate image diff and optional manual ROI score.
5. Run intervention in shadow mode.
6. Run no-op parity intervention.
7. Run suppression intervention.
8. Compare generated images under fixed seed.
```

Acceptance:

- README includes practical settings for `big breasts`-style concept probing.

---

### P9-T03 Update planning docs after implementation

Priority: medium  
Status: completed

Update:

```text
INTERVENTION_SPEC.md
INTERVENTION_TASKS.md
INTERVENTION_PROGRESS.md
```

Acceptance:

- Completed tasks are marked completed.
- Manual validation results are recorded.

---

## P10 — Manual Validation

### P10-T01 Validate node appears separately in ComfyUI

Priority: highest after tests  
Status: not started

Acceptance:

- `Anima Concept Survey Model Patch` still appears.
- `Anima Concept Intervention Model Patch` appears separately.
- The intervention node description warns that output can change.

---

### P10-T02 Validate fixed-seed no-op parity

Priority: highest  
Status: not started

Run fixed seed:

```text
A: no intervention node
B: shadow mode
C: intervene + token_attention_scale strength=1.0
D: intervene + attention_logit_bias bias=0.0
```

Acceptance:

- Outputs are visually identical.
- Logs show observations but no meaningful deltas.

---

### P10-T03 Validate target suppression

Priority: highest  
Status: not started

Run:

```text
intervention_terms=big breasts
branch_mode=positive_only
target_call_indices=<candidate call>
intervention_kind=attention_logit_bias
logit_bias=-5.0
```

Acceptance:

- JSONL shows reduced concept attention mass.
- Image changes, if any, are compared against baseline under fixed seed.
- Results are recorded in `INTERVENTION_PROGRESS.md`.

---

### P10-T04 Validate fixed-seed prompt-pair image difference

Priority: highest  
Status: not started

Run:

```text
A: original prompt with big breasts
B: same prompt with big breasts removed
C: same prompt with small breasts or flat chest, if useful
```

Acceptance:

- The comparison uses the same seed, sampler, steps, CFG, negative prompt, and latent size.
- `compare_intervention_images.py` produces diff outputs.
- A manually supplied chest ROI bbox is recorded if used.
- Results are recorded as evidence that prompt-level causality can exist even when observe-only heatmaps are not localized.

---

### P10-T05 Validate controls

Priority: high  
Status: not started

Controls:

- unrelated token suppression;
- wrong call suppression;
- weaker bias sweep;
- stronger bias sweep;
- negative_only if branch layout allows.

Acceptance:

- Control results are not overinterpreted.
- Any unexpected result is documented as a risk.

---

## Later / Optional

Image difference and manual ROI bbox scoring were promoted to P7 because fixed-seed prompt-pair evidence is required before LoRA/merge interpretation.

### L-T01 Module-path resolver

Status: not started

Improve mapping from eligible call index to actual transformer block/module path.

---

## Completion Checklist

- [x] Existing survey node remains observe-only.
- [x] Existing survey tests pass.
- [x] Separate intervention node is added.
- [x] Intervention config validates correctly.
- [x] Shadow mode returns original output.
- [x] Intervene mode can modify supported cross-attention calls.
- [x] No-op intervention matches original output in tests.
- [x] Target suppression reduces target token attention mass in tests.
- [x] Branch/call/step filters work.
- [x] Fallback mode is conservative.
- [x] Intervention JSONL records are emitted.
- [x] Intervention report script works.
- [x] Paired image diff script writes diff PNG and JSON summary.
- [x] Manual ROI bbox scoring is available for fixed-seed prompt/intervention comparisons.
- [x] README documents two-node workflow.
- [ ] Manual fixed-seed validation is completed.
