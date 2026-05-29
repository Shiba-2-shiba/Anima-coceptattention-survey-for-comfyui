# INTERVENTION_PROGRESS.md — Separate Intervention Node Refactor Progress

Project: Anima ConceptAttention Survey for ComfyUI  
Target branch: `dev`  
Focus: add a separate, explicit attention intervention node while preserving the existing observe-only survey node.

---

## Current Status

Status: M9 documentation completed. M10 manual ComfyUI validation remains.

The previous follow-up refactor made the survey side substantially safer:

- same-span concept duplicate handling is implemented;
- concept identity and aggregate manifests are stable;
- run-filtered reporting exists;
- concept-specific attention candidate ranking exists;
- preview normalization warnings exist;
- branch delta and JSONL-only run comparison exist.

Recent ComfyUI validation confirmed that the survey pipeline can run on a real Anima/Qwen workflow and produce a single `big breasts` concept occurrence with token span `[78, 79]` and an aggregate concept manifest. However, the observed heatmap did not provide the desired causal explanation of final image content.

A later fixed-seed prompt-pair check showed that removing `big breasts` visibly reduced the generated breast region. This confirms both points: the prompt concept can affect the expected visual region, and observe-only attention heatmaps alone are not enough to explain that effect. The refactor now needs paired image-difference evidence plus explicit intervention, not intervention alone.

---

## Decision Summary

### D1. Use separate-node architecture

Decision:

```text
Add Anima Concept Intervention Model Patch as a separate node.
```

Reason:

The existing survey node is valuable because it is observe-only. Mixing intervention into `mode=observe` would make the node harder to trust and easier to misuse.

### D2. Keep survey node frozen as an observer

Decision:

The intervention refactor must first protect existing survey passthrough behavior with tests.

Reason:

The survey node is the candidate-discovery and logging base. It should not become generation-changing.

### D3. Start with attention-token interventions

Decision:

Initial intervention kinds:

```text
token_attention_scale
attention_logit_bias
```

Reason:

They directly target matched prompt token keys and are easier to test than final-image or latent-space interventions. The public name should be `attention_logit_bias` because the bias is applied to attention logits, not final model/image logits.

### D4. Add `shadow` mode before `intervene` mode

Decision:

The new node should support:

```text
off | shadow | intervene
```

Reason:

`shadow` mode allows Codex and manual validation to confirm token resolution, call filtering, and would-be deltas without changing generation.

### D5. Treat intervention results as heuristic

Decision:

Reports should say intervention results are causal-probe heuristics, not ground-truth semantic causality or segmentation.

Reason:

Diffusion model behavior is nonlinear. Suppressing one token key can affect many visual factors indirectly.

### D6. Promote paired image difference and manual ROI scoring

Decision:

Fixed-seed prompt-pair image comparison and manual ROI bbox scoring are required workflow components, not later optional polish.

Reason:

The `big breasts` present vs removed comparison produced the clearest evidence that the prompt affected the intended region. That evidence should be captured with a repeatable diff script and optional user-supplied ROI before any LoRA/merge ranking is interpreted as region-specific.

---

## Known Risks

### R1. Output layout mismatch

Risk:

The optimized attention override signature may require a return layout that differs from the simple `[batch, heads, query, dim]` output expected by direct `probs @ v` computation.

Mitigation:

- Add mock no-op parity tests.
- In real runtime, fallback if output layout is not verified.
- Do not ship `intervene` mode without no-op parity tests.

### R2. Intervention may be slower or use more VRAM

Risk:

Computing full logits/probs and modified outputs can be expensive.

Mitigation:

- Keep `max_logits_mib` guard.
- Use scoped call indices.
- Start with small candidate call sets from survey reports.

### R3. Unsupported masks / reshape modes

Risk:

Some attention calls may use masks or non-`skip_reshape` layouts.

Mitigation:

- Fallback by default.
- Emit clear `intervention_fallback` reasons.

### R4. Branch layout uncertainty

Risk:

Positive/negative branch selection depends on `cond_or_uncond` layout.

Mitigation:

- Reuse existing branch selection helpers.
- Fallback if branch layout is unknown for branch-specific modes.

### R5. Attention intervention does not guarantee region-specific visual change

Risk:

Changing attention to `big breasts` may change face, clothing, pose, composition, or nothing obvious.

Mitigation:

- Require fixed-seed baseline/intervention comparison.
- Require fixed-seed prompt-pair comparison, such as concept present vs removed.
- Generate image-difference artifacts before interpreting region-specific effects.
- Allow manual ROI bbox scoring early; automatic semantic segmentation remains out of scope.
- Add controls with unrelated tokens and wrong calls.

### R6. Module path remains unknown

Risk:

`eligible_call_index` may not map to a stable transformer module path, limiting LoRA/merge usefulness.

Mitigation:

- Preserve block/metadata fields.
- Document `block=unknown` when unresolved.
- Add module-path resolver as later work.

### R7. Randomized seed invalidates image comparisons

Risk:

If ComfyUI workflow uses seed randomization, baseline/intervention visual comparison is invalid.

Mitigation:

- Manual protocol requires fixed seed.
- README must warn against randomize mode for intervention testing.

---

## Milestone Status

### M0. Planning Documents

Status: completed

Goal:

- Define separate-node intervention refactor for Codex.

Completed:

- [x] Create `INTERVENTION_SPEC.md`.
- [x] Create `INTERVENTION_TASKS.md`.
- [x] Create `INTERVENTION_PROGRESS.md`.

---

### M1. Survey Behavior Protection

Status: completed

Goal:

- Ensure the existing survey node remains observe-only.

Planned:

- [x] Add or confirm survey passthrough test.
- [x] Confirm survey schema remains stable.
- [x] Run baseline `python -m pytest -q`.

---

### M2. Shared Attention Math and Intervention Primitives

Status: completed

Goal:

- Implement tested tensor primitives before node integration.

Planned:

- [x] Add attention math helper tests.
- [x] Implement `token_attention_scale` primitive.
- [x] Implement `attention_logit_bias` primitive.
- [x] Confirm survey scoring tests still pass.

---

### M3. Intervention Config and Records

Status: completed

Goal:

- Add typed config and JSONL record builders for intervention.

Planned:

- [x] Add `InterventionConfig`.
- [x] Add step scope parser or reuse call scope parser.
- [x] Add intervention record builders.
- [x] Add tests for config validation and record JSON serialization.

---

### M4. Separate Intervention Node

Status: completed

Goal:

- Expose a second ComfyUI node without altering the survey node.

Planned:

- [x] Add `AnimaConceptInterventionModelPatch`.
- [x] Update extension node list to include both nodes.
- [x] Prevent override stacking conflicts.
- [x] Add import/schema smoke tests.

---

### M5. Intervention Override — Shadow Mode

Status: completed

Goal:

- Log would-be intervention effects while returning original output.

Planned:

- [x] Implement override skeleton.
- [x] Resolve concept token matches from `clip` + `prompt_text`.
- [x] Emit `intervention_plan`.
- [x] Implement eligibility/fallback logic.
- [x] Compute original vs modified attention statistics.
- [x] Confirm `shadow` mode returns original output.

---

### M6. Intervention Override — Intervene Mode

Status: completed

Goal:

- Safely return modified attention output for supported calls.

Planned:

- [x] Implement output reconstruction.
- [x] Add no-op parity tests for scale=1 and bias=0.
- [x] Apply intervention only to selected branches.
- [x] Apply call and step filters.
- [x] Emit intervention observation records.

---

### M7. Intervention and Image Difference Reporting

Status: completed

Goal:

- Summarize intervention logs into useful CSV/Markdown outputs.
- Generate fixed-seed image-difference evidence for prompt-pair and intervention comparisons.

Planned:

- [x] Add `scripts/summarize_intervention.py` or equivalent report helper.
- [x] Add `intervention_summary.json`.
- [x] Add `intervention_by_call.csv`.
- [x] Add `intervention_by_concept.csv`.
- [x] Add `intervention_report.md`.
- [x] Add `scripts/compare_intervention_images.py`.
- [x] Add `diff_abs.png`.
- [x] Add `diff_summary.json`.
- [x] Add manual ROI bbox scoring.
- [x] Add report tests.

---

### M8. Survey-to-Intervention Bridge

Status: completed

Goal:

- Make it easier to choose intervention targets from survey runs.

Planned:

- [x] Document `recommended_concept_targets.csv` as attention-only candidates.
- [x] Document that survey candidates must be cross-checked with fixed-seed prompt/image differences.
- [x] Add call-by-call aggregate concept heatmaps if feasible.
- [x] Preserve and document module metadata / unknown block state.

---

### M9. Documentation

Status: completed

Goal:

- Explain the two-node workflow and intervention limitations.

Planned:

- [x] Update README with survey vs intervention distinction.
- [x] Add fixed-seed intervention workflow examples.
- [x] Add no-op parity and suppression validation instructions.
- [x] Update planning docs after implementation.

---

### M10. Manual ComfyUI Validation

Status: not started

Goal:

- Confirm intervention node behavior in real ComfyUI workflow.

Planned:

- [ ] Confirm both nodes appear separately.
- [ ] Run fixed-seed no-op parity tests.
- [ ] Run fixed-seed prompt-pair image diff, including `big breasts` present vs removed.
- [ ] Run target suppression test for one survey candidate call.
- [ ] Run unrelated-token and wrong-call controls.
- [ ] Record results and screenshots/log paths.

---

## Current Next Step for Codex

Start M10 manual ComfyUI validation.

Recommended next validation:

1. Confirm both nodes appear separately in ComfyUI.
2. Run fixed-seed no-op parity checks for `shadow`, `token_attention_scale strength=1.0`, and `attention_logit_bias logit_bias=0.0`.
3. Run fixed-seed prompt-pair image diff for `big breasts` present vs removed.
4. Run one scoped suppression test from a survey candidate call.
5. Run unrelated-token and wrong-call controls.

Record ComfyUI outputs, screenshots/log paths, image diff paths, and JSONL report paths before marking M10 complete.

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

Manual validation:
- ...

Notes:
- ...

Next:
- ...
```

---

## Change Log

### 2026-05-29 — M9 documentation completed

Changed:

- Updated README with separate Survey and Intervention node workflow guidance.
- Added fixed-seed prompt-pair and baseline-vs-intervention image difference examples.
- Added intervention `shadow`, no-op parity, suppression, and control validation instructions.
- Documented intervention summary and image diff scripts in the public workflow.
- Updated intervention planning documents for M9 completion and M10 handoff.

Tests:

- command: `python -m pytest -q`
- result: `83 passed, 28 subtests passed`
- command: `git diff --check`
- result: passed with existing LF-to-CRLF working-copy warnings

Manual validation:

- Not run. Real ComfyUI node visibility, no-op parity, target suppression, prompt-pair diff, and controls remain M10 items.

Next:

- Start M10 manual ComfyUI validation.

### 2026-05-29 — Added survey-to-intervention bridge outputs

Changed:

- Documented `recommended_concept_targets.csv` as an attention-only candidate ranking.
- Added report text requiring fixed-seed prompt/image differences before LoRA or merge interpretation.
- Added call-by-call concept aggregate heatmaps under `concepts/aggregate_by_call`.
- Added `block` and `metadata` aggregation to `survey_by_concept.csv`.
- Added a Markdown `Block Metadata` section that explicitly reports `block=unknown`.
- Updated README heatmap/report guidance for call-level intervention target inspection.

Tests:

- command: `python -m pytest tests/test_survey_attention.py tests/test_reporting.py -q`
- result: `31 passed, 5 subtests passed`
- command: `python -m compileall -q anima_concept_survey tests scripts nodes.py`
- result: passed
- command: `python -m pytest -q`
- result: `83 passed, 28 subtests passed`
- command: `git diff --check`
- result: passed with existing LF-to-CRLF working-copy warnings

Manual validation:

- Not run. Real call-level heatmap inspection and target selection remain M10 items.

Next:

- Start M9: update README with the full two-node survey/intervention workflow and validation examples.

---

### 2026-05-29 — Added intervention reports and image diff tooling

Changed:

- Added `scripts/summarize_intervention.py` for JSONL-only intervention summaries.
- Added `intervention_summary.json`, `intervention_by_call.csv`, `intervention_by_concept.csv`, and `intervention_report.md` outputs.
- Added `scripts/compare_intervention_images.py` for fixed-seed image differences.
- Added `diff_abs.png`, `diff_summary.json`, and `diff_report.md` outputs.
- Added manual ROI bbox scoring via `--roi-bbox x,y,w,h`.
- Added tests for intervention reporting, empty observation logs, image diff artifacts, ROI scoring, and size mismatch errors.

Tests:

- command: `python -m pytest tests/test_intervention_reporting.py tests/test_image_diff.py -q`
- result: `4 passed`
- command: `python -m compileall -q anima_concept_survey tests scripts nodes.py`
- result: passed
- command: `python -m pytest -q`
- result: `81 passed, 28 subtests passed`
- command: `git diff --check`
- result: passed with existing LF-to-CRLF working-copy warnings

Manual validation:

- Not run. Real fixed-seed prompt-pair and intervention image comparisons remain M10 items.

Next:

- Start M8: improve the survey-to-intervention bridge and document survey candidate limits.

---

### 2026-05-29 — Added intervene output reconstruction

Changed:

- Implemented supported `intervene` mode output reconstruction for cross-attention calls.
- Applied `attention_logit_bias` and `token_attention_scale` only to selected concept-token branches.
- Preserved out-of-scope call/step behavior by returning original backend output.
- Kept unsupported layouts conservative through fallback handling.
- Connected the public intervention node so `mode=intervene` installs the intervention override.
- Added no-op parity, branch selection, call filter, step filter, fallback, and node installation tests.

Tests:

- command: `python -m pytest tests/test_intervention_override.py tests/test_nodes_intervention.py -q`
- result: `18 passed, 3 subtests passed`
- command: `python -m compileall -q anima_concept_survey tests nodes.py`
- result: passed
- command: `python -m pytest -q`
- result: `77 passed, 28 subtests passed`
- command: `git diff --check`
- result: passed with existing LF-to-CRLF working-copy warnings

Manual validation:

- Not run. Real ComfyUI no-op parity and target suppression remain M10 items.

Next:

- Start M7: add intervention JSONL reporting and fixed-seed image difference tooling with manual ROI bbox scoring.

---

### 2026-05-29 — Added shadow intervention override

Changed:

- Added `AnimaConceptInterventionAttentionOverride` with shadow-only behavior.
- Shadow mode resolves concept tokens, emits `intervention_plan`, checks eligibility, and writes observation/skipped/fallback records.
- Added would-be intervention stats: original/modified concept mass, score delta, probability delta, and attention-output delta norm.
- Wired the intervention node to install the shadow override while leaving `intervene` mode for M6.
- Added tests for original-output passthrough, no-op zero deltas, suppression nonzero deltas, skipped records, fallback records, and node installation.

Tests:

- command: `python -m pytest tests/test_intervention_override.py tests/test_nodes_intervention.py -q`
- result: `11 passed, 3 subtests passed`
- command: `python -m compileall -q anima_concept_survey tests nodes.py`
- result: passed
- command: `python -m pytest -q`
- result: `70 passed, 28 subtests passed`

Manual validation:

- Not run. Real ComfyUI shadow-mode validation remains a later M10 item.

Next:

- Start M6: implement supported `intervene` mode output reconstruction and no-op parity for modified attention output.

---

### 2026-05-29 — Added separate intervention node skeleton

Changed:

- Added `AnimaConceptInterventionModelPatch` as a separate ComfyUI node with explicit experimental warning text.
- Updated extension export list to return both Survey and Intervention nodes.
- Added no-op `off` mode passthrough and existing `optimized_attention_override` stacking rejection.
- Kept actual shadow/intervene runtime unimplemented until M5/M6, with a clear `NotImplementedError`.
- Added fake-`comfy_api` import smoke tests for node export, schema identity, `off` mode, and stacking guard.

Tests:

- command: `python -m pytest tests/test_nodes_intervention.py tests/test_node_schema_static.py -q`
- result: `6 passed`
- command: `python -m compileall -q anima_concept_survey tests nodes.py`
- result: passed
- command: `python -m pytest -q`
- result: `63 passed, 25 subtests passed`

Manual validation:

- Not run. Real ComfyUI node visibility remains an M10 manual validation item.

Next:

- Start M5: implement the intervention override skeleton and `shadow` mode that logs would-be intervention stats while returning original backend output.

---

### 2026-05-29 — Added intervention config and records

Changed:

- Added `InterventionConfig` with no-op-safe defaults, enum validation, numeric validation, and parsed call/step scopes.
- Added `parse_step_index_scope` using the same syntax as call scopes.
- Added intervention JSONL record builders for plan, observation, skipped, fallback, and summary events.
- Added tests for config validation, scope parsing, JSON serialization, concept match fields, and delta stat records.

Tests:

- command: `python -m pytest tests/test_intervention_config.py tests/test_intervention_records.py -q`
- result: `8 passed, 18 subtests passed`
- command: `python -m compileall -q anima_concept_survey tests`
- result: passed
- command: `python -m pytest -q`
- result: `58 passed, 25 subtests passed`

Manual validation:

- Not run. This pass only adds config/record boundaries and pure validation logic.

Next:

- Start M4: add the separate `Anima Concept Intervention Model Patch` node skeleton and export smoke tests without implementing runtime intervention yet.

---

### 2026-05-29 — Added tested attention intervention primitives

Changed:

- Added pure tensor attention math helpers for logits, probabilities, output reconstruction, token probability scaling, and attention logit bias.
- Added regression tests for helper shapes, normalization, no-op behavior, suppression, boosting, and copy/no-mutation behavior.
- Confirmed existing survey passthrough behavior remains protected by tests.
- Added a static schema guard for the existing survey node inputs because `comfy_api` is not available in the normal pytest environment.

Tests:

- command: `python -m pytest tests/test_attention_math.py -q`
- result: `6 passed`
- command: `python -m pytest tests/test_node_schema_static.py tests/test_attention_math.py -q`
- result: `7 passed`
- command: `python -m compileall -q anima_concept_survey tests`
- result: passed
- command: `python -m pytest -q`
- result: `50 passed, 7 subtests passed`

Manual validation:

- Not run. This pass is pure unit-testable helper groundwork only.

Next:

- Start M3: add `InterventionConfig`, step scope parsing, and intervention JSONL record builders.

---

### 2026-05-29 — Updated plan after prompt-removal validation

Changed:

- Recorded that fixed-seed removal of `big breasts` reduced the generated breast region.
- Promoted paired image-difference reporting and manual ROI bbox scoring from optional work to required evidence.
- Renamed public intervention kind from `token_logit_bias` to `attention_logit_bias`.
- Changed intervention defaults to no-op-safe values: `strength=1.0`, `logit_bias=0.0`.

Tests:

- Not run. Documentation-only update.

Next:

- Start M1/M2 with survey passthrough protection and tested attention intervention primitives.

---

### 2026-05-29 — Intervention planning documents created

Changed:

- Created separate-node intervention specification.
- Created Codex task breakdown.
- Created intervention progress tracker.
- Defined survey preservation, shadow mode, intervene mode, no-op parity, fallback behavior, and fixed-seed manual validation requirements.

Tests:

- Not run. Documentation-only artifact generation.

Next:

- Start M1: protect existing survey observe-only behavior with tests.
