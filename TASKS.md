# Tasks

## Milestone 1: Project Skeleton

- [x] Add `README.md` with install, purpose, and MVP usage.
- [x] Add `pyproject.toml` or minimal test config.
- [x] Add `__init__.py` for ComfyUI custom node loading.
- [x] Add `nodes.py` using ComfyUI V3 `ComfyExtension` and `define_schema()`.
- [x] Add package directory, likely `anima_concept_survey/`.
- [x] Add initial unit test directory.

## Milestone 2: Attention Observer Core

- [x] Create `survey_attention.py`.
- [x] Implement `SurveyConfig`.
- [x] Implement call-index parser for `all`, comma lists, and ranges.
- [x] Implement square spatial inference.
- [x] Implement Anima-like model detection.
- [x] Implement observe-only attention override.
- [x] Ensure observe mode returns original attention output exactly.
- [x] Add fallback reasons for unsupported shapes/signatures.
- [x] Add VRAM estimates and guards.

## Milestone 3: Runtime Records

- [x] Define JSONL schema version 1.
- [x] Write eligible attention observation records.
- [x] Record `step_index`, `num_steps`, and `eligible_call_index`.
- [x] Record q/k/v shape and inferred `image_len`, `text_len`, `spatial`.
- [x] Record branch information when `cond_or_uncond` is available.
- [x] Record block/module metadata when available.
- [x] Add compact top-token statistics.
- [x] Add concept/phrase score records for `concept_terms`.
- [x] Add summary records at step end or run finalize.

## Milestone 4: ComfyUI Node

- [x] Implement `Anima Concept Survey Model Patch`.
- [x] Inputs: `model`, `mode`, `capture_level`, `target_call_indices`, `branch_mode`, `jsonl_path`.
- [x] Inputs: `clip`, `prompt_text`, `concept_terms`.
- [x] Advanced inputs: `max_tokens`, `max_logits_mib`, `fail_mode`, `save_heatmaps`, `heatmap_dir`.
- [x] Output patched `MODEL`.
- [x] Reject existing `optimized_attention_override` with clear error.
- [x] Log install summary.
- [x] Keep `mode=off` as pass-through.
- [x] Resolve relative `jsonl_path` and `heatmap_dir` under ComfyUI `output`.
- [x] Use separate default output roots for JSONL logs and heatmaps.

## Milestone 5: Report Script

- [x] Create `scripts/parse_survey_log.py`.
- [x] Create `scripts/summarize_survey.py`.
- [x] Output `survey_summary.json`.
- [x] Output `survey_by_call.csv`.
- [x] Output `survey_by_token.csv`.
- [x] Output `survey_by_step.csv`.
- [x] Output `survey_by_branch.csv`.
- [x] Output `recommended_lora_targets.csv`.
- [x] Add Markdown report generation.

## Milestone 6: Tests

- [x] Test call-index parser.
- [x] Test square spatial inference.
- [x] Test branch selection.
- [x] Test observe mode exact passthrough.
- [x] Test unsupported shape fallback.
- [x] Test JSONL serialization.
- [x] Test report aggregation from fixture JSONL.
- [x] Test top-token ranking.
- [x] Test recommended target generation.
- [x] Test ComfyUI-output-relative path resolution.
- [x] Test concept phrase token matching.
- [x] Test concept phrase heatmap export.

## Milestone 7: Manual Validation

- [x] Install/symlink project under ComfyUI `custom_nodes`.
- [x] Confirm node appears/imports in ComfyUI.
- [ ] Run true baseline without survey node.
- [ ] Run observe mode with same seed/prompt/sampler.
- [ ] Confirm generated image does not change in observe mode.
- [x] Confirm JSONL contains eligible attention calls.
- [x] Confirm q/k/v shape matches expected Anima path.
- [ ] Confirm report script reads the JSONL.
- [ ] Confirm `concept_terms` phrase heatmap output in a fresh ComfyUI run.
- [ ] Confirm JSONL/heatmap defaults are clear in saved workflows after node refresh.

## Milestone 8: Heatmap Export

- [x] Add attention map aggregation by token.
- [x] Add 64x64 map export as `.npy`.
- [x] Add PNG heatmap export.
- [x] Add step/block/call filters.
- [x] Add top-token heatmap batch export.
- [x] Add 512px color preview PNGs.
- [x] Add aggregate token heatmaps under `heatmaps/aggregate`.
- [x] Add heatmap manifest files.
- [x] Add phrase/concept heatmaps from `concept_terms` under `heatmaps/concepts`.
- [x] Add aggregate phrase/concept heatmaps under `heatmaps/concepts/aggregate`.

## Milestone 9: Slider and Merge Integration

- [ ] Map high-scoring call/block indices to Anima module names where possible.
- [ ] Generate candidate `network_reg_dims` snippets for slider LoRA.
- [ ] Generate prompt YAML suggestions from high-impact tokens/concepts.
- [ ] Add comparison mode for prompt A/B activation deltas.
- [ ] Add report section for merge-sensitive blocks.

## Later

- [x] Add explicit concept list input.
- [ ] Explore concept embedding extraction through ComfyUI CLIP/T5.
- [ ] Add ConceptAttention-style output-space vector comparison.
- [ ] Support rectangular image layouts.
- [ ] Support video/Cosmos layouts.
- [ ] Investigate composition with other attention override nodes.
