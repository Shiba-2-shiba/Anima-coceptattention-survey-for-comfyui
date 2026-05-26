# Progress

## Current Status

Status: MVP implementation is working and unit-tested; ComfyUI import and attention logging have been manually observed.

The repository now contains a ComfyUI V3 node, observe-only attention override, JSONL records, report scripts, token heatmaps, aggregate heatmaps, phrase heatmaps via `concept_terms`, and synthetic tensor tests.

## Decisions Made

### 2026-05-26

- Chose a ComfyUI `optimized_attention_override` based observer as the first implementation path.
- Decided not to port the full ConceptAttention Flux pipeline into Anima for MVP.
- Decided to reuse the proven AFM observation path as the closest local reference.
- Decided that the first useful artifact is machine-readable attention survey data, not a trained LoRA or merge.
- Decided that output should support downstream slider/merge workflows through CSV/JSON reports.

### 2026-05-27

- Decided that `heatmap` mode should still write JSONL because PNG/NPY files cannot preserve token text, call, step, branch, and score metadata.
- Separated default JSONL and heatmap output locations:
  - JSONL: `anima_concept_survey/logs/survey.jsonl`
  - heatmaps: `anima_concept_survey/heatmaps`
- Added `concept_terms` as a phrase-level heatmap input rather than requiring users to manually interpret individual tokenizer tokens.
- Kept `concept_terms` as token-attention summation over the existing prompt tokens, not a separate ConceptAttention concept stream.

## Evidence Collected

### AFM Runtime Evidence

`AFM-for-anima-experimental` already captures Anima/Cosmos cross-attention calls with stable square image-query layout:

```text
q=(2, 16, 4096, 128)
k=(2, 16, 512, 128)
v=(2, 16, 512, 128)
spatial=(64, 64)
```

This validates that an observer node can inspect the attention calls needed for a first survey implementation.

### ConceptAttention Design Evidence

`ConceptAttention` computes heatmaps from vectors by comparing image vectors and concept vectors across selected layers/timesteps.

MVP adaptation:

- Use existing prompt/text keys first.
- Add explicit concept vectors later.

### exploring-mmdit Design Evidence

`exploring-mmdit` separates T2I and I2T attention regions for Flux/SD3 and aggregates attention by timestep/block.

MVP adaptation:

- Use this as the analysis/reporting model.
- Do not copy the diffusers attention processor path directly.

### Slider Node Connection

`Comfyui-anima-slider-node` already targets Anima modules such as:

- cross attention projections
- self attention projections
- MLP layers

Survey reports should rank block/call/token importance so those target patterns can be narrowed.

### Manual Runtime Evidence

A ComfyUI run with `prompt_text="big breasts,"` produced eligible cross-attention observations:

```text
q_shape=[2, 16, 4096, 128]
k_shape=[2, 16, 512, 128]
v_shape=[2, 16, 512, 128]
spatial=[64, 64]
branch=positive
text_len=512
image_len=4096
```

The token map decoded:

- `token_index=0`, `token_text="big"`, `token_source="qwen3_06b"`
- `token_index=1`, `token_text=" breasts"`, `token_source="qwen3_06b"`

The first aggregate implementation also showed many other top tokens, including undecoded tokens. This confirmed that token aggregate heatmaps answer "which individual tokens ranked high", not "show only this phrase". `concept_terms` was added to address that gap.

## Completed

- Reviewed reference repository layout.
- Reviewed README files for all four reference repositories.
- Identified AFM observe path as the lowest-risk ComfyUI integration point.
- Identified ConceptAttention vector/heatmap logic as a later-stage enhancement.
- Defined MVP node and report outputs.
- Created project planning documents:
  - `SPEC.md`
  - `PROGRESS.md`
  - `TASKS.md`
- Added ComfyUI package entrypoint and V3 node registration.
- Added `Anima Concept Survey Model Patch`.
- Added observe-only `AnimaConceptSurveyAttentionOverride`.
- Added JSONL attention observation/fallback/skipped/run-summary records.
- Added report aggregation helpers and CLI scripts.
- Added CLIP/prompt-based token text restoration.
- Added top-token `.npy` and grayscale `.png` heatmap export.
- Added ComfyUI-output-relative path resolution for `jsonl_path` and `heatmap_dir`.
- Added default JSONL path under `anima_concept_survey/logs`.
- Added top-token 512px color preview PNGs.
- Added token aggregate heatmaps under `heatmaps/aggregate`.
- Added heatmap manifest files.
- Added `concept_terms` input for phrase-level heatmaps.
- Added phrase/concept heatmaps under `heatmaps/concepts`.
- Added phrase/concept aggregate heatmaps under `heatmaps/concepts/aggregate`.
- Added unit tests for parser, progress, branch selection, observe passthrough, fallback, and reporting.
- Added tests for path resolution, concept phrase matching, and concept heatmap export.
- Verified with `python -m pytest -q`: 14 passed, 5 subtests passed.
- Confirmed ComfyUI can import the node.
- Confirmed JSONL contains eligible Anima-like cross-attention calls in a manual run.

## In Progress

- Fresh ComfyUI validation of the new `concept_terms` output path.

## Next Milestone

Manual ComfyUI validation:

1. Restart ComfyUI so the new `concept_terms` input is loaded.
2. Run a fresh workflow with:
   - `capture_level=heatmap`
   - `save_heatmaps=true`
   - `prompt_text` equal to the actual generation prompt
   - `concept_terms=big breasts`
   - a clean `heatmap_dir`, such as `anima_concept_survey/heatmaps_big_breasts`
3. Confirm `heatmaps_big_breasts/concepts/aggregate/aggregate_positive_concept_big_breasts_preview.png` is created.
4. Confirm `concept_scores` appears in JSONL attention observation records.
5. Run fixed-seed baseline without the node.
6. Run fixed-seed observe mode with the node.
7. Confirm image output is unchanged by observe mode.

## Open Questions

- Can token text be recovered reliably from every connected `CLIP` variant, or do some Anima loaders require model-specific decoder adapters?
- Does Anima always expose text length 512 for relevant runs, or should text length be inferred per call?
- Should heatmap export remain in the node itself, or should heavier aggregate/overlay outputs move to a report/postprocess script?
- Should the project directory be renamed from `coceptattention` to `conceptattention`?
- Should an image overlay node be added to render concept heatmaps over the final generated image?
- Should `concept_terms` support explicit tokenizer-source selection when multiple encoders expose overlapping token streams?

## Known Risks

- Runtime token ordering must be verified from Anima logs before labeling T2I/I2T semantics.
- Observe mode must call the original attention backend to avoid changing generated images.
- Full attention logits are large; top-k/sampled summaries may be necessary for practical VRAM use.
- Composing with other `optimized_attention_override` nodes is out of scope for MVP.
- Token aggregate heatmaps can include undecoded internal/special tokens, so phrase-level inspection should use `concept_terms`.
- `concept_terms` currently sums matched prompt-token attention; it is not a separate encoded concept stream and is not a final-image segmentation mask.
