# Progress

## Current Status

Status: MVP implementation started and unit-tested.

The repository now contains a ComfyUI V3 node skeleton, observe-only attention override, JSONL records, report scripts, and synthetic tensor tests.

## Decisions Made

### 2026-05-26

- Chose a ComfyUI `optimized_attention_override` based observer as the first implementation path.
- Decided not to port the full ConceptAttention Flux pipeline into Anima for MVP.
- Decided to reuse the proven AFM observation path as the closest local reference.
- Decided that the first useful artifact is machine-readable attention survey data, not a trained LoRA or merge.
- Decided that output should support downstream slider/merge workflows through CSV/JSON reports.

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
- Added unit tests for parser, progress, branch selection, observe passthrough, fallback, and reporting.
- Verified with `python -m pytest -q`: 11 passed, 5 subtests passed.

## In Progress

- None.

## Next Milestone

Manual ComfyUI validation:

1. Symlink/install this folder under ComfyUI `custom_nodes`.
2. Confirm the node appears under `model_patches/anima`.
3. Run fixed-seed baseline without the node.
4. Run fixed-seed observe mode with the node.
5. Confirm image output is unchanged and JSONL contains eligible observations.

## Open Questions

- Can token text be recovered reliably from every connected `CLIP` variant, or do some Anima loaders require model-specific decoder adapters?
- Does Anima always expose text length 512 for relevant runs, or should text length be inferred per call?
- Should heatmap export be in the node itself, or only in the report script?
- Should the project directory be renamed from `coceptattention` to `conceptattention`?

## Known Risks

- Runtime token ordering must be verified from Anima logs before labeling T2I/I2T semantics.
- Observe mode must call the original attention backend to avoid changing generated images.
- Full attention logits are large; top-k/sampled summaries may be necessary for practical VRAM use.
- Composing with other `optimized_attention_override` nodes is out of scope for MVP.
