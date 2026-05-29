# INTERVENTION_SPEC.md — Separate Intervention Node Refactor Specification

Project: Anima ConceptAttention Survey for ComfyUI  
Target branch: `dev`  
Planning date: 2026-05-29  
Primary decision: implement intervention as a separate node, not as an extension of the observe-only survey node.
Revision note: updated after fixed-seed prompt-removal validation showed that removing `big breasts` changes the breast region even when the observe-only attention heatmap is not breast-localized.

---

## 0. Summary

The existing `Anima Concept Survey Model Patch` should remain an observe-only inspection node. It records cross-attention observations, concept token matches, concept heatmaps, reports, and comparison data, but it must not change generated output.

The new work should add a second explicit node:

```text
Anima Concept Intervention Model Patch
```

This new node is allowed to change generation by suppressing or boosting attention paid to selected prompt concept tokens at selected attention calls / steps / branches. It should be treated as an experimental causal-probing tool, not as a replacement for the survey node.

The long-term goal is to move from:

```text
Where did the model attend to this prompt token?
```

toward:

```text
What changes when this prompt token's attention contribution is suppressed or boosted here?
```

This is still not proof of semantic causality. It is a controlled intervention heuristic that can support LoRA / slider / merge investigation after repeated validation.

The practical analysis target is now explicitly three-view:

```text
observe attention candidates
+ compare fixed-seed prompt/output differences
+ run scoped token interventions
```

No single view is sufficient for LoRA or merge decisions.

---

## 1. Background and Evidence

Recent ComfyUI validation showed that the survey pipeline now works as an observation system:

- the workflow used `AnimaConceptSurveyModelPatch` with `mode=observe`, `capture_level=heatmap`, `branch_mode=positive_only`, and `heatmap_output=concepts_only`;
- `big breasts` was matched as one concept occurrence in the Qwen token stream;
- the aggregate manifest showed `concept_uid=bigbreasts__qwen3_06b__occ0__tok078-079`, `token_indices=[78, 79]`, and `observation_count=840`;
- the duplicate same-span concept problem did not appear in that run.
- a fixed-seed prompt-pair test later showed that removing `big breasts` visibly reduced the breast region in the generated image.

However, the generated aggregate heatmap did not behave like a final-image body-part segmentation map. The heatmap is a latent cross-attention summary, and the observed location did not directly answer whether the concept caused the final breast-region appearance. The prompt-pair result confirms the concept can affect the intended region even when observe-only attention is not visually localized there.

Therefore, the next refactor should not try to reinterpret observe-only heatmaps as causal evidence. It should add explicit intervention and paired-image comparison workflows that change one controlled factor at a time and then compare output behavior under fixed seed / fixed workflow conditions.

---

## 2. Non-negotiable Constraints

### 2.1 Keep survey observe-only

The existing survey node must remain a pure observer.

Requirements:

- `AnimaConceptSurveyModelPatch` must keep exact passthrough behavior.
- The survey override must continue returning the original attention backend output.
- Existing survey tests must continue to pass.
- No intervention option should be hidden inside `mode=observe`.

### 2.2 Intervention must be explicit

The new node may modify generation, but it must make that clear in naming, schema, tooltips, README, and logs.

Required display name:

```text
Anima Concept Intervention Model Patch
```

Required category:

```text
model_patches/anima
```

Required description should contain wording equivalent to:

```text
Experimental: changes cross-attention output for selected concept tokens.
```

### 2.3 Separate failure semantics

The intervention node must be conservative.

- `fail_mode=fallback`: if unsupported shape / mask / branch layout / output layout is encountered, return the original backend output and emit an intervention fallback record.
- `fail_mode=raise`: raise the error for debugging.
- Unsupported calls must not silently produce partially modified or malformed tensors.

### 2.4 JSONL compatibility

The survey JSONL schema stays compatible. Intervention logs may use their own events while still using `schema_version=1`.

Allowed new events:

```text
intervention_plan
intervention_observation
intervention_skipped
intervention_fallback
intervention_summary
```

### 2.5 No claim of final-image segmentation

Intervention records and reports must avoid implying that attention maps are final-image masks.

Use labels like:

```text
attention intervention
attention delta
output delta candidate
causal-probe heuristic
```

Avoid labels like:

```text
true causal map
segmentation map
LoRA target proof
```

---

## 3. Public Node Specification

### 3.1 Node class

Add a new node class in `nodes.py` or a new node module imported by `nodes.py`:

```python
class AnimaConceptInterventionModelPatch(io.ComfyNode):
    ...
```

Update the extension entrypoint so both nodes are available:

```python
return [AnimaConceptSurveyModelPatch, AnimaConceptInterventionModelPatch]
```

### 3.2 Inputs

Minimum recommended inputs:

```text
model: MODEL
clip: CLIP
mode: off | shadow | intervene
intervention_kind: token_attention_scale | attention_logit_bias
prompt_text: multiline string
intervention_terms: multiline string
target_call_indices: string, default="all"
target_step_indices: string, default="all"
branch_mode: both | positive_only | negative_only
strength: float, default=1.0
logit_bias: float, default=0.0
max_steps: int, default=0
jsonl_path: string
max_logits_mib: float, default=1024.0
fail_mode: fallback | raise
```

Optional later inputs:

```text
token_component: phrase | each_token | first_token | explicit_indices
explicit_token_indices: string
save_attention_delta_heatmaps: boolean
heatmap_dir: string
```

### 3.3 Output

```text
model: MODEL
```

### 3.4 Mode semantics

```text
off
  Return input model unchanged.

shadow
  Resolve concept tokens, compute original and would-be-modified attention statistics,
  emit JSONL, but return the original backend output.
  This is for safety validation.

intervene
  Resolve concept tokens, modify attention probabilities/logits for scoped calls,
  return modified attention output.
```

### 3.5 Strength semantics

For `token_attention_scale`:

```text
strength = 1.0  -> no-op
strength = 0.0  -> suppress selected concept attention mass as much as possible before renormalization
strength > 1.0  -> boost selected concept attention mass before renormalization
```

For `attention_logit_bias`:

```text
logit_bias < 0  -> suppress selected token keys
logit_bias = 0  -> no-op
logit_bias > 0  -> boost selected token keys
```

Earlier drafts used the name `token_logit_bias`; public UI and records should prefer `attention_logit_bias` because the bias is applied to attention logits, not final image/model logits. A compatibility alias is acceptable only if documented.

If both `strength` and `logit_bias` are present, only the field relevant to the selected `intervention_kind` is used.

---

## 4. Internal Architecture

### 4.1 New modules

Recommended module layout:

```text
anima_concept_survey/
  intervention_config.py
  intervention.py
  intervention_records.py
  intervention_reporting.py        # optional in first pass
  attention_math.py                # shared math helpers, if extraction is useful
```

The package name can remain `anima_concept_survey` to avoid a larger repository rename.

### 4.2 Reuse existing utilities

Reuse, do not duplicate:

```text
concepts.py       concept parsing and token matching
branches.py       branch selection helpers
selectors.py      call scope and shape helpers
progress.py       sampler progress extraction
paths.py          ComfyUI output path resolution
writer.py         JSONL writer
metadata.py       transformer metadata discovery
```

### 4.3 Shared attention math

Consider extracting shared helpers from the current observer:

```python
def compute_attention_logits(q, k, scale): ...
def compute_attention_probs(logits): ...
def attention_output_from_probs(probs, v): ...
def apply_token_attention_scale(probs, token_indices, scale): ...
def apply_attention_logit_bias(logits, token_indices, bias): ...
```

Extraction is recommended only if it simplifies tests. Do not destabilize the survey node for a cosmetic refactor.

---

## 5. Intervention Algorithm

### 5.1 Eligibility checks

The intervention override should use similar checks to the survey observer:

- attention args must be valid 4D tensors;
- `skip_reshape` must be true unless a tested output adapter exists;
- masks are unsupported at first unless tested;
- query length and text length must indicate cross-attention;
- query length must map to a square latent grid if heatmaps are requested;
- branch selection must be known for branch-specific intervention;
- memory estimate must be below `max_logits_mib`.

### 5.2 Concept matching

Use existing `build_concept_token_matches()` against `prompt_text` and `clip` tokenization.

If intervention terms are unmatched or ambiguous, emit diagnostics and do not intervene for that term.

If token indices are out of range for a given runtime `text_len`, emit an alignment warning and skip that call.

### 5.3 Scope filters

Intervention applies only when all filters match:

```text
step in target_step_indices or all
eligible_call_index in target_call_indices or all
branch in branch_mode selection
concept token indices valid for current text_len
```

Unselected calls and unselected branches must be mathematically unmodified.

### 5.4 Probability modification: token_attention_scale

Given:

```python
probs: [batch, heads, image_query, text_key]
token_indices: list[int]
scale: float
```

Apply:

```python
modified = probs.clone()
modified[..., token_indices] *= scale
modified = modified / modified.sum(dim=-1, keepdim=True).clamp_min(eps)
```

Notes:

- `scale=1.0` must be a no-op within numerical tolerance.
- `scale=0.0` should suppress selected token keys while preserving a normalized distribution.
- Use an epsilon guard.

### 5.5 Logit modification: attention_logit_bias

Given:

```python
logits: [batch, heads, image_query, text_key]
token_indices: list[int]
bias: float
```

Apply:

```python
modified_logits = logits.clone()
modified_logits[..., token_indices] += bias
modified_probs = softmax(modified_logits, dim=-1)
```

Notes:

- `bias=0.0` must be a no-op within numerical tolerance.
- Large negative bias approximates ablation.
- This mode is usually more numerically stable than direct probability scaling.

### 5.6 Output reconstruction

For supported calls, return the modified attention output.

Expected internal computation:

```python
out = torch.matmul(modified_probs.to(v.dtype), v)
```

But the exact return layout must be validated against the original backend for the current `optimized_attention_override` signature.

Required safety test:

```text
intervention no-op output == original backend output, within tolerance
```

If the modified output layout cannot be proven compatible, fallback to the original backend and emit `intervention_fallback`.

### 5.7 Original backend comparison

For `shadow` mode:

- call the original backend and return it;
- compute what would have been changed;
- emit stats such as original concept mass, modified concept mass, and mean probability delta.

For `intervene` mode:

- do not call the original backend twice unless needed for a debug/no-op parity mode;
- avoid doubling runtime cost by default.

---

## 6. JSONL Events

### 6.1 intervention_plan

Emit once at initialization when possible.

```json
{
  "schema_version": 1,
  "event": "intervention_plan",
  "run_id": "intervention-...",
  "mode": "intervene",
  "intervention_kind": "attention_logit_bias",
  "prompt_text": "...",
  "intervention_terms": "big breasts",
  "matches": [...],
  "target_call_indices": "10,14",
  "target_step_indices": "all",
  "branch_mode": "positive_only",
  "strength": 0.0,
  "logit_bias": -5.0
}
```

### 6.2 intervention_observation

Emit for each scoped eligible call.

Required fields:

```json
{
  "schema_version": 1,
  "event": "intervention_observation",
  "run_id": "intervention-...",
  "step_index": 4,
  "num_steps": 30,
  "eligible_call_index": 10,
  "branch": "positive",
  "block": "unknown-or-module-path",
  "metadata": {},
  "intervention_kind": "attention_logit_bias",
  "term": "big breasts",
  "concept_uid": "...",
  "token_indices": [78, 79],
  "token_texts": [" big", " breasts"],
  "original_score_mean": 0.001,
  "modified_score_mean": 0.0002,
  "score_delta": -0.0008,
  "mean_abs_prob_delta": 0.0001,
  "max_abs_prob_delta": 0.01,
  "output_delta_norm": 0.12,
  "estimated_logits_mib": 512.0
}
```

### 6.3 intervention_skipped

Emit when a call is eligible but out of scope.

Reasons:

```text
max_steps_reached
target_step_skipped
target_call_skipped
branch_not_selected
concept_not_matched
concept_token_index_out_of_range
```

### 6.4 intervention_fallback

Emit when intervention could not be safely applied.

Reasons:

```text
missing_sigmas
mask_shape_unsupported
not_skip_reshape
not_cross_attention
cannot_infer_spatial_shape
branch_layout_unknown
vram_guard_exceeded
output_layout_unverified
runtime_error
```

### 6.5 intervention_summary

Emit at finalize if a reliable finalize hook exists; otherwise the report script can summarize records.

Fields:

```json
{
  "schema_version": 1,
  "event": "intervention_summary",
  "run_id": "intervention-...",
  "observed_calls": 100,
  "modified_calls": 20,
  "skipped_calls": 80,
  "fallback_calls": 0,
  "fallback_reasons": {}
}
```

---

## 7. Reporting Requirements

First pass reporting can be simple, but it must support the fixed-seed comparison workflow that exposed the current limitation: `big breasts` changed the generated breast region even though the observe-only attention heatmap was not breast-localized.

Add intervention summary script:

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

Minimum useful columns:

```text
run_id
term
concept_uid
branch
eligible_call_index
observation_count
original_score_mean
modified_score_mean
score_delta
mean_abs_prob_delta
max_abs_prob_delta
output_delta_norm
intervention_kind
strength
logit_bias
block
```

Markdown should clearly state:

```text
This report describes attention intervention effects, not final-image ground-truth causality.
```

Add a required paired-image comparison script before treating intervention results as LoRA / merge guidance:

```bash
python scripts/compare_intervention_images.py baseline.png variant.png --out-dir diff_report
```

Outputs:

```text
diff_abs.png
diff_signed_preview.png        # optional if useful
diff_summary.json
diff_report.md
```

Minimum `diff_summary.json` fields:

```text
image_shape
mean_abs_delta
max_abs_delta
percentile_95_abs_delta
changed_pixel_ratio
roi_delta_over_whole_delta      # present only when ROI is supplied
roi_bbox                        # present only when ROI is supplied
```

The script should accept an optional manual ROI bbox first, before any automatic body-part detection:

```bash
python scripts/compare_intervention_images.py baseline.png variant.png --out-dir diff_report --roi-bbox x,y,w,h
```

Automatic semantic body-part detection remains out of scope for the first pass.

---

## 8. Relationship to Current Survey Outputs

The survey node remains the recommended first step:

1. Run survey with `positive_only`, `concepts_only`, fixed seed.
2. Inspect `recommended_concept_targets.csv` as attention-only candidate calls.
3. Generate a fixed-seed prompt-pair baseline such as `big breasts` present vs removed, and save image-difference outputs.
4. Choose one or a few candidate calls using attention ranking plus prompt-pair image evidence.
5. Run the new intervention node with fixed seed and target calls.
6. Compare generated outputs, paired-image diffs, optional manual ROI scores, and intervention logs.

The current `recommended_concept_targets.csv` should be relabeled in documentation as an attention-candidate ranking, not as a proven LoRA target ranking.

---

## 9. Call / Module Metadata Requirement

A major limitation remains that `eligible_call_index` may not map to a stable model module path.

The intervention refactor should not block on full module-path support, but it should preserve and improve metadata wherever possible.

Minimum:

- include `eligible_call_index`, shape, text_len, image_len, heads, and block metadata in all intervention observations;
- preserve `metadata.py` discovery behavior;
- document when `block=unknown`.

Preferred later improvement:

- identify actual Anima transformer block / attention module path;
- expose module path in survey and intervention records;
- use module path in reports and LoRA/merge candidate ranking.

---

## 10. Testing Requirements

Every behavior-changing patch must include tests.

Required automated tests:

1. Survey node remains observe-only passthrough.
2. Intervention config validates modes and numeric ranges.
3. New node is exported while existing survey node remains exported.
4. Concept matching is reused and preserves `concept_uid`.
5. `token_attention_scale` with `scale=1.0` is no-op.
6. `token_attention_scale` with `scale=0.0` reduces target token probability mass.
7. `attention_logit_bias` with `bias=0.0` is no-op.
8. Negative `attention_logit_bias` reduces target token probability mass.
9. Branch selection modifies only selected branch in a mock tensor.
10. Target call filter modifies only selected eligible call.
11. Unsupported mask / non-`skip_reshape` falls back in fallback mode.
12. `shadow` mode returns original output and emits would-be delta stats.
13. `intervene` mode returns modified output for a supported mock attention call.
14. JSONL records are emitted with expected event names and required fields.
15. Summary script writes CSV/Markdown outputs.
16. Image comparison script writes diff PNG and JSON summary.
17. Image comparison ROI bbox scoring reports `roi_delta_over_whole_delta` when ROI is supplied.

Run after each patch:

```bash
python -m pytest -q
```

---

## 11. Manual Validation Protocol

### 11.1 Fixed-seed setup

Use fixed seed, fixed prompt, fixed negative prompt, fixed sampler, fixed steps, fixed CFG, fixed latent size.

Do not use randomize mode during intervention comparison.

### 11.2 No-op parity

Run:

```text
A: no intervention node
B: intervention node, mode=shadow
C: intervention node, mode=intervene, token_attention_scale strength=1.0
D: intervention node, mode=intervene, attention_logit_bias bias=0.0
```

Expected:

- A/B/C/D should match visually and, if images are available as arrays, numerically within expected deterministic tolerance.
- JSONL should show observations for B/C/D.

### 11.3 Suppression test

Run:

```text
intervention_terms=big breasts
target_call_indices=<top survey candidate, e.g. 10>
branch_mode=positive_only
intervention_kind=attention_logit_bias
logit_bias=-5.0
```

Expected:

- intervention JSONL shows reduced concept mass;
- generated image may change;
- if image changes, compare with baseline under same seed.

### 11.4 Prompt-pair image difference

Run a prompt-pair comparison before interpreting intervention results:

```text
A: original prompt containing big breasts
B: same prompt with big breasts removed
C: same prompt with small breasts or flat chest, if the model accepts the wording
```

Expected:

- fixed seed and all sampler settings are identical;
- image difference output shows whether the concept affects the expected visual region;
- optional manual ROI bbox quantifies whether the expected region changed more than the whole image average.

### 11.5 Control tests

Run controls:

```text
unrelated token suppression
wrong call suppression
negative_only suppression
small bias / strong bias sweep
```

Expected:

- unrelated token should not produce the same region-specific difference as target concept;
- wrong call should have smaller or different effect;
- stronger bias should generally produce larger attention/output deltas, but visual behavior may be nonlinear.

---

## 12. Out of Scope for This Refactor

Do not implement in the first intervention pass:

- LoRA training automation;
- model merge automation;
- final-image semantic segmentation;
- automatic breast/body-part detection;
- automatic prompt rewriting;
- separate ConceptAttention concept-vector stream;
- composition with other existing `optimized_attention_override` patches;
- full multi-encoder attention-key resolver unless a validation failure requires it.

Manual ROI bbox scoring and plain image-difference reporting are in scope. They are not semantic segmentation and should be treated as user-supplied measurement aids.

---

## 13. Completion Criteria

The separate-node intervention refactor is complete when:

- [x] existing survey node still passes observe-only passthrough tests;
- [ ] new intervention node appears separately in ComfyUI;
- [x] `shadow` mode logs would-be intervention stats but does not change output;
- [x] `intervene` mode modifies supported cross-attention calls;
- [x] no-op intervention is numerically equivalent to original output in tests;
- [x] target-token suppression reduces target attention mass in tests;
- [x] branch and call filters are respected;
- [x] unsupported calls safely fallback;
- [x] intervention JSONL records are usable for reporting;
- [x] paired-image diff script produces diff PNG and JSON summary;
- [x] manual ROI bbox scoring is available for fixed-seed prompt/intervention comparisons;
- [x] README explains the difference between survey and intervention;
- [ ] manual fixed-seed no-op and suppression validations are completed.
