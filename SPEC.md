# Anima ConceptAttention Survey for ComfyUI - Specification

## Purpose

ComfyUI上でAnima/Cosmos系モデルの画像生成中に、どのattention key/token/block/stepが強く使われているかを観測する。

主な利用目的:

- `Comfyui-anima-slider-node` のLoRA学習対象選定
- Anima向けprompt slider設計
- マージモデル作成時のblock/token重要度分析
- ConceptAttention系のheatmap可視化をAnima実行環境へ段階的に導入するための土台作り

## Scope

### MVP

MVPでは、モデル内部を大きく改造せず、ComfyUIの `MODEL` に `optimized_attention_override` を追加する観測専用ノードを作る。

対象:

- Anima/Cosmos DiT系のComfyUI実行パス
- square image query grid
- text/image cross-attention
- CFG branch別の観測
- JSONL/CSV向けの機械可読ログ
- 64x64 heatmap生成に使えるattention map統計

非対象:

- 学習、LoRA生成、モデルマージの直接実行
- attentionの編集
- 非square latent、動画、長方形レイアウト
- Diffusers pipelineへの移植
- ConceptAttentionのconcept stream完全移植

## Reference Findings

### AFM-for-anima-experimental

最重要の土台。

既存AFMノードはComfyUI上のAnima実行で、以下のクロスアテンション形状を捕捉できている。

```text
q=(2, 16, 4096, 128)
k=(2, 16, 512, 128)
v=(2, 16, 512, 128)
spatial=(64, 64)
```

これは「4096個の画像query tokenが、512個のtext key tokenへどうattentionしているか」を調査するのに十分な入口になる。

利用する考え方:

- `optimized_attention_override`
- `mode=observe`
- `eligible_call_index`
- `step_index`
- CFG branch識別
- JSONL schema
- observe/edit比較ではなく、まずobserve単独の安定記録

### ConceptAttention

研究実装として、以下の考え方を参考にする。

- concept vectorとimage patch vectorのdot product
- output-space heatmap
- cross-attention heatmap
- layer/timestep集約

ただしMVPでは、Anima内部に追加concept streamを入れない。
まず既存prompt tokenへのattentionを観測し、後続フェーズでconcept promptを別入力として扱う。

### exploring-mmdit

参考にする設計:

- T2I: image tokens attending to text tokens
- I2T: text tokens attending to image tokens
- block/timestepごとの集約
- token indexからword attention mapを作る流れ

Animaの実装では、実際のtensor shapeとtoken orderをログで確定してから同等の分解を行う。

### Comfyui-anima-slider-node

調査結果の利用先。

既存のLoRA target候補:

- `model.diffusion_model.blocks.*.self_attn.*_proj`
- `model.diffusion_model.blocks.*.cross_attn.*_proj`
- `model.diffusion_model.blocks.*.mlp.layer1`
- `model.diffusion_model.blocks.*.mlp.layer2`

survey側では、block/call/token単位の重要度を出し、slider nodeの `network_reg_dims` やprompt YAML設計に接続できる形を目指す。

## Node Design

### Node 1: Anima Concept Survey Model Patch

Category:

```text
model_patches/anima
```

Inputs:

- `model`: ComfyUI MODEL
- `clip`: ComfyUI CLIP, used for token text restoration
- `mode`: `observe`, `off`
- `capture_level`: `summary`, `tokens`, `heatmap`
- `prompt_text`: prompt used to build token index labels
- `target_call_indices`: `all`, comma list, range
- `diagnostic_call_indices`: `all`, comma list, range
- `branch_mode`: `both`, `positive_only`, `negative_only`
- `max_tokens`: token上位N件
- `max_steps`: optional cap for smoke runs
- `jsonl_path`: optional output path
- `save_heatmaps`: bool
- `heatmap_dir`: optional output dir
- `max_logits_mib`: VRAM guard
- `fail_mode`: `fallback`, `raise`

Outputs:

- patched `MODEL`

Behavior:

- `mode=off`: 元のmodelを返す
- `mode=observe`: original attention backendを呼び、観測だけ行う
- `prompt_text` が指定された場合、`clip.tokenize()` から `token_index -> token_id/token_text/token_source` を復元する
- 既存の `optimized_attention_override` がある場合はエラーにする
- fallback reasonを必ず記録する
- 観測ができないattention callはoriginal backendへ戻す

### Node 2: Anima Survey Report

Inputs:

- `jsonl_path`
- `report_format`: `json`, `csv`, `markdown`, `all`
- `late_start_step`
- `top_k`

Outputs:

- `report_json`
- `report_path`
- `summary_csv_path`

Behavior:

- JSONLを読み、block/call/step/branch/token単位で集計
- slider/merge向けの推奨target候補を生成

## Data Schema

### JSONL Record

Each eligible observation should include:

```json
{
  "schema_version": 1,
  "event": "attention_observation",
  "step_index": 0,
  "num_steps": 24,
  "eligible_call_index": 7,
  "branch": "positive",
  "batch_indices": [1],
  "block": "unknown",
  "module_path": "unknown",
  "q_shape": [2, 16, 4096, 128],
  "k_shape": [2, 16, 512, 128],
  "v_shape": [2, 16, 512, 128],
  "spatial": [64, 64],
  "text_len": 512,
  "image_len": 4096,
  "prompt_text": "A dragon standing on a rock.",
  "token_scores": [
    {
      "token_index": 123,
      "token_id": 456,
      "token_text": "dragon",
      "token_source": "t5xxl",
      "score_mean": 0.01,
      "score_max": 0.08,
      "score_entropy": 0.75
    }
  ],
  "estimated_logits_mib": 64.0
}
```

### Report Outputs

Expected files:

- `survey_summary.json`
- `survey_by_call.csv`
- `survey_by_token.csv`
- `survey_by_step.csv`
- `survey_by_branch.csv`
- `recommended_lora_targets.csv`
- optional heatmap PNG files
- optional heatmap NPY files

## Metrics

Initial metrics:

- mean attention per token
- max attention per token
- top-k patch concentration
- token entropy
- per-step token score trajectory
- per-call token score trajectory
- positive/negative CFG branch difference
- late-step emphasis score

Later metrics:

- output-space vector similarity
- ConceptAttention-style concept-to-image heatmap
- prompt-pair activation delta
- slider direction score
- merge candidate block sensitivity

## Implementation Plan

### Phase 1: Read-only Survey MVP

- Create ComfyUI custom node skeleton
- Port safe pieces from AFM observe path
- Add JSONL writer
- Add token/call/step summary stats
- Add unit tests using synthetic q/k/v
- Validate node metadata with ComfyUI V3 schema style

### Phase 2: Report Generator

- Add parser for survey JSONL
- Generate CSV/JSON/Markdown reports
- Add top token and top call ranking
- Add slider-target recommendation output

### Phase 3: Heatmap Export

- Aggregate image-query to text-key attention into 64x64 maps
- Export PNG/NPY per token/concept
- Add configurable layer/step aggregation

### Phase 4: Concept Prompt Survey

- Add separate concept list input
- Encode concept prompts through existing CLIP/T5 path if feasible
- Compare concept embeddings against observed text/image vectors
- Keep this separate from MVP until base observation is stable

## Verification Strategy

Unit tests:

- call-index parser
- square spatial inference
- branch selection
- token score aggregation
- JSONL serialization
- report aggregation
- fallback handling

Smoke tests:

- synthetic q/k/v attention call
- `mode=observe` returns original output exactly
- unsupported shapes fall back cleanly

Manual ComfyUI validation:

- fixed seed, fixed prompt, fixed sampler
- true baseline without node
- observe run with node
- confirm same output image when observe mode only calls original backend
- confirm JSONL contains eligible calls and summaries

## Risks

- Token index to readable token text may require access to CLIP/T5 tokenizer metadata from ComfyUI conditioning.
- Anima token order must be confirmed from runtime logs, not assumed from Flux.
- Full logits can be large; MVP must keep VRAM guards and top-k/sampled summaries.
- Existing `optimized_attention_override` patches cannot compose safely yet.
- ComfyUI internals may expose block metadata inconsistently; logs must tolerate `unknown`.

## Naming Note

The current directory name is `Anima-coceptattention-survey-for-comfyui`.
The intended project spelling appears to be `Anima-conceptattention-survey-for-comfyui`.
Renaming can be done later if desired; this spec keeps the existing directory name to avoid implicit filesystem changes.
