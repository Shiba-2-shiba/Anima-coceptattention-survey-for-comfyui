# Anima ConceptAttention Survey for ComfyUI

ComfyUI上でAnima/Cosmos系モデルのcross-attentionを観測し、画像生成中にどのtext key/token/call/stepが強く使われているかをJSONL/CSV/Markdownへ出力する実験用custom nodeです。

## Current MVP

- `Anima Concept Survey Model Patch`
- observe-only attention override
- original attention backendの戻り値をそのまま返す
- eligible callごとにJSONL記録
- token/call/step/branch集計スクリプト
- slider/merge検討用の `recommended_lora_targets.csv`

## Install

このフォルダをComfyUIの `custom_nodes` 配下に置くか、シンボリックリンクしてください。

```powershell
cd C:\path\to\ComfyUI\custom_nodes
New-Item -ItemType SymbolicLink -Path Anima-coceptattention-survey-for-comfyui -Target "C:\Users\inott\Downloads\新しいフォルダー (4)\Anima-coceptattention-survey-for-comfyui"
```

ComfyUIを再起動すると、`model_patches/anima` に `Anima Concept Survey Model Patch` が追加されます。

## Basic Workflow

1. Anima/Cosmos model loaderの後、samplerの前に `Anima Concept Survey Model Patch` を接続します。
2. `mode=observe` にします。
3. `capture_level=tokens` にします。
4. `prompt_text` に生成promptを入れます。
5. `jsonl_path` に保存先を指定します。
6. 固定seed/prompt/samplerで生成します。

observe modeはattentionを編集しません。観測できないcallはfallbackとして元のattention backendへ戻ります。

`CLIP` と `prompt_text` を接続/入力すると、ComfyUIの `clip.tokenize()` からtoken indexに対する `token_id`, `token_text`, `token_source` を復元し、上位token scoreに付与します。decodeできない環境では `<token:ID>` 形式にフォールバックします。

## Report

JSONLを集計します。

```powershell
python scripts\summarize_survey.py path\to\survey.jsonl --out-dir path\to\survey_report --top-k 32 --late-start-step 16
```

出力:

- `survey_summary.json`
- `survey_by_call.csv`
- `survey_by_step.csv`
- `survey_by_branch.csv`
- `survey_by_token.csv`
- `recommended_lora_targets.csv`
- `survey_report.md`

## Development

```powershell
python -m pytest -q
```

## Limits

- MVPはsquare image-query gridのみ対象です。
- token textの復元は未実装で、現時点ではtoken indexを記録します。
- 既存の `optimized_attention_override` とは併用できません。
- `capture_level=heatmap` は上位tokenの `.npy` とgrayscale `.png` を保存します。
