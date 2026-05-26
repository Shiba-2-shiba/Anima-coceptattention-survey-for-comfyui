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
5. phrase heatmapが必要な場合は `concept_terms` に語句を1行ずつ入れます。
6. phrase heatmapだけ見たい場合は `heatmap_output=concepts_only` にします。
7. `jsonl_path` に保存先を指定します。
8. 固定seed/prompt/samplerで生成します。

observe modeはattentionを編集しません。観測できないcallはfallbackとして元のattention backendへ戻ります。

`CLIP` と `prompt_text` を接続/入力すると、ComfyUIの `clip.tokenize()` からtoken indexに対する `token_id`, `token_text`, `token_source` を復元し、上位token scoreに付与します。decodeできない環境では `<token:ID>` 形式にフォールバックします。

### Output Paths

`jsonl_path` と `heatmap_dir` は、絶対パスまたはComfyUIの `output` ディレクトリ基準の相対パスを受け付けます。

デフォルト:

- `jsonl_path`: `anima_concept_survey/logs/survey.jsonl`
- `heatmap_dir`: `anima_concept_survey/heatmaps`

たとえばComfyUIのoutputが `C:\ComfyUI\output` の場合、デフォルトのJSONLは `C:\ComfyUI\output\anima_concept_survey\logs\survey.jsonl` に保存されます。

`jsonl_path` を空欄にするとJSONLファイル出力を無効化し、ComfyUIログへの出力だけになります。`save_heatmaps=true` のとき `heatmap_dir` が空欄なら、デフォルトのheatmapディレクトリへ保存します。

### Heatmap Interpretation

`capture_level=heatmap` はJSONLも出力します。JSONLにはtoken text、step、call、branch、scoreが入り、PNG/NPYだけでは失われる情報を補います。

Heatmap PNG/NPYは最終画像そのものへのセグメンテーションではありません。各attention callで「画像latent query位置が、指定text token keyへどれだけattentionしていたか」を64x64などのlatent gridへ戻したものです。

出力:

- `step*_call*_token*.npy`: raw latent-grid heatmap
- `step*_call*_token*.png`: raw grayscale PNG
- `step*_call*_token*_preview.png`: 512x512 color preview
- `manifest.json`: step/callごとのtoken、score、ファイル対応
- `aggregate/aggregate_*_token*_preview.png`: tokenごとにstep/callを平均した見やすいpreview
- `aggregate/manifest.json`: aggregate heatmapのtoken、score、観測数
- `concepts/aggregate/aggregate_*_concept_*_preview.png`: `concept_terms` の複数tokenを合算したphrase preview
- `concepts/aggregate/manifest.json`: phraseと対応token、score、観測数

Manifestにはattention scoreに加えて、保存されたheatmap配列そのものの統計も含まれます。

- `heatmap_mean`
- `heatmap_max`
- `heatmap_std`
- `heatmap_max_over_mean`

まず見るべきものは `aggregate/*_preview.png` と `aggregate/manifest.json` です。個別の `step*_call*` はデバッグ用で、callごとの差が大きく粗く見えます。

`concept_terms` を使うと、たとえば `big breasts` を tokenizer 結果の `big` + ` breasts` として探し、複数tokenのattentionを合算したheatmapを `concepts/aggregate` に出力します。スペースやカンマはtokenizer次第で個別tokenや単語先頭マーカーになりますが、phrase matchingでは記号と空白を正規化して連続token列を探します。

`heatmap_output`:

- `concepts_only`: `concept_terms` のphrase heatmapだけを保存します。通常はこちらを使います。
- `tokens_only`: 上位tokenごとのheatmapだけを保存します。
- `tokens_and_concepts`: 両方を保存します。診断用です。

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
- token textの復元は `CLIP` と `prompt_text` が利用できる場合のみ行います。decodeできない環境では `<token:ID>` 形式にフォールバックします。
- 既存の `optimized_attention_override` とは併用できません。
- `capture_level=heatmap` は上位tokenの `.npy`、grayscale `.png`、512px preview、aggregate previewを保存します。
