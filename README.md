# Anima ConceptAttention Survey for ComfyUI

ComfyUI上でAnima/Cosmos系モデルのcross-attentionを観測し、画像生成中にどのtext key/token/call/stepが強く使われているかをJSONL/CSV/Markdownへ出力する実験用custom nodeです。

## Current Refactored MVP

- `Anima Concept Survey Model Patch`
- observe-only attention override
- original attention backendの戻り値をそのまま返す
- eligible callごとにJSONL記録
- source-scoped `concept_terms` phrase matching
- unmatched/ambiguous/out-of-range concept diagnostics
- token/concept heatmap export and aggregate manifests
- token/call/step/branch集計スクリプト
- concept集計用の `survey_by_concept.csv`
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
3. concept heatmapを見る場合は `capture_level=heatmap` にします。
4. `prompt_text` に生成promptを入れます。
5. phrase heatmapが必要な場合は `concept_terms` に語句を1行ずつ入れます。
6. phrase heatmapだけ見たい場合は `heatmap_output=concepts_only` にします。
7. `jsonl_path` に保存先を指定します。
8. 固定seed/prompt/samplerで生成します。

observe modeはattentionを編集しません。観測できないcallはfallbackとして元のattention backendへ戻ります。

`CLIP` と `prompt_text` を接続/入力すると、ComfyUIの `clip.tokenize()` からtoken indexに対する `token_id`, `token_text`, `token_source`, `source_token_index` を復元し、上位token scoreやconcept matchに付与します。decodeできない環境では `<token:ID>` 形式にフォールバックします。

単一conceptの局在確認は、まず `branch_mode=positive_only` と `heatmap_output=concepts_only` を推奨します。`branch_mode=both` はpositive/negative差分の診断には有用ですが、negative branch heatmapはnear-uniformになりやすく、preview色だけで局在と判断しないでください。reportの `survey_branch_concept_delta.csv` と `Preview Warnings` を確認してから解釈します。

推奨する単一concept確認設定:

```text
mode=observe
capture_level=heatmap
branch_mode=positive_only
save_heatmaps=true
heatmap_output=concepts_only
prompt_text=<exact generation prompt>
concept_terms=<phrase>
jsonl_path=<one file per run, or use --run-id later>
heatmap_dir=<one directory per run>
```

`max_tokens` は上位token heatmapの保存数だけを制御します。`concept_terms` のconcept heatmapは一致したphrase tokenのattention massから作るため、`max_tokens` を増やしてもconcept heatmapは鋭くなりません。

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
- `preview_normalization`

まず見るべきものは `aggregate/*_preview.png` と `aggregate/manifest.json` です。個別の `step*_call*` はデバッグ用で、callごとの差が大きく粗く見えます。

Preview PNGは `preview_normalization=per_file_minmax` でファイルごとにmin-max正規化されています。色の強さはrun間、branch間、call間で絶対比較しないでください。比較やtarget選定では `.npy` と manifest / JSONL の `heatmap_mean`, `heatmap_max_over_mean`, `score_mean_over_uniform`, `near_uniform` を優先します。`heatmap_max_over_mean < 1.05` はnear-uniform、`1.05 <= heatmap_max_over_mean < 1.15` はweak focusとしてreportに警告されます。

`concept_terms` を使うと、たとえば `big breasts` を tokenizer 結果の `big` + ` breasts` として探し、複数tokenのattention probability massを合算したheatmapを `concepts/aggregate` に出力します。スペース、カンマ、`▁`、`Ġ` などはtokenizer次第で個別tokenや単語先頭マーカーになりますが、phrase matchingでは記号と空白を正規化してsource内のtoken列を探します。

### Concept Terms and Diagnostics

`concept_terms` は改行、セミコロン、カンマ区切りで複数指定できます。

```text
big breasts
red hair; blue eyes
qwen3_06b:big breasts
clip_l:red hair
```

source prefixなしのtermは、各 `token_source` 内だけで照合されます。複数sourceで同じtermが見つかった場合はambiguousとして扱い、誤ったheatmapを作らず `concept_alignment_warning` をJSONLに出します。sourceを明示したい場合は `<token_source>:<term>` を使ってください。

Concept matchingの主な診断event:

- `concept_match_summary`: term、match、unmatched、ambiguousの一覧
- `concept_unmatched`: source内で連続token列として見つからなかったterm
- `concept_alignment_warning`: ambiguous matchやruntime `text_len` 外のtoken indexなど

`attention_observation[].concept_scores` と manifest には、`token_indices`, `source_token_indices`, `token_texts`, `token_sources`, `ignored_token_indices`, `occurrence_index`, `match_warnings` が入ります。`token_index >= text_len` のmatchは、そのcallではscore/heatmapを作らず `concept_token_index_out_of_range` warningにします。

`heatmap_output`:

- `concepts_only`: `concept_terms` のphrase heatmapだけを保存します。通常はこちらを使います。
- `tokens_only`: 上位tokenごとのheatmapだけを保存します。
- `tokens_and_concepts`: 両方を保存します。診断用です。

## Report

JSONLを集計します。

```powershell
python scripts\summarize_survey.py path\to\survey.jsonl --out-dir path\to\survey_report --top-k 32 --late-start-step 16
```

append-only JSONLに複数runが入っている場合、run filterなしの集計は失敗します。意図せず複数runを混ぜないため、次のいずれかを指定してください。

```powershell
python scripts\summarize_survey.py path\to\survey.jsonl --list-runs
python scripts\summarize_survey.py path\to\survey.jsonl --out-dir path\to\report --run-id survey-...
python scripts\summarize_survey.py path\to\survey.jsonl --out-dir path\to\report --latest-run
python scripts\summarize_survey.py path\to\survey.jsonl --out-dir path\to\report --allow-mixed-runs
```

出力:

- `survey_summary.json`
- `survey_by_call.csv`
- `survey_by_step.csv`
- `survey_by_branch.csv`
- `survey_by_token.csv`
- `survey_by_concept.csv`
- `recommended_lora_targets.csv`
- `recommended_concept_targets.csv`
- `survey_branch_concept_delta.csv`
- `survey_runs.csv`
- `survey_report.md`

`survey_by_concept.csv` の主要列:

```text
term,concept_uid,token_source,occurrence_index,branch,eligible_call_index,observation_count,score_mean,score_max,score_entropy,heatmap_mean,heatmap_max_over_mean,uniform_baseline,score_mean_over_uniform,near_uniform,token_indices,source_token_indices,token_texts,token_sources
```

`heatmap_output=concepts_only` などで `token_scores` が空でも、report生成は継続し、concept sectionと `recommended_concept_targets.csv` を出します。MarkdownのPreview Warningsは、min-max previewが弱い/ほぼ一様なheatmapを強く見せる可能性がある箇所を示します。

`survey_branch_concept_delta.csv` は同じcall/concept identityのpositive/negativeを比較し、`positive-localized`, `negative-uniform`, `both-diffuse`, `branch-ambiguous` の解釈ラベルを付けます。Markdownでは `Positive Concept Targets`, `Negative Concept Diagnostics`, `Branch Delta` を分けて表示します。

2つのrunを比較する場合:

```powershell
python scripts\compare_survey_runs.py path\to\survey.jsonl `
  --run-a survey-... `
  --run-b survey-... `
  --out-dir path\to\compare_report
```

比較reportは `compare_summary.json`, `concept_score_delta.csv`, `concept_call_delta.csv`, `branch_delta.csv`, `prompt_token_span_delta.csv`, `compare_report.md` を出力します。

## Package Layout

主要責務は以下に分離されています。

- `config.py`: node設定と定数
- `paths.py`: ComfyUI output path解決
- `progress.py`: sampler step推定
- `branches.py`: positive/negative branch選択
- `selectors.py`: call scope、spatial shape、logits memory推定
- `metadata.py`: model/transformer metadata
- `token_text.py`: token flattening、decode、`source_token_index`
- `concepts.py`: `concept_terms` parser/source-scoped matcher
- `scoring.py`: token/concept score計算
- `heatmaps.py`: heatmap/manifest/aggregate保存
- `records.py`: JSONL record builder
- `writer.py`: JSONL writer
- `override.py`: runtime attention observer
- `survey_attention.py`: 既存import互換のre-export layer

## Manual Validation Protocol

自動テスト後、ComfyUI上では次を確認してください。

1. 固定seed/prompt/samplerでsurvey nodeなしのbaseline画像を保存します。
2. 同じ条件でsurvey nodeを挟み、以下を設定します。

```text
mode=observe
capture_level=heatmap
save_heatmaps=true
heatmap_output=concepts_only
prompt_text=<exact generation prompt>
concept_terms=big breasts
```

3. 生成画像がbaselineと変わらないことを確認します。
4. `heatmaps/concepts/aggregate/aggregate_positive_concept_big_breasts_preview.png` と `manifest.json` を確認します。
5. JSONLに `concept_match_summary` と `attention_observation.concept_scores` があり、unexpectedな `concept_unmatched` がないことを確認します。
6. `text_len >= max(token_indices) + 1`、`token_texts` が意図したphrase、`token_sources` が混在していないことを確認します。
7. `scripts/summarize_survey.py` を実行し、`survey_by_concept.csv` に対象termが出ることを確認します。

## Development

```powershell
python -m pytest -q
```

## Limits

- MVPはsquare image-query gridのみ対象です。
- token textの復元は `CLIP` と `prompt_text` が利用できる場合のみ行います。decodeできない環境では `<token:ID>` 形式にフォールバックします。
- `concept_terms` は既存prompt tokenへのattention massを合算するMVP機能です。別concept promptをencodeするConceptAttention concept-vector streamではありません。
- flattened token indexとruntime attention key indexの対応はrun summaryとdiagnosticsで監査できますが、最終確認は実際のComfyUI workflowで行ってください。
- 既存の `optimized_attention_override` とは併用できません。
- `capture_level=heatmap` は上位tokenの `.npy`、grayscale `.png`、512px preview、aggregate previewを保存します。
