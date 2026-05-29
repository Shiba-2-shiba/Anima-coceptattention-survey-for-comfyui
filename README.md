# Anima ConceptAttention Survey for ComfyUI

ComfyUI上でAnima/Cosmos系モデルのcross-attentionを観測し、画像生成中にどのtext key/token/call/stepが強く使われているかをJSONL/CSV/Markdownへ出力する実験用custom nodeです。別nodeとして、選択したconcept tokenへのattentionを固定seedで抑制/強調するintervention probeも含みます。

## Current Refactored MVP

- `Anima Concept Survey Model Patch`: observe-only、original attention backendの戻り値をそのまま返す
- `Anima Concept Intervention Model Patch`: experimental、`mode=intervene` で選択tokenへのcross-attentionを変更できる
- eligible callごとにSurvey/Intervention JSONL記録
- source-scoped `concept_terms` phrase matching
- unmatched/ambiguous/out-of-range concept diagnostics
- token/concept heatmap export and aggregate manifests
- concept heatmap aggregate_by_call manifests for intervention target inspection
- token/call/step/branch集計スクリプト
- concept集計用の `survey_by_concept.csv`
- attention-only候補としての `recommended_lora_targets.csv` / `recommended_concept_targets.csv`
- intervention JSONL summary and fixed-seed image difference tooling

## Install

このフォルダをComfyUIの `custom_nodes` 配下に置くか、シンボリックリンクしてください。

```powershell
cd C:\path\to\ComfyUI\custom_nodes
New-Item -ItemType SymbolicLink -Path Anima-coceptattention-survey-for-comfyui -Target "C:\Users\inott\Downloads\新しいフォルダー (4)\Anima-coceptattention-survey-for-comfyui"
```

ComfyUIを再起動すると、`model_patches/anima` に次の2つのnodeが追加されます。

- `Anima Concept Survey Model Patch`
- `Anima Concept Intervention Model Patch`

## Two-Node Workflow

このcustom nodeは、観測用のSurvey nodeと、生成結果を変更しうるIntervention nodeを分けています。

- `Anima Concept Survey Model Patch`: observe-onlyです。attention backendの出力を変えず、token/concept/call/step/branchの候補を記録します。
- `Anima Concept Intervention Model Patch`: experimentalなcausal-probeです。`mode=intervene` では選択したconcept tokenへのcross-attentionを抑制/強調し、生成結果が変わる可能性があります。

推奨順序:

1. Survey nodeで `positive_only` / `concepts_only` の候補を作ります。
2. `recommended_concept_targets.csv` と call別concept heatmapを確認します。
3. 同じ固定seedでprompt-pair画像差分を作ります。例: `big breasts` あり vs 削除。
4. Intervention nodeをまず `mode=shadow` で動かし、token解決と対象call/step/branchを確認します。
5. `mode=intervene` のno-op parityを確認します。
6. suppression/boostを固定seedで試し、生成画像、画像diff、intervention JSONLを合わせて解釈します。

Surveyのattention候補だけ、またはInterventionのログだけをLoRA/mergeの層指定根拠として扱わないでください。最終画像の領域差分は、固定seedのprompt-pair / baseline-vs-intervention画像比較で別途確認します。

## Survey Workflow

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

## Intervention Workflow

`Anima Concept Intervention Model Patch` はSurveyとは別nodeです。`mode=shadow` は元のattention backend出力を返し、would-be interventionの統計だけをJSONLに出します。`mode=intervene` は対応済みのcross-attention callで変更後のattention出力を返します。

最初はSurvey reportの `recommended_concept_targets.csv` と `concepts/aggregate_by_call/manifest.json` からcall候補を1つか少数に絞ります。たとえば `big breasts` を調べる場合:

```text
mode=shadow
intervention_kind=attention_logit_bias
prompt_text=<exact generation prompt>
intervention_terms=big breasts
target_call_indices=<candidate call, e.g. 10>
target_step_indices=all
branch_mode=positive_only
logit_bias=-5.0
jsonl_path=anima_concept_survey/logs/intervention_big_breasts.jsonl
fail_mode=fallback
```

Intervention kind:

- `attention_logit_bias`: selected token keyのattention logitsへ `logit_bias` を加えます。`0.0` はno-op、負値は抑制、正値は強調です。
- `token_attention_scale`: selected token keyのattention probability massへ `strength` を掛けて再正規化します。`1.0` はno-op、`0.0` は抑制、`>1.0` は強調です。

固定seedで次を順に確認してください。

1. `mode=shadow`: 画像はbaselineと変わらず、JSONLに `intervention_plan` / `intervention_observation` が出ること。
2. `mode=intervene`, `token_attention_scale`, `strength=1.0`: no-opとしてbaselineと一致すること。
3. `mode=intervene`, `attention_logit_bias`, `logit_bias=0.0`: no-opとしてbaselineと一致すること。
4. `mode=intervene`, `attention_logit_bias`, `logit_bias=-5.0`: JSONL上で対象concept massが下がること。
5. unrelated token、wrong call、必要なら `negative_only` をcontrolとして比較すること。

Intervention report:

```powershell
python scripts\summarize_intervention.py path\to\intervention.jsonl --out-dir path\to\intervention_report
```

出力:

- `intervention_summary.json`
- `intervention_by_call.csv`
- `intervention_by_concept.csv`
- `intervention_report.md`

このreportはattention intervention効果の要約であり、最終画像のground-truth causalityやsemantic segmentationではありません。

## Fixed-Seed Image Difference

InterventionやLoRA/merge候補を解釈する前に、同じseed/sampler/steps/CFG/negative prompt/latent sizeで画像差分を作ります。

Prompt-pair例:

```text
A: original prompt containing big breasts
B: same prompt with big breasts removed
C: same prompt with small breasts or flat chest, if useful
```

Baseline-vs-intervention例:

```text
A: no intervention node
B: intervention node with scoped suppression
```

画像diff:

```powershell
python scripts\compare_intervention_images.py baseline.png variant.png --out-dir path\to\diff_report
python scripts\compare_intervention_images.py baseline.png variant.png --out-dir path\to\diff_report --roi-bbox x,y,w,h
```

出力:

- `diff_abs.png`
- `diff_summary.json`
- `diff_report.md`

`--roi-bbox` は手動で指定した測定範囲です。自動semantic segmentationではありません。`roi_delta_over_whole_delta` は、そのROI内の変化量が画像全体平均に比べて大きいかを見る補助指標として扱ってください。

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
- `concepts/aggregate_by_call/call*_concept_*_preview.png`: call/branchごとのphrase aggregate preview
- `concepts/aggregate_by_call/manifest.json`: intervention候補確認用のcall別phrase manifest

Manifestにはattention scoreに加えて、保存されたheatmap配列そのものの統計も含まれます。

- `heatmap_mean`
- `heatmap_max`
- `heatmap_std`
- `heatmap_max_over_mean`
- `preview_normalization`

まず見るべきものは `aggregate/*_preview.png` と `aggregate/manifest.json` です。intervention対象のcallを絞るときは `concepts/aggregate_by_call/manifest.json` と `call*_concept_*_preview.png` を確認します。個別の `step*_call*` はデバッグ用で、callごとの差が大きく粗く見えます。

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
term,concept_uid,token_source,occurrence_index,branch,eligible_call_index,observation_count,score_mean,score_max,score_entropy,heatmap_mean,heatmap_max_over_mean,uniform_baseline,score_mean_over_uniform,near_uniform,block,metadata,token_indices,source_token_indices,token_texts,token_sources
```

`heatmap_output=concepts_only` などで `token_scores` が空でも、report生成は継続し、concept sectionと `recommended_concept_targets.csv` を出します。`recommended_concept_targets.csv` はattention-only candidate rankingです。最終画像の特定領域を因果的に作った証拠ではないため、intervention対象選定では固定seedのprompt/image differenceとintervention JSONLを合わせて確認してください。MarkdownのPreview Warningsは、min-max previewが弱い/ほぼ一様なheatmapを強く見せる可能性がある箇所を示します。

`survey_branch_concept_delta.csv` は同じcall/concept identityのpositive/negativeを比較し、`positive-localized`, `negative-uniform`, `both-diffuse`, `branch-ambiguous` の解釈ラベルを付けます。Markdownでは `Positive Concept Targets`, `Negative Concept Diagnostics`, `Branch Delta` を分けて表示します。

Markdownの `Block Metadata` section は、concept observationに含まれる `block` を要約します。`block=unknown` の場合、そのrunではeligible call indexから安定したtransformer module pathを取得できていないという意味です。LoRA/merge判断では、unknown blockを層指定の根拠として扱わないでください。

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

- `attention_math.py`: attention logits/probs/outputとtoken interventionの純粋関数
- `config.py`: node設定と定数
- `paths.py`: ComfyUI output path解決
- `progress.py`: sampler step推定
- `branches.py`: positive/negative branch選択
- `selectors.py`: call scope、spatial shape、logits memory推定
- `metadata.py`: model/transformer metadata
- `token_text.py`: token flattening、decode、`source_token_index`
- `concepts.py`: `concept_terms` parser/source-scoped matcher
- `intervention_config.py`: intervention node設定とscope validation
- `intervention.py`: runtime attention intervention override
- `intervention_records.py`: intervention JSONL record builder
- `intervention_reporting.py`: intervention JSONL summary writer
- `image_diff.py`: fixed-seed image difference and ROI scoring
- `scoring.py`: token/concept score計算
- `heatmaps.py`: heatmap/manifest/aggregate保存
- `records.py`: JSONL record builder
- `writer.py`: JSONL writer
- `override.py`: runtime attention observer
- `survey_attention.py`: 既存import互換のre-export layer

## Manual Validation Protocol

自動テスト後、ComfyUI上では次を確認してください。

### Survey observe-only validation

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

### Intervention fixed-seed validation

1. 固定seed/prompt/samplerでintervention nodeなしのbaseline画像を保存します。
2. 同じ条件で `mode=shadow` を実行し、画像がbaselineと変わらないことを確認します。
3. `mode=intervene`, `token_attention_scale`, `strength=1.0` を実行し、no-op parityを確認します。
4. `mode=intervene`, `attention_logit_bias`, `logit_bias=0.0` を実行し、no-op parityを確認します。
5. Survey候補callを1つ選び、`attention_logit_bias=-5.0` などでtarget suppressionを実行します。
6. `scripts/summarize_intervention.py` で `score_delta`, `mean_abs_prob_delta`, `output_delta_norm` を確認します。
7. `scripts/compare_intervention_images.py` でbaselineとsuppression画像を比較します。
8. unrelated token、wrong call、bias sweepをcontrolとして記録します。

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
- Interventionは対応済みcross-attention callだけを変更します。unsupported shape/mask/layoutは `fail_mode=fallback` で元のbackend出力を返し、fallback recordを出します。
- Intervention結果はcausal-probe heuristicです。最終画像のsemantic causalityやsegmentationの証明ではありません。
