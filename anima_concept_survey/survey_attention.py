from __future__ import annotations

from .branches import branch_index_groups, selected_branch_indices
from .concepts import (
    ConceptMatchReport,
    ConceptTermSpec,
    ConceptTokenMatch,
    build_concept_token_matches,
    normalize_concept_text,
    parse_concept_terms,
)
from .config import (
    BRANCH_MODES,
    CAPTURE_LEVELS,
    DEFAULT_HEATMAP_RELATIVE_DIR,
    DEFAULT_JSONL_RELATIVE_PATH,
    DEFAULT_OUTPUT_SUBDIR,
    FAIL_MODES,
    HEATMAP_OUTPUTS,
    MODES,
    SurveyConfig,
)
from .heatmaps import (
    ConceptHeatmapAccumulator,
    HeatmapAccumulator,
    HeatmapStore,
    colorize_heatmap,
    heatmap_for_token,
    heatmap_stats,
    safe_filename_text,
    save_heatmap_png,
    write_heatmap_manifest,
)
from .metadata import discover_transformer_metadata, is_anima_like_model, safe_metadata_value
from .override import (
    AnimaConceptSurveyAttentionOverride,
    ConceptTokenGroup,
    StepStats,
    SurveyFallback,
    SurveyStats,
    build_concept_token_groups,
)
from .paths import comfy_output_dir, resolve_comfy_jsonl_path, resolve_comfy_output_path
from .progress import ProgressInfo, progress_from_sigmas
from .records import (
    attention_fallback_record,
    attention_skipped_record,
    concept_alignment_warning_record,
    concept_ambiguity_warning_records,
    concept_match_summary_record,
    concept_unmatched_records,
    public_concept_record,
    run_summary_record,
)
from .scoring import concept_scores_from_attention, normalized_entropy, token_scores_from_attention
from .selectors import estimate_logits_mib, infer_square_spatial_shape, parse_call_index_scope, shape_key


_safe_metadata_value = safe_metadata_value
_heatmap_for_token = heatmap_for_token
_safe_filename_text = safe_filename_text
_colorize_heatmap = colorize_heatmap
_heatmap_stats = heatmap_stats
_public_concept_record = public_concept_record
_normalized_entropy = normalized_entropy
_token_scores_from_attention = token_scores_from_attention


__all__ = [
    "BRANCH_MODES",
    "CAPTURE_LEVELS",
    "DEFAULT_HEATMAP_RELATIVE_DIR",
    "DEFAULT_JSONL_RELATIVE_PATH",
    "DEFAULT_OUTPUT_SUBDIR",
    "FAIL_MODES",
    "HEATMAP_OUTPUTS",
    "MODES",
    "AnimaConceptSurveyAttentionOverride",
    "ConceptHeatmapAccumulator",
    "ConceptMatchReport",
    "ConceptTermSpec",
    "ConceptTokenGroup",
    "ConceptTokenMatch",
    "HeatmapAccumulator",
    "HeatmapStore",
    "ProgressInfo",
    "StepStats",
    "SurveyConfig",
    "SurveyFallback",
    "SurveyStats",
    "attention_fallback_record",
    "attention_skipped_record",
    "branch_index_groups",
    "build_concept_token_groups",
    "build_concept_token_matches",
    "colorize_heatmap",
    "comfy_output_dir",
    "concept_alignment_warning_record",
    "concept_ambiguity_warning_records",
    "concept_match_summary_record",
    "concept_scores_from_attention",
    "concept_unmatched_records",
    "discover_transformer_metadata",
    "estimate_logits_mib",
    "heatmap_for_token",
    "heatmap_stats",
    "infer_square_spatial_shape",
    "is_anima_like_model",
    "normalize_concept_text",
    "normalized_entropy",
    "parse_call_index_scope",
    "parse_concept_terms",
    "progress_from_sigmas",
    "public_concept_record",
    "resolve_comfy_jsonl_path",
    "resolve_comfy_output_path",
    "run_summary_record",
    "safe_filename_text",
    "save_heatmap_png",
    "selected_branch_indices",
    "shape_key",
    "token_scores_from_attention",
    "write_heatmap_manifest",
]
