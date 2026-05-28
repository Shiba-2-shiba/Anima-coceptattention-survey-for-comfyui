from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .selectors import parse_call_index_scope


MODES = ["observe", "off"]
CAPTURE_LEVELS = ["summary", "tokens", "heatmap"]
BRANCH_MODES = ["both", "positive_only", "negative_only"]
FAIL_MODES = ["fallback", "raise"]
HEATMAP_OUTPUTS = ["concepts_only", "tokens_only", "tokens_and_concepts"]
DEFAULT_OUTPUT_SUBDIR = "anima_concept_survey"
DEFAULT_JSONL_RELATIVE_PATH = f"{DEFAULT_OUTPUT_SUBDIR}/logs/survey.jsonl"
DEFAULT_HEATMAP_RELATIVE_DIR = f"{DEFAULT_OUTPUT_SUBDIR}/heatmaps"


@dataclass(frozen=True)
class SurveyConfig:
    mode: str = "observe"
    capture_level: str = "tokens"
    target_call_indices: str = "all"
    diagnostic_call_indices: str = "all"
    branch_mode: str = "both"
    max_tokens: int = 16
    max_steps: int = 0
    jsonl_path: str | None = None
    save_heatmaps: bool = False
    heatmap_dir: str | None = None
    heatmap_output: str = "concepts_only"
    max_logits_mib: float = 1024.0
    fail_mode: str = "fallback"
    prompt_text: str = ""
    concept_terms: str = ""
    token_text_map: dict[int, dict[str, Any]] = field(default_factory=dict)

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"Unsupported survey mode: {self.mode!r}")
        if self.capture_level not in CAPTURE_LEVELS:
            raise ValueError(f"Unsupported capture_level: {self.capture_level!r}")
        if self.branch_mode not in BRANCH_MODES:
            raise ValueError(f"Unsupported branch_mode: {self.branch_mode!r}")
        if self.fail_mode not in FAIL_MODES:
            raise ValueError(f"Unsupported fail_mode: {self.fail_mode!r}")
        if self.heatmap_output not in HEATMAP_OUTPUTS:
            raise ValueError(f"Unsupported heatmap_output: {self.heatmap_output!r}")
        parse_call_index_scope(self.target_call_indices)
        parse_call_index_scope(self.diagnostic_call_indices)
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.max_steps < 0:
            raise ValueError("max_steps must be zero or positive")
        if self.max_logits_mib <= 0:
            raise ValueError("max_logits_mib must be positive")
        if self.save_heatmaps and not self.heatmap_dir:
            raise ValueError("heatmap_dir is required when save_heatmaps is true")
