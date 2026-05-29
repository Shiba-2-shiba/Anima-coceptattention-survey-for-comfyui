from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import BRANCH_MODES, FAIL_MODES
from .selectors import parse_call_index_scope, parse_step_index_scope


INTERVENTION_MODES = ["off", "shadow", "intervene"]
INTERVENTION_KINDS = ["token_attention_scale", "attention_logit_bias"]
DEFAULT_INTERVENTION_JSONL_RELATIVE_PATH = "anima_concept_survey/logs/intervention.jsonl"


@dataclass(frozen=True)
class InterventionConfig:
    mode: str = "shadow"
    intervention_kind: str = "attention_logit_bias"
    prompt_text: str = ""
    intervention_terms: str = ""
    target_call_indices: str = "all"
    target_step_indices: str = "all"
    branch_mode: str = "positive_only"
    strength: float = 1.0
    logit_bias: float = 0.0
    max_steps: int = 0
    jsonl_path: str | None = None
    max_logits_mib: float = 1024.0
    fail_mode: str = "fallback"
    token_text_map: dict[int, dict[str, Any]] = field(default_factory=dict)

    def validate(self) -> None:
        if self.mode not in INTERVENTION_MODES:
            raise ValueError(f"Unsupported intervention mode: {self.mode!r}")
        if self.intervention_kind not in INTERVENTION_KINDS:
            raise ValueError(f"Unsupported intervention_kind: {self.intervention_kind!r}")
        if self.branch_mode not in BRANCH_MODES:
            raise ValueError(f"Unsupported branch_mode: {self.branch_mode!r}")
        if self.fail_mode not in FAIL_MODES:
            raise ValueError(f"Unsupported fail_mode: {self.fail_mode!r}")
        self.target_call_scope()
        self.target_step_scope()
        if self.strength < 0:
            raise ValueError("strength must be non-negative")
        if self.max_steps < 0:
            raise ValueError("max_steps must be zero or positive")
        if self.max_logits_mib <= 0:
            raise ValueError("max_logits_mib must be positive")

    def target_call_scope(self) -> set[int] | None:
        return parse_call_index_scope(self.target_call_indices)

    def target_step_scope(self) -> set[int] | None:
        return parse_step_index_scope(self.target_step_indices)
