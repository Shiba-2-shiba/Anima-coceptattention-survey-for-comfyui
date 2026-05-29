from __future__ import annotations

import logging

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .anima_concept_survey import (
    BRANCH_MODES,
    CAPTURE_LEVELS,
    DEFAULT_INTERVENTION_JSONL_RELATIVE_PATH,
    DEFAULT_HEATMAP_RELATIVE_DIR,
    DEFAULT_JSONL_RELATIVE_PATH,
    FAIL_MODES,
    HEATMAP_OUTPUTS,
    INTERVENTION_KINDS,
    INTERVENTION_MODES,
    MODES,
    AnimaConceptInterventionAttentionOverride,
    AnimaConceptSurveyAttentionOverride,
    InterventionConfig,
    SurveyConfig,
    is_anima_like_model,
    resolve_comfy_jsonl_path,
    resolve_comfy_output_path,
)


LOGGER = logging.getLogger(__name__)


class AnimaConceptSurveyModelPatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AnimaConceptSurveyModelPatch",
            display_name="Anima Concept Survey Model Patch",
            category="model_patches/anima",
            description="Observe Anima/Cosmos cross-attention token activation during sampling without editing attention output.",
            search_aliases=["anima concept survey", "concept attention survey", "anima attention inspect"],
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Combo.Input("mode", options=MODES, default="observe"),
                io.Combo.Input("capture_level", options=CAPTURE_LEVELS, default="tokens"),
                io.String.Input("prompt_text", multiline=True, default="", tooltip="Prompt text used to restore token text labels with the connected CLIP."),
                io.String.Input("concept_terms", multiline=True, default="", tooltip="Optional phrases to export as combined phrase heatmaps, one per line. Example: big breasts"),
                io.String.Input("target_call_indices", default="all", advanced=True),
                io.String.Input("diagnostic_call_indices", default="all", advanced=True),
                io.Combo.Input("branch_mode", options=BRANCH_MODES, default="both", advanced=True),
                io.Int.Input("max_tokens", default=16, min=1, max=512, step=1, advanced=True),
                io.Int.Input("max_steps", default=0, min=0, max=10000, step=1, advanced=True),
                io.String.Input(
                    "jsonl_path",
                    default=DEFAULT_JSONL_RELATIVE_PATH,
                    advanced=True,
                    tooltip="Absolute .jsonl file, or relative .jsonl file under ComfyUI output directory. Empty disables JSONL output.",
                ),
                io.Boolean.Input("save_heatmaps", default=False, advanced=True),
                io.String.Input(
                    "heatmap_dir",
                    default=DEFAULT_HEATMAP_RELATIVE_DIR,
                    advanced=True,
                    tooltip="Absolute directory, or relative directory under ComfyUI output directory.",
                ),
                io.Combo.Input(
                    "heatmap_output",
                    options=HEATMAP_OUTPUTS,
                    default="concepts_only",
                    advanced=True,
                    tooltip="Choose concepts_only for concept_terms phrase maps without unrelated top-token heatmaps.",
                ),
                io.Float.Input("max_logits_mib", default=1024.0, min=1.0, max=65536.0, step=16.0, advanced=True),
                io.Combo.Input("fail_mode", options=FAIL_MODES, default="fallback", advanced=True),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        clip,
        mode,
        capture_level,
        prompt_text,
        concept_terms,
        target_call_indices,
        diagnostic_call_indices,
        branch_mode,
        max_tokens,
        max_steps,
        jsonl_path,
        save_heatmaps,
        heatmap_dir,
        heatmap_output,
        max_logits_mib,
        fail_mode,
    ) -> io.NodeOutput:
        if mode == "off":
            return io.NodeOutput(model)
        if not is_anima_like_model(model):
            raise ValueError("Anima Concept Survey requires an Anima-like MODEL.")

        config = SurveyConfig(
            mode=mode,
            capture_level=capture_level,
            target_call_indices=target_call_indices,
            diagnostic_call_indices=diagnostic_call_indices,
            branch_mode=branch_mode,
            max_tokens=max_tokens,
            max_steps=max_steps,
            jsonl_path=resolve_comfy_jsonl_path(jsonl_path),
            save_heatmaps=save_heatmaps,
            heatmap_dir=resolve_comfy_output_path(
                heatmap_dir,
                default_relative=DEFAULT_HEATMAP_RELATIVE_DIR if save_heatmaps else None,
            ),
            heatmap_output=heatmap_output,
            max_logits_mib=max_logits_mib,
            fail_mode=fail_mode,
            prompt_text=prompt_text,
            concept_terms=concept_terms,
        )
        config.validate()

        patched = model.clone()
        transformer_options = patched.model_options.setdefault("transformer_options", {})
        if "optimized_attention_override" in transformer_options:
            raise ValueError("Anima Concept Survey cannot be combined with an existing optimized_attention_override patch yet.")
        transformer_options["optimized_attention_override"] = AnimaConceptSurveyAttentionOverride(config, clip=clip)
        LOGGER.info(
            "[AnimaConceptSurvey] installed model patch mode=%s capture_level=%s branch_mode=%s jsonl_path=%s",
            mode,
            capture_level,
            branch_mode,
            config.jsonl_path,
        )
        return io.NodeOutput(patched)


class AnimaConceptInterventionModelPatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AnimaConceptInterventionModelPatch",
            display_name="Anima Concept Intervention Model Patch",
            category="model_patches/anima",
            description="Experimental: changes cross-attention output for selected concept tokens in intervene mode.",
            search_aliases=["anima concept intervention", "concept attention intervention", "anima attention edit"],
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Combo.Input("mode", options=INTERVENTION_MODES, default="shadow"),
                io.Combo.Input("intervention_kind", options=INTERVENTION_KINDS, default="attention_logit_bias"),
                io.String.Input("prompt_text", multiline=True, default="", tooltip="Prompt text used to resolve intervention token labels with the connected CLIP."),
                io.String.Input("intervention_terms", multiline=True, default="", tooltip="Phrases to intervene on, one per line. Example: big breasts"),
                io.String.Input("target_call_indices", default="all", advanced=True),
                io.String.Input("target_step_indices", default="all", advanced=True),
                io.Combo.Input("branch_mode", options=BRANCH_MODES, default="positive_only", advanced=True),
                io.Float.Input("strength", default=1.0, min=0.0, max=100.0, step=0.05, advanced=True),
                io.Float.Input("logit_bias", default=0.0, min=-100.0, max=100.0, step=0.25, advanced=True),
                io.Int.Input("max_steps", default=0, min=0, max=10000, step=1, advanced=True),
                io.String.Input(
                    "jsonl_path",
                    default=DEFAULT_INTERVENTION_JSONL_RELATIVE_PATH,
                    advanced=True,
                    tooltip="Absolute .jsonl file, or relative .jsonl file under ComfyUI output directory. Empty disables JSONL output.",
                ),
                io.Float.Input("max_logits_mib", default=1024.0, min=1.0, max=65536.0, step=16.0, advanced=True),
                io.Combo.Input("fail_mode", options=FAIL_MODES, default="fallback", advanced=True),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        clip,
        mode,
        intervention_kind,
        prompt_text,
        intervention_terms,
        target_call_indices,
        target_step_indices,
        branch_mode,
        strength,
        logit_bias,
        max_steps,
        jsonl_path,
        max_logits_mib,
        fail_mode,
    ) -> io.NodeOutput:
        if mode == "off":
            return io.NodeOutput(model)
        if not is_anima_like_model(model):
            raise ValueError("Anima Concept Intervention requires an Anima-like MODEL.")

        config = InterventionConfig(
            mode=mode,
            intervention_kind=intervention_kind,
            prompt_text=prompt_text,
            intervention_terms=intervention_terms,
            target_call_indices=target_call_indices,
            target_step_indices=target_step_indices,
            branch_mode=branch_mode,
            strength=strength,
            logit_bias=logit_bias,
            max_steps=max_steps,
            jsonl_path=resolve_comfy_jsonl_path(jsonl_path),
            max_logits_mib=max_logits_mib,
            fail_mode=fail_mode,
        )
        config.validate()

        patched = model.clone()
        transformer_options = patched.model_options.setdefault("transformer_options", {})
        if "optimized_attention_override" in transformer_options:
            raise ValueError("Anima Concept Intervention cannot be combined with an existing optimized_attention_override patch yet.")
        transformer_options["optimized_attention_override"] = AnimaConceptInterventionAttentionOverride(config, clip=clip)
        LOGGER.info(
            "[AnimaConceptIntervention] installed model patch mode=%s intervention_kind=%s branch_mode=%s jsonl_path=%s",
            mode,
            intervention_kind,
            branch_mode,
            config.jsonl_path,
        )
        return io.NodeOutput(patched)


class AnimaConceptSurveyExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [AnimaConceptSurveyModelPatch, AnimaConceptInterventionModelPatch]


async def comfy_entrypoint() -> AnimaConceptSurveyExtension:
    return AnimaConceptSurveyExtension()
