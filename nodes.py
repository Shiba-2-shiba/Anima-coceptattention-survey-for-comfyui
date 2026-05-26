from __future__ import annotations

import logging

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .anima_concept_survey import (
    BRANCH_MODES,
    CAPTURE_LEVELS,
    FAIL_MODES,
    MODES,
    AnimaConceptSurveyAttentionOverride,
    SurveyConfig,
    is_anima_like_model,
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
                io.String.Input("target_call_indices", default="all", advanced=True),
                io.String.Input("diagnostic_call_indices", default="all", advanced=True),
                io.Combo.Input("branch_mode", options=BRANCH_MODES, default="both", advanced=True),
                io.Int.Input("max_tokens", default=16, min=1, max=512, step=1, advanced=True),
                io.Int.Input("max_steps", default=0, min=0, max=10000, step=1, advanced=True),
                io.String.Input("jsonl_path", default="", advanced=True),
                io.Boolean.Input("save_heatmaps", default=False, advanced=True),
                io.String.Input("heatmap_dir", default="", advanced=True),
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
        target_call_indices,
        diagnostic_call_indices,
        branch_mode,
        max_tokens,
        max_steps,
        jsonl_path,
        save_heatmaps,
        heatmap_dir,
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
            jsonl_path=jsonl_path.strip() or None,
            save_heatmaps=save_heatmaps,
            heatmap_dir=heatmap_dir.strip() or None,
            max_logits_mib=max_logits_mib,
            fail_mode=fail_mode,
            prompt_text=prompt_text,
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


class AnimaConceptSurveyExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [AnimaConceptSurveyModelPatch]


async def comfy_entrypoint() -> AnimaConceptSurveyExtension:
    return AnimaConceptSurveyExtension()
