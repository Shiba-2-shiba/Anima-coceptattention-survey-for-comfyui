from pathlib import Path


def test_survey_node_schema_keeps_required_public_inputs():
    source = Path("nodes.py").read_text(encoding="utf-8")

    assert "class AnimaConceptSurveyModelPatch" in source
    assert 'node_id="AnimaConceptSurveyModelPatch"' in source
    assert 'display_name="Anima Concept Survey Model Patch"' in source

    for input_name in (
        "mode",
        "capture_level",
        "prompt_text",
        "concept_terms",
        "target_call_indices",
        "diagnostic_call_indices",
        "branch_mode",
        "max_tokens",
        "max_steps",
        "jsonl_path",
        "save_heatmaps",
        "heatmap_dir",
        "heatmap_output",
        "max_logits_mib",
        "fail_mode",
    ):
        assert f'"{input_name}"' in source


def test_intervention_node_schema_is_separate_and_exported():
    source = Path("nodes.py").read_text(encoding="utf-8")

    assert "class AnimaConceptInterventionModelPatch" in source
    assert 'node_id="AnimaConceptInterventionModelPatch"' in source
    assert 'display_name="Anima Concept Intervention Model Patch"' in source
    assert "AnimaConceptSurveyModelPatch, AnimaConceptInterventionModelPatch" in source
    assert "existing optimized_attention_override" in source

    for input_name in (
        "mode",
        "intervention_kind",
        "prompt_text",
        "intervention_terms",
        "target_call_indices",
        "target_step_indices",
        "branch_mode",
        "strength",
        "logit_bias",
        "max_steps",
        "jsonl_path",
        "max_logits_mib",
        "fail_mode",
    ):
        assert f'"{input_name}"' in source
