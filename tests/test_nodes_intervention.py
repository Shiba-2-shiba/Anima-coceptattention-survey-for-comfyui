import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path


class _FakeInputGroup:
    @staticmethod
    def Input(name, **kwargs):
        return {"name": name, **kwargs}

    @staticmethod
    def Output(**kwargs):
        return {"output": True, **kwargs}


class _FakeCombo:
    @staticmethod
    def Input(name, **kwargs):
        return {"name": name, **kwargs}


class _FakeString:
    @staticmethod
    def Input(name, **kwargs):
        return {"name": name, **kwargs}


class _FakeFloat:
    @staticmethod
    def Input(name, **kwargs):
        return {"name": name, **kwargs}


class _FakeInt:
    @staticmethod
    def Input(name, **kwargs):
        return {"name": name, **kwargs}


class _FakeBoolean:
    @staticmethod
    def Input(name, **kwargs):
        return {"name": name, **kwargs}


class _FakeSchema:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeNodeOutput:
    def __init__(self, *values):
        self.values = values


class _FakeIo:
    ComfyNode = object
    Schema = _FakeSchema
    NodeOutput = _FakeNodeOutput
    Model = _FakeInputGroup
    Clip = _FakeInputGroup
    Combo = _FakeCombo
    String = _FakeString
    Float = _FakeFloat
    Int = _FakeInt
    Boolean = _FakeBoolean


class _FakeComfyExtension:
    pass


class Anima:
    pass


class _FakeInnerModel:
    def __init__(self):
        self.diffusion_model = Anima()


class _FakeModel:
    def __init__(self, transformer_options=None):
        self.model = _FakeInnerModel()
        self.model_options = {"transformer_options": dict(transformer_options or {})}

    def clone(self):
        return _FakeModel(self.model_options.get("transformer_options"))


def _load_plugin_with_fake_comfy_api():
    root = Path(__file__).resolve().parents[1]
    package_name = "_anima_plugin_under_test"
    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            del sys.modules[name]

    comfy_api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.latest")
    latest.ComfyExtension = _FakeComfyExtension
    latest.io = _FakeIo
    sys.modules["comfy_api"] = comfy_api
    sys.modules["comfy_api.latest"] = latest

    spec = importlib.util.spec_from_file_location(
        package_name,
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return sys.modules[f"{package_name}.nodes"]


class InterventionNodeTests(unittest.TestCase):
    def test_extension_exports_survey_and_intervention_nodes(self):
        nodes = _load_plugin_with_fake_comfy_api()

        node_list = asyncio.run(nodes.AnimaConceptSurveyExtension().get_node_list())

        self.assertEqual(node_list[0].__name__, "AnimaConceptSurveyModelPatch")
        self.assertEqual(node_list[1].__name__, "AnimaConceptInterventionModelPatch")

    def test_intervention_node_schema_uses_separate_identity_and_warning(self):
        nodes = _load_plugin_with_fake_comfy_api()

        schema = nodes.AnimaConceptInterventionModelPatch.define_schema()
        input_names = {item["name"] for item in schema.inputs if "name" in item}

        self.assertEqual(schema.node_id, "AnimaConceptInterventionModelPatch")
        self.assertEqual(schema.display_name, "Anima Concept Intervention Model Patch")
        self.assertEqual(schema.category, "model_patches/anima")
        self.assertIn("changes cross-attention output", schema.description)
        self.assertIn("mode", input_names)
        self.assertIn("intervention_kind", input_names)
        self.assertIn("intervention_terms", input_names)
        self.assertIn("target_step_indices", input_names)

    def test_intervention_off_mode_returns_input_model(self):
        nodes = _load_plugin_with_fake_comfy_api()
        model = _FakeModel()

        output = nodes.AnimaConceptInterventionModelPatch.execute(
            model,
            object(),
            "off",
            "attention_logit_bias",
            "",
            "",
            "all",
            "all",
            "positive_only",
            1.0,
            0.0,
            0,
            "",
            1024.0,
            "fallback",
        )

        self.assertIs(output.values[0], model)

    def test_intervention_rejects_existing_attention_override(self):
        nodes = _load_plugin_with_fake_comfy_api()
        model = _FakeModel({"optimized_attention_override": object()})

        with self.assertRaisesRegex(ValueError, "existing optimized_attention_override"):
            nodes.AnimaConceptInterventionModelPatch.execute(
                model,
                object(),
                "shadow",
                "attention_logit_bias",
                "1girl, big breasts",
                "big breasts",
                "all",
                "all",
                "positive_only",
                1.0,
                0.0,
                0,
                "",
                1024.0,
                "fallback",
            )

    def test_intervention_shadow_installs_attention_override(self):
        nodes = _load_plugin_with_fake_comfy_api()
        model = _FakeModel()

        output = nodes.AnimaConceptInterventionModelPatch.execute(
            model,
            object(),
            "shadow",
            "attention_logit_bias",
            "1girl, big breasts",
            "big breasts",
            "all",
            "all",
            "positive_only",
            1.0,
            0.0,
            0,
            "",
            1024.0,
            "fallback",
        )

        patched = output.values[0]
        override = patched.model_options["transformer_options"]["optimized_attention_override"]
        self.assertEqual(override.__class__.__name__, "AnimaConceptInterventionAttentionOverride")
        self.assertIsNot(patched, model)

    def test_intervention_intervene_installs_attention_override(self):
        nodes = _load_plugin_with_fake_comfy_api()
        model = _FakeModel()

        output = nodes.AnimaConceptInterventionModelPatch.execute(
            model,
            object(),
            "intervene",
            "attention_logit_bias",
            "1girl, big breasts",
            "big breasts",
            "all",
            "all",
            "positive_only",
            1.0,
            0.0,
            0,
            "",
            1024.0,
            "fallback",
        )

        patched = output.values[0]
        override = patched.model_options["transformer_options"]["optimized_attention_override"]
        self.assertEqual(override.config.mode, "intervene")
        self.assertEqual(override.__class__.__name__, "AnimaConceptInterventionAttentionOverride")


if __name__ == "__main__":
    unittest.main()
