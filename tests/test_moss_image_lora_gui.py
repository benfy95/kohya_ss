"""Regression tests for MOSS-Image LoRA training support in the GUI
(sd-scripts >= 0.12.1, moss_image_train_network.py + networks.lora_generic).
"""

import unittest
from unittest.mock import patch

from kohya_gui import lora_gui
from conftest import (
    build_train_model_kwargs,
    run_train_model_and_load_toml,
    mock_executor,
)
from test_lumina_lora_gui import FIXTURE, NUMERIC_FIXUPS, STRING_OVERRIDES

MOSS_STRING_OVERRIDES = STRING_OVERRIDES + (
    "moss_text_encoder",
    "moss_vae",
    "moss_tokenizer_cache_dir",
    "moss_timestep_sampling",
    "moss_attn_mode",
    "moss_sample_uncond_skip_layers",
    "moss_convrot_int8_bwd",
)

MOSS_OVERRIDES = {
    "LoRA_type": "MOSS-Image",
    "moss_image_checkbox": True,
    "moss_text_encoder": "/models/model-00001-of-00004.safetensors",
    "moss_vae": "/models/flux2-vae.safetensors",
    "moss_tokenizer_cache_dir": "",
    "moss_max_token_length": 4096,
    "moss_timestep_sampling": "truncated_logit_normal",
    "moss_discrete_flow_shift": 1.0,
    "moss_sigmoid_scale": 1.0,
    "moss_logit_mean": -1.09,
    "moss_logit_std": 0.95,
    "moss_timestep_cutoff": 0.95,
    "moss_timestep_uniform_ratio": 0.01,
    "moss_attn_mode": "torch",
    "moss_split_attn": False,
    "moss_sample_uncond_skip_layers": "22",
    "moss_convrot_int8": False,
    "moss_convrot_int8_text_encoder": False,
    "moss_convrot_int8_bwd": "bf16",
    "moss_block_swap_h2d_only": False,
    "moss_unsloth_offload_checkpointing": False,
    "moss_compile": False,
    "moss_enable_multiline_caption": False,
    "moss_train_text_embedder": False,
    "moss_cache_text_encoder_outputs": True,
    "moss_cache_text_encoder_outputs_to_disk": True,
    "bucket_reso_steps": 64,
    "blocks_to_swap": 0,
    "mixed_precision": "bf16",
}


def _kwargs(overrides):
    return build_train_model_kwargs(
        lora_gui.train_model,
        FIXTURE,
        numeric_fixups=NUMERIC_FIXUPS,
        string_overrides=MOSS_STRING_OVERRIDES,
        overrides={**MOSS_OVERRIDES, **overrides},
    )


class TestMossImageLoraConfigOutput(unittest.TestCase):
    def setUp(self):
        # Keep results independent of this machine's Hugging Face cache.
        patcher = patch.object(
            lora_gui, "resolve_moss_tokenizer_dir", side_effect=lambda v="": v
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_cached_tokenizer_folder_is_forwarded(self):
        with patch.object(
            lora_gui, "resolve_moss_tokenizer_dir", return_value="/hf/tokenizer"
        ):
            config = self._run_and_load_toml({})
        self.assertEqual(config.get("tokenizer_cache_dir"), "/hf/tokenizer")

    def _run_and_load_toml(self, overrides):
        return run_train_model_and_load_toml(lora_gui, _kwargs(overrides))

    def _run_cmd(self, overrides):
        with patch.object(lora_gui, "print_command_and_toml") as mocked:
            mock_executor(lora_gui)
            with patch.object(
                lora_gui, "get_executable_path", return_value="accelerate"
            ):
                lora_gui.train_model(**_kwargs(overrides))
        return mocked

    def test_network_module_is_lora_generic(self):
        config = self._run_and_load_toml({})
        self.assertEqual(config.get("network_module"), "networks.lora_generic")

    def test_model_paths_are_forwarded(self):
        config = self._run_and_load_toml({})
        self.assertEqual(
            config.get("text_encoder"), "/models/model-00001-of-00004.safetensors"
        )
        self.assertEqual(config.get("vae"), "/models/flux2-vae.safetensors")
        self.assertNotIn("tokenizer_cache_dir", config)

    def test_pretraining_timestep_schedule_is_forwarded(self):
        config = self._run_and_load_toml({})
        self.assertEqual(config.get("timestep_sampling"), "truncated_logit_normal")
        self.assertEqual(config.get("discrete_flow_shift"), 1.0)
        self.assertAlmostEqual(config.get("logit_mean"), -1.09)
        self.assertAlmostEqual(config.get("logit_std"), 0.95)
        self.assertAlmostEqual(config.get("timestep_cutoff"), 0.95)
        self.assertAlmostEqual(config.get("timestep_uniform_ratio"), 0.01)
        self.assertEqual(config.get("moss_max_token_length"), 4096)

    def test_truncation_params_only_for_truncated_logit_normal(self):
        config = self._run_and_load_toml({"moss_timestep_sampling": "shift"})
        self.assertEqual(config.get("timestep_sampling"), "shift")
        self.assertNotIn("timestep_cutoff", config)
        self.assertNotIn("timestep_uniform_ratio", config)

    def test_cached_te_outputs_force_dit_only(self):
        config = self._run_and_load_toml({"text_encoder_lr": 0.0001, "unet_lr": 0.0001})
        self.assertTrue(config.get("cache_text_encoder_outputs"))
        self.assertTrue(config.get("cache_text_encoder_outputs_to_disk"))
        self.assertTrue(config.get("network_train_unet_only"))

    def test_te_lora_allowed_without_cache(self):
        config = self._run_and_load_toml(
            {
                "moss_cache_text_encoder_outputs": False,
                "moss_cache_text_encoder_outputs_to_disk": False,
                "text_encoder_lr": 0.0001,
                "unet_lr": 0.0001,
            }
        )
        self.assertNotIn("network_train_unet_only", config)
        self.assertNotIn("cache_text_encoder_outputs", config)

    def test_convrot_and_memory_options_forwarded(self):
        config = self._run_and_load_toml(
            {
                "moss_convrot_int8": True,
                "moss_convrot_int8_text_encoder": True,
                "moss_convrot_int8_bwd": "int8",
                "blocks_to_swap": 20,
                "moss_block_swap_h2d_only": True,
                "gradient_checkpointing": True,
                "moss_compile": True,
            }
        )
        self.assertTrue(config.get("convrot_int8"))
        self.assertTrue(config.get("convrot_int8_text_encoder"))
        self.assertEqual(config.get("convrot_int8_bwd"), "int8")
        self.assertEqual(config.get("blocks_to_swap"), 20)
        self.assertTrue(config.get("block_swap_h2d_only"))
        self.assertTrue(config.get("compile"))

    def test_incompatible_sd_args_are_excluded(self):
        config = self._run_and_load_toml(
            {"fp8_base": True, "v2": True, "v_parameterization": True}
        )
        for key in (
            "fp8_base",
            "v2",
            "v_parameterization",
            "clip_skip",
            "max_token_length",
            "convrot_int8_bwd",
            "xformers",
            "sdpa",
        ):
            self.assertNotIn(key, config)

    def test_attn_mode_always_explicit(self):
        """The global xformers setting must not leak into MOSS-Image: the backend
        would switch to xformers attention, which fails without split_attn."""
        config = self._run_and_load_toml({"xformers": "xformers"})
        self.assertEqual(config.get("attn_mode"), "torch")
        self.assertNotIn("xformers", config)

    def test_text_embedder_and_multiline_caption(self):
        config = self._run_and_load_toml(
            {"moss_train_text_embedder": True, "moss_enable_multiline_caption": True}
        )
        self.assertIn("train_text_embedder=True", config.get("network_args", []))
        self.assertTrue(config.get("enable_multiline_caption"))

    def test_empty_uncond_skip_layers_is_kept(self):
        config = self._run_and_load_toml({"moss_sample_uncond_skip_layers": ""})
        self.assertEqual(config.get("sample_uncond_skip_layers"), "")

    def test_native_lokr_allowed_on_moss(self):
        config = self._run_and_load_toml({"LoRA_type": "Kohya LoKr"})
        self.assertEqual(config.get("network_module"), "networks.lokr")

    def test_script_selection_invokes_moss_image_train_network(self):
        mocked = self._run_cmd({})
        self.assertTrue(mocked.called)
        run_cmd = mocked.call_args[0][0]
        self.assertTrue(any("moss_image_train_network.py" in part for part in run_cmd))

    def test_rejects_mismatched_lora_type(self):
        self.assertFalse(self._run_cmd({"LoRA_type": "Standard"}).called)
        self.assertFalse(self._run_cmd({"moss_image_checkbox": False}).called)

    def test_rejects_invalid_combinations(self):
        for bad in (
            {"moss_text_encoder": ""},
            {"moss_vae": ""},
            {"bucket_reso_steps": 8},
            {"blocks_to_swap": 41},
            {"blocks_to_swap": 10, "moss_unsloth_offload_checkpointing": True},
            {"moss_attn_mode": "xformers", "moss_split_attn": False},
            {"shuffle_caption": True},
        ):
            with self.subTest(bad=bad):
                self.assertFalse(self._run_cmd(bad).called)


if __name__ == "__main__":
    unittest.main()
