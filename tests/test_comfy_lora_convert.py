"""Optional post-training ComfyUI conversion of LoRA files
(kohya_gui/comfy_lora_convert.py, LoRA tab checkbox, executor hook)."""

import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import gradio as gr

from kohya_gui import class_command_executor as ce
from kohya_gui import comfy_lora_convert as clc
from kohya_gui import lora_gui


def _touch(path, mtime=None):
    open(path, "wb").close()
    if mtime is not None:
        os.utime(path, (mtime, mtime))


class TestConverterLookup(unittest.TestCase):
    def test_supported_types_have_converters_in_sd_scripts(self):
        for lora_type in ("Anima", "HunyuanImage-2.1"):
            path = clc.comfy_converter_for(lora_type)
            self.assertIsNotNone(path, lora_type)
            self.assertTrue(os.path.isfile(path))

    def test_other_types_have_none(self):
        for lora_type in ("MOSS-Image", "Flux1", "Standard", "Kohya LoKr"):
            self.assertIsNone(clc.comfy_converter_for(lora_type))


class TestFindRunOutputs(unittest.TestCase):
    def test_picks_this_runs_lora_files_only(self):
        with tempfile.TemporaryDirectory() as d:
            start = time.time()
            for name in (
                "my.safetensors",
                "my-000005.safetensors",
                "my-step00000100.safetensors",
                "my.comfy-tmp.safetensors",
                "my.ckpt",
                "other.safetensors",
                "myother.safetensors",
            ):
                _touch(os.path.join(d, name))
            _touch(os.path.join(d, "my-000001.safetensors"), mtime=start - 3600)
            found = [
                os.path.basename(p)
                for p in clc.find_run_outputs(d, "my", since=start - 1)
            ]
        self.assertEqual(
            found,
            ["my-000005.safetensors", "my-step00000100.safetensors", "my.safetensors"],
        )


def _fake_converter(calls):
    def run(cmd, **kwargs):
        calls.append(cmd)
        with open(cmd[3], "w") as f:  # converter writes dst
            f.write("comfy")
        return SimpleNamespace(returncode=0)

    return run


class TestConvertRunOutputs(unittest.TestCase):
    def test_converts_in_place_keeping_file_names(self):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            for name in ("my.safetensors", "my-000002.safetensors"):
                with open(os.path.join(d, name), "w") as f:
                    f.write("sd-scripts")
            converted = clc.convert_run_outputs(
                "Anima",
                d,
                "my",
                since=0,
                run=_fake_converter(calls),
                needs_conversion=lambda p: True,
            )
            names = sorted(os.listdir(d))
            contents = {n: open(os.path.join(d, n)).read() for n in names}
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0][1].endswith("convert_anima_lora_to_comfy.py"))
        self.assertEqual(len(converted), 2)
        self.assertEqual(names, ["my-000002.safetensors", "my.safetensors"])
        self.assertEqual(set(contents.values()), {"comfy"})

    def test_failed_conversion_keeps_original_and_no_temp_left(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "my.safetensors")
            with open(src, "w") as f:
                f.write("sd-scripts")

            def failing(cmd, **kw):
                open(cmd[3], "w").close()  # partial output
                return SimpleNamespace(returncode=1)

            converted = clc.convert_run_outputs(
                "HunyuanImage-2.1",
                d,
                "my",
                since=0,
                run=failing,
                needs_conversion=lambda p: True,
            )
            names = os.listdir(d)
            content = open(src).read()
        self.assertEqual(converted, [])
        self.assertEqual(names, ["my.safetensors"])
        self.assertEqual(content, "sd-scripts")

    def test_already_converted_files_are_skipped(self):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            _touch(os.path.join(d, "my.safetensors"))
            converted = clc.convert_run_outputs(
                "Anima",
                d,
                "my",
                since=0,
                run=_fake_converter(calls),
                needs_conversion=lambda p: False,
            )
        self.assertEqual(calls, [])
        self.assertEqual(converted, [])

    def test_is_sd_scripts_lora_detects_key_format(self):
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as d:
            sd = os.path.join(d, "a.safetensors")
            comfy = os.path.join(d, "b.safetensors")
            save_file({"lora_unet_blocks_0_x.lora_down.weight": torch.zeros(1)}, sd)
            save_file(
                {"diffusion_model.blocks.0.x.lora_A.weight": torch.zeros(1)}, comfy
            )
            self.assertTrue(clc.is_sd_scripts_lora(sd))
            self.assertFalse(clc.is_sd_scripts_lora(comfy))


class TestBuildConversionStep(unittest.TestCase):
    def _step(self, **over):
        kw = dict(
            enabled=True,
            lora_type="Anima",
            save_model_as="safetensors",
            output_dir="o",
            output_name="n",
        )
        kw.update(over)
        return lora_gui.build_comfy_conversion_step(**kw)

    def test_enabled_supported_type_returns_callable(self):
        self.assertTrue(callable(self._step()))

    def test_disabled_unsupported_or_ckpt_returns_none(self):
        self.assertIsNone(self._step(enabled=False))
        self.assertIsNone(self._step(lora_type="MOSS-Image"))
        self.assertIsNone(self._step(lora_type="Kohya LoKr"))
        self.assertIsNone(self._step(save_model_as="ckpt"))


class TestExecutorOnFinish(unittest.TestCase):
    def _finish(self, returncode, stopped=False):
        with gr.Blocks():
            ex = ce.CommandExecutor(headless=True)
        proc = MagicMock()
        proc.poll.return_value = returncode
        proc.returncode = returncode
        ex.process = proc
        ex._stopped_by_user = stopped
        hook = MagicMock()
        ex._on_finish = hook
        ex.wait_for_training_to_end()
        return hook, ex

    def test_called_on_success(self):
        hook, ex = self._finish(0)
        hook.assert_called_once()
        self.assertIsNone(ex._on_finish)

    def test_called_when_user_stopped(self):
        hook, _ = self._finish(1, stopped=True)
        hook.assert_called_once()

    def test_not_called_on_failure(self):
        hook, _ = self._finish(1)
        hook.assert_not_called()


if __name__ == "__main__":
    unittest.main()
