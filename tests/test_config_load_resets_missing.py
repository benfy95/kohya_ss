"""Loading a config file must not keep values from the previous config.

Settings missing from a loaded file (e.g. added after the file was saved)
used to keep whatever was on screen: loading a MOSS-Image config and then an
older Anima config left "MOSS-Image" ticked. Loading a *file* now resets
missing settings to the GUI defaults; applying a *preset* (which lists only
some settings on purpose) still keeps the current values.
"""

import importlib
import inspect
import json
import os
import tempfile
import unittest

import gradio as gr

from kohya_gui.class_gui_config import KohyaSSGUIConfig
from kohya_gui.common_gui import loaded_field_value

TABS = {
    "lora_gui": "lora_tab",
    "dreambooth_gui": "dreambooth_tab",
    "finetune_gui": "finetune_tab",
    "textual_inversion_gui": "ti_tab",
    "leco_gui": "leco_tab",
    "anima_lllite_gui": "anima_lllite_tab",
}
SKIP = {"ask_for_file", "apply_preset", "file_path"}


class TestLoadedFieldValue(unittest.TestCase):
    def test_rules(self):
        defaults = {"moss": False, "dim": 8}
        data = {"dim": 32, "nulled": None}
        self.assertEqual(loaded_field_value("dim", data, 16, defaults, True), 32)
        self.assertFalse(loaded_field_value("moss", data, True, defaults, True))
        self.assertTrue(loaded_field_value("moss", data, True, defaults, False))
        self.assertEqual(loaded_field_value("unknown", data, 5, defaults, True), 5)
        self.assertEqual(loaded_field_value("nulled", data, 5, {"nulled": 1}, True), 1)


def _build(module_name, tab_fn):
    module = importlib.import_module(f"kohya_gui.{module_name}")
    with gr.Blocks():
        getattr(module, tab_fn)(headless=True, config=KohyaSSGUIConfig())
    return module


def _flipped(value):
    if isinstance(value, bool):
        return not value
    return value


def _call_open(module, file_path, current, apply_preset=False):
    sig = inspect.signature(module.open_configuration)
    kwargs = {}
    for name in sig.parameters:
        if name == "ask_for_file":
            kwargs[name] = False
        elif name == "apply_preset":
            kwargs[name] = apply_preset
        elif name == "file_path":
            kwargs[name] = file_path
        elif name == "training_preset":
            kwargs[name] = "none"
        else:
            kwargs[name] = current.get(name)
    result = module.open_configuration(**kwargs)
    names = [n for n in sig.parameters if n not in SKIP]
    return dict(zip(names, list(result)[1:]))


class TestConfigFileReplacesPreviousConfig(unittest.TestCase):
    def test_every_tab_resets_missing_fields_to_defaults(self):
        for module_name, tab_fn in TABS.items():
            with self.subTest(tab=module_name):
                module = _build(module_name, tab_fn)
                defaults = dict(module.FIELD_DEFAULTS)
                self.assertTrue(defaults)
                bool_fields = [k for k, v in defaults.items() if isinstance(v, bool)]
                self.assertTrue(bool_fields)
                # "previous config" on screen: every checkbox flipped
                current = {k: _flipped(v) for k, v in defaults.items()}
                keep_key = bool_fields[0]
                with tempfile.TemporaryDirectory() as d:
                    path = os.path.join(d, "older_config.json")
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump({keep_key: current[keep_key]}, f)
                    loaded = _call_open(module, path, current)
                self.assertEqual(loaded[keep_key], current[keep_key])
                lingering = [
                    k
                    for k in bool_fields[1:]
                    if k in loaded and loaded[k] != defaults[k]
                ]
                self.assertEqual(lingering, [], "values kept from previous config")

    def test_moss_tick_cleared_by_older_anima_config(self):
        module = _build("lora_gui", "lora_tab")
        current = dict(module.FIELD_DEFAULTS)
        current.update(moss_image_checkbox=True, LoRA_type="MOSS-Image")
        older_anima = {"LoRA_type": "Anima", "anima_checkbox": True, "network_dim": 32}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "anima_20261108.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(older_anima, f)
            loaded = _call_open(module, path, current)
        self.assertFalse(loaded["moss_image_checkbox"])
        self.assertTrue(loaded["anima_checkbox"])
        self.assertEqual(loaded["LoRA_type"], "Anima")
        self.assertEqual(loaded["network_dim"], 32)

    def test_preset_still_overlays_current_values(self):
        module = _build("lora_gui", "lora_tab")
        current = dict(module.FIELD_DEFAULTS)
        current.update(output_dir="D:/keep/me", moss_image_checkbox=True)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "preset.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"optimizer": "Prodigy"}, f)
            # apply_preset resolves its own path from training_preset; emulate
            # a preset load by pointing the file at our preset JSON.
            loaded = _call_open(module, path, current, apply_preset=True)
        self.assertEqual(loaded["output_dir"], "D:/keep/me")
        self.assertTrue(loaded["moss_image_checkbox"])


if __name__ == "__main__":
    unittest.main()
