"""Loaded configs must stick: no change handler may echo a saved field's value.

Loading a config updates many fields in one go. Handlers triggered by those
changes read the *stale* on-screen values of their inputs; if they write those
back as ``value`` they silently undo part of the loaded config (this reset
Network Rank 32 -> 8 and LR warmup). This test runs every LoRA-tab handler
that fires on a field change and outputs to saved fields, with the tab's
default inputs, and fails if any of them sets a saved field's value, except
the deliberate rules listed in ALLOWED.
"""

import unittest

import gradio as gr

from kohya_gui import lora_gui
from kohya_gui.class_gui_config import KohyaSSGUIConfig

# (handler name, output field) pairs that intentionally set a value:
# model checkboxes derived from the model path, constant scheduler => no
# warmup, mutually exclusive compile options.
ALLOWED_FIELDS_BY_TRIGGER = {
    "pretrained_model_name_or_path": {
        "pretrained_model_name_or_path",
        "v2",
        "v_parameterization",
        "sdxl",
        "sd3_checkbox",
        "flux1_checkbox",
        "hunyuan_image_checkbox",
        "anima_checkbox",
        "lumina_checkbox",
        "moss_image_checkbox",
    },
    "lr_scheduler": {"lr_warmup", "lr_warmup_steps"},  # only for "constant"
    "anima_compile": {"anima_torch_compile"},
    "anima_torch_compile": {"anima_compile"},
}


def _events(demo, registry_ids):
    fns = demo.fns.values() if hasattr(demo.fns, "values") else demo.fns
    for f in fns:
        triggers = []
        for block, event in getattr(f, "targets", []):
            comp = demo.blocks.get(block) if isinstance(block, int) else block
            if event in ("change", "input") and id(comp) in registry_ids:
                triggers.append(registry_ids[id(comp)])
        outs = [o for o in f.outputs if id(o) in registry_ids]
        if triggers and outs:
            yield f, triggers, outs


class TestConfigLoadSticks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gr.Blocks() as demo:
            lora_gui.lora_tab(headless=True, config=KohyaSSGUIConfig())
        cls.demo = demo
        cls.names = {id(c): n for n, c in lora_gui.last_built_field_registry}
        cls.events = list(_events(demo, cls.names))

    def test_handlers_found(self):
        triggers = {t for _, ts, _ in self.events for t in ts}
        for expected in ("LoRA_type", "lr_scheduler", "anima_checkbox"):
            self.assertIn(expected, triggers)

    def test_no_handler_echoes_saved_values(self):
        offenders = []
        for f, triggers, outs in self.events:
            inputs = [getattr(c, "value", None) for c in f.inputs]
            if f.preprocess is False or not callable(f.fn):
                continue
            result = f.fn(*inputs)
            if not isinstance(result, (tuple, list)):
                result = [result]
            allowed = set().union(
                *(ALLOWED_FIELDS_BY_TRIGGER.get(t, set()) for t in triggers)
            )
            for comp, update in zip(f.outputs, result):
                name = self.names.get(id(comp))
                args = getattr(update, "constructor_args", {}) or {}
                if name and "value" in args and name not in allowed:
                    offenders.append(f"{triggers} -> {name}={args['value']!r}")
        self.assertEqual(offenders, [], "handlers overwrite loaded config values")


class TestLrSchedulerWarmup(unittest.TestCase):
    """Constant forces warmup to 0; other schedulers never touch the value."""

    @classmethod
    def setUpClass(cls):
        with gr.Blocks() as demo:
            lora_gui.lora_tab(headless=True, config=KohyaSSGUIConfig())
        names = {id(c): n for n, c in lora_gui.last_built_field_registry}
        cls.fn = next(
            f.fn
            for f, triggers, _ in _events(demo, names)
            if "lr_scheduler" in triggers
        )

    def test_constant_zeroes_warmup(self):
        warmup, steps = type(self).fn("constant", 10, 100)
        self.assertEqual(warmup.constructor_args["value"], 0)
        self.assertEqual(steps.constructor_args["value"], 0)

    def test_other_scheduler_keeps_loaded_warmup(self):
        type(self).fn("constant", 10, 100)
        warmup, steps = type(self).fn("cosine", 0, 0)
        self.assertNotIn("value", warmup.constructor_args)
        self.assertNotIn("value", steps.constructor_args)
        self.assertTrue(warmup.constructor_args["interactive"])


if __name__ == "__main__":
    unittest.main()
