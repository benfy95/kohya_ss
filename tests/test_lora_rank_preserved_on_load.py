"""Loading a config must keep its Network Rank / Conv Rank.

Loading changes LoRA_type and network_dim in one update; the LoRA_type change
handler then read the stale on-screen rank and wrote it back as the slider
value, so a config saved with rank 32 trained at the previous value (8).
The handler may only adjust the sliders' maximum.
"""

import unittest

import gradio as gr

from kohya_gui import lora_gui
from kohya_gui.class_gui_config import KohyaSSGUIConfig


class TestRankNotOverwrittenByLoraTypeChange(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gr.Blocks() as demo:
            lora_gui.lora_tab(headless=True, config=KohyaSSGUIConfig())
        fns = demo.fns.values() if hasattr(demo.fns, "values") else demo.fns
        cls.handler = staticmethod(
            next(
                f.fn
                for f in fns
                if getattr(f.fn, "__name__", "") == "update_LoRA_settings"
            )
        )
        registry = dict(lora_gui.last_built_field_registry)
        cls.network_dim = registry["network_dim"]
        cls.conv_dim = registry["conv_dim"]
        dep = next(
            f for f in fns if getattr(f.fn, "__name__", "") == "update_LoRA_settings"
        )
        cls.outputs = list(dep.outputs)

    def _updates(self, lora_type, conv_dim=4, network_dim=8):
        result = self.handler(lora_type, conv_dim, network_dim)
        return dict(zip([id(c) for c in self.outputs], result))

    def test_rank_values_are_not_written_back(self):
        for lora_type in ("Anima", "Standard", "MOSS-Image", "LyCORIS/LoKr"):
            with self.subTest(lora_type=lora_type):
                updates = self._updates(lora_type)
                for comp in (self.network_dim, self.conv_dim):
                    args = updates[id(comp)].constructor_args
                    self.assertNotIn("value", args)
                    self.assertIn("maximum", args)

    def test_maximum_still_follows_lora_type(self):
        lycoris = self._updates("LyCORIS/LoKr")[id(self.network_dim)]
        anima = self._updates("Anima")[id(self.network_dim)]
        self.assertEqual(lycoris.constructor_args["maximum"], 100000)
        self.assertEqual(anima.constructor_args["maximum"], 512)


if __name__ == "__main__":
    unittest.main()
