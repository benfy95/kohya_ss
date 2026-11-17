"""BLIP captioning: tools/blip_caption.py (transformers-native BLIP) and the
GUI tab that launches it instead of sd-scripts' legacy make_captions.py,
which does not run on transformers 5."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kohya_gui import blip_caption_gui

_REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "blip_caption_tool", _REPO / "tools" / "blip_caption.py"
)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


class TestBlipCaptionTool(unittest.TestCase):
    def test_strip_prompt(self):
        self.assertEqual(tool.strip_prompt("a picture of a cat"), "a cat")
        self.assertEqual(tool.strip_prompt("A Picture Of a cat "), "a cat")
        self.assertEqual(tool.strip_prompt("a dog"), "a dog")

    def test_generation_kwargs_match_original_blip(self):
        beam = tool.generation_kwargs(
            SimpleNamespace(
                beam_search=True, num_beams=3, top_p=0.9, max_length=75, min_length=5
            )
        )
        self.assertEqual(beam["num_beams"], 3)
        self.assertNotIn("do_sample", beam)
        sample = tool.generation_kwargs(
            SimpleNamespace(
                beam_search=False, num_beams=3, top_p=0.8, max_length=75, min_length=5
            )
        )
        self.assertTrue(sample["do_sample"])
        self.assertEqual(sample["top_p"], 0.8)
        self.assertEqual(sample["repetition_penalty"], 1.1)

    def test_find_images_filters_and_recurses(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("a.png", "b.JPG", "c.txt", "sub/d.webp"):
                p = Path(d, name)
                p.parent.mkdir(exist_ok=True)
                p.touch()
            flat = [p.name for p in tool.find_images(Path(d), recursive=False)]
            deep = [p.name for p in tool.find_images(Path(d), recursive=True)]
        self.assertEqual(flat, ["a.png", "b.JPG"])
        self.assertEqual(sorted(deep), ["a.png", "b.JPG", "d.webp"])

    def test_missing_folder_fails(self):
        self.assertEqual(tool.main(["Z:/definitely/not/here"]), 1)


class TestBlipCaptionGui(unittest.TestCase):
    def _run(self, returncode=0):
        with (
            patch.object(
                blip_caption_gui.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=returncode),
            ) as run,
            patch.object(blip_caption_gui, "add_pre_postfix") as post,
        ):
            blip_caption_gui.caption_images(
                train_data_dir="/imgs",
                caption_file_ext=".txt",
                batch_size=2,
                num_beams=3,
                top_p=0.9,
                max_length=75,
                min_length=5,
                beam_search=True,
                prefix="pre",
            )
        return run, post

    def test_launches_transformers_native_tool(self):
        run, post = self._run()
        cmd = run.call_args[0][0]
        self.assertTrue(cmd[1].replace("\\", "/").endswith("tools/blip_caption.py"))
        self.assertNotIn("--caption_weights", cmd)
        self.assertIn("--beam_search", cmd)
        self.assertEqual(cmd[cmd.index("--batch_size") + 1], "2")
        self.assertTrue(post.called)

    def test_failure_skips_prefix_step(self):
        _, post = self._run(returncode=1)
        self.assertFalse(post.called)


if __name__ == "__main__":
    unittest.main()
