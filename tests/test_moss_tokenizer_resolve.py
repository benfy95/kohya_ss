"""resolve_moss_tokenizer_dir: reuse a cached tokenizer so MOSS-Image runs
skip the Hugging Face Hub checks."""

import os
import tempfile
import unittest
from unittest.mock import patch

from kohya_gui import class_moss_image as cmi


class TestResolveMossTokenizerDir(unittest.TestCase):
    def test_user_value_wins(self):
        self.assertEqual(cmi.resolve_moss_tokenizer_dir("D:/tok"), "D:/tok")

    def test_cached_folder_is_used(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("tokenizer_config.json", "tokenizer.json"):
                open(os.path.join(d, name), "w").close()
            with patch(
                "huggingface_hub.try_to_load_from_cache",
                return_value=os.path.join(d, "tokenizer_config.json"),
            ):
                self.assertEqual(cmi.resolve_moss_tokenizer_dir(""), d)

    def test_incomplete_cache_falls_back_to_download(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "tokenizer_config.json"), "w").close()
            with patch(
                "huggingface_hub.try_to_load_from_cache",
                return_value=os.path.join(d, "tokenizer_config.json"),
            ):
                self.assertEqual(cmi.resolve_moss_tokenizer_dir(""), "")

    def test_not_cached_falls_back_to_download(self):
        with patch("huggingface_hub.try_to_load_from_cache", return_value=None):
            self.assertEqual(cmi.resolve_moss_tokenizer_dir(""), "")


if __name__ == "__main__":
    unittest.main()
