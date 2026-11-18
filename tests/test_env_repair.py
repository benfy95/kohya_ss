"""Startup repair of package records left behind by old uv upgrades
(kohya_gui/env_repair.py).

Reproduces the reported failure: a leftover transformers-4.54.1.dist-info with
only licenses/LICENSE next to the real transformers-5.17.0.dist-info made
importlib.metadata.version("transformers") return None, so diffusers crashed
with "TypeError: expected string or bytes-like object, got 'NoneType'".
"""

import importlib.metadata as md
import os
import tempfile
import unittest

from kohya_gui import env_repair


def _make(site, rel_files):
    for rel, text in rel_files.items():
        path = os.path.join(site, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


def _version(site, name):
    dists = list(md.distributions(name=name, path=[site]))
    return dists[0].version if dists else "missing"


class TestRepairLeftovers(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.site = self._tmp.name
        _make(
            self.site,
            {
                # residue from the old-uv upgrade (as found on the affected PC)
                "transformers-4.54.1.dist-info/licenses/LICENSE": "Apache",
                "click-8.2.1.dist-info/licenses/LICENSE.txt": "BSD",
                # the real, complete records
                "transformers-5.17.0.dist-info/METADATA": (
                    "Metadata-Version: 2.4\nName: transformers\nVersion: 5.17.0\n"
                ),
                "transformers-5.17.0.dist-info/licenses/LICENSE": "Apache",
                "click-8.5.1.dist-info/METADATA": (
                    "Metadata-Version: 2.4\nName: click\nVersion: 8.5.1\n"
                ),
                # broken in a different way: must be left alone
                "odd-1.0.dist-info/RECORD": "",
                # unrelated package files
                "transformers/__init__.py": "",
            },
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_reproduces_none_version_before_repair(self):
        self.assertIsNone(_version(self.site, "transformers"))

    def test_repair_restores_real_versions(self):
        removed = env_repair.remove_orphaned_dist_info([self.site], require_venv=False)
        self.assertEqual(
            sorted(os.path.basename(p) for p in removed),
            ["click-8.2.1.dist-info", "transformers-4.54.1.dist-info"],
        )
        self.assertEqual(_version(self.site, "transformers"), "5.17.0")
        self.assertEqual(_version(self.site, "click"), "8.5.1")

    def test_only_residue_is_touched(self):
        env_repair.remove_orphaned_dist_info([self.site], require_venv=False)
        left = sorted(os.listdir(self.site))
        self.assertIn("odd-1.0.dist-info", left)
        self.assertIn("transformers-5.17.0.dist-info", left)
        self.assertIn("transformers", left)
        self.assertTrue(
            os.path.isfile(
                os.path.join(self.site, "transformers-5.17.0.dist-info", "METADATA")
            )
        )

    def test_second_run_is_a_no_op(self):
        env_repair.remove_orphaned_dist_info([self.site], require_venv=False)
        self.assertEqual(
            env_repair.remove_orphaned_dist_info([self.site], require_venv=False),
            [],
        )

    def test_never_runs_outside_a_virtualenv(self):
        from unittest.mock import patch

        with patch.object(env_repair, "in_virtualenv", return_value=False):
            self.assertEqual(env_repair.remove_orphaned_dist_info([self.site]), [])
        self.assertIn("transformers-4.54.1.dist-info", os.listdir(self.site))


class TestThisEnvironment(unittest.TestCase):
    def test_project_venv_has_no_leftovers(self):
        self.assertEqual(
            env_repair.find_orphaned_dist_info(env_repair._site_packages_dirs()), []
        )


if __name__ == "__main__":
    unittest.main()
