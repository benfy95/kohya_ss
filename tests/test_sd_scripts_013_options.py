"""GUI support for the sd-scripts 0.13.x options.

- --gradient_checkpointing_skip_blocks (Anima, MOSS-Image: LoRA, fine-tune, LLLite)
- --replica_devices (single-process multi-GPU LoRA for Anima / MOSS-Image)
- --text_encoder_blocks_to_swap (MOSS-Image)
- --run_control (LoRA) and the GUI buttons that send its commands
- resize_lora.py --svd_method
- networks/lora_post_hoc_ema.py (new LoRA tool tab)
"""

import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch

from conftest import (
    build_train_model_kwargs,
    mock_executor,
    run_train_model_and_load_toml,
)

from kohya_gui import (
    class_run_control as rc,
    common_gui,
    finetune_gui,
    lora_gui,
    lora_post_hoc_ema_gui as ema,
    resize_lora_gui,
)
import test_anima_finetune_gui as anima_ft
import test_anima_lllite_gui as anima_lllite
import test_anima_lora_gui as anima_lora
import test_moss_image_lora_gui as moss

REPLICA_READY = {
    "cache_latents": True,
    "text_encoder_lr": 0.0,
    "unet_lr": 0.0001,
    "gradient_checkpointing": True,
    "gpu_ids": "",
}


def _anima_lora_kwargs(overrides):
    return build_train_model_kwargs(
        lora_gui.train_model,
        anima_lora.FIXTURE,
        numeric_fixups=anima_lora.NUMERIC_FIXUPS,
        string_overrides=anima_lora.STRING_OVERRIDES + ("anima_replica_devices",),
        overrides={**anima_lora.ANIMA_OVERRIDES, **overrides},
    )


def _moss_kwargs(overrides):
    return build_train_model_kwargs(
        lora_gui.train_model,
        moss.FIXTURE,
        numeric_fixups=moss.NUMERIC_FIXUPS,
        string_overrides=moss.MOSS_STRING_OVERRIDES + ("moss_replica_devices",),
        overrides={**moss.MOSS_OVERRIDES, **overrides},
    )


def _launched(kwargs) -> bool:
    with patch.object(lora_gui, "print_command_and_toml") as mocked:
        mock_executor(lora_gui)
        with (
            tempfile.TemporaryDirectory() as d,
            patch.object(lora_gui, "get_executable_path", return_value="accelerate"),
        ):
            lora_gui.train_model(**dict(kwargs, output_dir=d, output_name="t"))
        return mocked.called


class TestHelpers(unittest.TestCase):
    def test_skip_blocks_value(self):
        f = common_gui.gradient_checkpointing_skip_blocks_value
        self.assertIsNone(f(0, True))
        self.assertIsNone(f("", True))
        self.assertEqual(f(12, True), 12)
        self.assertEqual(f(12.0, True), 12)
        self.assertIsNone(f(12, False))  # needs gradient checkpointing

    def test_replica_values(self):
        self.assertEqual(common_gui.replica_devices_value(" cuda:1 , 2 ,"), "cuda:1,2")
        self.assertIsNone(common_gui.replica_devices_value("  "))
        self.assertEqual(common_gui.replica_device_indexes("1, cuda:2"), [1, 2])
        self.assertEqual(common_gui.replica_device_indexes("gpu1"), [None])

    def test_replica_validation(self):
        v = common_gui.validate_replica_settings
        ok = dict(
            cache_latents=True, cache_text_encoder_outputs=True, train_unet_only=True
        )
        self.assertTrue(
            v(
                "",
                cache_latents=False,
                cache_text_encoder_outputs=False,
                train_unet_only=False,
            )
        )
        self.assertTrue(v("1", **ok))
        self.assertTrue(v("cuda:1", **ok, gpu_ids="0,1"))
        self.assertFalse(v("1", **ok, gpu_ids="0"))  # GPU IDs hides GPU 1
        self.assertFalse(v("x", **ok))
        for missing in ok:
            with self.subTest(missing=missing):
                self.assertFalse(v("1", **dict(ok, **{missing: False})))


class TestAnimaLora(unittest.TestCase):
    def test_new_options_absent_by_default(self):
        config = run_train_model_and_load_toml(lora_gui, _anima_lora_kwargs({}))
        for key in (
            "gradient_checkpointing_skip_blocks",
            "replica_devices",
            "run_control",
        ):
            self.assertNotIn(key, config)

    def test_options_forwarded(self):
        config = run_train_model_and_load_toml(
            lora_gui,
            _anima_lora_kwargs(
                {
                    **REPLICA_READY,
                    "anima_cache_text_encoder_outputs": True,
                    "anima_gradient_checkpointing_skip_blocks": 28,
                    "anima_replica_devices": "cuda:1, cuda:2",
                    "run_control": True,
                }
            ),
        )
        self.assertEqual(config["gradient_checkpointing_skip_blocks"], 28)
        self.assertEqual(config["replica_devices"], "cuda:1,cuda:2")
        self.assertIs(config["run_control"], True)

    def test_skip_blocks_dropped_without_gradient_checkpointing(self):
        config = run_train_model_and_load_toml(
            lora_gui,
            _anima_lora_kwargs(
                {
                    "gradient_checkpointing": False,
                    "anima_gradient_checkpointing_skip_blocks": 10,
                }
            ),
        )
        self.assertNotIn("gradient_checkpointing_skip_blocks", config)

    def test_replica_requirements_block_launch(self):
        good = {
            **REPLICA_READY,
            "anima_cache_text_encoder_outputs": True,
            "anima_replica_devices": "1",
        }
        self.assertTrue(_launched(_anima_lora_kwargs(good)))
        for bad in (
            {"cache_latents": False},
            {"anima_cache_text_encoder_outputs": False},
            {"text_encoder_lr": 0.0001},
            {"gpu_ids": "0"},
        ):
            with self.subTest(bad=bad):
                self.assertFalse(_launched(_anima_lora_kwargs({**good, **bad})))


class TestMossLora(unittest.TestCase):
    def setUp(self):
        p = patch.object(
            lora_gui, "resolve_moss_tokenizer_dir", side_effect=lambda v="": v
        )
        p.start()
        self.addCleanup(p.stop)

    def test_options_forwarded(self):
        config = run_train_model_and_load_toml(
            lora_gui,
            _moss_kwargs(
                {
                    **REPLICA_READY,
                    "moss_gradient_checkpointing_skip_blocks": 42,
                    "moss_text_encoder_blocks_to_swap": 28,
                    "moss_replica_devices": "1",
                    # the Anima fields must not leak into a MOSS run
                    "anima_gradient_checkpointing_skip_blocks": 5,
                    "anima_replica_devices": "3",
                }
            ),
        )
        self.assertEqual(config["gradient_checkpointing_skip_blocks"], 42)
        self.assertEqual(config["text_encoder_blocks_to_swap"], 28)
        self.assertEqual(config["replica_devices"], "1")

    def test_te_swap_absent_when_zero(self):
        config = run_train_model_and_load_toml(lora_gui, _moss_kwargs({}))
        self.assertNotIn("text_encoder_blocks_to_swap", config)


class TestAnimaFinetuneAndLllite(unittest.TestCase):
    def test_finetune_skip_blocks(self):
        config = anima_ft.TestAnimaFinetuneConfigOutput._run_and_load_toml(
            None,
            {
                "gradient_checkpointing": True,
                "anima_gradient_checkpointing_skip_blocks": 14,
            },
        )
        self.assertEqual(config["gradient_checkpointing_skip_blocks"], 14)

    def test_lllite_skip_blocks(self):
        _cmd, config, _ = anima_lllite._run_and_load_toml(
            {
                "gradient_checkpointing": True,
                "anima_gradient_checkpointing_skip_blocks": 7,
            }
        )
        self.assertEqual(config["gradient_checkpointing_skip_blocks"], 7)


class TestRunControl(unittest.TestCase):
    def test_run_dir(self):
        self.assertEqual(rc.run_dir_for("out", "my"), os.path.join("out", "my_run"))
        self.assertEqual(rc.run_dir_for("out", ""), os.path.join("out", "last_run"))

    def test_send_command_writes_json_atomically(self):
        with tempfile.TemporaryDirectory() as d:
            name = rc.send_command(d, {"command": "save"})
            cmd_dir = os.path.join(d, "commands")
            self.assertEqual(os.listdir(cmd_dir), [name])
            self.assertTrue(name.endswith("_save.json"))
            with open(os.path.join(cmd_dir, name), encoding="utf-8") as f:
                self.assertEqual(json.load(f), {"command": "save"})

    def test_format_status(self):
        status = {
            "state": "running",
            "progress": {
                "global_step": 120,
                "max_train_steps": 900,
                "epoch": 3,
                "num_train_epochs": 75,
            },
            "loss": {"current": 0.1234, "average": 0.11},
            "lr": {"unet": 1.0},
            "speed": {"sec_per_step": 1.5, "elapsed_sec": 180, "eta_sec": 3725},
            "vram": [{"device": "cuda:0", "allocated_mb": 12000}],
            "last_checkpoint": {"path": "/o/my-000003.safetensors", "step": 108},
            "last_sample": {
                "dir": "/o/sample",
                "files": ["a.png", "b.png"],
                "step": 100,
            },
            "last_command": {"command": "save", "status": "done"},
        }
        text = rc.format_status(status)
        for part in (
            "running",
            "120",
            "900",
            "0.1234",
            "1:02:05",
            "cuda:0: 12000 MB",
            "my-000003.safetensors",
            "b.png",
            "save → done",
        ):
            self.assertIn(part, text)
        self.assertIn("No status yet", rc.format_status(None))

    def test_buttons_only_send_to_a_live_training(self):
        with tempfile.TemporaryDirectory() as d:
            run_dir = rc.run_dir_for(d, "x")
            handler = rc.command_button_handler({"command": "pause"}, "pause")
            with patch.object(rc.gr, "Warning"), patch.object(rc.gr, "Info"):
                handler(d, "x")  # no status.json yet
                self.assertFalse(os.path.isdir(os.path.join(run_dir, "commands")))
                os.makedirs(run_dir)
                with open(os.path.join(run_dir, "status.json"), "w") as f:
                    json.dump({"state": "finished"}, f)
                handler(d, "x")
                self.assertFalse(os.path.isdir(os.path.join(run_dir, "commands")))
                with open(os.path.join(run_dir, "status.json"), "w") as f:
                    json.dump({"state": "running"}, f)
                handler(d, "x")
            self.assertEqual(len(os.listdir(os.path.join(run_dir, "commands"))), 1)


class TestResizeSvdMethod(unittest.TestCase):
    def _cmd(self, svd_method):
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            with patch.object(resize_lora_gui.subprocess, "run") as run:
                resize_lora_gui.resize_lora(
                    path,
                    8,
                    "out.safetensors",
                    "fp16",
                    "cuda",
                    "None",
                    "0.9",
                    False,
                    2,
                    svd_method,
                )
            return run.call_args[0][0]
        finally:
            os.remove(path)

    def test_qr(self):
        cmd = self._cmd("qr")
        self.assertEqual(cmd[cmd.index("--svd_method") + 1], "qr")
        self.assertNotIn("--svd_lowrank_niter", cmd)  # only used by 'merged'

    def test_merged_is_default(self):
        cmd = self._cmd("merged")
        self.assertNotIn("--svd_method", cmd)
        self.assertIn("--svd_lowrank_niter", cmd)


class TestPostHocEma(unittest.TestCase):
    def _make(self, d, names):
        t = time.time() - 1000
        for i, n in enumerate(names):
            p = os.path.join(d, n)
            open(p, "wb").close()
            os.utime(p, (t + i, t + i))

    def test_select_order_filter_last_n_and_exclude(self):
        with tempfile.TemporaryDirectory() as d:
            self._make(
                d,
                [
                    "my-000004.safetensors",
                    "other.safetensors",
                    "my-000002.safetensors",
                    "my-000006.safetensors",
                    "my.json",
                    "my_ema.safetensors",
                ],
            )
            # written in the order above: mtime order != name order on purpose
            names = lambda files: [os.path.basename(p) for p in files]
            self.assertEqual(
                names(ema.select_checkpoints(d, "my-")),
                [
                    "my-000004.safetensors",
                    "my-000002.safetensors",
                    "my-000006.safetensors",
                ],
            )
            self.assertEqual(
                names(ema.select_checkpoints(d, "my-", 2)),
                ["my-000002.safetensors", "my-000006.safetensors"],
            )
            self.assertNotIn(
                "my_ema.safetensors",
                names(
                    ema.select_checkpoints(
                        d, "my*", 0, os.path.join(d, "my_ema.safetensors")
                    )
                ),
            )
            self.assertEqual(ema.select_checkpoints(os.path.join(d, "nope"), ""), [])

    def test_build_command(self):
        cmd = ema.build_command(["a", "b"], "o.safetensors", 0.9, 0, 0)
        self.assertTrue(
            cmd[1].endswith(os.path.join("networks", "lora_post_hoc_ema.py"))
        )
        self.assertEqual(cmd[2:4], ["a", "b"])
        self.assertIn("--no_sort", cmd)
        self.assertEqual(cmd[cmd.index("--beta") + 1], "0.9")
        self.assertNotIn("--beta2", cmd)
        cmd = ema.build_command(["a", "b"], "o", 0.9, 0.99, 0)
        self.assertEqual(cmd[cmd.index("--beta2") + 1], "0.99")
        cmd = ema.build_command(["a", "b"], "o", 0.9, 0.99, 0.2)
        self.assertEqual(cmd[cmd.index("--sigma_rel") + 1], "0.2")
        self.assertNotIn("--beta", cmd)

    def test_first_file_weight_matches_docs(self):
        self.assertAlmostEqual(ema.first_file_weight(4, 0.9), 0.729)
        self.assertAlmostEqual(ema.first_file_weight(8, 0.95), 0.6983, places=3)

    def test_script_runs_on_tiny_loras(self):
        import torch
        from safetensors.torch import load_file, save_file

        with tempfile.TemporaryDirectory() as d:
            for i, v in enumerate((0.0, 1.0, 2.0)):
                p = os.path.join(d, f"t-{i}.safetensors")
                save_file(
                    {"lora_unet_x.lora_down.weight": torch.full((2, 2), v)},
                    p,
                    metadata={"ss_network_module": "networks.lora"},
                )
                os.utime(p, (1000 + i, 1000 + i))
            out = os.path.join(d, "ema.safetensors")
            with patch.object(ema.gr, "Info"), patch.object(ema.gr, "Warning"):
                ema.merge(d, "t-", 0, 0.5, 0, 0, out)
            w = load_file(out)["lora_unet_x.lora_down.weight"]
            # 0 -> 0.5*0+0.5*1 = 0.5 -> 0.5*0.5+0.5*2 = 1.25
            self.assertAlmostEqual(float(w[0, 0]), 1.25, places=3)


if __name__ == "__main__":
    unittest.main()
