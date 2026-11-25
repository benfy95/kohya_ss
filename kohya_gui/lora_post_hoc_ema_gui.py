"""LoRA Post-Hoc EMA: merge several checkpoints of one training run into one LoRA.

Runs sd-scripts/networks/lora_post_hoc_ema.py (sd-scripts 0.13.1). Averaging the
checkpoints saved near the end of a training reduces the step-to-step fluctuation of
the weights; see sd-scripts/docs/lora_post_hoc_ema.md.
"""

import fnmatch
import os
import subprocess
import sys

import gradio as gr

from .common_gui import (
    get_folder_path,
    get_saveasfilename_path,
    scriptdir,
    setup_environment,
)
from .custom_logging import setup_logging

log = setup_logging()

folder_symbol = "\U0001f4c2"  # 📂
PYTHON = sys.executable
LORA_EXTS = (".safetensors",)


def select_checkpoints(
    folder: str, name_filter: str = "", last_n=0, exclude: str = ""
) -> list:
    """LoRA files of `folder` matching `name_filter` (wildcards, or a plain prefix),
    oldest to newest by modification time, limited to the newest `last_n` (0 = all).
    `exclude` (the output file) is never selected."""
    if not folder or not os.path.isdir(folder):
        return []
    pattern = (name_filter or "").strip() or "*"
    if not any(c in pattern for c in "*?["):
        pattern += "*"
    exclude_abs = os.path.normcase(os.path.abspath(exclude)) if exclude else None
    files = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if not name.lower().endswith(LORA_EXTS) or not os.path.isfile(path):
            continue
        if not fnmatch.fnmatch(name.lower(), pattern.lower()):
            continue
        if exclude_abs and os.path.normcase(os.path.abspath(path)) == exclude_abs:
            continue
        files.append(path)
    files.sort(key=lambda p: (os.path.getmtime(p), p))
    n = int(last_n or 0)
    return files[-n:] if n > 0 else files


def first_file_weight(count: int, beta: float) -> float:
    """Share of the oldest file in the result with a fixed beta (docs table)."""
    return beta ** max(0, count - 1)


def preview(folder, name_filter, last_n, beta, sigma_rel, output_file) -> str:
    files = select_checkpoints(folder, name_filter, last_n, output_file)
    if not files:
        return "No matching `.safetensors` files."
    lines = [f"**{len(files)} checkpoint(s)**, oldest first:"]
    lines += [f"{i + 1}. `{os.path.basename(p)}`" for i, p in enumerate(files)]
    if len(files) < 2:
        lines.append("\nAt least 2 checkpoints are needed.")
    elif not sigma_rel:
        w = first_file_weight(len(files), float(beta or 0.95))
        lines.append(
            f"\nWith beta {float(beta or 0.95):g}, the oldest file makes up {w:.0%} of the result. "
            "Use only checkpoints from the end of the training, or lower beta / use Sigma rel."
        )
    return "  \n".join(lines)


def build_command(files, output_file, beta, beta2, sigma_rel) -> list:
    cmd = [
        PYTHON,
        os.path.join(scriptdir, "sd-scripts", "networks", "lora_post_hoc_ema.py"),
    ]
    cmd += list(files)
    cmd += [
        "--output_file",
        output_file,
        "--no_sort",
    ]  # files are already in time order
    if sigma_rel:
        cmd += ["--sigma_rel", str(float(sigma_rel))]
    else:
        cmd += ["--beta", str(float(beta))]
        if beta2:
            cmd += ["--beta2", str(float(beta2))]
    return cmd


def merge(folder, name_filter, last_n, beta, beta2, sigma_rel, output_file):
    if not output_file:
        gr.Warning("Choose an output file.")
        return
    if not output_file.lower().endswith(".safetensors"):
        output_file += ".safetensors"
    files = select_checkpoints(folder, name_filter, last_n, output_file)
    if len(files) < 2:
        gr.Warning("At least 2 matching checkpoints are needed.")
        return
    cmd = build_command(files, output_file, beta, beta2, sigma_rel)
    log.info(f"Executing command: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=setup_environment())
    if result.returncode == 0 and os.path.isfile(output_file):
        log.info(f"Saved Post-Hoc EMA LoRA: {output_file}")
        gr.Info(f"Saved {os.path.basename(output_file)}")
    else:
        log.error(f"Post-Hoc EMA merge failed (exit code {result.returncode}).")
        gr.Warning("Post-Hoc EMA merge failed, see the console.")


def gradio_lora_post_hoc_ema_tab(headless=False):
    with gr.Tab("Post-Hoc EMA"):
        gr.Markdown(
            "Merge several checkpoints of **one training run** into one LoRA with an "
            "exponential moving average. Averaging the checkpoints from the end of a training "
            "smooths out step-to-step noise and can give a more stable LoRA than the last one. "
            "Needs only the checkpoint files (no base model, no GPU). Save checkpoints during "
            "training with *Save every N epochs/steps*."
        )
        with gr.Group(), gr.Row():
            folder = gr.Textbox(
                label="Checkpoint folder",
                placeholder="Usually the output folder",
                interactive=True,
            )
            folder_button = gr.Button(
                folder_symbol,
                elem_id="open_folder_small",
                elem_classes=["tool"],
                visible=(not headless),
            )
            folder_button.click(get_folder_path, outputs=folder, show_progress=False)
            name_filter = gr.Textbox(
                label="File name filter",
                placeholder="e.g. my_lora-0000 or my_lora-*.safetensors",
                info="Plain text matches the start of the name; * and ? are wildcards",
                interactive=True,
            )
            last_n = gr.Number(
                label="Use the last N checkpoints",
                value=0,
                minimum=0,
                step=1,
                precision=0,
                info="0 = all matching files. Typically the last quarter of the training",
                interactive=True,
            )
        with gr.Row():
            beta = gr.Number(
                label="Beta",
                value=0.95,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                info="Decay rate. Higher = more weight on older checkpoints",
                interactive=True,
            )
            beta2 = gr.Number(
                label="Beta 2",
                value=0,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                info="Decay rate at the last file (linear from Beta). 0 = off",
                interactive=True,
            )
            sigma_rel = gr.Number(
                label="Sigma rel (Power Function EMA)",
                value=0,
                minimum=0.0,
                step=0.01,
                info="EDM2 Power Function EMA instead of Beta, e.g. 0.2. 0 = off",
                interactive=True,
            )
        with gr.Group(), gr.Row():
            output_file = gr.Textbox(
                label="Output file",
                placeholder="e.g. D:/loras/my_lora_ema.safetensors",
                interactive=True,
            )
            output_button = gr.Button(
                folder_symbol,
                elem_id="open_folder_small",
                elem_classes=["tool"],
                visible=(not headless),
            )
            exts = gr.Textbox(value="*.safetensors", visible=False)
            exts_name = gr.Textbox(value="LoRA", visible=False)
            output_button.click(
                get_saveasfilename_path,
                inputs=[output_file, exts, exts_name],
                outputs=output_file,
                show_progress=False,
            )
        selection = gr.Markdown("")
        with gr.Row():
            preview_button = gr.Button("Preview selection", variant="secondary")
            merge_button = gr.Button("Merge checkpoints", variant="primary")
        inputs = [folder, name_filter, last_n, beta, sigma_rel, output_file]
        preview_button.click(
            preview, inputs=inputs, outputs=selection, show_progress=False
        )
        merge_button.click(
            merge,
            inputs=[folder, name_filter, last_n, beta, beta2, sigma_rel, output_file],
            show_progress=False,
        ).then(preview, inputs=inputs, outputs=selection, show_progress=False)
