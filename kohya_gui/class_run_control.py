"""Run control: observe and operate a running LoRA training (sd-scripts --run_control).

With ``--run_control`` the training writes its state to ``<output_dir>/<output_name>_run``
(manifest.json, status.json, events.jsonl, loss.jsonl) and executes command files put
into ``<run dir>/commands/`` after the next optimizer step (sample, save, save_state,
stop, pause, resume). See sd-scripts/docs/run_control.md.

The helpers here only read and write those plain files, the same way as
sd-scripts/tools/run_control_cli.py, so they need no access to the training process.
"""

import json
import os
import time

import gradio as gr

from .custom_logging import setup_logging

log = setup_logging()

RUN_DIR_SUFFIX = "_run"
DEFAULT_OUTPUT_NAME = "last"  # sd-scripts DEFAULT_LAST_OUTPUT_NAME
STATUS_FILE = "status.json"
COMMANDS_DIR = "commands"

# (button label, command dict, confirmation shown after sending)
COMMANDS = [
    ("Sample now", {"command": "sample"}, "sample images with the current weights"),
    ("Save now", {"command": "save"}, "save the LoRA now"),
    ("Save state", {"command": "save_state"}, "save the training state now"),
    ("Pause", {"command": "pause"}, "pause (frees VRAM, process stays)"),
    ("Resume", {"command": "resume"}, "resume"),
    ("Save & stop", {"command": "stop"}, "save, then stop the training"),
]


def run_dir_for(output_dir: str, output_name: str) -> str:
    """Run directory of a training, as sd-scripts library/run_control.get_run_dir."""
    name = output_name or DEFAULT_OUTPUT_NAME
    return os.path.join(output_dir or "", name + RUN_DIR_SUFFIX)


def read_status(run_dir: str):
    """status.json of the run, or None if missing / being replaced."""
    try:
        with open(os.path.join(run_dir, STATUS_FILE), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def send_command(run_dir: str, command: dict) -> str:
    """Write a command file atomically; returns its file name (also the result file name)."""
    commands_dir = os.path.join(run_dir, COMMANDS_DIR)
    os.makedirs(commands_dir, exist_ok=True)
    name = f"{time.time_ns()}_{command['command']}.json"  # executed in name order
    path = os.path.join(commands_dir, name)
    tmp = path + ".tmp"  # the training only reads *.json
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(command, f, ensure_ascii=False)
    os.replace(tmp, path)
    return name


def _duration(sec) -> str:
    if sec is None:
        return "-"
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _num(value, fmt="{:.4g}") -> str:
    return "-" if value is None else fmt.format(value)


def format_status(status) -> str:
    """Markdown summary of status.json."""
    if not status:
        return "No status yet. Start the training with **Enable run control** ticked."
    p = status.get("progress") or {}
    loss = status.get("loss") or {}
    speed = status.get("speed") or {}
    step, total = p.get("global_step"), p.get("max_train_steps")
    lines = [
        f"**State:** {status.get('state', '?')}",
        f"**Step:** {_num(step, '{}')} / {_num(total, '{}')}"
        f" · **Epoch:** {_num(p.get('epoch'), '{}')} / {_num(p.get('num_train_epochs'), '{}')}",
        f"**Loss:** {_num(loss.get('current'))} (avg {_num(loss.get('average'))})",
        f"**Speed:** {_num(speed.get('sec_per_step'), '{:.2f}')} s/step"
        f" · **Elapsed:** {_duration(speed.get('elapsed_sec'))}"
        f" · **ETA:** {_duration(speed.get('eta_sec'))}",
    ]
    lr = status.get("lr") or {}
    if lr:
        lines.append(
            "**LR:** " + ", ".join(f"{k} {_num(v, '{:.3g}')}" for k, v in lr.items())
        )
    vram = status.get("vram") or []
    if vram:
        parts = []
        for v in vram:
            used = v.get("allocated_mb") or v.get("reserved_mb") or v.get("used_mb")
            parts.append(f"{v.get('device', 'GPU')}: {_num(used, '{}')} MB")
        lines.append("**VRAM:** " + ", ".join(parts))
    for key, label in (
        ("last_checkpoint", "Last save"),
        ("last_sample", "Last sample"),
    ):
        item = status.get(key)
        if isinstance(item, dict):
            path = item.get("path") or (item.get("files") or [None])[-1]
            if path:
                lines.append(
                    f"**{label}:** `{os.path.basename(str(path))}` (step {item.get('step', '?')})"
                )
    cmd = status.get("last_command")
    if isinstance(cmd, dict):
        lines.append(f"**Last command:** {cmd.get('command')} → {cmd.get('status')}")
    if status.get("error"):
        lines.append(f"**Error:** {status['error']}")
    return "  \n".join(lines)


def status_markdown(output_dir: str, output_name: str) -> str:
    run_dir = run_dir_for(output_dir, output_name)
    return f"`{run_dir}`  \n" + format_status(read_status(run_dir))


def command_button_handler(command: dict, description: str):
    def handler(output_dir, output_name):
        run_dir = run_dir_for(output_dir, output_name)
        status = read_status(run_dir)
        if not status:
            gr.Warning(
                "No running training found. Was it started with Enable run control?"
            )
            return status_markdown(output_dir, output_name)
        if status.get("state") in (
            "finished",
            "stopped",
            "failed",
            "interrupted",
            "aborted",
        ):
            gr.Warning(f"The training has already ended ({status.get('state')}).")
            return status_markdown(output_dir, output_name)
        send_command(run_dir, dict(command))
        log.info(f"Run control: sent '{command['command']}' to {run_dir}")
        gr.Info(f"Sent: {description}. Runs after the next training step.")
        return status_markdown(output_dir, output_name)

    return handler


class RunControl:
    """'Run control' accordion of the LoRA tab."""

    def __init__(self, config: dict = {}, output_dir=None, output_name=None) -> None:
        self.config = config
        with gr.Accordion("Run control (experimental)", open=False):
            gr.Markdown(
                "Writes live status files to `<output dir>/<output name>_run` and lets you "
                "operate the training while it runs. Commands take effect after the next step."
            )
            self.run_control = gr.Checkbox(
                label="Enable run control",
                value=self.config.get("run_control.enabled", False),
                info="--run_control: status.json, events, per-image loss log (tools/analyze_run_loss.py) and the commands below",
                interactive=True,
            )
            if output_dir is None or output_name is None:
                return
            with gr.Row():
                buttons = [
                    (gr.Button(label, size="sm"), cmd, desc)
                    for label, cmd, desc in COMMANDS
                ]
                refresh = gr.Button("Refresh status", size="sm", variant="secondary")
            self.status = gr.Markdown("")
            for button, cmd, desc in buttons:
                button.click(
                    command_button_handler(cmd, desc),
                    inputs=[output_dir, output_name],
                    outputs=[self.status],
                    show_progress=False,
                )
            refresh.click(
                status_markdown,
                inputs=[output_dir, output_name],
                outputs=[self.status],
                show_progress=False,
            )
