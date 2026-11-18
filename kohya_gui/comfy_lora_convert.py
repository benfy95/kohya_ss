"""Optional post-training conversion of LoRA files to ComfyUI format.

sd-scripts ships converters for the models whose LoRA key names ComfyUI does
not load directly (``sd-scripts/networks/convert_*_lora_to_comfy.py``). When
the user enables "Save in ComfyUI format" on the LoRA tab, every LoRA file
written by the run (final file and epoch/step saves) is converted after
training **in place**: the file keeps its name and is replaced by the
ComfyUI-format version. Each file is converted to a temporary file first and
only swapped in when the conversion succeeded, so a failure never leaves a
broken LoRA behind. (The converters' ``--reverse`` option restores the
sd-scripts format if ever needed.)

Only plain LoRA files are supported by these converters (``lora_up`` /
``lora_down`` weights), so conversion is offered per LoRA type, not per model.
"""

import os
import subprocess
import sys
from typing import Callable, List, Optional

from .common_gui import scriptdir, setup_environment
from .custom_logging import setup_logging

log = setup_logging()

# LoRA type (as shown in the LoRA tab) -> converter in sd-scripts/networks.
COMFY_CONVERTERS = {
    "Anima": "convert_anima_lora_to_comfy.py",
    "HunyuanImage-2.1": "convert_hunyuan_image_lora_to_comfy.py",
}
_TMP_SUFFIX = ".comfy-tmp"
# Key prefixes of sd-scripts-format LoRA files (DiT / text encoder).
_SD_SCRIPTS_PREFIXES = ("lora_unet_", "lora_te")


def comfy_converter_for(lora_type: str) -> Optional[str]:
    """Absolute path of the converter for ``lora_type``, or None if none exists."""
    name = COMFY_CONVERTERS.get(lora_type)
    if not name:
        return None
    path = os.path.join(scriptdir, "sd-scripts", "networks", name)
    return path if os.path.isfile(path) else None


def find_run_outputs(output_dir: str, output_name: str, since: float) -> List[str]:
    """LoRA files of this run: ``<name>.safetensors`` and ``<name>-*.safetensors``
    (epoch / step saves) modified at or after ``since``."""
    if not output_dir or not output_name or not os.path.isdir(output_dir):
        return []
    found = []
    for entry in os.listdir(output_dir):
        stem, ext = os.path.splitext(entry)
        if ext.lower() != ".safetensors" or stem.endswith(_TMP_SUFFIX):
            continue
        if stem != output_name and not stem.startswith(output_name + "-"):
            continue
        path = os.path.join(output_dir, entry)
        try:
            if os.path.getmtime(path) >= since:
                found.append(path)
        except OSError:
            continue
    return sorted(found)


def is_sd_scripts_lora(path: str) -> bool:
    """True if the file still uses sd-scripts key names (not yet converted)."""
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            return any(k.startswith(_SD_SCRIPTS_PREFIXES) for k in f.keys())
    except Exception:
        return False


def temp_output_path(path: str) -> str:
    stem, ext = os.path.splitext(path)
    return f"{stem}{_TMP_SUFFIX}{ext}"


def convert_run_outputs(
    lora_type: str,
    output_dir: str,
    output_name: str,
    since: float,
    run: Callable = subprocess.run,
    needs_conversion: Callable[[str], bool] = is_sd_scripts_lora,
) -> List[str]:
    """Convert this run's LoRA files in place; returns the files converted."""
    converter = comfy_converter_for(lora_type)
    if converter is None:
        log.warning(f"No ComfyUI converter is available for LoRA type '{lora_type}'.")
        return []
    sources = find_run_outputs(output_dir, output_name, since)
    if not sources:
        log.warning(
            f"ComfyUI conversion: no '{output_name}' .safetensors files from this "
            f"run were found in {output_dir}."
        )
        return []
    env = setup_environment()
    converted = []
    for src in sources:
        name = os.path.basename(src)
        if not needs_conversion(src):
            log.info(f"Already in ComfyUI format, skipping: {name}")
            continue
        tmp = temp_output_path(src)
        log.info(f"Converting LoRA to ComfyUI format: {name}")
        result = run(
            [sys.executable, converter, src, tmp],
            env=env,
            cwd=os.path.join(scriptdir, "sd-scripts"),
        )
        if getattr(result, "returncode", 1) == 0 and os.path.isfile(tmp):
            try:
                os.replace(tmp, src)  # atomic swap; keeps the original name
                converted.append(src)
                continue
            except OSError as e:
                log.error(f"Could not replace {name} with its converted version: {e}")
        else:
            log.error(f"ComfyUI conversion failed for {name}; original kept.")
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    if converted:
        log.info(
            f"ComfyUI conversion done: {len(converted)} file(s) now in ComfyUI "
            "format (same file names)."
        )
    return converted
