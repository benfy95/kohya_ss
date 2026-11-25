# Kohya's GUI

[![GitHub stars](https://img.shields.io/github/stars/bmaltais/kohya_ss?style=social)](https://github.com/bmaltais/kohya_ss/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/bmaltais/kohya_ss?style=social)](https://github.com/bmaltais/kohya_ss/network/members)
[![License](https://img.shields.io/github/license/bmaltais/kohya_ss)](LICENSE.md)
[![GitHub issues](https://img.shields.io/github/issues/bmaltais/kohya_ss)](https://github.com/bmaltais/kohya_ss/issues)

> **This is a community build of [bmaltais/kohya_ss](https://github.com/bmaltais/kohya_ss)** that runs on the latest
> [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) backend (**v0.13.1**) and adds
> **[MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) LoRA training** to the GUI.
> See [About this build](#about-this-build) for what changed and how to install it.

This is a GUI and CLI for training diffusion models.

This project provides a user-friendly Gradio-based Graphical User Interface (GUI) for [Kohya's Stable Diffusion training scripts](https://github.com/kohya-ss/sd-scripts). 
Stable Diffusion training empowers users to customize image generation models by fine-tuning existing models, creating unique artistic styles, 
and training specialized models like LoRA (Low-Rank Adaptation).

Key features of this GUI include:
*   Easy-to-use interface for setting a wide range of training parameters.
*   Automatic generation of the command-line interface (CLI) commands required to run the training scripts.
*   Support for LoRA, LoHa, LoKr, Dreambooth, fine-tuning, and Textual Inversion training.
*   Support for a wide range of base models: Stable Diffusion 1.5/2.x, SDXL, SD3, Flux.1, Lumina Image 2.0, Anima, HunyuanImage-2.1, and **MOSS-Image** (LoRA).
*   LECO (concept erasure/editing) training.

Support for Linux and macOS is also available. While Linux support is actively maintained through community contributions, macOS compatibility may vary.

## About this build

This build keeps everything from the original GUI and updates it to the newest training backend, so the latest
models supported by sd-scripts can be trained from the GUI.

### What's new

- **Backend: sd-scripts v0.13.1** (from v0.11.1), with dependencies aligned to it: `transformers` 5.17.0,
  `diffusers` 0.40.0, `huggingface-hub` 1.32.0, `safetensors` 0.8.0, `accelerate` ≥ 1.15.0.
- **New in sd-scripts 0.13, available in the GUI:**
  - **Run control** (LoRA tab, experimental): live status (step, loss, speed, ETA, VRAM) and buttons to
    *Sample now*, *Save now*, *Save state*, *Pause* (frees VRAM), *Resume* and *Save & stop* a running training.
    Also writes a per-image loss log (`tools/analyze_run_loss.py` finds images that don't learn well).
  - **Gradient Checkpointing Skip Blocks** (Anima and MOSS-Image: LoRA, fine-tuning, LLLite): trades spare VRAM
    for speed. MOSS-Image: about 1 GB more and 8.6 ms faster per block; Anima: 6.5 GB / 0.85 s per step up to
    19.9 GB / 0.53 s per step with all 28 blocks.
  - **Replica Devices** (Anima and MOSS-Image LoRA, experimental): multi-GPU training in one process, no DDP,
    also on Windows. Needs cached latents, cached text encoder outputs and DiT-only training; the GUI checks these
    and that *Accelerate launch → GPU IDs* doesn't hide the extra GPUs.
  - **Text Encoder Blocks to Swap** (MOSS-Image): cache text encoder outputs on an 8 GB GPU.
  - **Post-Hoc EMA** (LoRA → Tools): merge the last checkpoints of a training into one smoother LoRA.
  - **Resize LoRA → SVD method `qr`**: exact and over 100× faster.
  - Anima variants with a different number of blocks (e.g. Anima-2.9B) now load.
- **MOSS-Image LoRA training** on the LoRA tab (details below).
- **Readable training console:** colours, one-line full-width progress bars, a timestamp on every log line, and
  "saving checkpoint: …" messages shown as each epoch is saved.
- **BLIP captioning fixed for transformers 5:** the BLIP tab now uses transformers' own BLIP-large captioning model
  (`Salesforce/blip-image-captioning-large`) instead of sd-scripts' legacy BLIP code, which no longer runs.
- **Optional ComfyUI format:** for Anima and HunyuanImage-2.1 LoRAs, tick *Save in ComfyUI format* (next to
  *Save precision*) to convert every saved LoRA to ComfyUI format after training, keeping the same file names
  (uses sd-scripts' own converters; the sd-scripts-format version is replaced).
  MOSS-Image LoRAs, and Anima LoRAs that don't train the text encoder, already load in ComfyUI as saved.
- All existing model types (SD 1.5/2.x, SDXL, SD3, Flux.1, Lumina, Anima, HunyuanImage-2.1) were re-checked
  against the new backend.

### ⭐ MOSS-Image

[MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) (OpenMOSS) is a 5B single-stream DiT that uses
**Qwen3.5-9B** as its text encoder and the **FLUX.2 VAE**. LoRAs trained here load in ComfyUI without conversion.

**How to train one:**
1. **LoRA tab → Model:** set *Pretrained model* to the MOSS-Image DiT: either the first shard
   `diffusion_pytorch_model-00001-of-00003.safetensors` from
   [OpenMOSS-Team/MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base), or a ComfyUI single file
   such as `moss_image_base_bf16.safetensors` / `moss_image_base_int8_convrot.safetensors`.
2. Tick the **MOSS-Image** checkbox, and set **LoRA type** to **MOSS-Image** (`networks.lora_generic`).
   Kohya LoHa / LoKr also work.
3. In the **MOSS-Image** panel, set the **Qwen3.5-9B text encoder** (`model-00001-of-00004.safetensors`, or the
   ComfyUI `qwen3.5_9b_int8_convrot.safetensors`) and the **FLUX.2 VAE** (`flux2-vae.safetensors`).
4. Use **bf16** mixed precision and gradient checkpointing, and keep **Cache Text Encoder Outputs** on.
   Turn **Shuffle caption** off: it can't be combined with cached text encoder outputs (the GUI tells you).

**Included options:**
- Timestep sampling preset that reproduces MOSS-Image's pre-training schedule (`truncated_logit_normal`).
- **ConvRot INT8** for the DiT and/or the text encoder (roughly halves weight VRAM; replaces fp8). Pre-quantized
  ComfyUI `*_int8_convrot` files are detected automatically.
- Block swap (up to 40 blocks) with the faster **H2D-only** mode, Unsloth activation offloading, and `torch.compile`.
- Multi-line captions, text-embedder LoRA, and cached text encoder outputs.
- The tokenizer (~20 MB) is downloaded once, then reused from the local cache.

**Rough VRAM guide** (from the [sd-scripts documentation](https://github.com/kohya-ss/sd-scripts/blob/main/docs/moss_image_train_network.md),
1024px, batch 1):

| GPU   | Suggested setup |
|-------|-----------------|
| 24 GB | bf16 DiT, text encoder outputs cached (INT8 text encoder if loading runs out of memory) |
| 16 GB | ConvRot INT8 for both the DiT and the text encoder |
| 12 GB | ConvRot INT8 for both + Blocks to swap 20–36 (needs ~15 GB system RAM) |

> **Tip (Windows):** install [`triton-windows`](https://github.com/woct0rdho/triton-windows) matching your PyTorch
> version for fast INT8 kernels. Without it, INT8 still saves memory but runs slower.

### Install this build

Requirements are the same as the original: **Python 3.10 or 3.11**, Git, and an NVIDIA GPU with a recent driver.
Always clone with `--recursive` so the matching sd-scripts version is downloaded too.

**Windows (uv, recommended)**

```powershell
git clone --recursive https://github.com/benfy95/kohya_ss.git
cd kohya_ss
.\gui-uv.bat
```

**Windows (pip)**

```powershell
git clone --recursive https://github.com/benfy95/kohya_ss.git
cd kohya_ss
.\setup.bat
.\gui.bat
```

**Linux**

```bash
git clone --recursive https://github.com/benfy95/kohya_ss.git
cd kohya_ss
./gui-uv.sh        # or: ./setup.sh && ./gui.sh
```

The detailed guides under [Installation Options](#installation-options) apply unchanged. Just use this repository's
URL in the `git clone` step.

**Updating an existing install of this build**

```bash
git pull
git submodule update --init --recursive
```

Then start it again with `gui-uv.bat` (it syncs dependencies automatically), or re-run `setup.bat` for the pip
install.

> If you are switching an existing *original* kohya_ss install to this build, re-run the installer: the backend
> needs the newer `transformers` 5 / `diffusers` 0.40 packages.

> **Update uv first** (`uv self update`, or `winget upgrade astral-sh.uv` if you installed it with winget). Old uv
> versions (seen with 0.7.21) leave half-removed package records behind when upgrading, which made training fail
> with `TypeError: expected string or bytes-like object, got 'NoneType'` (Python reports no version for
> `transformers`). The GUI now removes these leftovers automatically at startup and prints what it removed.

### Contributors

- **[benfy95](https://github.com/benfy95)**: maintainer of this build, testing and MOSS-Image training validation
- **Claude (Anthropic), via Claude Code**: implementation of the backend update, MOSS-Image GUI support and fixes

All credit for the GUI goes to [bmaltais](https://github.com/bmaltais/kohya_ss) and its contributors, and for the
training scripts to [kohya-ss](https://github.com/kohya-ss/sd-scripts).

## Table of Contents

- [About this build](#about-this-build)
  - [What's new](#whats-new)
  - [MOSS-Image](#-moss-image)
  - [Install this build](#install-this-build)
  - [Contributors](#contributors)
- [Installation Options](#installation-options)
  - [Local Installation Overview](#local-installation-overview)
    - [`uv` vs `pip` – What's the Difference?](#uv-vs-pip--whats-the-difference)
  - [Cloud Installation Overview](#cloud-installation-overview)
    - [Colab](#-colab)
    - [Runpod, Novita, Docker](#runpod-novita-docker)
  - [Offline / air-gapped GUI shell](#offline--air-gapped-gui-shell)
- [Custom Path Defaults](#custom-path-defaults)
- [Server Options](#server-options)
  - [`allowed_paths`](#allowed_paths)
  - [Remote / SSH: use `--headless`](#remote--ssh-use---headless)
- [LoRA](#lora)
  - [Sample image generation during training](#sample-image-generation-during-training)
  - [Troubleshooting](#troubleshooting)
  - [Page File Limit](#page-file-limit)
  - [No module called tkinter](#no-module-called-tkinter)
  - [LORA Training on TESLA V100 - GPU Utilization Issue](#lora-training-on-tesla-v100---gpu-utilization-issue)
- [SDXL training](#sdxl-training)
- [Masked loss](#masked-loss)
- [Guides](#guides)
  - [Using Accelerate Lora Tab to Select GPU ID](#using-accelerate-lora-tab-to-select-gpu-id)
    - [Starting Accelerate in GUI](#starting-accelerate-in-gui)
    - [Running Multiple Instances (linux)](#running-multiple-instances-linux)
    - [Monitoring Processes](#monitoring-processes)
- [Interesting Forks](#interesting-forks)
- [Contributing](#contributing)
- [License](#license)
- [Change History](#change-history)
  - [v26.0.0](#v2600)


## Installation Options

You can run `kohya_ss` either **locally on your machine** or via **cloud-based solutions** like Colab or Runpod.

- If you have a GPU-equipped PC and want full control: install it locally using `uv` or `pip`.
- If your system doesn’t meet requirements or you prefer a browser-based setup: use Colab or a paid GPU provider like Runpod or Novita.
- If you are a developer or DevOps user, Docker is also supported.

---

### Local Installation Overview

You can install `kohya_ss` locally using either the `uv` or `pip` method. Choose one depending on your platform and preferences:

| Platform     | Recommended Method | Instructions                                |
|--------------|----------------|---------------------------------------------|
| Linux        | `uv`           | [uv_linux.md](./docs/Installation/uv_linux.md) |
| Linux or Mac | `pip`              | [pip_linux.md](./docs/Installation/pip_linux.md)               |
| Windows      | `uv`           | [uv_windows.md](./docs/Installation/uv_windows.md)             |
| Windows      | `pip`          | [pip_windows.md](./docs/Installation/pip_windows.md)           |

#### `uv` vs `pip` – What's the Difference?

- `uv` is faster and isolates dependencies more cleanly, ideal if you want minimal setup hassle.
- `pip` is more traditional, easier to debug if issues arise, and works better with some IDEs or Python tooling.
- If unsure: try `uv`. If it doesn't work for you, fall back to `pip`.

### Cloud Installation Overview

#### 🦒 Colab

For browser-based training without local setup, use this Colab notebook:  
<https://github.com/camenduru/kohya_ss-colab>

- No installation required
- Free to use (GPU availability may vary)
- Maintained by **camenduru**, not the original author

| Colab                                                                                                                                                                          | Info               |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------ |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/kohya_ss-colab/blob/main/kohya_ss_colab.ipynb) | kohya_ss_gui_colab |

> 💡 If you encounter issues, please report them on camenduru’s repo.

**Special thanks**  
I would like to express my gratitude to camenduru for their valuable contribution.

#### Runpod, Novita, Docker

These options are for users running training on hosted GPU infrastructure or containers.

- **[Runpod setup](docs/installation_runpod.md)** – Ready-made GPU background training via templates.
- **[Novita setup](docs/installation_novita.md)** – Similar to Runpod, but integrated into the Novita UI.
- **[Docker setup](docs/installation_docker.md)** – For developers/sysadmins using containerized environments.

### Offline / air-gapped GUI shell

After a normal install, the **local Gradio UI shell** is intended to run without the browser (or server process) fetching third-party CDN hosts for core assets:

| Layer | Offline after install? | Notes |
|-------|------------------------|--------|
| GUI shell (HTML/JS/CSS, theme fonts, About tab) | Yes | No Google Fonts, cdnjs iframe-resizer, About remote badges, or Gradio analytics by default |
| Install (PyPI / torch wheels / `uv` sync) | No | One-time network (or a pre-populated cache / mirror) is still required |
| Captioning / HF models / W&B / HF upload | Feature-dependent | Needs local model caches or intentional network; for hard offline training set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` |

The public GitHub `README.md` may still use marketing badges; the **in-app About → README** view strips remote images so the browser does not load them.


## Custom Path Defaults with `config.toml`

The GUI supports a configuration file named `config.toml` that allows you to set default paths for many of the input fields. This is useful for avoiding repetitive manual selection of directories every time you start the GUI.

**Purpose of `config.toml`:**

*   Pre-fill default directory paths for pretrained models, datasets, output folders, LoRA models, etc.
*   Streamline your workflow by having the GUI remember your preferred locations.

**How to Use and Customize:**

1.  **Create your configuration file:**
    *   In the root directory of the `kohya_ss` repository, you'll find a file named `config example.toml`.
    *   Copy this file and rename the copy to `config.toml`. This `config.toml` file will be automatically loaded when the GUI starts.
2.  **Edit `config.toml`:**
    *   Open `config.toml` with a text editor.
    *   The file uses TOML (Tom's Obvious, Minimal Language) format, which consists of `key = "value"` pairs.
    *   Modify the paths for the keys according to your local directory structure.
    *   **Important:**
        *   Use absolute paths (e.g., `C:/Users/YourName/StableDiffusion/Models` or `/home/yourname/sd-models`).
        *   Alternatively, you can use paths relative to the `kohya_ss` root directory.
        *   Ensure you use forward slashes (`/`) for paths, even on Windows, as this is generally more compatible with TOML and Python.
        *   Make sure the specified directories exist on your system.

**Structure of `config.toml`:**

The `config.toml` file can have several sections, typically corresponding to different training modes or general settings. Common keys you might want to set include:

*   `model_dir`: Default directory for loading base Stable Diffusion models.
*   `lora_model_dir`: Default directory for saving and loading LoRA models.
*   `output_dir`: Default base directory for training outputs (images, logs, model checkpoints).
*   `dataset_dir`: A general default if you store all your datasets in one place.
*   Specific input paths for different training tabs like Dreambooth, Finetune, LoRA, etc. (e.g., `db_model_dir`, `ft_source_model_name_or_path`).

**Example Configurations:**

Here's an example snippet of what your `config.toml` might look like:

```toml
# General settings
model_dir = "C:/ai_stuff/stable-diffusion-webui/models/Stable-diffusion"
lora_model_dir = "C:/ai_stuff/stable-diffusion-webui/models/Lora"
vae_dir = "C:/ai_stuff/stable-diffusion-webui/models/VAE"
output_dir = "C:/ai_stuff/kohya_ss_outputs"
logging_dir = "C:/ai_stuff/kohya_ss_outputs/logs"

# Dreambooth specific paths
db_model_dir = "C:/ai_stuff/stable-diffusion-webui/models/Stable-diffusion"
db_reg_image_dir = "C:/ai_stuff/datasets/dreambooth_regularization_images"
# Add other db_... paths as needed

# Finetune specific paths
ft_model_dir = "C:/ai_stuff/stable-diffusion-webui/models/Stable-diffusion"
# Add other ft_... paths as needed

# LoRA / LoCon specific paths
lc_model_dir = "C:/ai_stuff/stable-diffusion-webui/models/Stable-diffusion" # Base model for LoRA training
lc_output_dir = "C:/ai_stuff/kohya_ss_outputs/lora"
lc_dataset_dir = "C:/ai_stuff/datasets/my_lora_project"
# Add other lc_... paths as needed

# You can find a comprehensive list of all available keys in the `config example.toml` file.
# Refer to it to customize paths for all supported options in the GUI.
```

**Using a Custom Config File Path:**

If you prefer to name your configuration file differently or store it in another location, you can specify its path using the `--config` command-line argument when launching the GUI:

*   On Windows: `gui.bat --config D:/my_configs/kohya_settings.toml`
*   On Linux/macOS: `./gui.sh --config /home/user/my_configs/kohya_settings.toml`

By effectively using `config.toml`, you can significantly speed up your training setup process. Always refer to the `config example.toml` for the most up-to-date list of configurable paths.

## Server Options

The `config.toml` file can also be used to configure the Gradio server.

### `allowed_paths`

The `allowed_paths` option allows you to specify a list of directories that the application can access. This is useful if you want to store your models, datasets, or other files on an external drive or in a location outside of the application's root directory.

**Example:**

```toml
[server]
allowed_paths = ["/mnt/external_drive/models", "/home/user/datasets"]
```

### Remote / SSH: use `--headless`

When you run the GUI on a remote machine (SSH, Runpod, Docker, cloud GPU) and open it in a browser, **start with `--headless`**.

Without it, native OS dialogs (easygui overwrite confirmation, local file pickers) run on the **server** process. Over SSH there is usually no interactive display, so those prompts can **block training indefinitely**.

`--headless` does two things that matter for remote use:

* Hides local file/folder picker buttons in the Gradio UI (type paths instead).
* Skips the easygui “overwrite existing model?” prompt and overwrites if the output name already exists.

Examples:

```bash
# Linux / macOS
./gui.sh --listen 0.0.0.0 --server_port 7860 --headless

# Windows
gui.bat --listen 0.0.0.0 --server_port 7860 --headless
```

If you start without `--headless` over SSH or without a display, the process logs a warning recommending `--headless`. There is no separate `--skip_overwrite` flag; use `--headless` for that behavior on train start.

## LoRA

To train a LoRA, you can currently use the `train_network.py` code. You can create a LoRA network by using the all-in-one GUI.

Once you have created the LoRA network, you can generate images using auto1111 by installing [this extension](https://github.com/kohya-ss/sd-webui-additional-networks).

For more detailed information on LoRA training options and advanced configurations, please refer to our LoRA documentation:
- [LoRA Training Guide](docs/LoRA/top_level.md)
- [LoRA Training Options](docs/LoRA/options.md)

## Sample image generation during training

A prompt file might look like this, for example:

```txt
# prompt 1
masterpiece, best quality, (1girl), in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy, bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

# prompt 2
masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n (low quality, worst quality), bad anatomy, bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
```

Lines beginning with `#` are comments. You can specify options for the generated image with options like `--n` after the prompt. The following options can be used:

- `--n`: Negative prompt up to the next option.
- `--w`: Specifies the width of the generated image.
- `--h`: Specifies the height of the generated image.
- `--d`: Specifies the seed of the generated image.
- `--l`: Specifies the CFG scale of the generated image.
- `--s`: Specifies the number of steps in the generation.

The prompt weighting such as `( )` and `[ ]` is working.

## Troubleshooting

If you encounter any issues, refer to the troubleshooting steps below.

### Page File Limit

If you encounter an X error related to the page file, you may need to increase the page file size limit in Windows.

### No module called tkinter

If you encounter an error indicating that the module `tkinter` is not found, try reinstalling Python 3.10 on your system.

### LORA Training on TESLA V100 - GPU Utilization Issue

See [Troubleshooting LORA Training on TESLA V100](docs/troubleshooting_tesla_v100.md) for details.

## SDXL training

For detailed guidance on SDXL training, please refer to the [official sd-scripts documentation](https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_SDXL-en.md) and relevant sections in our [LoRA Training Guide](docs/LoRA/top_level.md).

## Masked loss

The masked loss is supported in each training script. To enable the masked loss, specify the `--masked_loss` option.

> [!WARNING]
> The feature is not fully tested, so there may be bugs. If you find any issues, please open an Issue.

ControlNet dataset is used to specify the mask. The mask images should be the RGB images. The pixel value 255 in R channel is treated as the mask (the loss is calculated only for the pixels with the mask), and 0 is treated as the non-mask. The pixel values 0-255 are converted to 0-1 (i.e., the pixel value 128 is treated as the half weight of the loss). See details for the dataset specification in the [LLLite documentation](./docs/train_lllite_README.md#preparing-the-dataset).

## Guides

The following are guides extracted from issues discussions

### Using Accelerate Lora Tab to Select GPU ID

#### Starting Accelerate in GUI

- Open the kohya GUI on your desired port.
- Open the `Accelerate launch` tab
- Ensure the Multi-GPU checkbox is unchecked.
- Set GPU IDs to the desired GPU (like 1).

#### Running Multiple Instances (linux)

- For tracking multiple processes, use separate kohya GUI instances on different ports (e.g., 7860, 7861).
- Start instances using `nohup ./gui.sh --listen 0.0.0.0 --server_port <port> --headless > log.log 2>&1 &`.

#### Monitoring Processes

- Open each GUI in a separate browser tab.
- For terminal access, use SSH and tools like `tmux` or `screen`.

For more details, visit the [GitHub issue](https://github.com/bmaltais/kohya_ss/issues/2577).

## Interesting Forks

To finetune HunyuanDiT models or create LoRAs, visit this [fork](https://github.com/Tencent/HunyuanDiT/tree/main/kohya_ss-hydit)

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please consider the following:
- For bug reports or feature requests, please open an issue on the [GitHub Issues page](https://github.com/bmaltais/kohya_ss/issues).
- If you'd like to submit code changes, please open a pull request. Ensure your changes are well-tested and follow the existing code style.
- For security-related concerns, please refer to our `SECURITY.md` file.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE.md](LICENSE.md) file for details.

## Change History

Only the upcoming release is detailed here. For the full history of published releases, see [GitHub Releases](https://github.com/bmaltais/kohya_ss/releases).

### benfy95 build (2026-11-25)

**Changed**
- sd-scripts backend v0.12.3 → v0.13.1 (no dependency changes).

**Added**
- Run control for LoRA training (`--run_control`): status panel and Sample / Save / Save state / Pause / Resume /
  Save & stop buttons.
- Gradient Checkpointing Skip Blocks for Anima (LoRA, fine-tuning, LLLite) and MOSS-Image.
- Replica Devices (single-process multi-GPU LoRA) for Anima and MOSS-Image, with pre-launch checks.
- Text Encoder Blocks to Swap for MOSS-Image.
- LoRA → Tools → Post-Hoc EMA (`networks/lora_post_hoc_ema.py`).
- Resize LoRA: SVD method (`merged` / `qr`).

### benfy95 build (2026-11-17)

**Added**
- MOSS-Image LoRA training (sd-scripts `moss_image_train_network.py`, `networks.lora_generic`), including ConvRot
  INT8, H2D-only block swap and Kohya LoHa/LoKr support.
- "Save in ComfyUI format" option for Anima / HunyuanImage-2.1 LoRAs (converts every saved file in place after training).

**Changed**
- sd-scripts backend v0.11.1 → v0.12.3; `transformers` 5.17.0, `diffusers` 0.40.0, `huggingface-hub` 1.32.0,
  `safetensors` 0.8.0, `schedulefree` 1.4.1, `accelerate` ≥ 1.15.0.
- BLIP captioning now uses transformers' BLIP-large model (`tools/blip_caption.py`).

**Fixed**
- Training console output: colours, full-width one-line progress bars, timestamps on every line, visible
  checkpoint-save messages, and stop/end messages on their own line.
- Repeated Starlette `HTTP_422_UNPROCESSABLE_ENTITY` deprecation warning from Gradio is filtered.
- Updating an existing install with an old uv could leave half-removed `*.dist-info` folders that made
  `importlib.metadata` report no version for `transformers`, crashing training at `import diffusers`. The GUI
  removes these leftovers at startup.
- Loaded configs now stick: GUI handlers no longer write stale on-screen values back over a freshly loaded
  config. This reset Network Rank / Conv Rank (a config saved with rank 32 trained at 8), LR warmup, and
  v-parameterization whenever the loaded LoRA type or scheduler differed from the one on screen (e.g. right
  after starting the GUI). Verified by a load -> save round-trip with all 312 settings unchanged.

### v26.0.0

**Added**
- Lumina Image 2.0 LoRA (#3553) and full fine-tune (#3521, #3554) training support.
- Anima LoRA (#3541), full fine-tune (#3523, #3551), ControlNet-LLLite (#3549), and advanced options: torch.compile, Qwen-Image VAE 2D, timestep visualization (#3542).
- Native LoHa/LoKr training (`networks.loha`/`networks.lokr`) for SDXL and Anima (#3550).
- HunyuanImage-2.1 LoRA training support (#3537).
- LECO (concept erasure/editing) training for SD1.x/2.x and SDXL (#3539).
- Inpainting model training support for SD1.5/SDXL (#3540).
- `--show_timesteps` / `--show_timesteps_resolution` exposed in the GUI for DiT training, with a console/matplotlib visualization mode (#3535).
- `--svd_lowrank_niter` exposed in the Resize LoRA tab (#3534).
- `wd-eva02-large-tagger-v3` added to the WD14 tagger (#3315).
- In-app documentation pointer for multi-resolution dataset TOML config (#3536).
- "Expand all accordions" option (#3310).
- Flux Kontext manual captioning tool, with aspect-ratio correction, delete button, target-folder suggestion, and pagination fixes (#3346 series).
- GUI layout polish: `info=` hint text is now hover-revealed instead of always shown, and tool-button icons are bottom-aligned with their input box.

**Changed**
- Upgraded the `sd-scripts` submodule to v0.11.1 and adjusted GUI args for its refactor (#3519, #3520).
- Refactored `lora_gui.py`, `leco_gui.py`, and the remaining GUI files onto a `FIELD_REGISTRY` + dict-adapter pattern for positional-argument safety (#3543, #3544, #3545, #3547, #3548).
- Cleaned up HunyuanImage-2.1 LoRA arg-forwarding cosmetics (#3555).
- Upgraded PyTorch for XPU (#3366).
- Updated Intel packages for oneAPI compatibility (#3341).

**Fixed**
- Console scripts are now found correctly when the venv is not activated (#3552).
- Fixed requirement installation when using system pip (#3515).
- Fixed a TensorBoard crash on non-x86 platforms via a cross-platform AVX check (#3345).
- Fixed installation error on Apple Silicon (M-series) Macs (#3353).
- Fixed broken documentation links in the README (#3505).
- Addressed review feedback across the v0.11.1 upgrade, LoRA+/TI, and FIELD_REGISTRY PRs (#3518, #3533, #3546).
