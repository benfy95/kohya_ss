import os

import gradio as gr
from .common_gui import (
    get_any_file_path,
    get_folder_path,
    document_symbol,
    folder_symbol,
)

# Backend defaults (sd-scripts library/moss_image_train_utils.py / args.py).
# The timestep defaults reproduce the MOSS-Image pre-training distribution as
# recommended by sd-scripts/docs/moss_image_train_network.md, section 7.2.
MOSS_DEFAULT_MAX_TOKEN_LENGTH = 4096
MOSS_MAX_BLOCKS_TO_SWAP = 40  # 42 DiT blocks - 2
MOSS_DIT_BLOCKS = 42
MOSS_TEXT_ENCODER_LAYERS = 32
MOSS_TIMESTEP_SAMPLING_CHOICES = [
    "sigma",
    "uniform",
    "sigmoid",
    "shift",
    "flux_shift",
    "truncated_logit_normal",
]


MOSS_TOKENIZER_REPO = "OpenMOSS-Team/MOSS-Image-Base"
MOSS_TOKENIZER_SUBFOLDER = "tokenizer"
_MOSS_TOKENIZER_FILES = ("tokenizer_config.json", "tokenizer.json")


def resolve_moss_tokenizer_dir(tokenizer_cache_dir: str = "") -> str:
    """Return the tokenizer location to hand to moss_image_train_network.py.

    A user-set value is returned unchanged. Otherwise, if a previous run
    already downloaded the tokenizer into the Hugging Face cache, return that
    local folder: the backend then loads it with ``local_files_only=True``
    instead of re-checking ~10 files on the Hub at every start. Returns ""
    (backend downloads it) when nothing is cached yet.
    """
    if tokenizer_cache_dir:
        return tokenizer_cache_dir
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return ""
    config_path = try_to_load_from_cache(
        MOSS_TOKENIZER_REPO,
        f"{MOSS_TOKENIZER_SUBFOLDER}/{_MOSS_TOKENIZER_FILES[0]}",
    )
    if not isinstance(config_path, str):
        return ""
    folder = os.path.dirname(config_path)
    if all(os.path.isfile(os.path.join(folder, f)) for f in _MOSS_TOKENIZER_FILES):
        return folder
    return ""


class mossImageTraining:
    def __init__(
        self,
        headless: bool = False,
        finetuning: bool = False,
        training_type: str = "",
        config: dict = {},
        moss_image_checkbox: gr.Checkbox = None,
    ) -> None:
        self.headless = headless
        self.finetuning = finetuning
        self.training_type = training_type
        self.config = config
        if moss_image_checkbox is None:
            raise ValueError("moss_image_checkbox must be provided")
        self.moss_image_checkbox = moss_image_checkbox

        with gr.Accordion(
            "MOSS-Image",
            open=True,
            visible=False,
            elem_classes=["moss_image_background"],
        ) as moss_image_accordion:
            with gr.Group():
                gr.Markdown(
                    "Select the DiT (first shard `diffusion_pytorch_model-00001-of-00003.safetensors` "
                    "or a ComfyUI single file) as the pretrained model above, and use LoRA type "
                    "**MOSS-Image** (`networks.lora_generic`), Kohya LoHa or Kohya LoKr."
                )
                with gr.Row():
                    self.text_encoder = gr.Textbox(
                        label="Qwen3.5-9B Text Encoder Path",
                        placeholder="First shard model-00001-of-00004.safetensors or ComfyUI qwen3.5_9b_int8_convrot.safetensors",
                        value=self.config.get("moss_image.text_encoder", ""),
                        interactive=True,
                    )
                    self.text_encoder_button = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                        visible=(not headless),
                        interactive=True,
                    )
                    self.text_encoder_button.click(
                        get_any_file_path,
                        outputs=self.text_encoder,
                        show_progress=False,
                    )

                    self.vae = gr.Textbox(
                        label="FLUX.2 VAE Path",
                        placeholder="Path to flux2-vae.safetensors (ComfyUI / BFL or diffusers format)",
                        value=self.config.get("moss_image.vae", ""),
                        interactive=True,
                    )
                    self.vae_button = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                        visible=(not headless),
                        interactive=True,
                    )
                    self.vae_button.click(
                        get_any_file_path,
                        outputs=self.vae,
                        show_progress=False,
                    )

                with gr.Row():
                    self.tokenizer_cache_dir = gr.Textbox(
                        label="Tokenizer Cache Dir",
                        placeholder="(Optional) HF cache dir or local tokenizer folder for offline training",
                        info="If empty, the tokenizer is downloaded from OpenMOSS-Team/MOSS-Image-Base (~20MB)",
                        value=self.config.get("moss_image.tokenizer_cache_dir", ""),
                        interactive=True,
                    )
                    self.tokenizer_cache_dir_button = gr.Button(
                        folder_symbol,
                        elem_id="open_folder_small",
                        visible=(not headless),
                        interactive=True,
                    )
                    self.tokenizer_cache_dir_button.click(
                        get_folder_path,
                        outputs=self.tokenizer_cache_dir,
                        show_progress=False,
                    )
                    self.max_token_length = gr.Number(
                        label="Max Token Length",
                        value=self.config.get(
                            "moss_image.max_token_length",
                            MOSS_DEFAULT_MAX_TOKEN_LENGTH,
                        ),
                        info="Text tokens incl. chat template. Default and maximum: 4096",
                        minimum=1,
                        maximum=MOSS_DEFAULT_MAX_TOKEN_LENGTH,
                        step=1,
                        interactive=True,
                    )

                with gr.Row():
                    self.timestep_sampling = gr.Dropdown(
                        label="Timestep Sampling",
                        choices=MOSS_TIMESTEP_SAMPLING_CHOICES,
                        value=self.config.get(
                            "moss_image.timestep_sampling", "truncated_logit_normal"
                        ),
                        info="truncated_logit_normal + mean -1.09 / std 0.95 / shift 1.0 reproduces the pre-training schedule",
                        interactive=True,
                    )
                    self.discrete_flow_shift = gr.Number(
                        label="Discrete Flow Shift",
                        value=self.config.get("moss_image.discrete_flow_shift", 1.0),
                        info="Used by sigma / shift / truncated_logit_normal. Default 1.0",
                        minimum=-1024,
                        maximum=1024,
                        step=0.01,
                        interactive=True,
                    )
                    self.sigmoid_scale = gr.Number(
                        label="Sigmoid Scale",
                        value=self.config.get("moss_image.sigmoid_scale", 1.0),
                        info="Used by sigmoid / shift / flux_shift",
                        minimum=0.0,
                        maximum=1024,
                        step=0.01,
                        interactive=True,
                    )

                with gr.Row():
                    self.logit_mean = gr.Number(
                        label="Logit Mean",
                        value=self.config.get("moss_image.logit_mean", -1.09),
                        info="Used by sigma / truncated_logit_normal",
                        step=0.01,
                        interactive=True,
                    )
                    self.logit_std = gr.Number(
                        label="Logit Std",
                        value=self.config.get("moss_image.logit_std", 0.95),
                        info="Used by sigma / truncated_logit_normal",
                        step=0.01,
                        interactive=True,
                    )
                    self.timestep_cutoff = gr.Number(
                        label="Timestep Cutoff",
                        value=self.config.get("moss_image.timestep_cutoff", 0.95),
                        info="truncated_logit_normal only. Default 0.95",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        interactive=True,
                    )
                    self.timestep_uniform_ratio = gr.Number(
                        label="Timestep Uniform Ratio",
                        value=self.config.get(
                            "moss_image.timestep_uniform_ratio", 0.01
                        ),
                        info="truncated_logit_normal only. Default 0.01",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.001,
                        interactive=True,
                    )

                with gr.Row():
                    self.attn_mode = gr.Dropdown(
                        label="Attention Mode",
                        choices=["torch", "xformers", "flash"],
                        value=self.config.get("moss_image.attn_mode", "torch"),
                        info="xformers requires Split Attention (sageattn is inference-only)",
                        interactive=True,
                    )
                    self.split_attn = gr.Checkbox(
                        label="Split Attention",
                        value=self.config.get("moss_image.split_attn", False),
                        info="Lower memory; required for xformers",
                        interactive=True,
                    )
                    self.sample_uncond_skip_layers = gr.Textbox(
                        label="Sample Uncond Skip Layers",
                        value=self.config.get(
                            "moss_image.sample_uncond_skip_layers", "22"
                        ),
                        info="DiT blocks skipped in the CFG negative branch for samples. Default 22 (official). Empty disables",
                        interactive=True,
                    )

                with gr.Row():
                    self.convrot_int8 = gr.Checkbox(
                        label="ConvRot INT8 (DiT)",
                        value=self.config.get("moss_image.convrot_int8", False),
                        info="Halves DiT weight VRAM. Replaces fp8_base (unsupported). Not needed for *_int8_convrot files",
                        interactive=True,
                    )
                    self.convrot_int8_text_encoder = gr.Checkbox(
                        label="ConvRot INT8 (Text Encoder)",
                        value=self.config.get(
                            "moss_image.convrot_int8_text_encoder", False
                        ),
                        info="~16GB -> ~8GB. Not needed for the ComfyUI int8 file",
                        interactive=True,
                    )
                    self.convrot_int8_bwd = gr.Dropdown(
                        label="ConvRot INT8 Backward",
                        choices=["bf16", "int8"],
                        value=self.config.get("moss_image.convrot_int8_bwd", "bf16"),
                        info="int8 requires triton (triton-windows on Windows)",
                        interactive=True,
                    )

                with gr.Row():
                    self.block_swap_h2d_only = gr.Checkbox(
                        label="H2D-only Block Swap",
                        value=self.config.get("moss_image.block_swap_h2d_only", False),
                        info="Faster block swap for LoRA (set Blocks to swap, max 40). Requires gradient checkpointing",
                        interactive=True,
                    )
                    self.unsloth_offload_checkpointing = gr.Checkbox(
                        label="Unsloth Offload Checkpointing",
                        value=self.config.get(
                            "moss_image.unsloth_offload_checkpointing", False
                        ),
                        info="Offload activations to CPU RAM. Not with Blocks to swap",
                        interactive=True,
                    )
                    self.compile = gr.Checkbox(
                        label="Per-block Torch Compile",
                        value=self.config.get("moss_image.compile", False),
                        info="--compile (requires triton)",
                        interactive=True,
                    )

                with gr.Row():
                    self.gradient_checkpointing_skip_blocks = gr.Number(
                        label="Gradient Checkpointing Skip Blocks",
                        value=self.config.get(
                            "moss_image.gradient_checkpointing_skip_blocks", 0
                        ),
                        info="Last N of 42 DiT blocks keep their activations: ~1GB more VRAM and ~8.6ms faster per block. Needs Gradient checkpointing. 0 = off",
                        minimum=0,
                        maximum=MOSS_DIT_BLOCKS,
                        step=1,
                        interactive=True,
                    )
                    self.text_encoder_blocks_to_swap = gr.Number(
                        label="Text Encoder Blocks to Swap",
                        value=self.config.get(
                            "moss_image.text_encoder_blocks_to_swap", 0
                        ),
                        info="Stream N of 32 text encoder layers from CPU while caching its outputs (28: ~5GB, 32: ~3.5GB peak). 0 = off",
                        minimum=0,
                        maximum=MOSS_TEXT_ENCODER_LAYERS,
                        step=1,
                        interactive=True,
                    )
                    self.replica_devices = gr.Textbox(
                        label="Replica Devices (multi-GPU, experimental)",
                        value=self.config.get("moss_image.replica_devices", ""),
                        placeholder="e.g. 1 or cuda:1,cuda:2",
                        info="Extra GPUs for the LoRA, no DDP needed (works on Windows). Requires Cache latents and Cache Text Encoder Outputs. Batch size is per GPU",
                        interactive=True,
                    )

                with gr.Row():
                    self.enable_multiline_caption = gr.Checkbox(
                        label="Enable Multiline Caption",
                        value=self.config.get(
                            "moss_image.enable_multiline_caption", False
                        ),
                        info="Keep line breaks in captions; '---===---' lines separate caption candidates",
                        interactive=True,
                    )
                    self.train_text_embedder = gr.Checkbox(
                        label="Train Text Embedder",
                        value=self.config.get("moss_image.train_text_embedder", False),
                        info="Also apply LoRA to the DiT text embedder (network_args train_text_embedder=True)",
                        interactive=True,
                    )

                with gr.Row():
                    self.cache_text_encoder_outputs = gr.Checkbox(
                        label="Cache Text Encoder Outputs",
                        value=self.config.get(
                            "moss_image.cache_text_encoder_outputs", True
                        ),
                        info="Strongly recommended. Forces DiT-only LoRA (network_train_unet_only)",
                        interactive=True,
                    )
                    self.cache_text_encoder_outputs_to_disk = gr.Checkbox(
                        label="Cache Text Encoder Outputs to Disk",
                        value=self.config.get(
                            "moss_image.cache_text_encoder_outputs_to_disk", True
                        ),
                        info="Writes <image>_moss_te.npz next to each image",
                        interactive=True,
                    )

                self.moss_image_checkbox.change(
                    lambda moss_image_checkbox: gr.Accordion(
                        visible=moss_image_checkbox
                    ),
                    inputs=[self.moss_image_checkbox],
                    outputs=[moss_image_accordion],
                )
