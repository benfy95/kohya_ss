"""Caption images with BLIP (large) using transformers' own BLIP implementation.

Replaces sd-scripts' ``finetune/make_captions.py`` for the GUI's BLIP tab.
That script bundles a 2022 copy of the BLIP model code which no longer runs on
transformers 5 (required by sd-scripts >= 0.12 for MOSS-Image): helpers it
imports were moved/removed and its decoder relies on the generation API
transformers 5 dropped. ``Salesforce/blip-image-captioning-large`` is the same
BLIP-large captioning model, maintained inside transformers.

Behaviour mirrors make_captions.py: same options, the "a picture of " prompt
(stripped from the output), beam search or nucleus sampling, one caption file
per image written next to it (overwriting).

    python tools/blip_caption.py <image folder> [--caption_extension .txt] ...
"""

import argparse
import os
import sys
from pathlib import Path

DEFAULT_MODEL = "Salesforce/blip-image-captioning-large"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".jxl")
# Same conditioning prompt as BLIP's original captioner (blip.py BLIP_Decoder).
PROMPT = "a picture of "


def find_images(folder: Path, recursive: bool) -> list:
    pattern = "**/*" if recursive else "*"
    return sorted(
        p
        for p in folder.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def strip_prompt(caption: str, prompt: str = PROMPT) -> str:
    caption = caption.strip()
    if caption.lower().startswith(prompt.strip().lower()):
        caption = caption[len(prompt.strip()) :].strip()
    return caption


def generation_kwargs(args) -> dict:
    """Match BLIP_Decoder.generate() in the original repository."""
    common = {"max_length": args.max_length, "min_length": args.min_length}
    if args.beam_search:
        return {**common, "num_beams": args.num_beams, "repetition_penalty": 1.0}
    return {
        **common,
        "do_sample": True,
        "top_p": args.top_p,
        "repetition_penalty": 1.1,
    }


def pick_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def setup_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("train_data_dir", help="folder with the images to caption")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local folder")
    p.add_argument("--caption_extension", default=".caption")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--beam_search", action="store_true")
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_length", type=int, default=75)
    p.add_argument("--min_length", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--debug", action="store_true", help="print each caption")
    return p


def main(argv=None) -> int:
    args = setup_parser().parse_args(argv)
    folder = Path(args.train_data_dir)
    if not folder.is_dir():
        print(f"Image folder not found: {folder}", file=sys.stderr)
        return 1

    images = find_images(folder, args.recursive)
    print(f"found {len(images)} images in {folder}")
    if not images:
        return 0

    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import BlipForConditionalGeneration, BlipProcessor

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"loading BLIP: {args.model} on {device}")
    processor = BlipProcessor.from_pretrained(args.model)
    model = BlipForConditionalGeneration.from_pretrained(args.model).to(device).eval()
    gen_kwargs = generation_kwargs(args)

    batch_size = max(1, int(args.batch_size))
    for start in tqdm(range(0, len(images), batch_size), smoothing=0.0):
        paths, pil_images = [], []
        for path in images[start : start + batch_size]:
            try:
                pil_images.append(Image.open(path).convert("RGB"))
                paths.append(path)
            except Exception as e:  # corrupted / unsupported image
                print(f"skipping {path}: {e}", file=sys.stderr)
        if not paths:
            continue
        inputs = processor(
            images=pil_images, text=[PROMPT] * len(pil_images), return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        captions = processor.batch_decode(out, skip_special_tokens=True)
        for path, caption in zip(paths, captions):
            caption = strip_prompt(caption)
            with open(
                path.with_suffix(args.caption_extension), "w", encoding="utf-8"
            ) as f:
                f.write(caption + "\n")
            if args.debug:
                print(f"{path.name}: {caption}")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
