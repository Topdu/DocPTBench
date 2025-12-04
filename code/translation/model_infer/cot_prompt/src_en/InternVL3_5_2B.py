#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import re
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# ========== 1. Multi-language Configuration ==========
LANG_CONFIG = {
    "zh": {
        "name": "Simplified Chinese",
        "tag": "chinese_md",
        "ext": ".zh.mmd",
        "style_note": "Fluent, professional zh-CN for narrative text; use Chinese punctuation in prose."
    },
    "de": {
        "name": "German",
        "tag": "german_md",
        "ext": ".de.mmd",
        "style_note": "Fluent, professional German; use standard German punctuation."
    },
    "fr": {
        "name": "French",
        "tag": "french_md",
        "ext": ".fr.mmd",
        "style_note": "Fluent, professional French; use standard French punctuation."
    },
    "ru": {
        "name": "Russian",
        "tag": "russian_md",
        "ext": ".ru.mmd",
        "style_note": "Fluent, professional Russian; use standard Russian punctuation."
    }
}

# ========== 2. Dynamic Prompt Template ==========
DOC_TRANSLATION_PROMPT_TEMPLATE = """You are a professional document translator.

Task: First OCR and reconstruct the readable English Markdown from the whole document image. 
Then produce a high-fidelity {lang_name} Markdown translation.

Output format (STRICT):
Return EXACTLY two fenced blocks, in this order, and nothing else:
<english_md>
...the English Markdown reconstruction...
</english_md>
<{lang_tag}>
...the {lang_name} Markdown translation...
</{lang_tag}>

Formulas (LaTeX ONLY):
- Detect ANY formula/equation and represent it in LaTeX.
- Wrap EVERY formula in display math delimiters: $$ ... $$ (use display math even if the source looked inline).
- Inside $$...$$, DO NOT alter symbols/operators/variables/numbers/units.
- In <english_md>, keep any human words in formulas as English (e.g., \\text{{Mean}}).
- In <{lang_tag}>, translate ONLY human words inside formulas with \\text{{...}} (e.g., translate terms to target language); keep math intact.

Tables (keep Markdown):
- Keep ALL tables as Markdown tables with the same column count, pipes `|`, alignment markers, and row structure.
- Translate only narrative text in cells; do NOT convert tables to LaTeX.

Other structure (preserve):
- Paragraphs, blank lines, lists (ordered/unordered), task lists (- [ ] / - [x]),
  blockquotes (>), horizontal rules, footnotes/references, anchors/IDs, figure numbers and captions.
- Hyperlinks/images: translate visible text (alt/title) ONLY; keep URLs and reference labels unchanged.

Do-not-translate / protect:
- Code: fenced blocks ```...``` (with language) and inline `...` (copy as-is).
- URLs/emails/paths/domains/ids/hashes and link reference labels ([ref-id]:).
- Stable product/brand/model names commonly written in English (e.g., “GitHub”, “ResNet-50”).
- Template variables/placeholders: {{var}}, {{{{handlebars}}}}, <PLACEHOLDER>, %s, {{0}}, $ENV_VAR.

Target Language Style:
- {style_note}
- NEVER change punctuation inside code, LaTeX math, or URLs.

Quality checks BEFORE finalizing:
- <english_md> is English-only, with all formulas as $$...$$ LaTeX.
- <{lang_tag}> translates narrative text; formulas use the same $$...$$ LaTeX with only \\text{{...}} translated.
- ANY formula/equation is represened in LaTeX.
- No added/omitted content; structure renders correctly.
- Output ONLY the two blocks. No explanations or reasoning."""

# ========== Preprocessing Constants ==========
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ========== Image Preprocessing ==========
def build_transform(input_size: int):
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image: Image.Image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12, use_thumbnail=True, device="cuda", dtype=torch.bfloat16):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=use_thumbnail, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(dtype=dtype)
    if device != "cpu":
        pixel_values = pixel_values.cuda()
    return pixel_values

# ========== Dual-block Parsing & Post-processing ==========

def parse_dual_blocks(text: str, lang_code: str):
    """Dynamically parse based on language config"""
    tag = LANG_CONFIG[lang_code]["tag"]
    
    # Compile regex
    target_re = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    en_re = re.compile(r"<english_md>\s*(.*?)\s*</english_md>", re.S | re.I)

    zh = target_re.search(text)
    en = en_re.search(text)
    
    # Fallback: if no tag found but text exists, assume full text is target
    target_md = zh.group(1).strip() if zh else text.strip()
    en_md = en.group(1).strip() if en else ""
    
    # Edge case: if en_md found but target_md empty (and text not empty), likely tag failure
    if not target_md and not en_md and text.strip():
        target_md = text.strip()

    return en_md, target_md

def force_h2_headings(md: str) -> str:
    """Normalize ATX/Setext headings to '## ' only in non-code blocks."""
    lines = md.splitlines()
    out, in_code = [], False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            in_code = not in_code
            out.append(line); i += 1; continue

        if not in_code:
            m = re.match(r'^(\s*)#{1,6}\s+(.*)$', line)
            if m:
                out.append(f"{m.group(1)}## {m.group(2)}"); i += 1; continue
            if i + 1 < len(lines):
                m2 = re.match(r'^\s*(=+|-+)\s*$', lines[i+1])
                if m2 and lines[i].strip():
                    out.append(f"## {lines[i].strip()}"); i += 2; continue

        out.append(line); i += 1
    return "\n".join(out)

# ========== Build Question ==========
def build_question(lang_code: str):
    cfg = LANG_CONFIG[lang_code]
    
    # Fill Prompt Template
    system_part = DOC_TRANSLATION_PROMPT_TEMPLATE.format(
        lang_name=cfg["name"],
        lang_tag=cfg["tag"],
        style_note=cfg["style_note"]
    )
    
    user_instruction = (
        f"Reconstruct and translate the document image as specified. "
        f"Return ONLY the two blocks: <english_md>...</english_md> and <{cfg['tag']}>...</{cfg['tag']}>."
    )

    # InternVL format: <image>\nUser Instruction
    return system_part.strip() + "\n\n<image>\n" + user_instruction

# ========== Inference ==========
@torch.inference_mode()
def translate_one(model, tokenizer, image_path: str, generation_config: dict,
                  lang_code: str,
                  input_size=448, max_num=12, use_thumbnail=True, device="cuda"):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    pixel_values = load_image(
        image_path, input_size=input_size, max_num=max_num,
        use_thumbnail=use_thumbnail, device=device, dtype=dtype
    )
    
    # Build prompt
    question = build_question(lang_code)
    
    full_text = model.chat(tokenizer, pixel_values, question, generation_config).strip()
    
    # Parse
    en_md, target_md = parse_dual_blocks(full_text, lang_code)
    
    if en_md: en_md = force_h2_headings(en_md)
    if target_md: target_md = force_h2_headings(target_md)
    
    return en_md, target_md, full_text

# ========== Main Entry ==========
def main():
    parser = argparse.ArgumentParser(description="InternVL3 Document Image -> Dual-block Markdown (Multi-Language)")
    
    # Core Arguments
    # Changed default to public HuggingFace ID to avoid local absolute paths
    parser.add_argument("--model", default="OpenGVLab/InternVL3_5-2B", help="HuggingFace model path or ID")
    parser.add_argument("--dir", required=True, help="Input image folder")
    parser.add_argument("--lang", default="zh", choices=["zh", "de", "fr", "ru"], help="Target language: zh, de, fr, ru")
    
    # Output Control
    parser.add_argument("--out_dir", default=None, help="Output directory (defaults to ./results_{lang})")
    parser.add_argument("--en_dir", default=None, help="English reconstruction output directory")
    parser.add_argument("--raw_dir", default=None, help="Full raw output directory")
    
    # Switches
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing files")
    parser.add_argument("--save_english", action="store_true", help="Save English reconstruction block")
    parser.add_argument("--save_full", action="store_true", help="Save full raw model output")
    
    # Model Parameters
    parser.add_argument("--input_size", type=int, default=448)
    parser.add_argument("--max_num", type=int, default=12)
    parser.add_argument("--no_thumbnail", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    args = parser.parse_args()

    # 1. Path Setup
    lang_cfg = LANG_CONFIG[args.lang]
    
    # Determine Output Directory
    # Removed hardcoded absolute paths. Now defaults to relative path.
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(f"translations_internvl35_2B_{args.lang}")
    
    out_dir.mkdir(parents=True, exist_ok=True)

    # English Output Directory
    if args.save_english:
        if args.en_dir:
            en_dir = Path(args.en_dir)
        else:
            en_dir = out_dir / "en_ocr"
        en_dir.mkdir(parents=True, exist_ok=True)
    else:
        en_dir = None

    # Raw Output Directory
    if args.save_full:
        if args.raw_dir:
            raw_dir = Path(args.raw_dir)
        else:
            raw_dir = out_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
    else:
        raw_dir = None

    # 2. Collect Images
    in_dir = Path(args.dir)
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    if not in_dir.exists():
        raise ValueError(f"Input directory does not exist: {in_dir}")
        
    image_paths = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])
    if not image_paths:
        raise ValueError("No images found in the input directory.")

    # 3. Load Model
    print(f"[INFO] Loading Model: {args.model}")
    print(f"[INFO] Target Language: {lang_cfg['name']} ({args.lang})")
    
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    
    # Ensure no API keys are printed or logged
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto").eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    
    generation_config = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=(args.temperature > 0),
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=1.05,
    )

    print(f"[INFO] Output Dir: {out_dir.resolve()}")

    # 4. Inference Loop
    for img in image_paths:
        target_path = out_dir / (img.stem + lang_cfg["ext"])
        if target_path.exists() and not args.overwrite:
            print(f"[SKIP] {target_path.name} exists.")
            continue

        try:
            en_md, target_md, full_text = translate_one(
                model, tokenizer, str(img), generation_config,
                lang_code=args.lang,
                input_size=args.input_size, max_num=args.max_num,
                use_thumbnail=not args.no_thumbnail,
                device=("cuda" if torch.cuda.is_available() else "cpu")
            )

            # Save Target Translation
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(target_md)

            # Save English Reconstruction
            if args.save_english and en_dir and en_md:
                en_path = en_dir / (img.stem + "_en.md")
                with open(en_path, "w", encoding="utf-8") as f:
                    f.write(en_md)

            # Save Full Output
            if args.save_full and raw_dir:
                raw_path = raw_dir / (img.stem + ".full.txt")
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(full_text)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"[OK] {img.name} -> {target_path.name}"
                  + (f" (+en)" if (args.save_english and en_md) else "")
                  + (f" (+raw)" if args.save_full else ""))

        except Exception as e:
            err_path = out_dir / (img.stem + ".error.txt")
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(f"Translation failed for {img.name}:\n{e}\n")
            print(f"[ERR] {img.name} -> see {err_path.name}")

if __name__ == "__main__":
    main()