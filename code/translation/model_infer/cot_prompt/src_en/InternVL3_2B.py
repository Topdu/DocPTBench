#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import argparse
from pathlib import Path
import re

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# ========== Preprocessing Constants ==========
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ========== 1. Multi-language Configuration ==========
LANG_CONFIG = {
    "zh": {
        "name": "Simplified Chinese",
        "tag": "chinese_md",
        "ext": ".zh.md",
        "style_note": "Fluent, professional zh-CN for narrative text; use standard Chinese punctuation."
    },
    "de": {
        "name": "German",
        "tag": "german_md",
        "ext": ".de.md",
        "style_note": "Fluent, professional German for narrative text; use standard German punctuation."
    },
    "fr": {
        "name": "French",
        "tag": "french_md",
        "ext": ".fr.md",
        "style_note": "Fluent, professional French for narrative text; use standard French punctuation."
    },
    "ru": {
        "name": "Russian",
        "tag": "russian_md",
        "ext": ".ru.md",
        "style_note": "Fluent, professional Russian for narrative text; use standard Russian punctuation."
    }
}

# ========== 2. Dynamic Prompt Template ==========
# Note: In Python format strings, literal { } need to be escaped as {{ }}
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
- In <{lang_tag}>, translate ONLY human words inside formulas with \\text{{...}} (e.g., translate "Mean" to target language); keep math intact.

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
- ANY formula/equation is represented in LaTeX.
- No added/omitted content; structure renders correctly.
- Output ONLY the two blocks. No explanations or reasoning."""

# ========== Image Preprocessing (Unchanged) ==========
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
    """
    Dynamically parse <english_md> and <{target}_md> based on lang_code.
    """
    tag = LANG_CONFIG[lang_code]["tag"]
    
    # Dynamic regex generation
    en_re = re.compile(r"<english_md>\s*(.*?)\s*</english_md>", re.S | re.I)
    target_re = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)

    m_en = en_re.search(text)
    en_md = m_en.group(1).strip() if m_en else ""

    m_target = target_re.search(text)
    if m_target:
        target_md = m_target.group(1).strip()
    else:
        # Fallback: If closed tag not found but text is not empty, try matching everything after the opening tag
        # Or if no tags found at all and en_md is empty, assume full text is target translation (depending on failure mode)
        m_loose = re.search(rf"<{tag}>\s*(.*)", text, re.S | re.I)
        if m_loose:
            target_md = m_loose.group(1).strip()
        elif not en_md and text.strip():
            # Extreme fallback: Output contains no standard tags, assume all is target text
            target_md = text.strip()
        else:
            target_md = ""

    return en_md, target_md

def force_h2_headings(md: str) -> str:
    """Normalize ATX/Setext headings to '## ' only in non-code blocks."""
    if not md: return ""
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
    # Fill template
    sys_prompt = DOC_TRANSLATION_PROMPT_TEMPLATE.format(
        lang_name=cfg["name"],
        lang_tag=cfg["tag"],
        style_note=cfg["style_note"]
    )
    
    user_instruction = (
        f"Reconstruct and translate the document image as specified. "
        f"Return ONLY the two blocks: <english_md>...</english_md> and <{cfg['tag']}>...</{cfg['tag']}>."
    )

    # InternVL3 standard format: System Prompt + <image> + User Instruction
    return (
        sys_prompt.strip()
        + "\n\n<image>\n"
        + user_instruction
    )

# ========== Inference ==========
@torch.inference_mode()
def translate_one(model, tokenizer, image_path: str, generation_config: dict, lang_code: str,
                  input_size=448, max_num=12, use_thumbnail=True, device="cuda"):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    pixel_values = load_image(
        image_path, input_size=input_size, max_num=max_num,
        use_thumbnail=use_thumbnail, device=device, dtype=dtype
    )
    
    # Dynamically build question
    question = build_question(lang_code)
    
    # Inference
    full_text = model.chat(tokenizer, pixel_values, question, generation_config).strip()
    
    # Parsing
    en_md, target_md = parse_dual_blocks(full_text, lang_code)
    
    # Heading normalization
    if en_md: en_md = force_h2_headings(en_md)
    if target_md: target_md = force_h2_headings(target_md)
    
    return en_md, target_md, full_text

# ========== Command Line Entry ==========
def main():
    parser = argparse.ArgumentParser(description="InternVL3 Document Image -> Dual-block Markdown (Multi-Language)")
    
    # Basic Arguments
    parser.add_argument("--model", default="OpenGVLab/InternVL3-2B", help="HuggingFace Model ID or Local Path")
    parser.add_argument("--dir", required=True, help="Input image folder")
    parser.add_argument("--out_dir", default=None, help="Output directory (default generated based on language)")
    
    # Language Selection
    parser.add_argument("--lang", default="zh", choices=["zh", "de", "fr", "ru"], 
                        help="Target language: zh (Chinese), de (German), fr (French), ru (Russian)")
    
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files")
    
    # Model Arguments
    parser.add_argument("--input_size", type=int, default=448)
    parser.add_argument("--max_num", type=int, default=12)
    parser.add_argument("--no_thumbnail", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--flash_attn", action="store_true", help="Enable flash-attn")

    # Save Options
    parser.add_argument("--save_english", action="store_true", help="Save English reconstruction block")
    parser.add_argument("--en_dir", type=str, default=None, help="English output directory")
    parser.add_argument("--save_full", action="store_true", help="Save full raw output")
    parser.add_argument("--raw_dir", type=str, default=None, help="Full output directory")
    
    args = parser.parse_args()

    # 1. Path and Config
    lang_cfg = LANG_CONFIG[args.lang]
    target_ext = lang_cfg["ext"]
    
    in_dir = Path(args.dir)
    
    # Default main output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(f"translations_internvl3_2B_{args.lang}")
    
    out_dir.mkdir(parents=True, exist_ok=True)

    # English output directory
    english_out_dir = None
    if args.save_english:
        if args.en_dir:
            english_out_dir = Path(args.en_dir)
        else:
            english_out_dir = out_dir / "en_ocr"
        english_out_dir.mkdir(parents=True, exist_ok=True)

    # Raw output directory
    raw_dir = None
    if args.save_full:
        if args.raw_dir:
            raw_dir = Path(args.raw_dir)
        else:
            raw_dir = out_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    image_paths = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])
    if not image_paths:
        raise ValueError("No images found in directory.")

    # 2. Load Model
    print(f"[INFO] Loading Model: {args.model}")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=args.flash_attn,
        trust_remote_code=True,
        device_map="auto",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)

    generation_config = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=(args.temperature > 0),
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=1.05,
    )

    print(f"[INFO] Target Language: {lang_cfg['name']} ({args.lang})")
    print(f"[INFO] Output Dir: {out_dir.resolve()}")

    # 3. Processing Loop
    for img in image_paths:
        target_path = out_dir / (img.stem + target_ext)
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

            # Save target translation
            if target_md:
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(target_md)
            else:
                print(f"[WARN] {img.name}: No translation block found.")

            # Save English reconstruction
            if args.save_english and english_out_dir and en_md:
                en_path = english_out_dir / (img.stem + ".en.md")
                with open(en_path, "w", encoding="utf-8") as f:
                    f.write(en_md)

            # Save full raw output
            if args.save_full and raw_dir:
                raw_path = raw_dir / (img.stem + ".full.txt")
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(full_text)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"[OK] {img.name} -> {target_path.name}"
                  + (f" (+en)" if (args.save_english and en_md) else "")
                  + (f" (+raw)" if raw_dir else ""))

        except Exception as e:
            err_path = out_dir / (img.stem + ".error.txt")
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(f"Translation failed for {img.name}:\n{e}\n")
            print(f"[ERR] {img.name} -> {e}")

if __name__ == "__main__":
    main()