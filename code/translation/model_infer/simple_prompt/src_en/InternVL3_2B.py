#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
import re

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
        "ext": ".zh.md",
        "style_note": "Use fluent, professional Simplified Chinese with standard punctuation."
    },
    "de": {
        "name": "German",
        "tag": "german_md",
        "ext": ".de.md",
        "style_note": "Use fluent, professional German with standard German punctuation."
    },
    "fr": {
        "name": "French",
        "tag": "french_md",
        "ext": ".fr.md",
        "style_note": "Use fluent, professional French with standard French punctuation."
    },
    "ru": {
        "name": "Russian",
        "tag": "russian_md",
        "ext": ".ru.md",
        "style_note": "Use fluent, professional Russian with standard Russian punctuation."
    }
}

# ========== 2. Preprocessing Constants ==========
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ========== 3. Dynamic Prompt Template ==========
DOC_TRANSLATION_PROMPT_TEMPLATE = """You are a professional document translator.

Task: Translate the content of the English document image directly into high-fidelity {lang_name} Markdown.

Output format (STRICT):
Return EXACTLY one fenced block, and nothing else:
<{lang_tag}>
...the {lang_name} Markdown translation...
</{lang_tag}>

Translation Rules:
- Formulas (LaTeX ONLY): Detect and represent any formula/equation in LaTeX, wrapped in display math delimiters: $$ ... $$. Inside the delimiters, translate only human words with \\text{{...}} (e.g., \\text{{Mean}} -> \\text{{translated_term}}).
- Tables (Markdown): Preserve Markdown tables. Translate only narrative text in cells.
- Structure: Preserve all structural elements like paragraphs, lists, blockquotes, etc.
- Do-not-translate: Keep code (`...`), URLs, emails, and stable English names (e.g., "GitHub") as is.
- Target Style: {style_note}
- Quality Check: Ensure the output is a single, complete, and accurate {lang_name} Markdown block. Do not add any explanations or extra content.
"""

# ========== 4. Image Preprocessing (InternVL Standard) ==========
def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
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
    target_ratios = sorted(set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    ), key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width, target_height = image_size * target_aspect_ratio[0], image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) > 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12, use_thumbnail=True, device="cuda", dtype=torch.bfloat16):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=use_thumbnail, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images]).to(dtype=dtype)
    if device != "cpu":
        pixel_values = pixel_values.cuda()
    return pixel_values

# ========== 5. Parsing and Post-processing ==========

def parse_target_block(text: str, lang_code: str) -> str:
    """Dynamically parse XML blocks based on language configuration."""
    tag = LANG_CONFIG[lang_code]["tag"]
    # Dynamic regex construction
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    
    m = pattern.search(text)
    return m.group(1).strip() if m else text.strip()

def force_h2_headings(md: str) -> str:
    """Unify all Markdown headings to '## ' (maintain structural consistency)."""
    lines = md.splitlines()
    out, in_code = [], False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            in_code = not in_code
            out.append(line); i += 1; continue
        if not in_code:
            if re.match(r'^\s*#{1,6}\s+.*$', line):
                out.append(re.sub(r'^\s*#{1,6}\s+', '## ', line)); i += 1; continue
            if i + 1 < len(lines) and re.match(r'^\s*(=+|-+)\s*$', lines[i+1]) and lines[i].strip():
                out.append(f"## {lines[i].strip()}"); i += 2; continue
        out.append(line); i += 1
    return "\n".join(out)

# ========== 6. Construct Question (Dynamic) ==========
def build_question(lang_code: str):
    cfg = LANG_CONFIG[lang_code]
    
    # Fill System Prompt
    system_prompt = DOC_TRANSLATION_PROMPT_TEMPLATE.format(
        lang_name=cfg["name"],
        lang_tag=cfg["tag"],
        style_note=cfg["style_note"]
    )
    
    # Construct User Instruction
    user_instruction = (
        f"Translate the document image directly into {cfg['name']} Markdown. "
        f"Return ONLY the <{cfg['tag']}>...</{cfg['tag']}> block."
    )
    
    return (
        system_prompt.strip()
        + "\n\n<image>\n"
        + user_instruction
    )

# ========== 7. Inference Logic ==========
@torch.inference_mode()
def translate_one(model, tokenizer, image_path: str, lang_code: str, generation_config: dict,
                  input_size=448, max_num=12, use_thumbnail=True, device="cuda") -> str:
    
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    
    # Load image
    pixel_values = load_image(image_path, input_size, max_num, use_thumbnail, device, dtype)
    
    # Build prompt for specific language
    question = build_question(lang_code)
    
    # Model inference
    full_text = model.chat(tokenizer, pixel_values, question, generation_config).strip()
    
    # Parse result
    target_md = parse_target_block(full_text, lang_code)
    
    if target_md:
        target_md = force_h2_headings(target_md)
        
    return target_md

# ========== 8. Main Program ==========
def main():
    parser = argparse.ArgumentParser(description="InternVL3 Document Image Direct Translation (Multilingual)")
    parser.add_argument("--model", default="OpenGVLab/InternVL3-2B", help="HuggingFace model path or local path")
    parser.add_argument("--dir", required=True, help="Input image folder")
    
    # Language options
    parser.add_argument("--lang", default="zh", choices=["zh", "de", "fr", "ru"], help="Target language (default: zh)")
    
    parser.add_argument("--out_dir", default=None, help="Output directory (default: auto-generated)")
    parser.add_argument("--ext", default=None, help="Output file extension (default: auto-selected)")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files")

    # Model and Preprocessing parameters
    parser.add_argument("--input_size", type=int, default=448, help="Tile size")
    parser.add_argument("--max_num", type=int, default=12, help="Max tiles")
    parser.add_argument("--no_thumbnail", action="store_true", help="Do not append thumbnail tile")
    parser.add_argument("--flash_attn", action="store_true", help="Enable Flash Attention 2")

    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    
    args = parser.parse_args()

    # Get language config
    lang_cfg = LANG_CONFIG[args.lang]

    # Auto-configure paths
    in_dir = Path(args.dir)
    
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(f"translations_internvl3_direct_en2{args.lang}")
    
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-configure extension
    file_ext = args.ext if args.ext else lang_cfg["ext"]

    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    image_paths = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])
    if not image_paths:
        raise ValueError(f"No supported image files found in directory '{in_dir}'.")

    # Load model
    print(f"[INFO] Loading model from: {args.model} ...")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    model = AutoModel.from_pretrained(
        args.model, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
        use_flash_attention_2=args.flash_attn, trust_remote_code=True, device_map="auto"
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
    print(f"[INFO] Output Directory: {out_dir.resolve()}")
    print(f"[INFO] Found {len(image_paths)} images.")

    # Sequential processing (local LLMs usually limited by VRAM, concurrency not recommended)
    for img_path in image_paths:
        out_path = out_dir / (img_path.stem + file_ext)
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] File exists: {out_path.name}")
            continue

        try:
            target_md = translate_one(
                model, tokenizer, str(img_path), 
                lang_code=args.lang,
                generation_config=generation_config,
                input_size=args.input_size, max_num=args.max_num,
                use_thumbnail=not args.no_thumbnail,
                device=("cuda" if torch.cuda.is_available() else "cpu")
            )
            out_path.write_text(target_md, encoding="utf-8")
            print(f"[OK] {img_path.name} -> {out_path.name}")

        except Exception as e:
            err_path = out_dir / (img_path.stem + ".error.txt")
            err_path.write_text(f"Translation failed for {img_path.name}:\n{e}\n", encoding="utf-8")
            print(f"[ERR] {img_path.name} -> Error log saved to {err_path.name}")

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()