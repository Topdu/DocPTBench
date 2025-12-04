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

# ==============================================================================
# 1. MULTI-LANGUAGE CONFIGURATION
# ==============================================================================

LANG_CONFIG = {
    "en": {
        "name": "English",
        "tag": "english_md",
        "ext": ".en.md",
        "style_note": "Use fluent, professional English with standard punctuation."
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

# ==============================================================================
# 2. DYNAMIC PROMPT TEMPLATES
# ==============================================================================

SYSTEM_PROMPT_TEMPLATE = """You are a professional document translator.

Task: Translate the content of the document image directly into high-fidelity {lang_name} Markdown.

Output format (STRICT):
Return EXACTLY one fenced block, and nothing else:
<{lang_tag}>
...the {lang_name} Markdown translation...
</{lang_tag}>

Translation Rules:
- Formulas (LaTeX ONLY): Detect and represent any formula/equation in LaTeX, wrapped in display math delimiters: $$ ... $$. Inside the delimiters, translate only human words (Chinese terms) into {lang_name} using \\text{{...}} (e.g., \\text{{mean}} → \\text{{translated_term}}).
- Tables (Markdown): Preserve Markdown tables. Translate only narrative text in cells.
- Structure: Preserve all structural elements like paragraphs, lists, blockquotes, etc.
- Do-not-translate: Keep code (`...`), URLs, emails, and stable names/brands as is.
- Target Style: {style_note}
- Quality Check: Ensure the output is a single, complete, and accurate {lang_name} Markdown block. Do not add any explanations or extra content.
"""

USER_INSTRUCTION_TEMPLATE = (
    "Translate the document image directly into {lang_name} Markdown. "
    "Return ONLY the <{lang_tag}>...</{lang_tag}> block as specified."
)

# ==============================================================================
# 3. PREPROCESSING (InternVL Standard Process)
# ==============================================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
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
    processed_images = [
        resized_img.crop((
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )) for i in range(blocks)
    ]
    if use_thumbnail and len(processed_images) != 1:
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

# ==============================================================================
# 4. PARSING AND POST-PROCESSING
# ==============================================================================

def parse_target_block(text: str, lang_code: str) -> str:
    """Dynamically parse the XML block based on language configuration."""
    tag = LANG_CONFIG[lang_code]["tag"]
    # Dynamically build regex
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    
    m = pattern.search(text)
    if m:
        return m.group(1).strip()
    
    # Fallback: If tag not found but text is not empty, return full text
    return text.strip()

def force_h2_headings(md: str) -> str:
    """Unify all Markdown headings to '## ' (optional optimization)."""
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

# ==============================================================================
# 5. PROMPT CONSTRUCTION
# ==============================================================================

def build_question(lang_code: str):
    cfg = LANG_CONFIG[lang_code]
    
    system_part = SYSTEM_PROMPT_TEMPLATE.format(
        lang_name=cfg["name"],
        lang_tag=cfg["tag"],
        style_note=cfg["style_note"]
    )
    
    user_part = USER_INSTRUCTION_TEMPLATE.format(
        lang_name=cfg["name"],
        lang_tag=cfg["tag"]
    )
    
    # InternVL Format: System Prompt + <image> + User Instruction
    return (
        system_part.strip()
        + "\n\n<image>\n"
        + user_part
    )

# ==============================================================================
# 6. INFERENCE LOGIC
# ==============================================================================

@torch.inference_mode()
def translate_one(model, tokenizer, image_path: str, lang_code: str, generation_config: dict,
                  input_size=448, max_num=12, use_thumbnail=True, device="cuda") -> str:
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    
    # Load image
    pixel_values = load_image(
        image_path, input_size=input_size, max_num=max_num,
        use_thumbnail=use_thumbnail, device=device, dtype=dtype
    )
    
    # Build Prompt
    question = build_question(lang_code)
    
    # Invoke model
    full_text = model.chat(tokenizer, pixel_values, question, generation_config).strip()
    
    # Parse result
    target_md = parse_target_block(full_text, lang_code)
    
    if target_md:
        target_md = force_h2_headings(target_md)
        
    return target_md

# ==============================================================================
# 7. MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="InternVL3 Multi-language Document Image Direct Translation")
    
    # Core Arguments
    parser.add_argument("--model", default="OpenGVLab/InternVL3_5-2B", help="HuggingFace model path or ID")
    parser.add_argument("--dir", required=True, help="Input image directory")
    parser.add_argument("--lang", default="en", choices=["en", "de", "fr", "ru"], help="Target language")
    parser.add_argument("--out_dir", default=None, help="Output directory (default: auto-generated based on language)")
    parser.add_argument("--ext", default=None, help="Output extension")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwrite")
    
    # Model Generation Arguments
    parser.add_argument("--input_size", type=int, default=448, help="Tile size")
    parser.add_argument("--max_num", type=int, default=12, help="Max tiles number")
    parser.add_argument("--no_thumbnail", action="store_true", help="Do not append thumbnail tile")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--flash_attn", action="store_true", help="Enable Flash Attention 2")
    
    args = parser.parse_args()

    # Initialize Configuration
    lang_cfg = LANG_CONFIG[args.lang]
    
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(f"translations_internvl_direct_{args.lang}")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = args.ext if args.ext else lang_cfg["ext"]

    # Collect Images
    in_dir = Path(args.dir)
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    image_paths = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])
    if not image_paths:
        raise ValueError(f"No supported image files found in directory '{in_dir}'.")

    # Load Model
    print(f"[INFO] Loading Model: {args.model}...")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_flash_attention_2=args.flash_attn,
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

    print(f"[INFO] Target Language: {lang_cfg['name']} ({args.lang})")
    print(f"[INFO] Output Dir: {out_dir.resolve()}")
    print(f"[INFO] Found {len(image_paths)} images.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for img_path in image_paths:
        out_path = out_dir / (img_path.stem + ext)
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] File exists: {out_path.name}")
            continue

        try:
            target_md = translate_one(
                model, tokenizer, str(img_path), args.lang, generation_config,
                input_size=args.input_size, max_num=args.max_num,
                use_thumbnail=not args.no_thumbnail,
                device=device
            )

            out_path.write_text(target_md, encoding="utf-8")
            print(f"[OK] {img_path.name} → {out_path.name}")

        except Exception as e:
            err_path = out_dir / (img_path.stem + ".error.txt")
            err_path.write_text(f"Failed to process '{img_path.name}':\n{e}\n", encoding="utf-8")
            print(f"[ERR] {img_path.name} → See details in {err_path.name}")
        
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()