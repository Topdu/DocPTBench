import os
import re
import argparse
import math
from pathlib import Path
from typing import Tuple, List

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# ==============================================================================
# 1. LANGUAGE CONFIGURATION
# ==============================================================================

LANG_CONFIG = {
    "en": {
        "name": "English",
        "tag": "english_md",
        "math_example": "Mean",
        "ext": ".en.md"
    },
    "de": {
        "name": "German",
        "tag": "german_md",
        "math_example": "Mittelwert",
        "ext": ".de.md"
    },
    "fr": {
        "name": "French",
        "tag": "french_md",
        "math_example": "Moyenne",
        "ext": ".fr.md"
    },
    "ru": {
        "name": "Russian",
        "tag": "russian_md",
        "math_example": "Среднее",
        "ext": ".ru.md"
    }
}

# ==============================================================================
# 2. PROMPT TEMPLATES
# ==============================================================================

SYSTEM_PROMPT_TEMPLATE = """You are a professional document translator.

Task: First OCR and reconstruct the readable Chinese Markdown from the whole document image.
Then produce a high-fidelity __LANG_NAME__ Markdown translation.

Output format (STRICT):
Return EXACTLY two fenced blocks, in this order, and nothing else:
<chinese_md>
...the Chinese Markdown reconstruction...
</chinese_md>
<__LANG_TAG__>
...the __LANG_NAME__ Markdown translation...
</__LANG_TAG__>

Formulas (LaTeX ONLY):
- Detect ANY formula/equation and represent it in LaTeX.
- Wrap EVERY formula in display math delimiters: $$ ... $$ (use display math even if the source looked inline).
- Inside $$...$$, DO NOT alter symbols/operators/variables/numbers/units.
- In <chinese_md>, keep any human words in formulas as Chinese (e.g., \\text{均值}).
- In <__LANG_TAG__>, translate ONLY human words inside formulas with \\text{...} (e.g., \\text{__MATH_EXAMPLE__}); keep math intact.

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
- Template variables/placeholders: {var}, {{handlebars}}, <PLACEHOLDER>, %s, {0}, $ENV_VAR.

__LANG_NAME__ style:
- Fluent, professional __LANG_NAME__ for narrative text; use standard __LANG_NAME__ punctuation in prose.
- NEVER change punctuation inside code, LaTeX math, or URLs.

Quality checks BEFORE finalizing:
- <chinese_md> is Chinese-only, with all formulas as $$...$$ LaTeX.
- <__LANG_TAG__> translates narrative text; formulas use the same $$...$$ LaTeX with only \\text{...} translated.
- ANY formula/equation is represented in LaTeX.
- No added/omitted content; structure renders correctly.
- Output ONLY the two blocks. No explanations or reasoning."""

USER_INSTRUCTION_TEMPLATE = (
    "Reconstruct and translate the document image as specified. "
    "Return ONLY the two blocks: <chinese_md>...</chinese_md> and <__LANG_TAG__>...</__LANG_TAG__>."
)

# ==============================================================================
# 3. INTERNVL IMAGE PREPROCESSING (Standard Implementation)
# ==============================================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

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

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12, use_thumbnail=True):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=use_thumbnail, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# ==============================================================================
# 4. PARSING & HELPERS
# ==============================================================================

def parse_dual_blocks(text: str, lang_code: str) -> Tuple[str, str]:
    """
    Parses <chinese_md> and <target_md> based on language config.
    """
    tag = LANG_CONFIG[lang_code]["tag"]
    
    target_re = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    zh_re = re.compile(r"<chinese_md>\s*(.*?)\s*</chinese_md>", re.S | re.I)

    target_md, zh_md = "", ""

    # 1. Try strict match for target
    m1 = target_re.search(text)
    if m1:
        target_md = m1.group(1).strip()
    else:
        # 2. Fallback: content after opening tag
        m2 = re.search(rf"<{tag}>\s*(.*)", text, re.S | re.I)
        if m2:
            target_md = m2.group(1).strip()
        else:
            # 3. Fallback: content after closing chinese tag
            m3 = re.search(r"</chinese_md>\s*(.*)", text, re.S | re.I)
            if m3:
                target_md = m3.group(1).strip()

    # Extract Chinese
    m_zh = zh_re.search(text)
    if m_zh:
        zh_md = m_zh.group(1).strip()

    # Ultimate fallback
    if not target_md and not zh_md:
        zh_md = text.strip()

    return target_md, zh_md

def force_h2_headings(md: str) -> str:
    """Normalizes headings to H2 (##) for consistency, ignoring code blocks."""
    lines = md.splitlines()
    out, in_code = [], False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            in_code = not in_code
            out.append(line); i += 1; continue

        if not in_code:
            # ATX (# Header)
            m = re.match(r'^(\s*)#{1,6}\s+(.*)$', line)
            if m:
                out.append(f"{m.group(1)}## {m.group(2)}"); i += 1; continue
            # Setext (Underline)
            if i + 1 < len(lines):
                m2 = re.match(r'^\s*(=+|-+)\s*$', lines[i+1])
                if m2 and lines[i].strip():
                    out.append(f"## {lines[i].strip()}"); i += 2; continue
        out.append(line); i += 1
    return "\n".join(out)

# ==============================================================================
# 5. INFERENCE
# ==============================================================================

@torch.inference_mode()
def translate_one(model, tokenizer, image_path: Path, lang_code: str, 
                  generation_config: dict, args):
    
    # 1. Prepare Prompt
    cfg = LANG_CONFIG[lang_code]
    sys_prompt = SYSTEM_PROMPT_TEMPLATE \
        .replace("__LANG_NAME__", cfg["name"]) \
        .replace("__LANG_TAG__", cfg["tag"]) \
        .replace("__MATH_EXAMPLE__", cfg["math_example"])
    
    user_instr = USER_INSTRUCTION_TEMPLATE.replace("__LANG_TAG__", cfg["tag"])
    
    # InternVL format: <image>\n{text}
    question = f"{sys_prompt}\n\n<image>\n{user_instr}"

    # 2. Load Image
    pixel_values = load_image(
        image_path, 
        input_size=args.input_size, 
        max_num=args.max_num,
        use_thumbnail=(not args.no_thumbnail)
    ).to(torch.bfloat16).cuda()

    # 3. Generate
    # Note: InternVL chat() handles the history internally if provided, here we pass None
    full_text = model.chat(tokenizer, pixel_values, question, generation_config).strip()

    # 4. Parse
    target_md, zh_md = parse_dual_blocks(full_text, lang_code)
    
    # 5. Post-process
    if target_md: target_md = force_h2_headings(target_md)
    if zh_md: zh_md = force_h2_headings(zh_md)

    return target_md, zh_md, full_text

# ==============================================================================
# 6. MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="InternVL Document Translation")
    
    # Core
    parser.add_argument("--dir", required=True, help="Input directory")
    parser.add_argument("--model", default="OpenGVLab/InternVL3_5-2B", help="HF Model ID or local path")
    parser.add_argument("--lang", default="en", choices=["en", "de", "fr", "ru"], help="Target language")
    parser.add_argument("--out_dir", default=None, help="Output directory")
    parser.add_argument("--overwrite", action="store_true")
    
    # InternVL Params
    parser.add_argument("--input_size", type=int, default=448)
    parser.add_argument("--max_num", type=int, default=12, help="Max tiles")
    parser.add_argument("--no_thumbnail", action="store_true")
    
    # Generation Params
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    
    # Output Control
    parser.add_argument("--save_chinese", action="store_true", help="Save Chinese OCR")
    parser.add_argument("--save_full", action="store_true", help="Save raw response")
    
    args = parser.parse_args()

    # Setup Paths
    in_dir = Path(args.dir)
    lang_cfg = LANG_CONFIG[args.lang]
    
    default_out = f"translations_internvl_{args.lang}"
    out_dir = Path(args.out_dir) if args.out_dir else Path(default_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    zh_dir = out_dir / "zh" if args.save_chinese else None
    if zh_dir: zh_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = out_dir / "raw" if args.save_full else None
    if raw_dir: raw_dir.mkdir(parents=True, exist_ok=True)

    # Find Images
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    images = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])
    if not images:
        raise ValueError(f"No images found in {in_dir}")

    # Load Model
    print(f"[INFO] Loading model: {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()

    gen_config = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=(args.temperature > 0),
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print(f"[INFO] Starting translation: Chinese -> {lang_cfg['name']}")
    print(f"[INFO] Output: {out_dir.resolve()}")

    for img in images:
        out_path = out_dir / (img.stem + lang_cfg["ext"])
        
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {img.name}")
            continue

        try:
            target_md, zh_md, full_text = translate_one(
                model, tokenizer, img, args.lang, gen_config, args
            )

            # Save Target
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(target_md)

            # Save Chinese
            if zh_dir and zh_md:
                with open(zh_dir / (img.stem + ".md"), "w", encoding="utf-8") as f:
                    f.write(zh_md)

            # Save Raw
            if raw_dir:
                with open(raw_dir / (img.stem + ".full.txt"), "w", encoding="utf-8") as f:
                    f.write(full_text)

            print(f"[OK] {img.name} -> {out_path.name}")

        except Exception as e:
            print(f"[ERR] {img.name}: {e}")
            with open(out_dir / (img.stem + ".error.txt"), "w") as f:
                f.write(str(e))
        
        # Cleanup VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()