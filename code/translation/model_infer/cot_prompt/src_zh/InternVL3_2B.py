import os
import argparse
import re
import math
import torch
import torchvision.transforms as T
from pathlib import Path
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
# 2. IMAGE PREPROCESSING (InternVL Standard)
# ==============================================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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

# ==============================================================================
# 3. PARSING & HELPERS
# ==============================================================================

def parse_dual_blocks(text: str, lang_code: str):
    """
    Parses <chinese_md> and <target_lang_md>.
    """
    tag = LANG_CONFIG[lang_code]["tag"]
    
    target_re = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    zh_re = re.compile(r"<chinese_md>\s*(.*?)\s*</chinese_md>", re.S | re.I)

    target_md, zh_md = "", ""

    # 1. Match Target Block
    m1 = target_re.search(text)
    if m1:
        target_md = m1.group(1).strip()
    else:
        # Fallback: content after opening tag
        m2 = re.search(rf"<{tag}>\s*(.*)", text, re.S | re.I)
        if m2:
            target_md = m2.group(1).strip()
        else:
            # Fallback: content after closing chinese tag
            m3 = re.search(r"</chinese_md>\s*(.*)", text, re.S | re.I)
            if m3:
                target_md = m3.group(1).strip()

    # 2. Match Chinese Block
    m_zh = zh_re.search(text)
    if m_zh:
        zh_md = m_zh.group(1).strip()

    # 3. Fallback: Everything is target if nothing matched
    if not target_md and not zh_md:
        zh_md = text.strip()

    return target_md, zh_md

def force_h2_headings(md: str) -> str:
    """Normalizes headings to H2 (##) unless inside code blocks."""
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

def build_prompt(lang_code: str) -> str:
    cfg = LANG_CONFIG[lang_code]
    sys_p = SYSTEM_PROMPT_TEMPLATE \
        .replace("__LANG_NAME__", cfg["name"]) \
        .replace("__LANG_TAG__", cfg["tag"]) \
        .replace("__MATH_EXAMPLE__", cfg["math_example"])
    
    user_p = USER_INSTRUCTION_TEMPLATE.replace("__LANG_TAG__", cfg["tag"])
    
    # InternVL format: <image>\nPrompt
    return f"{sys_p}\n\n<image>\n{user_p}"

# ==============================================================================
# 4. INFERENCE
# ==============================================================================

@torch.inference_mode()
def translate_one(model, tokenizer, image_path: str, generation_config: dict,
                  lang_code: str, args):
    
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load & Preprocess Image
    pixel_values = load_image(
        image_path, 
        input_size=args.input_size, 
        max_num=args.max_num,
        use_thumbnail=not args.no_thumbnail, 
        device=device, 
        dtype=dtype
    )

    # 2. Build Prompt
    question = build_prompt(lang_code)

    # 3. Inference
    # Note: InternVL usually handles the chat template internally if passed as string with <image>
    full_text = model.chat(tokenizer, pixel_values, question, generation_config).strip()

    # 4. Parse
    target_md, zh_md = parse_dual_blocks(full_text, lang_code)

    # 5. Post-process (Heading normalization)
    if target_md: target_md = force_h2_headings(target_md)
    if zh_md: zh_md = force_h2_headings(zh_md)

    return target_md, zh_md, full_text

# ==============================================================================
# 5. MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="InternVL3 Document Translation (Local)")
    
    # Core
    parser.add_argument("--dir", required=True, help="Input image directory")
    parser.add_argument("--model", default="OpenGVLab/InternVL3-2B", help="HuggingFace Model ID or Local Path")
    parser.add_argument("--lang", default="en", choices=["en", "de", "fr", "ru"], help="Target Language")
    parser.add_argument("--out_dir", default=None, help="Output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    # Model Params
    parser.add_argument("--input_size", type=int, default=448, help="Tile size")
    parser.add_argument("--max_num", type=int, default=12, help="Max tiles")
    parser.add_argument("--no_thumbnail", action="store_true", help="Disable thumbnail tile")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--flash_attn", action="store_true", help="Enable Flash Attention 2")

    # Saving
    parser.add_argument("--save_source", action="store_true", help="Save reconstructed Source (Chinese) Markdown")
    parser.add_argument("--save_full", action="store_true", help="Save raw model output")

    args = parser.parse_args()

    # Setup Paths
    in_dir = Path(args.dir)
    lang_cfg = LANG_CONFIG[args.lang]
    
    default_out_name = f"translations_internvl_{args.lang}"
    out_dir = Path(args.out_dir) if args.out_dir else Path(default_out_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_dir = out_dir / "source" if args.save_source else None
    if source_dir: source_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = out_dir / "raw" if args.save_full else None
    if raw_dir: raw_dir.mkdir(parents=True, exist_ok=True)

    # Find Images
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    image_paths = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])
    if not image_paths:
        print(f"No images found in {in_dir}")
        return

    # Load Model
    print(f"[INFO] Loading model: {args.model}...")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    
    try:
        model = AutoModel.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            load_in_8bit=False, # Can be exposed as arg if needed
            low_cpu_mem_usage=True,
            use_flash_attn=args.flash_attn,
            trust_remote_code=True,
            device_map="auto",
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    except Exception as e:
        print(f"[ERR] Failed to load model: {e}")
        return

    generation_config = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=(args.temperature > 0),
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=1.05,
    )

    print(f"[INFO] Processing {len(image_paths)} images. Target: {lang_cfg['name']}")

    # Process Loop
    for img in image_paths:
        out_path = out_dir / (img.stem + lang_cfg["ext"])
        
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {img.name}")
            continue

        try:
            target_md, source_md, full_text = translate_one(
                model, tokenizer, str(img), generation_config, args.lang, args
            )

            # Save Target
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(target_md)

            # Save Source (Optional)
            if source_dir and source_md:
                src_path = source_dir / (img.stem + ".zh.md")
                with open(src_path, "w", encoding="utf-8") as f:
                    f.write(source_md)

            # Save Raw (Optional)
            if raw_dir:
                raw_path = raw_dir / (img.stem + ".full.txt")
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(full_text)

            print(f"[OK] {img.name}")

        except Exception as e:
            print(f"[FAIL] {img.name}: {e}")
        
        # Cleanup VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()