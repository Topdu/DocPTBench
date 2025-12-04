#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import base64
import mimetypes
from pathlib import Path

from openai import OpenAI

try:
    from PIL import Image
    from io import BytesIO
except ImportError:
    Image = None

# ========== 1. Multi-language Configuration (Target Language) ==========
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

# ========== 2. Dynamic Prompt Template (Source: English) ==========
SYSTEM_PROMPT_TEMPLATE = """You are a professional document translator.

Task: Translate the content of the English document image directly into high-fidelity {lang_name} Markdown.

Output format (STRICT):
Return EXACTLY one fenced block, and nothing else:
<{lang_tag}>
...the {lang_name} Markdown translation...
</{lang_tag}>

Translation Rules:
- Formulas (LaTeX ONLY): Detect and represent any formula/equation in LaTeX, wrapped in display math delimiters: $$ ... $$. Inside the delimiters, translate only human words (English terms) into {lang_name} using \\text{{...}} (e.g., \\text{{mean}} → \\text{{translated_term}}).
- Tables (Markdown): Preserve Markdown tables. Translate only narrative text in cells.
- Structure: Preserve all structural elements like paragraphs, lists, blockquotes, etc.
- Do-not-translate: Keep code (`...`), URLs, emails, and stable names/brands as is.
- Target Style: {style_note}
- Quality Check: Ensure the output is a single, complete, and accurate {lang_name} Markdown block. Do not add any explanations or extra content.
"""

USER_INSTRUCTION_TEMPLATE = (
    "Translate the English document image directly into {lang_name} Markdown. "
    "Return ONLY the <{lang_tag}>...</{lang_tag}> block as specified."
)

# ===== Image Processing =====
def guess_mime(path: Path) -> str:
    """Guess MIME type based on file extension."""
    m, _ = mimetypes.guess_type(str(path))
    if not m:
        ext = path.suffix.lower()
        if ext in {".jpg", ".jpeg"}: m = "image/jpeg"
        elif ext == ".png": m = "image/png"
        elif ext == ".webp": m = "image/webp"
        elif ext == ".bmp": m = "image/bmp"
        elif ext == ".gif": m = "image/gif"
        else: m = "application/octet-stream"
    return m

def image_to_data_url(image_path: Path, max_long_edge: int = None, jpeg_quality: int = 92) -> str:
    """
    Read image and convert to Data URL.
    If Pillow is installed and max_long_edge is set, resize to save tokens.
    """
    mime = guess_mime(image_path)
    if max_long_edge and Image:
        try:
            with Image.open(image_path) as im:
                w, h = im.size
                m = max(w, h)
                if m > max_long_edge:
                    scale = max_long_edge / float(m)
                    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                    im = im.convert("RGB").resize(new_size, Image.LANCZOS)
                    buf = BytesIO()
                    im.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
                    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                    return f"data:image/jpeg;base64,{b64}"
        except Exception as e:
            print(f"[WARN] Failed to resize image {image_path.name}: {e}. Using original.")

    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"

# ===== Build Messages =====
def build_messages_for_single_image(data_url: str, lang_code: str):
    """Build prompt based on target language."""
    cfg = LANG_CONFIG[lang_code]
    
    system_content = SYSTEM_PROMPT_TEMPLATE.format(
        lang_name=cfg["name"],
        lang_tag=cfg["tag"],
        style_note=cfg["style_note"]
    )
    
    user_content = USER_INSTRUCTION_TEMPLATE.format(
        lang_name=cfg["name"],
        lang_tag=cfg["tag"]
    )

    return [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                        "detail": "high"
                    }
                },
                {"type": "text", "text": user_content},
            ],
        },
    ]

# ===== Parsing Logic =====
def parse_target_block(text: str, lang_code: str) -> str:
    """Dynamically parse XML block based on language config."""
    tag = LANG_CONFIG[lang_code]["tag"]
    # Dynamically build regex to extract content between tags
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    
    m = pattern.search(text)
    if m:
        return m.group(1).strip()
    
    # Fallback: if no tag found, return full text
    return text.strip()

# ===== API Call =====
def call_vision_api(client: OpenAI, model: str, messages,
                    max_tokens: int, temperature: float, top_p: float) -> str:
    """Call OpenAI-compatible interface."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    choices = getattr(resp, "choices", None)
    if choices and choices[0].message:
        content = choices[0].message.content
        if isinstance(content, str): return content
        if isinstance(content, list):
            # Handle rare multi-segment content responses
            return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    raise RuntimeError(f"Vision API request failed. Response: {resp}")

# ===== Main Process =====
def main():
    parser = argparse.ArgumentParser(description="Doubao Vision English Document Multi-language Direct Translation (EN -> ZH/DE/FR/RU)")
    
    # Core Arguments
    parser.add_argument("--dir", required=True, help="Input English image folder")
    parser.add_argument("--lang", default="zh", choices=["zh", "de", "fr", "ru"], help="Target language (default: zh)")
    parser.add_argument("--out_dir", default=None, help="Output directory (default: auto-generated)")
    parser.add_argument("--ext", default=None, help="Output extension (default: auto-selected)")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing files")

    # Model Parameters
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--model", default="doubao-seed-1-6-vision-250815", help="Model name")
    parser.add_argument("--api_key", type=str, default=None, help="Volcano Engine Ark API Key")
    parser.add_argument("--base_url", type=str, default="https://ark.cn-beijing.volces.com/api/v3", help="API endpoint")

    # Image Processing
    parser.add_argument("--max_long_edge", type=int, default=None, help="Max long edge limit (save tokens)")
    parser.add_argument("--jpeg_quality", type=int, default=92)

    args = parser.parse_args()

    # 1. Config Initialization
    lang_cfg = LANG_CONFIG[args.lang]
    
    # Auto-generate output directory name, e.g., translations_doubao_direct_en2zh
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(f"translations_doubao_direct_en2{args.lang}")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine extension
    ext = args.ext if args.ext else lang_cfg["ext"]

    # 2. Collect Images
    in_dir = Path(args.dir)
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
    
    if not in_dir.exists():
        raise ValueError(f"Input directory does not exist: {in_dir}")

    image_paths = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])
    if not image_paths:
        raise ValueError(f"No supported images found in directory '{in_dir}'.")

    # 3. API Client
    api_key = args.api_key or os.getenv("ARK_API_KEY")
    if not api_key:
        raise RuntimeError("Please set ARK_API_KEY environment variable or use --api_key parameter.")
    
    client = OpenAI(base_url=args.base_url, api_key=api_key)

    print(f"[INFO] Source Language: English")
    print(f"[INFO] Target Language: {lang_cfg['name']} ({args.lang})")
    print(f"[INFO] Output Dir: {out_dir.resolve()}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Found {len(image_paths)} images.")

    # 4. Processing Loop
    for img in image_paths:
        out_path = out_dir / (img.stem + ext)
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {out_path.name} exists.")
            continue

        # Image to Base64
        data_url = image_to_data_url(img, max_long_edge=args.max_long_edge, jpeg_quality=args.jpeg_quality)

        # Build Messages
        messages = build_messages_for_single_image(data_url, args.lang)

        try:
            print(f"[INFO] Processing {img.name} -> {lang_cfg['name']}...")
            full_text = call_vision_api(client, args.model, messages, args.max_tokens,
                                        args.temperature, args.top_p)
        except Exception as e:
            print(f"[ERR] Failed processing '{img.name}': {e}")
            (out_dir / (img.stem + ".error.txt")).write_text(str(e), encoding="utf-8")
            continue

        # Parse target language block
        target_md = parse_target_block(full_text, args.lang)

        # Write result
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(target_md)

        print(f"[OK] {img.name} -> {out_path.name}")

if __name__ == "__main__":
    if Image is None:
        print("[WARN] Pillow library not found. Image resizing (--max_long_edge) disabled.")
    main()