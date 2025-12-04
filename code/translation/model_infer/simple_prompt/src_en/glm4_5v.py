#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import base64
import mimetypes
from pathlib import Path
from typing import List, Dict, Any

from zai import ZhipuAiClient

# Optional: Downsampling support
try:
    from PIL import Image
    from io import BytesIO
except ImportError:
    Image = None

# ========== 1. Multi-language Configuration ==========
LANG_CONFIG = {
    "zh": {
        "name": "Simplified Chinese",
        "tag": "chinese_md",
        "ext": ".zh.md",
        "style_note": "Use fluent, professional Simplified Chinese (简体中文) with standard punctuation."
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

# ========== 2. Dynamic Prompt Template ==========
SYSTEM_PROMPT_TEMPLATE = """You are a professional document translator.

Task: Translate the content of the English document image directly into high-fidelity {lang_name} Markdown.

Output format (STRICT):
Return EXACTLY one fenced block, and nothing else:
<{lang_tag}>
...the {lang_name} Markdown translation...
</{lang_tag}>

Translation Rules:
- Formulas (LaTeX ONLY): Detect and represent any formula/equation in LaTeX, wrapped in display math delimiters: $$ ... $$. Inside the delimiters, translate only human words (English terms) into {lang_name} using \\text{{...}} (e.g., \\text{{Mean}} → \\text{{translated_term}}).
- Tables (Markdown): Preserve Markdown tables. Translate only narrative text in cells.
- Structure: Preserve all structural elements like paragraphs, lists, blockquotes, etc.
- Do-not-translate: Keep code (`...`), URLs, emails, and stable English names (e.g., "GitHub") as is.
- Target Style: {style_note}
- Quality Check: Ensure the output is a single, complete, and accurate {lang_name} Markdown block. Do not add any explanations or extra content.
"""

USER_INSTRUCTION_TEMPLATE = (
    "Translate the document image directly into {lang_name} Markdown. "
    "Return ONLY the <{lang_tag}>...</{lang_tag}> block as specified."
)

# ===== Image Processing =====
def guess_mime(path: Path) -> str:
    m, _ = mimetypes.guess_type(str(path))
    if not m:
        ext = path.suffix.lower()
        if ext in {".jpg", ".jpeg"}: m = "image/jpeg"
        elif ext == ".png": m = "image/png"
        else: m = "application/octet-stream"
    return m

def image_to_base64(image_path: Path, max_long_edge: int = None, jpeg_quality: int = 92) -> str:
    """Read image and convert to Base64 string (without data: prefix, adapted for zai SDK)."""
    if max_long_edge and Image:
        try:
            with Image.open(image_path) as im:
                w, h = im.size
                if max(w, h) > max_long_edge:
                    scale = max_long_edge / float(max(w, h))
                    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                    im = im.convert("RGB").resize(new_size, Image.LANCZOS)
                    buf = BytesIO()
                    im.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
                    return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as e:
            print(f"[WARN] Failed to resize image {image_path.name}: {e}. Using original.")
    return base64.b64encode(image_path.read_bytes()).decode("ascii")

# ===== Build Messages =====
def build_messages_for_single_image(b64_string: str, lang_code: str):
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
                {"type": "image_url", "image_url": {"url": b64_string}},
                {"type": "text", "text": user_content},
            ],
        },
    ]

# ===== Parsing Logic =====
def parse_target_block(text: str, lang_code: str) -> str:
    """Dynamically parse XML block based on language configuration."""
    tag = LANG_CONFIG[lang_code]["tag"]
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    
    m = pattern.search(text)
    return m.group(1).strip() if m else text.strip()

# ===== API Call =====
def call_glm45v(client: ZhipuAiClient, model: str, messages,
                max_tokens: int, temperature: float, top_p: float, thinking: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        thinking={"type": thinking} if thinking else None,
    )
    if hasattr(resp, "choices") and resp.choices:
        content = resp.choices[0].message.content
        return content if isinstance(content, str) else ""
    raise RuntimeError(f"GLM-4.5V request failed. Response: {resp}")

# ===== Main Process =====
def main():
    parser = argparse.ArgumentParser(description="Direct document translation (EN -> ZH/DE/FR/RU) with GLM-4.5V")
    
    # Core Arguments
    parser.add_argument("--dir", required=True, help="Input English image folder")
    parser.add_argument("--lang", default="zh", choices=["zh", "de", "fr", "ru"], help="Target language (default: zh)")
    parser.add_argument("--out_dir", default=None, help="Output directory (default: auto-generated based on language)")
    parser.add_argument("--ext", default=None, help="Output file extension")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files")

    # Generation parameters
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--model", default="glm-4.5v", help="Model name")
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="enabled",
                        help="Enable thinking capability (GLM-4.5V feature)")

    # Image processing
    parser.add_argument("--max_long_edge", type=int, default=2000, help="Downsample by longest edge before encoding, default 2000")
    parser.add_argument("--jpeg_quality", type=int, default=92, help="JPEG quality")

    # API credentials
    parser.add_argument("--api_key", type=str, default=None, help="API Key (or set ZHIPUAI_API_KEY env var)")

    args = parser.parse_args()

    # 1. Configuration Initialization
    lang_cfg = LANG_CONFIG[args.lang]
    
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(f"translations_glm45v_direct_en2{args.lang}")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = args.ext if args.ext else lang_cfg["ext"]

    # 2. Image Collection
    in_dir = Path(args.dir)
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
    image_paths = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])
    if not image_paths:
        raise ValueError(f"No supported image files found in directory '{in_dir}'.")

    # 3. Client Initialization
    api_key = args.api_key or os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set environment variable ZHIPUAI_API_KEY or specify via --api_key.")
    client = ZhipuAiClient(api_key=api_key)

    print(f"[INFO] Source Language: English")
    print(f"[INFO] Target Language: {lang_cfg['name']} ({args.lang})")
    print(f"[INFO] Output Dir: {out_dir.resolve()}")
    print(f"[INFO] Model: {args.model}, Thinking: {args.thinking}")
    print(f"[INFO] Found {len(image_paths)} images.")

    # 4. Processing Loop
    for img_path in image_paths:
        out_path = out_dir / (img_path.stem + ext)
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] File exists: {out_path.name}")
            continue

        b64_string = image_to_base64(img_path, max_long_edge=args.max_long_edge, jpeg_quality=args.jpeg_quality)
        
        # Build Prompt with language code
        messages = build_messages_for_single_image(b64_string, args.lang)

        try:
            print(f"[INFO] Processing {img_path.name} -> {lang_cfg['name']}...")
            full_text = call_glm45v(
                client=client, model=args.model, messages=messages,
                max_tokens=args.max_tokens, temperature=args.temperature,
                top_p=args.top_p, thinking=args.thinking,
            )
        except Exception as e:
            err_path = out_dir / (img_path.stem + ".error.txt")
            err_path.write_text(f"Failed to process '{img_path.name}':\n{e}\n", encoding="utf-8")
            print(f"[ERR] {img_path.name} -> Details in {err_path.name}")
            continue

        # Parse target language block
        target_md = parse_target_block(full_text, args.lang)

        # Write result
        out_path.write_text(target_md, encoding="utf-8")

        print(f"[OK] {img_path.name} -> {out_path.name}")

if __name__ == "__main__":
    main()