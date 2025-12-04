#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import base64
import mimetypes
from pathlib import Path
from typing import Tuple, List, Dict, Any

from zai import ZhipuAiClient

# Optional: Downsampling support
try:
    from PIL import Image
    from io import BytesIO
except Exception:
    Image = None

# ==============================================================================
# 1. Multi-language Configuration
# ==============================================================================

LANG_CONFIG = {
    "zh": {
        "name": "Simplified Chinese (zh-CN)",
        "tag": "chinese_md",
        "ext": ".zh.md",
        "style_note": "Fluent, professional zh-CN; use standard Chinese punctuation."
    },
    "de": {
        "name": "German",
        "tag": "german_md",
        "ext": ".de.md",
        "style_note": "Fluent, professional German; use standard German punctuation."
    },
    "fr": {
        "name": "French",
        "tag": "french_md",
        "ext": ".fr.md",
        "style_note": "Fluent, professional French; use standard French punctuation."
    },
    "ru": {
        "name": "Russian",
        "tag": "russian_md",
        "ext": ".ru.md",
        "style_note": "Fluent, professional Russian; use standard Russian punctuation."
    }
}

# ==============================================================================
# 2. Dynamic Prompt Templates
# Note: In Python format strings, literal { } must be escaped as {{ }}
# ==============================================================================

SYSTEM_PROMPT_TEMPLATE = """You are a professional document translator.

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

USER_INSTRUCTION_TEMPLATE = (
    "Reconstruct and translate the document image as specified. "
    "Return ONLY the two blocks: <english_md>...</english_md> and <{lang_tag}>...</{lang_tag}>."
)

# ==============================================================================
# 3. Helper Functions
# ==============================================================================

def get_prompts(lang_code: str) -> Tuple[str, str]:
    cfg = LANG_CONFIG[lang_code]
    sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        lang_name=cfg["name"],
        lang_tag=cfg["tag"],
        style_note=cfg["style_note"]
    )
    user_prompt = USER_INSTRUCTION_TEMPLATE.format(
        lang_tag=cfg["tag"]
    )
    return sys_prompt, user_prompt

def parse_dual_blocks(text: str, lang_code: str) -> Tuple[str, str]:
    """
    Returns (en_md, target_md)
    """
    tag = LANG_CONFIG[lang_code]["tag"]
    
    # Dynamically generate regex
    target_re = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    en_re = re.compile(r"<english_md>\s*(.*?)\s*</english_md>", re.S | re.I)

    # Extract English OCR
    m_en = en_re.search(text)
    en_md = m_en.group(1).strip() if m_en else ""

    # Extract target language translation
    m_target = target_re.search(text)
    if m_target:
        target_md = m_target.group(1).strip()
    else:
        # Fallback: if closing tag is missing, try matching everything after the tag
        m_target_loose = re.search(rf"<{tag}>\s*(.*)", text, re.S | re.I)
        target_md = m_target_loose.group(1).strip() if m_target_loose else ""

    # Extreme case: if parsing fails completely and text is not empty, assume entire text is English OCR
    if not en_md and not target_md and text.strip():
        en_md = text.strip()

    return en_md, target_md

# ==============================================================================
# 4. Image Processing
# ==============================================================================

def guess_mime(path: Path) -> str:
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

def image_to_base64(image_path: Path, max_long_edge: int = None, jpeg_quality: int = 92) -> str:
    if max_long_edge and Image is not None:
        try:
            im = Image.open(image_path)
            w, h = im.size
            m = max(w, h)
            if m > max_long_edge:
                scale = max_long_edge / float(m)
                new_size = (max(1, int(w*scale)), max(1, int(h*scale)))
                im = im.convert("RGB").resize(new_size, Image.LANCZOS)
                buf = BytesIO()
                im.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
                return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            pass
    data = image_path.read_bytes()
    return base64.b64encode(data).decode("ascii")

def image_to_data_url(image_path: Path, max_long_edge: int = None, jpeg_quality: int = 92) -> str:
    mime = guess_mime(image_path)
    if max_long_edge and Image is not None:
        try:
            im = Image.open(image_path)
            w, h = im.size
            m = max(w, h)
            if m > max_long_edge:
                scale = max_long_edge / float(m)
                new_size = (max(1, int(w*scale)), max(1, int(h*scale)))
                im = im.convert("RGB").resize(new_size, Image.LANCZOS)
                buf = BytesIO()
                im.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                return f"data:image/jpeg;base64,{b64}"
        except Exception:
            pass
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"

# ==============================================================================
# 5. API Call & Message Construction
# ==============================================================================

def build_messages_for_single_image(b64_or_dataurl: str, use_data_url: bool, sys_prompt: str, user_prompt: str):
    return [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": b64_or_dataurl}},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

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
    choices = getattr(resp, "choices", None)
    if choices:
        content = choices[0].message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
    base_resp = getattr(resp, "base_resp", None)
    raise RuntimeError(f"GLM-4.5V request failed. resp={resp}, base_resp={base_resp}")

# ==============================================================================
# 6. Main Process
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Per-image doc translation eval with GLM-4.5V (Multi-Language)")
    
    # Basic Input/Output
    parser.add_argument("--dir", required=True, help="Input image directory")
    parser.add_argument("--out_dir", default=None, help="Output directory (default translations_glm45v_{lang})")
    
    # Language Selection
    parser.add_argument("--lang", default="zh", choices=["zh", "de", "fr", "ru"], 
                        help="Target language: zh (Chinese), de (German), fr (French), ru (Russian)")

    # Overwrite Control
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files")

    # Model Parameters
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--model", default="glm-4.5v", help="Model name")
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="enabled", help="Enable/disable thinking capability")

    # Output Options
    parser.add_argument("--save_english", action="store_true", help="Save English reconstruction block (<english_md>)")
    parser.add_argument("--save_full", action="store_true", help="Save full raw output (.full.txt)")
    parser.add_argument("--raw_dir", type=str, default=None, help="Directory for raw output")
    parser.add_argument("--english_out_dir", type=str, default=None, help="Directory for English reconstruction output")

    # Image Processing
    parser.add_argument("--max_long_edge", type=int, default=None, help="Downsample before encoding")
    parser.add_argument("--jpeg_quality", type=int, default=92, help="JPEG quality")
    parser.add_argument("--use_data_url", action="store_true", help="Use data URL instead of pure base64")

    # API Key
    parser.add_argument("--api_key", type=str, default=None, help="API Key (or env var ZHIPUAI_API_KEY)")

    args = parser.parse_args()

    # 1. Config and Paths
    lang_cfg = LANG_CONFIG[args.lang]
    target_ext = lang_cfg["ext"]
    
    in_dir = Path(args.dir)
    
    # Default main output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(f"translations_glm45v_{args.lang}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # English output directory
    english_out_dir = None
    if args.save_english:
        if args.english_out_dir:
            english_out_dir = Path(args.english_out_dir)
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

    # 2. Collect Images
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
    image_paths = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])
    if not image_paths:
        raise ValueError(f"No images found. Directory: {in_dir}")

    # 3. Initialize Client
    api_key = args.api_key or os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set env var ZHIPUAI_API_KEY or specify via --api_key.")
    client = ZhipuAiClient(api_key=api_key)

    print(f"[INFO] Target Language: {lang_cfg['name']} ({args.lang})")
    print(f"[INFO] Found {len(image_paths)} images. Output: {out_dir.resolve()}")
    print(f"[INFO] Model: {args.model} | Thinking: {args.thinking}")

    # 4. Processing Loop
    for img in image_paths:
        out_path = out_dir / (img.stem + target_ext)
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {out_path.name} exists.")
            continue

        # Image encoding
        if args.use_data_url:
            b64_or_dataurl = image_to_data_url(img, max_long_edge=args.max_long_edge, jpeg_quality=args.jpeg_quality)
        else:
            b64_or_dataurl = image_to_base64(img, max_long_edge=args.max_long_edge, jpeg_quality=args.jpeg_quality)

        # Get Prompt and build messages
        sys_prompt, user_prompt = get_prompts(args.lang)
        messages = build_messages_for_single_image(b64_or_dataurl, args.use_data_url, sys_prompt, user_prompt)

        try:
            full_text = call_glm45v(
                client=client,
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                thinking=args.thinking,
            )
        except Exception as e:
            print(f"[ERR] {img.name}: {e}")
            continue

        # Parse
        en_md, target_md = parse_dual_blocks(full_text, args.lang)

        # Save target translation
        if target_md:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(target_md)
        else:
            print(f"[WARN] {img.name}: No translation block found.")

        # Save English reconstruction
        if args.save_english and english_out_dir and en_md:
            en_path = english_out_dir / (img.stem + ".en.md")
            with open(en_path, "w", encoding="utf-8") as f:
                f.write(en_md)

        # Save full output
        if raw_dir:
            raw_path = raw_dir / (img.stem + ".full.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(full_text)

        print(f"[OK] {img.name} → {out_path.name}"
              + (f" (+en)" if (args.save_english and en_md) else "")
              + (f" (+raw)" if raw_dir else ""))

if __name__ == "__main__":
    main()