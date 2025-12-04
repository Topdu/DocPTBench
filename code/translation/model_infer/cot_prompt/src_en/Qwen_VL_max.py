#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import base64
import mimetypes
import re
from pathlib import Path
from typing import Tuple

from openai import OpenAI

# ========== 1. Multi-language Configuration ==========
LANG_CONFIG = {
    "zh": {
        "name": "Simplified Chinese",
        "tag": "chinese_md",
        "ext": ".zh.md",
        "style_note": "Fluent, professional zh-CN for narrative text; use Chinese punctuation in prose."
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

# ========== 2. Dynamic Prompt Template ==========
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

# ========== 3. Dual-Block Parsing (Dynamic) ==========
def parse_dual_blocks(text: str, lang_code: str) -> Tuple[str, str]:
    tag = LANG_CONFIG[lang_code]["tag"]
    
    # Build regex dynamically
    target_re = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    en_re = re.compile(r"<english_md>\s*(.*?)\s*</english_md>", re.S | re.I)

    target_match = target_re.search(text)
    en_match = en_re.search(text)
    
    # Fallback: if no tags found but text exists, assume full text is target
    target_md = target_match.group(1).strip() if target_match else text.strip()
    en_md = en_match.group(1).strip() if en_match else ""
    
    # Logic fix: if en found but target empty (and text not empty), likely tag failure
    if not target_md and not en_md and text.strip():
        target_md = text.strip()

    return en_md, target_md

# ========== Image to Data URL ==========
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

def image_path_to_data_url(image_path: Path) -> str:
    mime = guess_mime(image_path)
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"

# ========== Build Messages ==========
def build_messages(data_url: str, lang_code: str):
    cfg = LANG_CONFIG[lang_code]
    
    # Fill System Prompt
    system_content = SYSTEM_PROMPT_TEMPLATE.format(
        lang_name=cfg["name"],
        lang_tag=cfg["tag"],
        style_note=cfg["style_note"]
    )
    
    # Dynamic User Instruction
    user_instruction = (
        f"Reconstruct and translate the document image as specified. "
        f"Return ONLY the two blocks: <english_md>...</english_md> and <{cfg['tag']}>...</{cfg['tag']}>."
    )

    return [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": user_instruction},
            ],
        },
    ]

# ========== API Call ==========
def call_qwen_vl_max(client: OpenAI, model: str, messages, stream: bool, max_tokens: int, temperature: float, top_p: float) -> str:
    if stream:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        out = []
        for chunk in resp:
            delta = getattr(chunk.choices[0].delta, "content", None)
            if delta:
                out.append(delta)
        return "".join(out)
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return "".join(choice.message.content or "" for choice in resp.choices)

# ========== Main Process ==========
def main():
    parser = argparse.ArgumentParser(description="Qwen-VL-Max Document Translation (Multi-Language)")
    
    # Core paths and language
    parser.add_argument("--dir", required=True, help="Input image folder")
    parser.add_argument("--lang", default="zh", choices=["zh", "de", "fr", "ru"], help="Target language: zh, de, fr, ru")
    
    # Output directory control
    parser.add_argument("--out_dir", default=None, help="Output directory for target language (default: auto-generated)")
    parser.add_argument("--english_out_dir", default=None, help="Output directory for English reconstruction")
    parser.add_argument("--raw_dir", default=None, help="Output directory for full raw responses")
    
    # Feature switches
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing files")
    parser.add_argument("--save_english", action="store_true", help="Save English reconstruction")
    parser.add_argument("--save_full", action="store_true", help="Save raw model output")
    
    # Model parameters
    parser.add_argument("--model", default="qwen-vl-max", help="Model name")
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--stream", action="store_true")

    # API Configuration
    parser.add_argument("--api_key", type=str, default="", help="API Key (or set DASHSCOPE_API_KEY env var)")
    parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    args = parser.parse_args()

    # 1. Path Setup
    lang_cfg = LANG_CONFIG[args.lang]
    
    # Determine output directory (Relative path by default)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(f"results_{args.lang}")
    
    out_dir.mkdir(parents=True, exist_ok=True)

    # English output directory
    if args.save_english:
        if args.english_out_dir:
            en_dir = Path(args.english_out_dir)
        else:
            en_dir = out_dir / "english_reconstruction"
        en_dir.mkdir(parents=True, exist_ok=True)
    else:
        en_dir = None

    # Raw output directory
    if args.save_full:
        if args.raw_dir:
            raw_dir = Path(args.raw_dir)
        else:
            raw_dir = out_dir / "raw_responses"
        raw_dir.mkdir(parents=True, exist_ok=True)
    else:
        raw_dir = None

    # 2. Collect Images
    in_dir = Path(args.dir)
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
    if not in_dir.exists():
        raise ValueError(f"Input directory does not exist: {in_dir}")

    image_paths = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])
    if not image_paths:
        raise ValueError("No images found in the input directory.")

    # 3. Initialize Client
    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("Need API Key via --api_key or DASHSCOPE_API_KEY env.")
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    print(f"[INFO] Target Language: {lang_cfg['name']} ({args.lang})")
    print(f"[INFO] Output Dir: {out_dir.resolve()}")
    print(f"[INFO] Found {len(image_paths)} images.")

    # 4. Processing Loop
    for img in image_paths:
        target_path = out_dir / (img.stem + lang_cfg["ext"])
        if target_path.exists() and not args.overwrite:
            print(f"[SKIP] {target_path.name} exists.")
            continue

        # Prepare data
        data_url = image_path_to_data_url(img)
        messages = build_messages(data_url, args.lang)

        try:
            full_text = call_qwen_vl_max(
                client=client,
                model=args.model,
                messages=messages,
                stream=args.stream,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        except Exception as e:
            print(f"[ERR] {img.name}: {e}")
            continue

        # Parse
        en_md, target_md = parse_dual_blocks(full_text, args.lang)

        # Save target translation
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(target_md)

        # Save English reconstruction
        if args.save_english and en_dir and en_md:
            en_path = en_dir / (img.stem + "_en.md")
            with open(en_path, "w", encoding="utf-8") as f:
                f.write(en_md)

        # Save raw output
        if args.save_full and raw_dir:
            raw_path = raw_dir / (img.stem + ".full.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(full_text)

        print(f"[OK] {img.name} -> {target_path.name}"
              + (f" (+en)" if (args.save_english and en_md) else "")
              + (f" (+raw)" if args.save_full else ""))

if __name__ == "__main__":
    main()