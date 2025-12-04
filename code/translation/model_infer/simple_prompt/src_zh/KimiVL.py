#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import base64
import mimetypes
import re
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI

# ========== 1. Multi-language Config ==========
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

# ========== 2. Dynamic Prompt Templates ==========
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

# ===== Image Processing =====
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
    """Read image and convert to Data URL format (required by Kimi Vision)."""
    mime = guess_mime(image_path)
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"

# ===== Build Messages =====
def build_messages_for_single_image(data_url: str, lang_code: str) -> List[Dict[str, Any]]:
    """Build prompt based on language code."""
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
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": user_content},
            ],
        },
    ]

# ===== Parsing Logic =====
def parse_target_block(text: str, lang_code: str) -> str:
    """Dynamically parse XML block based on language config."""
    tag = LANG_CONFIG[lang_code]["tag"]
    # Dynamically build regex: match <tag>...</tag>
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    
    m = pattern.search(text)
    if m:
        return m.group(1).strip()
    
    # Fallback: if tag not found but text is not empty, return full text
    return text.strip()

# ===== API Call =====
def call_kimi_vision(client: OpenAI, model: str, messages, stream: bool,
                     max_tokens: int, temperature: float, top_p: float) -> str:
    completion_params = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    
    if stream:
        resp = client.chat.completions.create(**completion_params)
        full_content = ""
        print(f"[STREAM] ", end="", flush=True)
        for chunk in resp:
            delta = chunk.choices[0].delta.content or ""
            full_content += delta
            print(".", end="", flush=True) # Simple progress indicator
        print()
        return full_content
    else:
        resp = client.chat.completions.create(**completion_params)
        return "".join(choice.message.content or "" for choice in resp.choices)

# ===== Main Process =====
def main():
    parser = argparse.ArgumentParser(description="Kimi Vision Multi-language Direct Translation (ZH -> EN/DE/FR/RU)")
    
    # Core Arguments
    parser.add_argument("--dir", required=True, help="Input image directory")
    parser.add_argument("--lang", default="en", choices=["en", "de", "fr", "ru"], help="Target language")
    parser.add_argument("--out_dir", default=None, help="Output directory (default: auto-generated based on language)")
    parser.add_argument("--ext", default=None, help="Output extension (default: auto-selected based on language)")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing files")

    # Model Arguments
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--stream", action="store_true", help="Use streaming decoding")
    parser.add_argument("--model", default="moonshot-v1-32k-vision-preview", 
                        help="Model name (e.g., moonshot-v1-32k-vision-preview)")
    
    # API Credentials
    parser.add_argument("--api_key", type=str, default=None, help="Moonshot API Key")
    parser.add_argument("--base_url", type=str, default="https://api.moonshot.cn/v1", help="API Endpoint")

    args = parser.parse_args()

    # 1. Config Initialization
    lang_cfg = LANG_CONFIG[args.lang]
    
    # Auto-generate output directory name
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(f"translations_kimi_direct_{args.lang}")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine extension
    ext = args.ext if args.ext else lang_cfg["ext"]

    # 2. Collect Images
    in_dir = Path(args.dir)
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
    image_paths = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])
    if not image_paths:
        raise ValueError(f"No supported images found in directory '{in_dir}'.")

    # 3. API Client
    api_key = args.api_key or os.getenv("MOONSHOT_API_KEY")
    if not api_key:
        raise RuntimeError("Please set MOONSHOT_API_KEY environment variable or use --api_key argument.")
    
    client = OpenAI(api_key=api_key, base_url=args.base_url)

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

        # Convert image to Data URL
        data_url = image_path_to_data_url(img)

        # Build messages
        messages = build_messages_for_single_image(data_url, args.lang)

        try:
            print(f"[INFO] Processing {img.name} -> {lang_cfg['name']}...")
            full_text = call_kimi_vision(
                client=client, model=args.model, messages=messages,
                stream=args.stream, max_tokens=args.max_tokens,
                temperature=args.temperature, top_p=args.top_p,
            )
        except Exception as e:
            print(f"[ERR] Failed processing '{img.name}': {e}")
            (out_dir / (img.stem + ".error.txt")).write_text(str(e), encoding="utf-8")
            continue

        # Parse target language block
        target_md = parse_target_block(full_text, args.lang)

        # Write result
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(target_md)

        print(f"[OK] {img.name} → {out_path.name}")

if __name__ == "__main__":
    main()