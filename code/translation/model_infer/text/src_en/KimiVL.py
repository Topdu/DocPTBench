#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

from openai import OpenAI

# ==============================================================================
# 1. CONFIG & PROMPTS
# ==============================================================================

LANG_CONFIG = {
    "zh": {
        "name": "Simplified Chinese",
        "tag": "chinese_md",
        "ext": ".zh.md",
        "style_note": "Fluent, professional Simplified Chinese, with Chinese punctuation."
    },
    "de": {
        "name": "German",
        "tag": "german_md",
        "ext": ".de.md",
        "style_note": "Fluent, professional German with standard punctuation."
    },
    "fr": {
        "name": "French",
        "tag": "french_md",
        "ext": ".fr.md",
        "style_note": "Fluent, professional French with standard punctuation."
    },
    "ru": {
        "name": "Russian",
        "tag": "russian_md",
        "ext": ".ru.md",
        "style_note": "Fluent, professional Russian with standard punctuation."
    }
}

# Dynamic System Prompt Template
SYSTEM_PROMPT_TEMPLATE = """You are a professional document translator.

Task: Translate the provided English Markdown into a high-fidelity {lang_name} Markdown.

Output format (STRICT):
Return EXACTLY one fenced block:
<{lang_tag}>
...the {lang_name} Markdown translation...
</{lang_tag}>

Translation rules:
- Preserve structure: headings (##), lists, blockquotes (>), tables, horizontal rules, references, footnotes.
- Translate only narrative text. Keep code blocks, inline code, LaTeX formulas ($$...$$), URLs, emails, IDs unchanged.
- In formulas ($$...$$), only translate human words inside \\text{{...}} (e.g., \\text{{Mean}} → \\text{{translated_term}}).
- In tables, preserve the structure, translate only narrative text in cells.
- Keep hyperlinks/images as [text](url) with translated visible text but unchanged URL.
- Target Style: {style_note}
- No added/omitted content. Return ONLY the <{lang_tag}> block, nothing else.
"""

USER_INSTRUCTION_TEMPLATE = "Translate the given English Markdown into {lang_name} Markdown, following the rules."

# ==============================================================================
# 2. PARSING & MESSAGE BUILDING
# ==============================================================================

def parse_target_block(text: str, lang_code: str) -> str:
    """Dynamically parses the XML block content based on language configuration."""
    tag = LANG_CONFIG[lang_code]["tag"]
    # re.S makes . match newlines
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    m = pattern.search(text)
    return m.group(1).strip() if m else text.strip()

def build_messages_for_text(md_text: str, lang_code: str) -> List[Dict[str, Any]]:
    """Builds the message body based on the target language."""
    cfg = LANG_CONFIG[lang_code]
    
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        lang_name=cfg["name"],
        lang_tag=cfg["tag"],
        style_note=cfg["style_note"]
    )
    user_instruction = USER_INSTRUCTION_TEMPLATE.format(
        lang_name=cfg["name"]
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": md_text + "\n\n" + user_instruction},
    ]

# ==============================================================================
# 3. MODEL CALLING
# ==============================================================================

def call_model(client: OpenAI, model: str, messages, stream: bool, max_tokens: int, temperature: float, top_p: float) -> str:
    if stream:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        chunks = []
        print("  > Streaming response...", end="", flush=True)
        for ch in resp:
            delta = getattr(ch.choices[0].delta, "content", None)
            if delta:
                chunks.append(delta)
        print(" Done.")
        return "".join(chunks)
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return "".join((c.message.content or "") for c in resp.choices)

# ==============================================================================
# 4. MAIN LOGIC
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Language-only Markdown translation with Kimi (Moonshot) - Multi-language Support")
    parser.add_argument("--dir", required=True, help="Input directory containing English Markdown files.")
    
    # Language selection
    parser.add_argument("--lang", default="zh", choices=["zh", "de", "fr", "ru"], 
                        help="Target language: zh (Chinese), de (German), fr (French), ru (Russian). Default: zh")

    parser.add_argument("--out_dir", default=None, help="Output directory (default: auto-generated based on language).")
    parser.add_argument("--ext", default=None, help="Output extension (default: auto-generated based on language).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--save_full", action="store_true", help="Save full raw output as .full.txt.")

    # Generation parameters
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--stream", action="store_true", help="Use streaming response.")

    # API parameters
    parser.add_argument("--api_key", type=str, default=None, help="Explicitly provide API Key; otherwise reads MOONSHOT_API_KEY env var.")
    parser.add_argument("--base_url", type=str, default="https://api.moonshot.cn/v1",
                        help="Kimi OpenAI compatible Base URL.")
    parser.add_argument("--model", type=str, default="moonshot-v1-32k-vision-preview",
                        help="Model name.")

    args = parser.parse_args()

    # Get language config
    lang_cfg = LANG_CONFIG[args.lang]

    # Auto-determine output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(f"translations_kimi_{args.lang}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine extension
    ext = args.ext if args.ext else lang_cfg["ext"]

    in_dir = Path(args.dir)
    md_files = sorted([
        p for p in in_dir.iterdir()
        if p.suffix.lower() in (".md", ".mmd") and not p.name.endswith(".full.txt")
    ])
    if not md_files:
        raise ValueError(f"No Markdown files found in {in_dir}. Supported extensions: .md .mmd")

    api_key = args.api_key or os.getenv("MOONSHOT_API_KEY")
    if not api_key:
        raise RuntimeError("Please set MOONSHOT_API_KEY environment variable or provide via --api_key.")

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    print(f"[INFO] Target Language: {lang_cfg['name']} ({args.lang})")
    print(f"[INFO] Found {len(md_files)} files. Output to: {out_dir.resolve()}")
    print(f"[INFO] model={args.model} stream={args.stream} max_tokens={args.max_tokens}")

    for md in md_files:
        out_path = out_dir / (md.stem + ext)
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {out_path.name} exists. Use --overwrite to regenerate.")
            continue

        print(f"[PROC] Translating {md.name} ...")
        src = md.read_text(encoding="utf-8")
        
        # Build multi-language Prompt
        messages = build_messages_for_text(src, args.lang)

        try:
            full_text = call_model(
                client=client,
                model=args.model,
                messages=messages,
                stream=args.stream,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        except Exception as e:
            print(f"[ERR] {md.name}: {e}")
            continue

        # Parse XML block based on language
        target_md = parse_target_block(full_text, args.lang)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(target_md)

        if args.save_full:
            raw_path = out_dir / (md.stem + ".full.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(full_text)

        print(f"[OK] {md.name} -> {out_path.name}"
              + (f" (+raw)" if args.save_full else ""))

if __name__ == "__main__":
    main()