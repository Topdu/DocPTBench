#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from pathlib import Path
from typing import Tuple

from openai import OpenAI

# ========== 1. Multilingual Configuration ==========
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

Task: Translate the provided Chinese Markdown into a high-fidelity {lang_name} Markdown.

Output format (STRICT):
Return EXACTLY one fenced block:
<{lang_tag}>
...the {lang_name} Markdown translation...
</{lang_tag}>

Translation rules:
- You MUST translate all Chinese narrative text into fluent {lang_name}.
- NO Chinese characters should remain in <{lang_tag}> (except inside code, LaTeX formulas, or URLs that naturally contain them).
- Preserve structure: headings (##), lists, blockquotes (>), tables, horizontal rules, references, footnotes.
- Keep code blocks, inline code, LaTeX formulas ($$...$$), URLs, emails, IDs unchanged.
- In formulas ($$...$$), translate ONLY human words inside \\text{{...}} (e.g., \\text{{均值}} → \\text{{translated_term}}); keep math symbols/variables/units intact.
- In tables, preserve the structure; translate only narrative text in cells.
- Keep hyperlinks/images as [text](url): translate visible text, keep URL unchanged.
- Target Style: {style_note}
- No added/omitted content. Return ONLY the <{lang_tag}> block, nothing else.
"""

USER_INSTRUCTION_TEMPLATE = "Translate the given Chinese Markdown into {lang_name} Markdown, following the rules."

# ========== 3. Parsing and Message Construction ==========

def parse_target_block(text: str, lang_code: str) -> str:
    """Dynamically parse XML block based on language configuration."""
    tag = LANG_CONFIG[lang_code]["tag"]
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    m = pattern.search(text)
    return m.group(1).strip() if m else text.strip()

def build_messages_for_text(md_text: str, lang_code: str):
    cfg = LANG_CONFIG[lang_code]
    
    # Format System Prompt
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        lang_name=cfg["name"],
        lang_tag=cfg["tag"],
        style_note=cfg["style_note"]
    )
    
    # Format User Instruction
    user_instruction = USER_INSTRUCTION_TEMPLATE.format(
        lang_name=cfg["name"]
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": md_text + "\n\n" + user_instruction},
    ]

# ========== 4. API Call Logic ==========
def call_model(client: OpenAI, model: str, messages, stream: bool,
               max_tokens: int, temperature: float, top_p: float) -> str:
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
        for ch in resp:
            delta = getattr(ch.choices[0].delta, "content", None)
            if delta:
                chunks.append(delta)
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

# ========== 5. Main Program ==========
def main():
    parser = argparse.ArgumentParser(description="Language-only Markdown CN->Multi-Lang translation with DashScope API")
    parser.add_argument("--dir", required=True, help="Input Chinese Markdown directory")
    
    # Language selection
    parser.add_argument("--lang", default="en", choices=["en", "de", "fr", "ru"], help="Target language (default: en)")
    
    parser.add_argument("--out_dir", default=None, help="Output directory (default: auto-generated based on language)")
    parser.add_argument("--ext", default=None, help="Output extension (default: auto-generated based on language)")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files")
    parser.add_argument("--save_full", action="store_true", help="Save full raw output to .full.txt")

    # Generation parameters
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--stream", action="store_true", help="Use streaming response")

    # API parameters
    parser.add_argument("--api_key", type=str, default=None, help="Optional: Explicit API Key; otherwise reads DASHSCOPE_API_KEY env var")
    parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="OpenAI compatible Base URL")

    parser.add_argument("--model", type=str, default="qwen-vl-max",
                        help="Model name (default: qwen-vl-max)")

    args = parser.parse_args()

    # Configuration initialization
    lang_cfg = LANG_CONFIG[args.lang]

    in_dir = Path(args.dir)
    
    # Automatically set output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(f"translations_lo_{args.lang}")
    
    out_dir.mkdir(parents=True, exist_ok=True)

    # Automatically set extension
    ext = args.ext if args.ext else lang_cfg["ext"]

    md_files = sorted([
        p for p in in_dir.iterdir()
        if p.suffix.lower() in (".md", ".mmd") and not p.name.endswith(".full.txt")
    ])
    if not md_files:
        raise ValueError("No Markdown files found. Supported extensions: .md .mmd")

    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("Please set DASHSCOPE_API_KEY environment variable or specify via --api_key.")

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    print(f"[INFO] Target Language: {lang_cfg['name']} ({args.lang})")
    print(f"[INFO] Found {len(md_files)} files. Output to: {out_dir.resolve()}")
    print(f"[INFO] model={args.model} stream={args.stream} max_tokens={args.max_tokens}")

    for md in md_files:
        out_path = out_dir / (md.stem + ext)
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {out_path.name} exists. Use --overwrite to regenerate.")
            continue

        src = md.read_text(encoding="utf-8")
        
        # Build language-specific messages
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

        # Parse language-specific Block
        target_md = parse_target_block(full_text, args.lang)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(target_md)

        if args.save_full:
            raw_path = out_dir / (md.stem + ".full.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(full_text)

        print(f"[OK] {md.name} -> {out_path.name}"
              + (f" (+raw:True)" if args.save_full else ""))

if __name__ == "__main__":
    main()