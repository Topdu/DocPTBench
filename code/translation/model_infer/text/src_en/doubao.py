#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import argparse
from pathlib import Path

from openai import OpenAI

# ========== 1. Multi-language Configuration ==========
LANG_CONFIG = {
    "zh": {
        "name": "Simplified Chinese",
        "tag": "chinese_md",
        "ext": ".zh.md",
        "style_note": "Fluent, professional Simplified Chinese."
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

# ========== 2. Dynamic Prompt Templates (English -> Multi-lang) ==========
SYSTEM_PROMPT_TEMPLATE = """You are a professional document translator.

Task: Translate the provided English Markdown into a high-fidelity {lang_name} Markdown.

Output format (STRICT):
Return EXACTLY one fenced block:
<{lang_tag}>
...the {lang_name} Markdown translation...
</{lang_tag}>

Translation rules:
- You MUST translate all English narrative text into fluent {lang_name}.
- Preserve structure: headings (##), lists, blockquotes (>), tables, horizontal rules, references, footnotes.
- Keep code blocks, inline code, LaTeX formulas ($$...$$), URLs, emails, IDs unchanged.
- In formulas ($$...$$), translate ONLY human words inside \\text{{...}} (e.g., \\text{{mean}} → \\text{{translated_term}}); keep math symbols/variables/units intact.
- In tables, preserve the structure; translate only narrative text in cells.
- Keep hyperlinks/images as [text](url): translate visible text, keep URL unchanged.
- Target Style: {style_note}
- No added/omitted content. Return ONLY the <{lang_tag}> block, nothing else.
"""

USER_INSTRUCTION_TEMPLATE = "Translate the given English Markdown into {lang_name} Markdown, following the rules."

# ========== 3. Parsing and Message Construction ==========

def parse_target_block(text: str, lang_code: str) -> str:
    """Dynamically parses the XML block content based on language configuration."""
    tag = LANG_CONFIG[lang_code]["tag"]
    # re.S makes . match newlines
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    m = pattern.search(text)
    return m.group(1).strip() if m else text.strip()

def build_messages_for_text(md_text: str, lang_code: str):
    """
    Constructs the message payload for the API call based on target language.
    """
    cfg = LANG_CONFIG[lang_code]
    
    # Fill Prompt
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

def call_model(client: OpenAI, model: str, messages, max_tokens: int,
               temperature: float, top_p: float, stream: bool) -> str:
    """
    Calls the VolcEngine ARK model using the OpenAI library.
    """
    try:
        if stream:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
            )
            
            full_content = []
            print("[STREAM] ", end="", flush=True)
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
                    full_content.append(content)
            print() # Newline after stream ends
            return "".join(full_content)
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=False,
            )
            content = response.choices[0].message.content
            if content is None:
                raise RuntimeError("API returned an empty response.")
            return content

    except Exception as e:
        raise RuntimeError(f"API request failed: {e}")


# ========== 4. Main Process ==========
def main():
    parser = argparse.ArgumentParser(description="Text-based EN -> Multi-Lang translation with Doubao (VolcEngine ARK) API")
    parser.add_argument("--dir", required=True, help="Input English Markdown directory")
    
    # Language selection
    parser.add_argument("--lang", default="zh", choices=["zh", "de", "fr", "ru"], 
                        help="Target language: zh (Chinese), de (German), fr (French), ru (Russian). Default: zh")
    
    parser.add_argument("--out_dir", default=None, help="Output directory (default: auto-generated based on language)")
    parser.add_argument("--ext", default=None, help="Output extension (default: auto-generated based on language)")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files")
    parser.add_argument("--save_full", action="store_true", help="Save full raw output as .full.txt")

    # Generation parameters
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--stream", action="store_true", help="Use streaming response and print to console")

    # API parameters
    parser.add_argument("--api_key", type=str, default=None, help="Optional: Explicit API Key; otherwise reads env var ARK_API_KEY")
    parser.add_argument("--base_url", type=str, default="https://ark.cn-beijing.volces.com/api/v3", help="VolcEngine Ark Base URL")
    parser.add_argument("--model", default="doubao-seed-1-6-vision-250815", help="Model name")

    args = parser.parse_args()

    # Config Initialization
    lang_cfg = LANG_CONFIG[args.lang]
    
    # Auto-determine output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(f"translations_doubao_{args.lang}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine extension
    ext = args.ext if args.ext else lang_cfg["ext"]

    in_dir = Path(args.dir)
    md_files = sorted([
        p for p in in_dir.iterdir()
        if p.suffix.lower() in (".md", ".mmd") and not p.name.endswith(".full.txt")
    ])
    if not md_files:
        print(f"Error: No Markdown files found in directory '{in_dir}'.", file=sys.stderr)
        exit(1)

    api_key = args.api_key or os.getenv("ARK_API_KEY")
    if not api_key:
        print("Error: Please set environment variable ARK_API_KEY or specify via --api_key.", file=sys.stderr)
        exit(1)

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    print(f"[INFO] Target Language: {lang_cfg['name']} ({args.lang})")
    print(f"[INFO] Found {len(md_files)} Markdown files. Output: {out_dir.resolve()}")
    print(f"[INFO] model={args.model} stream={args.stream}")

    for md_file in md_files:
        out_path = out_dir / (md_file.stem + ext)
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {out_path.name} exists. Use --overwrite to regenerate.")
            continue

        print(f"\n--- Translating {md_file.name} to {args.lang} ---")
        try:
            src_text = md_file.read_text(encoding="utf-8")
            
            # Build language-specific messages
            messages = build_messages_for_text(src_text, args.lang)

            full_text = call_model(
                client=client,
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                stream=args.stream,
            )
        except Exception as e:
            print(f"\n[ERR] {md_file.name}: {e}", file=sys.stderr)
            continue

        # Parse language-specific block
        target_md = parse_target_block(full_text, args.lang)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(target_md)

        if args.save_full:
            raw_path = out_dir / (md_file.stem + ".full.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(full_text)

        print(f"[OK] {md_file.name} → {out_path.name}"
              + (f" (+raw:True)" if args.save_full else ""))

if __name__ == "__main__":
    main()