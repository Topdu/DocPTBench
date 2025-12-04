#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from pathlib import Path

from zai import ZhipuAiClient

# ========== 1. Multi-language Configuration ==========
LANG_CONFIG = {
    "zh": {
        "name": "Simplified Chinese",
        "tag": "chinese_md",
        "ext": ".zh.md",
        "style_note": "Fluent, professional zh-CN, with Chinese punctuation in prose."
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

# ========== 2. Dynamic Prompt Templates ==========
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
- Style: {style_note}
- No added/omitted content. Return ONLY the <{lang_tag}> block, nothing else.
"""

USER_INSTRUCTION_TEMPLATE = "Translate the given English Markdown into {lang_name} Markdown, following the rules."

# ========== 3. Parsing and Message Construction ==========

def parse_target_block(text: str, lang_code: str) -> str:
    """Dynamically parse XML block content based on language config."""
    tag = LANG_CONFIG[lang_code]["tag"]
    # re.S allows . to match newlines
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    m = pattern.search(text)
    return m.group(1).strip() if m else text.strip()

def build_messages_for_text(md_text: str, lang_code: str):
    """Build message list containing target language specific instructions."""
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

def call_model(client: ZhipuAiClient, model: str, messages, max_tokens: int,
               temperature: float, top_p: float, thinking: str) -> str:
    """Call GLM model, supporting the thinking parameter."""
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
            # Handle multimodal or segmented returns (though this is a text-only task)
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
    base_resp = getattr(resp, "base_resp", None)
    raise RuntimeError(f"GLM-4.5V request failed. resp={resp}, base_resp={base_resp}")

# ========== 4. Main Process ==========
def main():
    parser = argparse.ArgumentParser(description="Language-only English -> Multi-Lang Markdown translation with GLM-4.5V")
    parser.add_argument("--dir", required=True, help="Input English Markdown directory")
    
    # Language selection
    parser.add_argument("--lang", default="zh", choices=["zh", "de", "fr", "ru"], 
                        help="Target language: zh (Chinese), de (German), fr (French), ru (Russian). Default: zh")

    parser.add_argument("--out_dir", default=None, help="Output directory (default: generated based on language, e.g., translations_lo_glm45v_zh)")
    parser.add_argument("--ext", default=None, help="Output extension (default: generated based on language, e.g., .zh.md)")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files")
    parser.add_argument("--save_full", action="store_true", help="Save full raw output as .full.txt")

    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--model", type=str, default="glm-4.5v",
                        help="Model name (default: glm-4.5v)")
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="enabled",
                        help="Enable thinking capability (default: enabled)")

    parser.add_argument("--api_key", type=str, default=None, help="Explicitly pass API Key; otherwise read environment variable ZHIPUAI_API_KEY")
    args = parser.parse_args()

    # Config Initialization
    lang_cfg = LANG_CONFIG[args.lang]

    # Automatically determine output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(f"translations_lo_glm45v_{args.lang}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine extension
    ext = args.ext if args.ext else lang_cfg["ext"]

    in_dir = Path(args.dir)
    md_files = sorted([
        p for p in in_dir.iterdir()
        if p.suffix.lower() in (".md", ".mmd") and not p.name.endswith(".full.txt")
    ])
    if not md_files:
        raise ValueError("No Markdown files found. Supported extensions: .md .mmd")

    api_key = args.api_key or os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set environment variable ZHIPUAI_API_KEY or specify via --api_key.")

    client = ZhipuAiClient(api_key=api_key)

    print(f"[INFO] Target Language: {lang_cfg['name']} ({args.lang})")
    print(f"[INFO] Found {len(md_files)} files. Output to: {out_dir.resolve()}")
    print(f"[INFO] model={args.model} thinking={args.thinking} max_tokens={args.max_tokens}")

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
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                thinking=args.thinking,
            )
        except Exception as e:
            print(f"[ERR] {md.name}: {e}")
            continue

        # Parse language-specific block
        target_md = parse_target_block(full_text, args.lang)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(target_md)

        if args.save_full:
            raw_path = out_dir / (md.stem + ".full.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(full_text)

        print(f"[OK] {md.name} → {out_path.name}"
              + (f" (+raw:True)" if args.save_full else ""))

if __name__ == "__main__":
    main()