#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from pathlib import Path

# Dependency: zai SDK (pip install zhipuai)
try:
    from zai import ZhipuAiClient
except ImportError:
    # Compatibility for older versions or different namespaces if the user provides the zai library code
    try:
        from zhipuai import ZhipuAI as ZhipuAiClient
    except ImportError:
        ZhipuAiClient = None

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

# ========== 3. Parsing & Message Building ==========
def parse_target_block(text: str, lang_code: str) -> str:
    """Dynamically parse XML block based on language config."""
    tag = LANG_CONFIG[lang_code]["tag"]
    # Dynamically build regex
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    
    m = pattern.search(text)
    return m.group(1).strip() if m else text.strip()

def build_messages_for_text(md_text: str, lang_code: str):
    cfg = LANG_CONFIG[lang_code]
    
    system_content = SYSTEM_PROMPT_TEMPLATE.format(
        lang_name=cfg["name"],
        lang_tag=cfg["tag"],
        style_note=cfg["style_note"]
    )
    
    user_instruction = USER_INSTRUCTION_TEMPLATE.format(
        lang_name=cfg["name"]
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": md_text + "\n\n" + user_instruction},
    ]

# ========== 4. API Call ==========
def call_model(client, model: str, messages, max_tokens: int,
               temperature: float, top_p: float, thinking: str) -> str:
    """
    Call GLM model.
    Note: ZhipuAI SDK usage may vary by version; adapted based on provided code snippets.
    """
    # Build parameter dictionary
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    
    # Pass this parameter only if thinking is enabled
    if thinking and thinking != "disabled":
         params["thinking"] = {"type": thinking}

    resp = client.chat.completions.create(**params)
    
    choices = getattr(resp, "choices", None)
    if choices:
        content = choices[0].message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Handle multimodal or segmented returns
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
            
    base_resp = getattr(resp, "base_resp", None)
    raise RuntimeError(f"GLM Request failed. resp={resp}, base_resp={base_resp}")

# ========== Main Process ==========
def main():
    parser = argparse.ArgumentParser(description="Language-only Markdown translation (CN -> Multi-Lang) with GLM-4.5V")
    
    # Core parameters
    parser.add_argument("--dir", required=True, help="Input Chinese Markdown directory")
    parser.add_argument("--lang", default="en", choices=["en", "de", "fr", "ru"], help="Target language (default: en)")
    parser.add_argument("--out_dir", default=None, help="Output directory (default: auto-generated based on language)")
    parser.add_argument("--ext", default=None, help="Output extension")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files")
    parser.add_argument("--save_full", action="store_true", help="Save full raw output as .full.txt")

    # Model parameters
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--model", type=str, default="glm-4.5v", help="Model name")
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="enabled",
                        help="Whether to enable thinking capability (default: enabled)")

    parser.add_argument("--api_key", type=str, default=None, help="API Key (default: reads env var ZHIPUAI_API_KEY)")
    args = parser.parse_args()

    if ZhipuAiClient is None:
        raise ImportError("Module 'zai' or 'zhipuai' not found. Please install: pip install zhipuai")

    # 1. Config Initialization
    lang_cfg = LANG_CONFIG[args.lang]
    
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(f"translations_lo_glm45v_{args.lang}")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = args.ext if args.ext else lang_cfg["ext"]

    # 2. Scan Files
    in_dir = Path(args.dir)
    md_files = sorted([
        p for p in in_dir.iterdir()
        if p.suffix.lower() in (".md", ".mmd") and not p.name.endswith(".full.txt")
    ])
    if not md_files:
        raise ValueError(f"No Markdown files found in directory '{in_dir}'.")

    # 3. Client Initialization
    api_key = args.api_key or os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set environment variable ZHIPUAI_API_KEY or specify via --api_key.")

    client = ZhipuAiClient(api_key=api_key)

    print(f"[INFO] Target Language: {lang_cfg['name']} ({args.lang})")
    print(f"[INFO] Output Dir: {out_dir.resolve()}")
    print(f"[INFO] Model: {args.model}, Thinking: {args.thinking}")
    print(f"[INFO] Found {len(md_files)} files.")

    # 4. Processing Loop
    for md in md_files:
        out_path = out_dir / (md.stem + ext)
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {out_path.name} exists.")
            continue

        print(f"\n--- Processing {md.name} -> {args.lang} ---")
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
            # Simple error logging
            (out_dir / (md.stem + ".error.txt")).write_text(str(e), encoding="utf-8")
            continue

        # Parse language-specific block
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