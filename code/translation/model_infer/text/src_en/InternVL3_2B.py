#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer
import re
import sys

# ==============================================================================
# 1. CONFIG & PROMPTS (Multi-language Configuration)
# ==============================================================================

LANG_CONFIG = {
    "zh": {
        "name": "Simplified Chinese",
        "tag": "chinese_md",
        "ext": ".zh.mmd",
        "style_note": "Fluent, professional Simplified Chinese, with Chinese punctuation."
    },
    "de": {
        "name": "German",
        "tag": "german_md",
        "ext": ".de.mmd",
        "style_note": "Fluent, professional German with standard punctuation."
    },
    "fr": {
        "name": "French",
        "tag": "french_md",
        "ext": ".fr.mmd",
        "style_note": "Fluent, professional French with standard punctuation."
    },
    "ru": {
        "name": "Russian",
        "tag": "russian_md",
        "ext": ".ru.mmd",
        "style_note": "Fluent, professional Russian with standard punctuation."
    }
}

# Dynamic System Prompt Template
DOC_TRANSLATION_TEMPLATE = """You are a professional document translator.

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

# ==============================================================================
# 2. PARSING LOGIC
# ==============================================================================

def parse_target_block(text: str, lang_code: str) -> str:
    """Dynamically parse XML block content based on language configuration."""
    tag = LANG_CONFIG[lang_code]["tag"]
    # re.S allows . to match newlines
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    m = pattern.search(text)
    return m.group(1).strip() if m else text.strip()

# ==============================================================================
# 3. TRANSLATION FUNCTION
# ==============================================================================

@torch.inference_mode()
def translate_one(model, tokenizer, en_md: str, generation_config: dict, lang_code: str):
    """
    Single file translation function.
    Builds language-specific prompt, calls model chat interface, and parses result.
    """
    cfg = LANG_CONFIG[lang_code]
    
    # 1. Build System Prompt
    system_prompt = DOC_TRANSLATION_TEMPLATE.format(
        lang_name=cfg["name"],
        lang_tag=cfg["tag"],
        style_note=cfg["style_note"]
    )
    
    # 2. Build User Input (containing English MD)
    # InternVL/Qwen models typically handle system prompts via history or prepending.
    # Here we follow the original logic of prepending the system prompt to the question.
    user_instruction = f"Translate the given English Markdown into {cfg['name']} Markdown, following the rules."
    
    question = (
        system_prompt.strip() + 
        "\n\n<english_md>\n" + en_md + "\n</english_md>\n" + 
        user_instruction
    )

    # 3. Model Inference
    output = model.chat(tokenizer, None, question, generation_config).strip()
    
    # 4. Parse Result
    target_md = parse_target_block(output, lang_code)
    
    return target_md, output

# ==============================================================================
# 4. MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Translate English Markdown -> Multi-language Markdown with InternVL3")
    parser.add_argument("--model", default="OpenGVLab/InternVL3-2B", help="HuggingFace model path or id")
    parser.add_argument("--dir", required=True, help="Input English Markdown directory")
    
    # Language selection
    parser.add_argument("--lang", default="zh", choices=["zh", "de", "fr", "ru"], 
                        help="Target language: zh (Chinese), de (German), fr (French), ru (Russian). Default: zh")
    
    parser.add_argument("--out_dir", default=None, help="Output directory (default: auto-generated based on language)")
    parser.add_argument("--ext", default=None, help="Output extension (default: auto-generated based on language)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--save_full", action="store_true", help="Save full model output")
    
    args = parser.parse_args()

    # --- Path Setup ---
    in_dir = Path(args.dir)
    lang_cfg = LANG_CONFIG[args.lang]

    # Determine output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(f"translations_internvl3_{args.lang}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine extension
    ext = args.ext if args.ext else lang_cfg["ext"]

    # --- Model Loading ---
    print(f"[INFO] Loading model: {args.model} ...")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    try:
        model = AutoModel.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    generation_config = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=(args.temperature > 0),
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=1.05,
    )

    # --- File Scanning ---
    files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in (".md", ".mmd")])
    if not files:
        print(f"[ERROR] No Markdown files found in directory '{in_dir}' (supports .md/.mmd)")
        sys.exit(1)

    print(f"[INFO] Target Language: {lang_cfg['name']} ({args.lang})")
    print(f"[INFO] Found {len(files)} Markdown files. Output to {out_dir.resolve()}")

    # --- Processing Loop ---
    # Note: Local model inference usually consumes high VRAM. Concurrent execution via ThreadPoolExecutor
    # is not recommended unless using inference frameworks like vLLM. Keeping single-threaded here.
    success_count = 0
    skip_count = 0
    fail_count = 0

    for f in files:
        out_path = out_dir / (f.stem + ext)
        
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {out_path.name} exists")
            skip_count += 1
            continue

        en_md = f.read_text(encoding="utf-8")
        print(f"[RUN] Translating {f.name} ...", end="", flush=True)
        
        try:
            target_md, full_text = translate_one(model, tokenizer, en_md, generation_config, args.lang)

            with open(out_path, "w", encoding="utf-8") as fout:
                fout.write(target_md)

            if args.save_full:
                raw_path = out_dir / (f.stem + ".full.txt")
                raw_path.write_text(full_text, encoding="utf-8")

            print(f"\r[OK] {f.name} -> {out_path.name}      ")
            success_count += 1
        except Exception as e:
            print(f"\r[ERR] {f.name}: {str(e)}")
            err_path = out_dir / (f.stem + ".error.txt")
            err_path.write_text(str(e), encoding="utf-8")
            fail_count += 1

    print("\n--- ✅ Processing Complete ---")
    print(f"Success: {success_count}, Skipped: {skip_count}, Failed: {fail_count}")

if __name__ == "__main__":
    main()