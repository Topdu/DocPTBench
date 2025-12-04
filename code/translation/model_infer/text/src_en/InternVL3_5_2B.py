#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer
import re
import sys

# ==============================================================================
# 1. CONFIG & PROMPTS (Multilingual Configuration)
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

# Dynamic system prompt template
DOC_TRANSLATION_PROMPT_TEMPLATE = """You are a professional document translator.

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
# 2. PARSING & TRANSLATION LOGIC
# ==============================================================================

def parse_target_block(text: str, lang_code: str):
    """Dynamically parse XML block content based on language config."""
    tag = LANG_CONFIG[lang_code]["tag"]
    # re.S allows . to match newlines
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    m = pattern.search(text)
    return m.group(1).strip() if m else text.strip()

@torch.inference_mode()
def translate_one(model, tokenizer, en_md: str, generation_config: dict, lang_code: str):
    """
    Single file translation function.
    Builds language-specific prompt, invokes model, and parses result.
    """
    cfg = LANG_CONFIG[lang_code]
    
    # 1. Build System Prompt
    system_prompt = DOC_TRANSLATION_PROMPT_TEMPLATE.format(
        lang_name=cfg["name"],
        lang_tag=cfg["tag"],
        style_note=cfg["style_note"]
    )
    
    # 2. Build complete User Query (wrapping original text in <english_md>)
    user_instruction = f"Translate the given English Markdown into {cfg['name']} Markdown, following the rules."
    question = (
        system_prompt.strip() + 
        "\n\n<english_md>\n" + en_md + "\n</english_md>\n" + 
        user_instruction
    )

    # 3. Model Inference
    # Note: InternVL chat interface usually handles history; set to None for single-turn
    output = model.chat(tokenizer, None, question, generation_config).strip()
    
    # 4. Parse target language block
    target_md, full_text = parse_target_block(output, lang_code), output
    
    return target_md, full_text

# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Translate English Markdown -> Multi-language Markdown with InternVL3")
    
    # Model path argument
    parser.add_argument("--model", default="OpenGVLab/InternVL3_5-2B", help="HuggingFace model id or local path")
    
    # Input/Output arguments
    parser.add_argument("--dir", required=True, help="Input English Markdown directory")
    
    # Language selection
    parser.add_argument("--lang", default="zh", choices=["zh", "de", "fr", "ru"], 
                        help="Target language: zh (Chinese), de (German), fr (French), ru (Russian). Default: zh")
    
    parser.add_argument("--out_dir", default=None, help="Output directory (default: auto-generated based on language)")
    parser.add_argument("--ext", default=None, help="Output extension (default: auto-generated based on language)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--save_full", action="store_true", help="Save full model output")
    
    args = parser.parse_args()

    # --- Config Initialization ---
    lang_cfg = LANG_CONFIG[args.lang]
    
    in_dir = Path(args.dir)
    
    # Automatically determine output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(f"./translations_internvl_{args.lang}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Automatically determine extension
    ext = args.ext if args.ext else lang_cfg["ext"]

    # --- Load Model ---
    print(f"[INFO] Loading model from: {args.model} ...")
    # Select dtype based on CUDA availability
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    
    try:
        model = AutoModel.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    except Exception as e:
        print(f"[FATAL] Failed to load model: {e}")
        sys.exit(1)

    generation_config = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=(args.temperature > 0),
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=1.05,  # Slight repetition penalty
    )

    # --- Scan Files ---
    files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in (".md", ".mmd")])
    if not files:
        print(f"[ERROR] No Markdown files found in directory '{in_dir}' (supports .md/.mmd)")
        return

    print(f"[INFO] Target Language: {lang_cfg['name']} ({args.lang})")
    print(f"[INFO] Found {len(files)} files. Output to: {out_dir.resolve()}")

    # --- Processing Loop ---
    success_count = 0
    for f in files:
        out_path = out_dir / (f.stem + ext)
        
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {out_path.name} exists")
            continue

        en_md = f.read_text(encoding="utf-8")
        
        try:
            # Call translation function with lang_code
            target_md, full_text = translate_one(model, tokenizer, en_md, generation_config, args.lang)

            # Write result
            with open(out_path, "w", encoding="utf-8") as fout:
                fout.write(target_md)

            # Save full output (optional)
            if args.save_full:
                raw_path = out_dir / (f.stem + ".full.txt")
                raw_path.write_text(full_text, encoding="utf-8")

            print(f"[OK] {f.name} -> {out_path.name}")
            success_count += 1
            
        except Exception as e:
            err_path = out_dir / (f.stem + ".error.txt")
            err_path.write_text(str(e), encoding="utf-8")
            print(f"[ERR] {f.name} -> {err_path.name} | Error: {e}")

    print(f"\n--- Done! Successfully processed: {success_count} / {len(files)} ---")

if __name__ == "__main__":
    main()