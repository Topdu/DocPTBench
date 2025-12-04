#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor 

# ==============================================================================
# 1. CONFIG & PROMPTS (Multilingual Configuration)
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

def build_messages_for_text(md_text: str, lang_code: str):
    """Build message body based on target language."""
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

def parse_target_block(text: str, lang_code: str):
    """Dynamically parse XML block content based on language config."""
    tag = LANG_CONFIG[lang_code]["tag"]
    # re.S allows . to match newlines
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    m = pattern.search(text)
    return m.group(1).strip() if m else text.strip()

# ==============================================================================
# 3. INFERENCE LOGIC
# ==============================================================================

def translate_one(model, processor, text: str, device: torch.device, lang_code: str,
                  max_new_tokens: int = 4096, temperature: float = 0.0, top_p: float = 1.0):
    """
    Execute single file translation task.
    """
    # 1. Build Prompt
    messages = build_messages_for_text(text, lang_code)
    
    # 2. Preprocess
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], return_tensors="pt").to(device)

    # 3. Generation parameters
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=(temperature > 0),
        repetition_penalty=1.05,
    )

    # 4. Inference
    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    # 5. Decode
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    full_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # 6. Parse language-specific block
    target_md = parse_target_block(full_text, lang_code)
    
    return target_md, full_text

# ==============================================================================
# 4. MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Translate English Markdown files to Multi-Language with Qwen Local Model")
    
    # Core paths
    parser.add_argument("--dir", required=True, help="Input English Markdown directory")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-4B-Instruct", help="Local model path or HuggingFace ID")
    
    # Language options
    parser.add_argument("--lang", default="zh", choices=["zh", "de", "fr", "ru"], 
                        help="Target language: zh (Chinese), de (German), fr (French), ru (Russian). Default: zh")
    
    # Output control
    parser.add_argument("--out_dir", default=None, help="Output directory (default: auto-generated based on language)")
    parser.add_argument("--ext", default=None, help="Output file extension (default: auto-generated based on language)")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files")
    parser.add_argument("--save_full", action="store_true", help="Save full raw model output as .full.txt")

    # Model parameters
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--flash_attn", action="store_true", help="Enable flash_attention_2")

    args = parser.parse_args()

    # --- Initialize Config ---
    lang_cfg = LANG_CONFIG[args.lang]
    
    # Determine output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(f"translations_qwen_{args.lang}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine extension
    ext = args.ext if args.ext else lang_cfg["ext"]

    in_dir = Path(args.dir)
    
    # Iterate English Markdown files (exclude .full.txt)
    md_files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in (".md", ".mmd") and not p.name.endswith(".full.txt")])
    if not md_files:
        raise ValueError(f"No Markdown files found in {in_dir}.")

    print(f"[INFO] Target Language: {lang_cfg['name']} ({args.lang})")
    print(f"[INFO] Model Path: {args.model_path}")
    print(f"[INFO] Found {len(md_files)} files. Output to: {out_dir.resolve()}")

    try:

        from transformers import Qwen3VLForConditionalGeneration as ModelClass
    except ImportError:
        print("[WARN] Qwen3VLForConditionalGeneration not found, please check your transformers version.")

    model_kwargs = {"torch_dtype": "auto", "device_map": "auto"}
    if args.flash_attn:
        model_kwargs.update({"attn_implementation": "flash_attention_2"})
    
    print(f"[INFO] Loading model from {args.model_path} ...")
    model = ModelClass.from_pretrained(args.model_path, **model_kwargs)
    processor = AutoProcessor.from_pretrained(args.model_path)

    device = model.device if hasattr(model, "device") else ("cuda" if torch.cuda.is_available() else "cpu")

    # --- Processing Loop ---
    success_count = 0
    skipped_count = 0

    for md_file in md_files:
        out_path = out_dir / (md_file.stem + ext)
        
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {out_path.name} exists.")
            skipped_count += 1
            continue

        try:
            text = md_file.read_text(encoding="utf-8")
            
            # Call translation function
            target_md, full_text = translate_one(
                model, processor, text, device,
                lang_code=args.lang,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )

            # Save translation
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(target_md)

            # Optional: Save full raw output
            if args.save_full:
                raw_path = out_dir / (md_file.stem + ".full.txt")
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(full_text)

            print(f"[OK] {md_file.name} -> {out_path.name}")
            success_count += 1

        except Exception as e:
            print(f"[ERROR] Failed to process {md_file.name}: {e}")

        # VRAM Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n--- Processing Complete ---")
    print(f"Success: {success_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Output Directory: {out_dir.resolve()}")

if __name__ == "__main__":
    main()