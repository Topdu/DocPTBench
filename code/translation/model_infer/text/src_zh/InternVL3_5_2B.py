#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer
import re
import sys

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

# ========== 2. Dynamic Prompt Template ==========
SYSTEM_PROMPT_TEMPLATE = """You are a professional document translator.

Task: Translate the provided Chinese Markdown into high-fidelity {lang_name} Markdown.

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

USER_INSTRUCTION_TEMPLATE = (
    "Translate the given Chinese Markdown into {lang_name} Markdown, following the rules."
)

# ========== 3. Parsing Logic ==========
def parse_target_block(text: str, lang_code: str) -> str:
    """Dynamically parse XML block based on language config."""
    tag = LANG_CONFIG[lang_code]["tag"]
    # Dynamically build regex, re.S allows dot to match newlines
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    
    m = pattern.search(text)
    return m.group(1).strip() if m else text.strip()

# ========== 4. Translation Function ==========
@torch.inference_mode()
def translate_one(model, tokenizer, zh_md: str, lang_code: str, generation_config: dict):
    cfg = LANG_CONFIG[lang_code]
    
    # Fill Prompt Template
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        lang_name=cfg["name"],
        lang_tag=cfg["tag"],
        style_note=cfg["style_note"]
    )
    
    user_instruction = USER_INSTRUCTION_TEMPLATE.format(
        lang_name=cfg["name"]
    )
    
    # Concatenate input (System + Input + Instruction)
    question = (
        system_prompt.strip()
        + "\n\n<chinese_md>\n" + zh_md + "\n</chinese_md>\n"
        + user_instruction
    )
    
    # Call model
    output = model.chat(tokenizer, None, question, generation_config).strip()
    
    # Parse output
    target_md = parse_target_block(output, lang_code)
    return target_md, output

# ========== 5. Main Execution ==========
def main():
    parser = argparse.ArgumentParser(description="Translate Chinese Markdown -> Multi-Lang with InternVL3_5")
    
    # Default model path updated to HuggingFace ID or generic placeholder
    parser.add_argument("--model", default="OpenGVLab/InternVL3_5-2B", help="HuggingFace model path/id")
    parser.add_argument("--dir", required=True, help="Input Chinese Markdown directory")
    
    # Language selection argument
    parser.add_argument("--lang", default="en", choices=["en", "de", "fr", "ru"], help="Target language (default: en)")
    
    parser.add_argument("--out_dir", default=None, help="Output directory (default: auto-generated based on language)")
    parser.add_argument("--ext", default=None, help="Output file extension")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--save_full", action="store_true", help="Save full model output")
    args = parser.parse_args()

    # Initialize configuration
    lang_cfg = LANG_CONFIG[args.lang]
    
    # Automatically determine output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(f"./translations_{args.lang}")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine extension
    ext = args.ext if args.ext else lang_cfg["ext"]

    # Load model
    print(f"[INFO] Loading model: {args.model} ...")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    
    try:
        model = AutoModel.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True,     # Preserving original code feature
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    except Exception as e:
        print(f"[ERR] Failed to load model: {e}")
        sys.exit(1)

    generation_config = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=(args.temperature > 0),
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=1.05,
    )

    in_dir = Path(args.dir)
    files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in (".md", ".mmd")])
    if not files:
        raise ValueError("No Markdown files found in directory (supports .md/.mmd)")

    print(f"[INFO] Target Language: {lang_cfg['name']} ({args.lang})")
    print(f"[INFO] Found {len(files)} Markdown files. Output to {out_dir.resolve()}")

    for f in files:
        out_path = out_dir / (f.stem + ext)
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {out_path.name} exists")
            continue

        zh_md = f.read_text(encoding="utf-8")
        try:
            # Pass lang_code argument
            target_md, full_text = translate_one(model, tokenizer, zh_md, args.lang, generation_config)

            with open(out_path, "w", encoding="utf-8") as fout:
                fout.write(target_md)

            if args.save_full:
                raw_path = out_dir / (f.stem + ".full.txt")
                raw_path.write_text(full_text, encoding="utf-8")

            print(f"[OK] {f.name} -> {out_path.name}")
        except Exception as e:
            err_path = out_dir / (f.stem + ".error.txt")
            err_path.write_text(str(e), encoding="utf-8")
            print(f"[ERR] {f.name} -> {err_path.name} : {e}")

if __name__ == "__main__":
    main()