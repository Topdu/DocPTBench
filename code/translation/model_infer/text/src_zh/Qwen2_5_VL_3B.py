#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
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

# ========== 3. Message Construction & Parsing ==========

def build_messages_for_text(md_text: str, lang_code: str):
    cfg = LANG_CONFIG[lang_code]
    
    system_content = SYSTEM_PROMPT_TEMPLATE.format(
        lang_name=cfg["name"],
        lang_tag=cfg["tag"],
        style_note=cfg["style_note"]
    )
    
    user_content = (
        "\n\n<chinese_md>\n" + md_text + "\n</chinese_md>\n" + 
        USER_INSTRUCTION_TEMPLATE.format(lang_name=cfg["name"])
    )
    
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

def parse_target_block(text: str, lang_code: str):
    tag = LANG_CONFIG[lang_code]["tag"]
    # Dynamically build regex; re.S allows dot to match newlines
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    m = pattern.search(text)
    return m.group(1).strip() if m else text.strip()

# ========== 4. Translation Function ==========
def translate_one(model, processor, text: str, lang_code: str, device: torch.device,
                  max_new_tokens: int = 4096, temperature: float = 0.0, top_p: float = 1.0):
    
    messages = build_messages_for_text(text, lang_code)
    
    # Qwen2.5-VL specific chat template processing
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Image/video input processing (image_inputs is empty for pure text, but processor needs structure)
    inputs = processor(text=[prompt], return_tensors="pt").to(device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=(temperature > 0),
        repetition_penalty=1.05,
    )
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    # Decode output (remove prompt part)
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    full_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    target_md = parse_target_block(full_text, lang_code)
    return target_md, full_text

# ========== 5. Main Execution ==========
def main():
    parser = argparse.ArgumentParser(description="Translate Chinese Markdown files to Multi-Lang with Qwen2.5-VL")
    parser.add_argument("--dir", required=True, help="Input Chinese Markdown directory")
    
    parser.add_argument("--lang", default="en", choices=["en", "de", "fr", "ru"], help="Target language (default: en)")
    
    parser.add_argument("--out_dir", default=None, help="Output directory (default: auto-generated based on language)")
    parser.add_argument("--ext", default=None, help="Output file extension")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--flash_attn", action="store_true", help="Enable flash_attention_2")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Model path or HuggingFace ID")
    parser.add_argument("--save_full", action="store_true", help="Save full raw model output as .full.txt")
    args = parser.parse_args()

    lang_cfg = LANG_CONFIG[args.lang]
    
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(f"translations_{args.lang}")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ext = args.ext if args.ext else lang_cfg["ext"]

    in_dir = Path(args.dir)
    md_files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in (".md", ".mmd") and not p.name.endswith(".full.txt")])
    if not md_files:
        raise ValueError("No Markdown files found. Supported extensions: .md .mmd")

    print(f"[INFO] Loading model: {args.model_path} ...")
    model_kwargs = {"torch_dtype": "auto", "device_map": "auto"}
    if args.flash_attn:
        model_kwargs.update({"attn_implementation": "flash_attention_2"})
    
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs)
        processor = AutoProcessor.from_pretrained(args.model_path)
    except Exception as e:
        print(f"[ERR] Failed to load model: {e}")
        sys.exit(1)

    device = model.device if hasattr(model, "device") else ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Target Language: {lang_cfg['name']} ({args.lang})")
    print(f"[INFO] Found {len(md_files)} files. Output to: {out_dir.resolve()}")

    for md_file in md_files:
        out_path = out_dir / (md_file.stem + ext)
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {out_path.name} exists. Use --overwrite to regenerate.")
            continue

        text = md_file.read_text(encoding="utf-8")
        
        try:
            target_md, full_text = translate_one(
                model, processor, text, args.lang, device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(target_md)

            if args.save_full:
                raw_path = out_dir / (md_file.stem + ".full.txt")
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(full_text)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"[OK] {md_file.name} -> {out_path.name}")
            
        except Exception as e:
            print(f"[ERR] Failed to translate {md_file.name}: {e}")

if __name__ == "__main__":
    main()