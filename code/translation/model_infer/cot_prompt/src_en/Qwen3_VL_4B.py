#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
from pathlib import Path
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import re

# ========== 1. Multi-language Configuration ==========
LANG_CONFIG = {
    "zh": {
        "name": "Simplified Chinese",
        "tag": "chinese_md",
        "ext": ".zh.mmd",
        "style_note": "Fluent, professional zh-CN for narrative text; use Chinese punctuation in prose."
    },
    "de": {
        "name": "German",
        "tag": "german_md",
        "ext": ".de.mmd",
        "style_note": "Fluent, professional German; use standard German punctuation."
    },
    "fr": {
        "name": "French",
        "tag": "french_md",
        "ext": ".fr.mmd",
        "style_note": "Fluent, professional French; use standard French punctuation."
    },
    "ru": {
        "name": "Russian",
        "tag": "russian_md",
        "ext": ".ru.mmd",
        "style_note": "Fluent, professional Russian; use standard Russian punctuation."
    }
}

# ========== 2. Dynamic Prompt Template ==========
SYSTEM_PROMPT_TEMPLATE = """You are a professional document translator.

Task: First OCR and reconstruct the readable English Markdown from the whole document image. 
Then produce a high-fidelity {lang_name} Markdown translation.

Output format (STRICT):
Return EXACTLY two fenced blocks, in this order, and nothing else:
<english_md>
...the English Markdown reconstruction...
</english_md>
<{lang_tag}>
...the {lang_name} Markdown translation...
</{lang_tag}>

Formulas (LaTeX ONLY):
- Detect ANY formula/equation and represent it in LaTeX.
- Wrap EVERY formula in display math delimiters: $$ ... $$ (use display math even if the source looked inline).
- Inside $$...$$, DO NOT alter symbols/operators/variables/numbers/units.
- In <english_md>, keep any human words in formulas as English (e.g., \\text{{Mean}}).
- In <{lang_tag}>, translate ONLY human words inside formulas with \\text{{...}} (e.g., translate terms to target language); keep math intact.

Tables (keep Markdown):
- Keep ALL tables as Markdown tables with the same column count, pipes `|`, alignment markers, and row structure.
- Translate only narrative text in cells; do NOT convert tables to LaTeX.

Other structure (preserve):
- Paragraphs, blank lines, lists (ordered/unordered), task lists (- [ ] / - [x]),
  blockquotes (>), horizontal rules, footnotes/references, anchors/IDs, figure numbers and captions.
- Hyperlinks/images: translate visible text (alt/title) ONLY; keep URLs and reference labels unchanged.

Do-not-translate / protect:
- Code: fenced blocks ```...``` (with language) and inline `...` (copy as-is).
- URLs/emails/paths/domains/ids/hashes and link reference labels ([ref-id]:).
- Stable product/brand/model names commonly written in English (e.g., “GitHub”, “ResNet-50”).
- Template variables/placeholders: {{var}}, {{{{handlebars}}}}, <PLACEHOLDER>, %s, {{0}}, $ENV_VAR.

Target Language Style:
- {style_note}
- NEVER change punctuation inside code, LaTeX math, or URLs.

Quality checks BEFORE finalizing:
- <english_md> is English-only, with all formulas as $$...$$ LaTeX.
- <{lang_tag}> translates narrative text; formulas use the same $$...$$ LaTeX with only \\text{{...}} translated.
- ANY formula/equation is represened in LaTeX.
- No added/omitted content; structure renders correctly.
- Output ONLY the two blocks. No explanations or reasoning."""

# ========== 3. Helper Functions ==========

def build_messages_for_single_image(image_path: str, lang_code: str):
    cfg = LANG_CONFIG[lang_code]
    
    # Fill System Prompt
    system_content = SYSTEM_PROMPT_TEMPLATE.format(
        lang_name=cfg["name"],
        lang_tag=cfg["tag"],
        style_note=cfg["style_note"]
    )
    
    # Fill User Instruction
    user_instruction = (
        f"Reconstruct and translate the document image as specified. "
        f"Return ONLY the two blocks: <english_md>...</english_md> and <{cfg['tag']}>...</{cfg['tag']}>."
    )

    return [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_instruction},
            ],
        },
    ]

def parse_dual_blocks(text: str, lang_code: str):
    tag = LANG_CONFIG[lang_code]["tag"]
    
    # Dynamic regex
    target_re = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    en_re = re.compile(r"<english_md>\s*(.*?)\s*</english_md>", re.S | re.I)

    zh = target_re.search(text)
    en = en_re.search(text)
    
    # Fallback: if no tag found, assume full text is target
    target_md = zh.group(1).strip() if zh else text.strip()
    en_md = en.group(1).strip() if en else ""
    
    # Prevent misjudgment: if text exists but target_md is empty
    if not target_md and not en_md and text.strip():
        target_md = text.strip()

    return en_md, target_md

def translate_one(model, processor, image_path: str, device: torch.device, lang_code: str,
                  max_new_tokens: int = 3072, temperature: float = 0.0, top_p: float = 1.0):
    
    messages = build_messages_for_single_image(image_path, lang_code)
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=(temperature > 0),
        repetition_penalty=1.05,
    )
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    # Keep only new generated tokens
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    full_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    en_md, target_md = parse_dual_blocks(full_text, lang_code)
    return en_md, target_md, full_text

# ========== Main Process ==========
def main():
    parser = argparse.ArgumentParser(description="Per-image doc translation (Multi-Lang) with Qwen3-VL")
    
    # Core Arguments
    parser.add_argument("--dir", required=True, help="Input image folder")
    parser.add_argument("--lang", default="zh", choices=["zh", "de", "fr", "ru"], help="Target language: zh, de, fr, ru")
    
    # Path Arguments
    parser.add_argument("--out_dir", default=None, help="Output directory (defaults to ./results_{lang})")
    parser.add_argument("--english_out_dir", default=None, help="English reconstruction output directory")
    parser.add_argument("--raw_dir", default=None, help="Full raw output directory")
    
    # Feature Switches
    parser.add_argument("--ext", default=None, help="Output extension (defaults based on language)")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing files")
    parser.add_argument("--save_english", action="store_true", help="Save English reconstruction block")
    parser.add_argument("--save_full", action="store_true", help="Save full raw output")
    
    # Model Parameters
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--min_pixels", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--flash_attn", action="store_true", help="Enable flash_attention_2")
    # Default changed to a generic public ID or placeholder to avoid local paths
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="HuggingFace model path or ID")

    args = parser.parse_args()

    # 1. Path Setup
    lang_cfg = LANG_CONFIG[args.lang]
    
    # Determine Output Directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(f"results_{args.lang}")
    
    out_dir.mkdir(parents=True, exist_ok=True)

    # English Output Directory
    if args.save_english:
        if args.english_out_dir:
            english_out_dir = Path(args.english_out_dir)
        else:
            english_out_dir = out_dir / "english_reconstruction"
        english_out_dir.mkdir(parents=True, exist_ok=True)
    else:
        english_out_dir = None

    # Raw Output Directory
    if args.save_full:
        if args.raw_dir:
            raw_dir = Path(args.raw_dir)
        else:
            raw_dir = out_dir / "raw_responses"
        raw_dir.mkdir(parents=True, exist_ok=True)
    else:
        raw_dir = None

    # Extension
    ext = args.ext if args.ext else lang_cfg["ext"]

    # 2. Collect Images
    in_dir = Path(args.dir)
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    
    if not in_dir.exists():
        raise ValueError(f"Input directory does not exist: {in_dir}")

    image_paths = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])
    if not image_paths:
        raise ValueError("No images found in directory. Supported formats: .png .jpg .jpeg .webp .bmp")

    # 3. Load Model
    print(f"[INFO] Loading Model from: {args.model_path}")
    model_kwargs = {"torch_dtype": "auto", "device_map": "auto"}
    if args.flash_attn:
        model_kwargs.update({"attn_implementation": "flash_attention_2"})
    
    # Note: Ensure Qwen3VLForConditionalGeneration is available in your environment
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs)

    proc_kwargs = {}
    if args.min_pixels and args.max_pixels:
        proc_kwargs.update({"min_pixels": args.min_pixels, "max_pixels": args.max_pixels})
    processor = AutoProcessor.from_pretrained(args.model_path, **proc_kwargs)

    device = model.device if hasattr(model, "device") else ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Target Language: {lang_cfg['name']} ({args.lang})")
    print(f"[INFO] Output Dir: {out_dir.resolve()}")
    print(f"[INFO] Found {len(image_paths)} images.")

    # 4. Processing Loop
    for img in image_paths:
        out_path = out_dir / (img.stem + ext)
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {out_path.name} exists.")
            continue

        try:
            en_md, target_md, full_text = translate_one(
                model, processor, str(img), device, args.lang,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
        except Exception as e:
            print(f"[ERR] {img.name}: {e}")
            continue

        # Write target language file
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(target_md)

        # Write English reconstruction file
        if args.save_english and english_out_dir and en_md:
            en_path = english_out_dir / (img.stem + "_en.md")
            with open(en_path, "w", encoding="utf-8") as f:
                f.write(en_md)

        # Write full raw output
        if args.save_full and raw_dir:
            raw_path = raw_dir / (img.stem + ".full.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(full_text)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[OK] {img.name} -> {out_path.name}"
              + (f" (+en)" if (args.save_english and en_md) else "")
              + (f" (+raw)" if args.save_full else ""))

if __name__ == "__main__":
    main()