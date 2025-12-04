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
        "ext": ".zh.md",
        "style_note": "Use fluent, professional Simplified Chinese with standard punctuation."
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

Task: Translate the content of the English document image directly into high-fidelity {lang_name} Markdown.

Output format (STRICT):
Return EXACTLY one fenced block, and nothing else:
<{lang_tag}>
...the {lang_name} Markdown translation...
</{lang_tag}>

Translation Rules:
- Formulas (LaTeX ONLY): Detect and represent any formula/equation in LaTeX, wrapped in display math delimiters: $$ ... $$. Inside the delimiters, translate only human words with \\text{{...}} (e.g., \\text{{Mean}} → \\text{{translated_term}}).
- Tables (Markdown): Preserve Markdown tables. Translate only narrative text in cells.
- Structure: Preserve all structural elements like paragraphs, lists, blockquotes, etc.
- Do-not-translate: Keep code (`...`), URLs, emails, and stable English names (e.g., "GitHub") as is.
- Target Style: {style_note}
- Quality Check: Ensure the output is a single, complete, and accurate {lang_name} Markdown block. Do not add any explanations or extra content.
"""

USER_INSTRUCTION_TEMPLATE = (
    "Translate the document image directly into {lang_name} Markdown. "
    "Return ONLY the <{lang_tag}>...</{lang_tag}> block as specified."
)

# ========== 3. Dynamic Parsing Logic ==========
def parse_target_block(text: str, lang_code: str) -> str:
    """Dynamically parse XML block based on language configuration."""
    tag = LANG_CONFIG[lang_code]["tag"]
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    
    m = pattern.search(text)
    return m.group(1).strip() if m else text.strip()

# ========== 4. Build Messages ==========
def build_messages_for_single_image(image_path: str, lang_code: str):
    cfg = LANG_CONFIG[lang_code]
    
    system_content = SYSTEM_PROMPT_TEMPLATE.format(
        lang_name=cfg["name"],
        lang_tag=cfg["tag"],
        style_note=cfg["style_note"]
    )
    
    user_content = USER_INSTRUCTION_TEMPLATE.format(
        lang_name=cfg["name"],
        lang_tag=cfg["tag"]
    )

    return [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_content},
            ],
        },
    ]

# ========== 5. Inference Logic ==========
def translate_one(model, processor, image_path: str, device: torch.device, lang_code: str,
                  max_new_tokens: int = 4096, temperature: float = 0.0, top_p: float = 1.0) -> str:
    
    # Build language-specific messages
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

    trimmed_ids = generated_ids[0, len(inputs.input_ids[0]):]
    full_text = processor.decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # Parse language-specific tags
    target_md = parse_target_block(full_text, lang_code)
    return target_md

# ========== Main Process ==========
def main():
    parser = argparse.ArgumentParser(description="Direct document translation (EN -> ZH/DE/FR/RU) with Qwen2.5-VL")
    
    # Core arguments
    parser.add_argument("--dir", required=True, help="Input image directory")
    parser.add_argument("--lang", default="zh", choices=["zh", "de", "fr", "ru"], help="Target language (default: zh)")
    parser.add_argument("--out_dir", default=None, help="Output directory (default: auto-generated based on language)")
    parser.add_argument("--ext", default=None, help="Output file extension")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files")
    
    # Model and generation parameters
    parser.add_argument("--model_path", type=str, required=True, help="Path to Qwen2.5-VL model")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--flash_attn", action="store_true", help="Enable flash_attention_2")
    
    args = parser.parse_args()

    # 1. Configuration Initialization
    lang_cfg = LANG_CONFIG[args.lang]
    
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(f"translations_qwen25vl_direct_en2{args.lang}")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = args.ext if args.ext else lang_cfg["ext"]

    # 2. Image Collection
    in_dir = Path(args.dir)
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    image_paths = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])
    if not image_paths:
        raise ValueError(f"No supported image files found in directory '{in_dir}'.")

    # 3. Model Loading
    model_kwargs = {"torch_dtype": "auto", "device_map": "auto"}
    if args.flash_attn:
        model_kwargs.update({"attn_implementation": "flash_attention_2"})
    
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs)

    except Exception as e:
        raise RuntimeError(f"Failed to load model. Ensure transformers version is correct: {e}")

    processor = AutoProcessor.from_pretrained(args.model_path)
    device = model.device

    print(f"[INFO] Model: {args.model_path}")
    print(f"[INFO] Target Language: {lang_cfg['name']} ({args.lang})")
    print(f"[INFO] Found {len(image_paths)} images. Output: {out_dir.resolve()}")

    # 4. Processing Loop
    for img_path in image_paths:
        out_path = out_dir / (img_path.stem + ext)
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] File exists: {out_path.name}")
            continue

        try:
            target_md = translate_one(
                model, processor, str(img_path), device,
                lang_code=args.lang,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
            
            out_path.write_text(target_md, encoding="utf-8")
            print(f"[OK] {img_path.name} -> {out_path.name}")

        except Exception as e:
            err_path = out_dir / (img_path.stem + ".error.txt")
            err_path.write_text(f"Failed to process '{img_path.name}':\n{e}\n", encoding="utf-8")
            print(f"[ERR] {img_path.name} -> Details in {err_path.name}")

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()