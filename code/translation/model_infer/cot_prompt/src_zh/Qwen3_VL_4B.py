import os
import argparse
import torch
import re
from pathlib import Path
from typing import Tuple, List, Dict, Any

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ==============================================================================
# 1. LANGUAGE CONFIGURATION & TEMPLATES
# ==============================================================================

LANG_CONFIG = {
    "en": {
        "name": "English",
        "tag": "english_md",
        "math_example": "Mean",
        "ext": ".en.md"
    },
    "de": {
        "name": "German",
        "tag": "german_md",
        "math_example": "Mittelwert",
        "ext": ".de.md"
    },
    "fr": {
        "name": "French",
        "tag": "french_md",
        "math_example": "Moyenne",
        "ext": ".fr.md"
    },
    "ru": {
        "name": "Russian",
        "tag": "russian_md",
        "math_example": "Среднее",
        "ext": ".ru.md"
    }
}

SYSTEM_PROMPT_TEMPLATE = """You are a professional document translator.

Task: First OCR and reconstruct the readable Chinese Markdown from the whole document image.
Then produce a high-fidelity __LANG_NAME__ Markdown translation.

Output format (STRICT):
Return EXACTLY two fenced blocks, in this order, and nothing else:
<chinese_md>
...the Chinese Markdown reconstruction...
</chinese_md>
<__LANG_TAG__>
...the __LANG_NAME__ Markdown translation...
</__LANG_TAG__>

Formulas (LaTeX ONLY):
- Detect ANY formula/equation and represent it in LaTeX.
- Wrap EVERY formula in display math delimiters: $$ ... $$ (use display math even if the source looked inline).
- Inside $$...$$, DO NOT alter symbols/operators/variables/numbers/units.
- In <chinese_md>, keep any human words in formulas as Chinese (e.g., \\text{均值}).
- In <__LANG_TAG__>, translate ONLY human words inside formulas with \\text{...} (e.g., \\text{__MATH_EXAMPLE__}); keep math intact.

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
- Template variables/placeholders: {var}, {{handlebars}}, <PLACEHOLDER>, %s, {0}, $ENV_VAR.

__LANG_NAME__ style:
- Fluent, professional __LANG_NAME__ for narrative text; use standard __LANG_NAME__ punctuation in prose.
- NEVER change punctuation inside code, LaTeX math, or URLs.

Quality checks BEFORE finalizing:
- <chinese_md> is Chinese-only, with all formulas as $$...$$ LaTeX.
- <__LANG_TAG__> translates narrative text; formulas use the same $$...$$ LaTeX with only \\text{...} translated.
- ANY formula/equation is represented in LaTeX.
- No added/omitted content; structure renders correctly.
- Output ONLY the two blocks. No explanations or reasoning."""

USER_INSTRUCTION_TEMPLATE = (
    "Reconstruct and translate the document image as specified. "
    "Return ONLY the two blocks: <chinese_md>...</chinese_md> and <__LANG_TAG__>...</__LANG_TAG__>."
)

# ==============================================================================
# 2. PARSING & PROMPT HELPERS
# ==============================================================================

def get_prompts(lang_code: str) -> Tuple[str, str]:
    """Generates the system and user prompts based on the target language."""
    cfg = LANG_CONFIG[lang_code]
    
    sys_p = SYSTEM_PROMPT_TEMPLATE \
        .replace("__LANG_NAME__", cfg["name"]) \
        .replace("__LANG_TAG__", cfg["tag"]) \
        .replace("__MATH_EXAMPLE__", cfg["math_example"])
        
    user_p = USER_INSTRUCTION_TEMPLATE \
        .replace("__LANG_TAG__", cfg["tag"])
        
    return sys_p, user_p

def parse_response(text: str, lang_code: str) -> Tuple[str, str]:
    """
    Dynamically parses the response based on the target language tag.
    Returns: (target_md, zh_md)
    """
    tag = LANG_CONFIG[lang_code]["tag"]
    
    target_re = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    zh_re = re.compile(r"<chinese_md>\s*(.*?)\s*</chinese_md>", re.S | re.I)

    target_md, zh_md = "", ""

    # 1. Try strict block match
    m1 = target_re.search(text)
    if m1:
        target_md = m1.group(1).strip()
    else:
        # 2. Try loose match (end of text)
        m2 = re.search(rf"<{tag}>\s*(.*)", text, re.S | re.I)
        if m2:
            target_md = m2.group(1).strip()
        else:
            # 3. Fallback: everything after chinese block
            m3 = re.search(r"</chinese_md>\s*(.*)", text, re.S | re.I)
            if m3:
                target_md = m3.group(1).strip()

    m_zh = zh_re.search(text)
    if m_zh:
        zh_md = m_zh.group(1).strip()

    # Fallback if nothing matched
    if not target_md and not zh_md:
        zh_md = text.strip()

    return target_md, zh_md

def build_messages(image_path: str, sys_prompt: str, user_instruction: str):
    return [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_instruction},
            ],
        },
    ]

# ==============================================================================
# 3. INFERENCE ENGINE
# ==============================================================================

def run_inference(
    model, 
    processor, 
    image_path: str, 
    sys_prompt: str, 
    user_instruction: str,
    device: torch.device,
    args: argparse.Namespace
) -> str:
    
    messages = build_messages(image_path, sys_prompt, user_instruction)
    
    # Prepare text inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Prepare vision inputs (Qwen-VL specific)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Generation args
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=(args.temperature > 0),
        repetition_penalty=1.05, # Slight penalty to prevent loops
    )

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    # Decode
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    full_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return full_text

# ==============================================================================
# 4. MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL Local Document Translation")
    
    # --- Core ---
    parser.add_argument("--dir", required=True, help="Input directory containing images")
    parser.add_argument("--lang", default="en", choices=["en", "de", "fr", "ru"], 
                        help="Target language: en, de, fr, ru")
    parser.add_argument("--out_dir", default=None, help="Output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    # --- Model ---
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-4B-Instruct", 
                        help="Local path or HF Hub ID for the model")
    parser.add_argument("--flash_attn", action="store_true", help="Enable Flash Attention 2")
    
    # --- Generation Params ---
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    # --- Outputs ---
    parser.add_argument("--save_chinese", action="store_true", help="Save Chinese OCR text")
    parser.add_argument("--save_full", action="store_true", help="Save raw model output")
    
    args = parser.parse_args()

    # 1. Setup Paths
    lang_cfg = LANG_CONFIG[args.lang]
    target_ext = lang_cfg["ext"]
    
    in_dir = Path(args.dir)
    default_out_name = f"translations_qwen_{args.lang}"
    out_dir = Path(args.out_dir) if args.out_dir else Path(default_out_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    zh_dir = out_dir / "zh" if args.save_chinese else None
    if zh_dir: zh_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = out_dir / "raw" if args.save_full else None
    if raw_dir: raw_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Model
    print(f"[INFO] Loading model: {args.model_path} ...")
    
    model_kwargs = {
        "torch_dtype": "auto", 
        "device_map": "auto",
        "trust_remote_code": True
    }
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs)
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception as e:
        print(f"[ERR] Failed to load model: {e}")
        return

    device = model.device
    print(f"[INFO] Model loaded on {device}")

    # 3. Process Images
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    image_paths = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in valid_exts])
    
    if not image_paths:
        print(f"[WARN] No images found in {in_dir}")
        return

    print(f"[INFO] Found {len(image_paths)} images. Target: {lang_cfg['name']}")

    sys_prompt, user_instruction = get_prompts(args.lang)

    for i, img in enumerate(image_paths):
        out_path = out_dir / (img.stem + target_ext)
        
        if out_path.exists() and not args.overwrite:
            print(f"[{i+1}/{len(image_paths)}] [SKIP] {img.name}")
            continue

        print(f"[{i+1}/{len(image_paths)}] Processing {img.name} ...", end="", flush=True)

        try:
            # Inference
            full_text = run_inference(
                model, processor, str(img), 
                sys_prompt, user_instruction, 
                device, args
            )

            # Parse
            target_md, zh_md = parse_response(full_text, args.lang)

            # Save Target
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(target_md)

            # Save Optional
            if zh_dir and zh_md:
                zh_path = zh_dir / (img.stem + ".md")
                with open(zh_path, "w", encoding="utf-8") as f:
                    f.write(zh_md)
            
            if raw_dir:
                raw_path = raw_dir / (img.stem + ".full.txt")
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(full_text)

            print(" Done.")

        except Exception as e:
            print(f" Error: {e}")
        
        # Cleanup VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[INFO] All tasks completed. Output: {out_dir.resolve()}")

if __name__ == "__main__":
    main()