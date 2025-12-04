#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import base64
import mimetypes
import re
import io
import concurrent.futures
from pathlib import Path
from typing import Tuple, List, Dict, Any

from openai import OpenAI
from PIL import Image
from tqdm import tqdm

# ==============================================================================
# 1. Multi-language Configuration & Dynamic PROMPT
# ==============================================================================

LANG_CONFIG = {
    "zh": {
        "name": "Simplified Chinese (zh-CN)",
        "tag": "chinese_md",
        "ext": ".zh.md",
        "style_note": "Fluent, professional zh-CN; use standard Chinese punctuation."
    },
    "de": {
        "name": "German",
        "tag": "german_md",
        "ext": ".de.md",
        "style_note": "Fluent, professional German; use standard German punctuation."
    },
    "fr": {
        "name": "French",
        "tag": "french_md",
        "ext": ".fr.md",
        "style_note": "Fluent, professional French; use standard French punctuation."
    },
    "ru": {
        "name": "Russian",
        "tag": "russian_md",
        "ext": ".ru.md",
        "style_note": "Fluent, professional Russian; use standard Russian punctuation."
    }
}

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
- In <{lang_tag}>, translate ONLY human words inside formulas with \\text{{...}} (e.g., translate "Mean" to target language); keep math intact.

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
- ANY formula/equation is represented in LaTeX.
- No added/omitted content; structure renders correctly.
- Output ONLY the two blocks. No explanations or reasoning."""

USER_INSTRUCTION_TEMPLATE = (
    "Reconstruct and translate the document image as specified. "
    "Return ONLY the two blocks: <english_md>...</english_md> and <{lang_tag}>...</{lang_tag}>."
)

def get_prompts(lang_code: str) -> Tuple[str, str]:
    cfg = LANG_CONFIG[lang_code]
    sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        lang_name=cfg["name"],
        lang_tag=cfg["tag"],
        style_note=cfg["style_note"]
    )
    user_prompt = USER_INSTRUCTION_TEMPLATE.format(
        lang_tag=cfg["tag"]
    )
    return sys_prompt, user_prompt

def parse_dual_blocks(text: str, lang_code: str) -> Tuple[str, str]:
    """Parse the English and target language parts from the full API response."""
    tag = LANG_CONFIG[lang_code]["tag"]
    
    # Dynamic regex
    target_re = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    en_re = re.compile(r"<english_md>\s*(.*?)\s*</english_md>", re.S | re.I)

    target_md, en_md = "", ""

    # 1. Parse target language part
    m_target = target_re.search(text)
    if m_target:
        target_md = m_target.group(1).strip()
    else:
        # Loose match: tag not closed
        m2 = re.search(rf"<{tag}>\s*(.*)", text, re.S | re.I)
        if m2:
            target_md = m2.group(1).strip()
        else:
            # Secondary fallback: after English block
            m3 = re.search(r"</english_md>\s*(.*)", text, re.S | re.I)
            if m3:
                target_md = m3.group(1).strip()

    # 2. Parse English part
    m_en = en_re.search(text)
    if m_en:
        en_md = m_en.group(1).strip()

    # 3. Fallback: if nothing matched, treat all text as English (OCR)
    if not en_md and not target_md:
        en_md = text.strip()

    return target_md, en_md

# ==============================================================================
# 2. IMAGE & API HELPERS
# ==============================================================================

def image_path_to_data_url_fallback(image_path: Path) -> str:
    """Original encoding method without preprocessing, used as backup."""
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type:
        mime_type = "application/octet-stream"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime_type};base64,{b64}"

def preprocess_and_encode_image(image_path: Path, max_dim: int = 2048, quality: int = 90) -> str:
    """
    Read image, preprocess (resize and compress), then encode to data URL.
    """
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")

            if max(img.size) > max_dim:
                img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            img_bytes = buffer.getvalue()

            b64_string = base64.b64encode(img_bytes).decode("ascii")
            return f"data:image/jpeg;base64,{b64_string}"
    except Exception as e:
        # Simple handling: fallback if preprocessing fails
        return image_path_to_data_url_fallback(image_path)

def build_messages(data_url: str, sys_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
    """Build message body for the vision model."""
    return [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

# ==============================================================================
# 3. CORE WORKER FUNCTION
# ==============================================================================

def process_image_translation(image_path: Path, client: OpenAI, args: argparse.Namespace) -> Tuple[str, Dict[str, Any]]:
    """
    Single image processing function, called concurrently.
    """
    try:
        # 1. Get Prompt for current language
        sys_prompt, user_prompt = get_prompts(args.lang)

        # 2. Image preprocessing and encoding (try preprocess first, fallback if fails)
        data_url = preprocess_and_encode_image(image_path)
        
        # 3. Build request messages
        messages = build_messages(data_url, sys_prompt, user_prompt)

        # 4. Call API
        resp = client.chat.completions.create(
            model=args.model,
            messages=messages,
            stream=False,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        full_text = "".join(choice.message.content or "" for choice in resp.choices)

        if not full_text:
            return ("error", {"message": f"API for '{image_path.name}' returned empty content."})

        # 5. Parse results (pass language code)
        target_md, en_md = parse_dual_blocks(full_text, args.lang)

        # 6. Return success status and all necessary data
        return ("success", {
            "en_md": en_md,
            "target_md": target_md,
            "full_text": full_text
        })

    except Exception as e:
        return ("error", {"message": f"An unexpected error occurred for '{image_path.name}': {e}"})


# ==============================================================================
# 4. MAIN ORCHESTRATION
# ==============================================================================

def main():
    """Main function handling argument parsing, task distribution, and result aggregation."""
    parser = argparse.ArgumentParser(
        description="Concurrent document image translation script (EN Source -> Multi-Target) via Gemini/OpenAI API",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- Core Arguments ---
    parser.add_argument("--dir", required=True, help="Input English image folder path.")
    parser.add_argument("--out_dir", default=None, help="Output directory (default: translations_gemini_{lang}).")
    parser.add_argument("--lang", default="zh", choices=["zh", "de", "fr", "ru"], 
                        help="Target language: zh (Chinese), de (German), fr (French), ru (Russian)")
    parser.add_argument("--overwrite", action="store_true", help="If true, overwrite existing output files.")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of threads for concurrent processing.")

    # --- API and Model Arguments ---
    parser.add_argument("--model", default="gemini-2.5-pro", help="Model name to use.")
    parser.add_argument("--api_key", type=str, default="", help="API Key. Defaults to built-in Key or env var JIEKOU_API_KEY.")
    parser.add_argument("--base_url", type=str, default="https://ai.liaobots.work/v1", help="API Base URL.")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Max tokens setting for API call.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for text generation.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling parameter.")

    # --- Optional Output Arguments ---
    parser.add_argument("--save_english", action="store_true", help="Also save OCR reconstructed English Markdown (.en.md).")
    parser.add_argument("--save_full", action="store_true", help="Also save full raw text returned by API (.full.txt).")

    args = parser.parse_args()

    # --- Check and Initialize ---
    api_key = args.api_key or os.getenv("JIEKOU_API_KEY")
    if not api_key:
        raise ValueError("API Key not provided. Please set JIEKOU_API_KEY env var or use --api_key argument.")

    # Get language config
    lang_cfg = LANG_CONFIG[args.lang]
    target_ext = lang_cfg["ext"]

    in_dir = Path(args.dir)
    
    # Determine output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(f"translations_gemini_{args.lang}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auxiliary output directories
    en_dir = out_dir / "en_ocr" if args.save_english else None
    if en_dir: en_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = out_dir / "raw" if args.save_full else None
    if raw_dir: raw_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    # --- Collect and Filter Tasks ---
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
    all_images = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])

    if not all_images:
        print(f"No supported images found in folder '{in_dir}'.")
        return

    tasks_to_process = []
    skipped_count = 0
    for img_path in all_images:
        # Check if target file exists
        target_path = out_dir / (img_path.stem + target_ext)
        if target_path.exists() and not args.overwrite:
            skipped_count += 1
        else:
            tasks_to_process.append(img_path)

    if not tasks_to_process:
        print("All images have been processed. Use --overwrite to re-process.")
        return

    print(f"=== Task Start ===")
    print(f"Target Language: {lang_cfg['name']} ({args.lang})")
    print(f"Model: {args.model}")
    print(f"Found {len(all_images)} images, skipped {skipped_count}.")
    print(f"Preparing to process {len(tasks_to_process)} new images, concurrency: {args.max_workers}...")

    # --- Concurrent Processing ---
    success_count = 0
    failed_count = 0
    
    # Use tqdm to show progress
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_path = {executor.submit(process_image_translation, img_path, client, args): img_path for img_path in tasks_to_process}

        for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(tasks_to_process), desc="Translating"):
            img_path = future_to_path[future]
            try:
                status, result_data = future.result()

                if status == "success":
                    # 1. Save target language translation
                    target_path = out_dir / (img_path.stem + target_ext)
                    if result_data["target_md"]:
                        with open(target_path, "w", encoding="utf-8") as f:
                            f.write(result_data["target_md"])
                    else:
                        tqdm.write(f"[WARN] {img_path.name}: No translation block found.")

                    # 2. Optional: Save English reconstruction
                    if args.save_english and en_dir and result_data["en_md"]:
                        en_path = en_dir / (img_path.stem + ".en.md")
                        with open(en_path, "w", encoding="utf-8") as f:
                            f.write(result_data["en_md"])

                    # 3. Optional: Save full raw output
                    if args.save_full and raw_dir:
                        raw_path = raw_dir / (img_path.stem + ".full.txt")
                        with open(raw_path, "w", encoding="utf-8") as f:
                            f.write(result_data["full_text"])

                    success_count += 1
                else:
                    tqdm.write(f"[FAIL] {img_path.name}: {result_data['message']}")
                    failed_count += 1
            except Exception as exc:
                tqdm.write(f"[CRITICAL] Exception processing {img_path.name}: {exc}")
                failed_count += 1

    # --- Final Report ---
    print("\n--- ✅ Batch processing complete! ---")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Main Output Directory: {out_dir.resolve()}")
    if en_dir:
        print(f"English OCR Directory: {en_dir.resolve()}")

if __name__ == "__main__":
    main()