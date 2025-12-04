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
# 1. Multi-language Configuration & Prompt Templates
# ==============================================================================

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

SYSTEM_PROMPT_TEMPLATE = """You are a professional document translator.

Task: Translate the content of the English document image directly into high-fidelity {lang_name} Markdown.

Output format (STRICT):
Return EXACTLY one fenced block, and nothing else:
<{lang_tag}>
...the {lang_name} Markdown translation...
</{lang_tag}>

Translation Rules:
- Formulas (LaTeX ONLY): Detect and represent any formula/equation in LaTeX, wrapped in display math delimiters: $$ ... $$. Inside the delimiters, translate only human words (English terms) into {lang_name} using \\text{{...}} (e.g., \\text{{Mean}} → \\text{{translated_term}}).
- Tables (Markdown): Preserve Markdown tables. Translate only narrative text in cells.
- Structure: Preserve all structural elements like paragraphs, lists, blockquotes, etc.
- Do-not-translate: Keep code (`...`), URLs, emails, and stable names/brands as is.
- Target Style: {style_note}
- Quality Check: Ensure the output is a single, complete, and accurate {lang_name} Markdown block. Do not add any explanations or extra content.
"""

USER_INSTRUCTION_TEMPLATE = (
    "Translate the English document image directly into {lang_name} Markdown. "
    "Return ONLY the <{lang_tag}>...</{lang_tag}> block as specified."
)

def parse_target_block(text: str, lang_code: str) -> str:
    """Dynamically parse the XML block based on language configuration."""
    tag = LANG_CONFIG[lang_code]["tag"]
    # Dynamic regex construction
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    
    m = pattern.search(text)
    if m:
        return m.group(1).strip()
    # Fallback: If no tag is found but text is not empty, return the full text
    return text.strip()

# ==============================================================================
# 2. Image Helper Functions
# ==============================================================================

def image_path_to_data_url_fallback(image_path: Path) -> str:
    """Raw encoding method without preprocessing, used as a fallback."""
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type:
        mime_type = "application/octet-stream"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime_type};base64,{b64}"

def preprocess_and_encode_image(image_path: Path, max_dim: int = 2048, quality: int = 90) -> str:
    """
    Read image, preprocess (resize and compress), and encode as data URL.
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
        # tqdm.write allows safe printing while progress bar exists
        # tqdm.write(f"[WARN] Pillow failed to process image {image_path.name}: {e}. Fallback to raw encoding.")
        return image_path_to_data_url_fallback(image_path)

def build_messages(data_url: str, lang_code: str) -> List[Dict[str, Any]]:
    """Build message body based on target language."""
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
                {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                {"type": "text", "text": user_content},
            ],
        },
    ]

# ==============================================================================
# 3. Core Worker Function
# ==============================================================================

def process_image_translation(image_path: Path, client: OpenAI, args: argparse.Namespace) -> Tuple[str, Dict[str, Any]]:
    """
    Processing function for a single image, called concurrently.
    """
    try:
        # 1. Image preprocessing and encoding
        data_url = preprocess_and_encode_image(image_path)

        # 2. Build request messages
        messages = build_messages(data_url, args.lang)

        # 3. Call API
        resp = client.chat.completions.create(
            model=args.model,
            messages=messages,
            stream=False,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        full_text = resp.choices[0].message.content or ""

        if not full_text:
            return ("error", {"message": f"API for '{image_path.name}' returned empty content."})

        # 4. Parse result
        target_md = parse_target_block(full_text, args.lang)

        # 5. Return success status and data
        return ("success", {
            "target_md": target_md,
            "full_text": full_text
        })

    except Exception as e:
        return ("error", {"message": f"An unexpected error occurred for '{image_path.name}': {e}"})


# ==============================================================================
# 4. Main Orchestration
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gemini Vision Concurrent Document Translation (EN -> ZH/DE/FR/RU)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- Core Arguments ---
    parser.add_argument("--dir", required=True, help="Input directory containing English images.")
    parser.add_argument("--lang", default="zh", choices=["zh", "de", "fr", "ru"], help="Target language (default: zh).")
    parser.add_argument("--out_dir", default=None, help="Output directory (auto-generated by default).")
    parser.add_argument("--ext", default=None, help="Output extension (auto-selected by default).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files if True.")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of threads for concurrent processing.")

    # --- API and Model Arguments ---
    parser.add_argument("--model", default="gemini-2.5-pro", help="Model name to use (e.g., gemini-1.5-pro).")
    parser.add_argument("--api_key", type=str, default=None, help="API Key. Defaults to reading from environment variable API_KEY.")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1", help="API Base URL.")
    
    parser.add_argument("--max_tokens", type=int, default=8192, help="Max tokens for API response.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for text generation.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling parameter.")

    # --- Optional Output Arguments ---
    parser.add_argument("--save_full", action="store_true", help="Save full raw API response text (.full.txt).")

    args = parser.parse_args()

    # --- Initialization ---
    lang_cfg = LANG_CONFIG[args.lang]
    
    # Auto-set output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(f"translations_gemini_direct_en2{args.lang}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-set extension
    ext = args.ext if args.ext else lang_cfg["ext"]

    # Raw response directory
    raw_dir = None
    if args.save_full:
        raw_dir = out_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

    # --- API Client ---
    api_key = args.api_key or os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API Key not provided. Please set the API_KEY environment variable or use the --api_key argument.")

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    # --- Collect Tasks ---
    in_dir = Path(args.dir)
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
    
    if not in_dir.exists():
         raise ValueError(f"Input directory does not exist: {in_dir}")

    all_images = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])

    if not all_images:
        print(f"No supported images found in directory '{in_dir}'.")
        return

    tasks_to_process = []
    skipped_count = 0
    for img_path in all_images:
        target_path = out_dir / (img_path.stem + ext)
        if target_path.exists() and not args.overwrite:
            skipped_count += 1
        else:
            tasks_to_process.append(img_path)

    print(f"[INFO] Source: English -> Target: {lang_cfg['name']} ({args.lang})")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Output: {out_dir.resolve()}")
    print(f"[INFO] Total Images: {len(all_images)} | Skipped: {skipped_count} | To Process: {len(tasks_to_process)}")

    if not tasks_to_process:
        print("All images have been processed.")
        return

    # --- Concurrent Processing ---
    success_count = 0
    failed_count = 0
    
    print(f"Starting processing with {args.max_workers} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit tasks
        future_to_path = {
            executor.submit(process_image_translation, img_path, client, args): img_path 
            for img_path in tasks_to_process
        }

        # Progress bar loop
        for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(tasks_to_process), desc=f"Translating (EN->{args.lang.upper()})"):
            img_path = future_to_path[future]
            try:
                status, result_data = future.result()

                if status == "success":
                    # Save translation result
                    target_path = out_dir / (img_path.stem + ext)
                    with open(target_path, "w", encoding="utf-8") as f:
                        f.write(result_data["target_md"])

                    # Optional: Save full raw output
                    if args.save_full and raw_dir:
                        raw_path = raw_dir / (img_path.stem + ".full.txt")
                        with open(raw_path, "w", encoding="utf-8") as f:
                            f.write(result_data["full_text"])

                    success_count += 1
                else:
                    tqdm.write(f"[FAIL] {img_path.name}: {result_data['message']}")
                    failed_count += 1
            except Exception as exc:
                tqdm.write(f"[CRITICAL] Uncaught exception for {img_path.name}: {exc}")
                failed_count += 1

    # --- Final Report ---
    print("\n--- Batch Processing Complete ---")
    print(f"Success: {success_count}")
    print(f"Failed:  {failed_count}")
    print(f"Skipped: {skipped_count}")

if __name__ == "__main__":
    main()