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
# 1. MULTI-LANGUAGE CONFIG & PROMPTS
# ==============================================================================

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

SYSTEM_PROMPT_TEMPLATE = """You are a professional document translator.

Task: Translate the content of the Chinese document image directly into high-fidelity {lang_name} Markdown.

Output format (STRICT):
Return EXACTLY one fenced block, and nothing else:
<{lang_tag}>
...the {lang_name} Markdown translation...
</{lang_tag}>

Translation Rules:
- Formulas (LaTeX ONLY): Detect and represent any formula/equation in LaTeX, wrapped in display math delimiters: $$ ... $$. Inside the delimiters, translate only human words (Chinese terms) into {lang_name} using \\text{{...}} (e.g., \\text{{mean}} → \\text{{translated_term}}).
- Tables (Markdown): Preserve Markdown tables. Translate only narrative text in cells.
- Structure: Preserve all structural elements like paragraphs, lists, blockquotes, etc.
- Do-not-translate: Keep code (`...`), URLs, emails, and stable names/brands as is.
- Target Style: {style_note}
- Quality Check: Ensure the output is a single, complete, and accurate {lang_name} Markdown block. Do not add any explanations or extra content.
"""

USER_INSTRUCTION_TEMPLATE = (
    "Translate the document image directly into {lang_name} Markdown. "
    "Return ONLY the <{lang_tag}>...</{lang_tag}> block as specified."
)

def parse_target_block(text: str, lang_tag: str) -> str:
    """
    Dynamically parse the content of a specific tag from the API response.
    Example: <german_md>...</german_md>
    """
    pattern = re.compile(rf"<{lang_tag}>\s*(.*?)\s*</{lang_tag}>", re.S | re.I)
    m = pattern.search(text)
    # If tag is found, return content; otherwise, return full text as fallback
    return m.group(1).strip() if m else text.strip()

# ==============================================================================
# 2. IMAGE & API HELPERS
# ==============================================================================

def image_path_to_data_url_fallback(image_path: Path) -> str:
    """Original fallback encoding method without preprocessing."""
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type:
        mime_type = "application/octet-stream"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime_type};base64,{b64}"

def preprocess_and_encode_image(image_path: Path, max_dim: int = 2048, quality: int = 90) -> str:
    """
    Read image, preprocess (resize and compress), and encode to data URL.
    """
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize if image is too large to save tokens and bandwidth
            if max(img.size) > max_dim:
                img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            img_bytes = buffer.getvalue()

            b64_string = base64.b64encode(img_bytes).decode("ascii")
            return f"data:image/jpeg;base64,{b64_string}"
    except Exception as e:
        # tqdm.write might race in threads, but usually fine in except block
        print(f"[WARN] Pillow failed to process image {image_path.name}: {e}. Falling back to raw encoding.")
        return image_path_to_data_url_fallback(image_path)

def build_messages(data_url: str, lang_code: str) -> List[Dict[str, Any]]:
    """Build message body for the vision model with dynamic prompt."""
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
# 3. CORE WORKER FUNCTION
# ==============================================================================

def process_image_translation(image_path: Path, client: OpenAI, args: argparse.Namespace) -> Tuple[str, Dict[str, Any]]:
    """
    Single image processing function called concurrently.
    """
    try:
        # 1. Image preprocessing and encoding
        data_url = preprocess_and_encode_image(image_path)

        # 2. Build request messages (pass current language selection)
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

        # 4. Parse result (dynamic parsing based on language tag)
        lang_tag = LANG_CONFIG[args.lang]["tag"]
        target_md = parse_target_block(full_text, lang_tag)

        # 5. Return success status and data
        return ("success", {
            "target_md": target_md,
            "full_text": full_text
        })

    except Exception as e:
        return ("error", {"message": f"An unexpected error occurred for '{image_path.name}': {e}"})


# ==============================================================================
# 4. MAIN ORCHESTRATION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gemini Multi-language Concurrent Document Image Translation Script",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- Core Arguments ---
    parser.add_argument("--dir", required=True, help="Input image directory path.")
    parser.add_argument("--lang", default="en", choices=["en", "de", "fr", "ru"], help="Target language (en/de/fr/ru).")
    parser.add_argument("--out_dir", default=None, help="Output directory (default: auto-generated based on language).")
    parser.add_argument("--overwrite", action="store_true", help="If true, overwrite existing output files.")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of concurrent worker threads.")

    # --- API and Model Arguments ---
    parser.add_argument("--model", default="gemini-2.5-pro", help="Model name to use.")
    parser.add_argument("--api_key", type=str, default=None, help="API Key.")
    parser.add_argument("--base_url", type=str, default="https://ai.liaobots.work/v1", help="API Base URL.")
    parser.add_argument("--max_tokens", type=int, default=16384, help="Max tokens for API call.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for text generation.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling parameter.")

    # --- Optional Output Arguments ---
    parser.add_argument("--save_full", action="store_true", help="Save full raw text returned by API (.full.txt).")
    parser.add_argument("--raw_dir", type=str, default=None, help="Directory to save full raw output.")

    args = parser.parse_args()

    # --- Check and Initialize ---
    if not args.api_key:
        # Try reading from environment variable
        args.api_key = os.getenv("JIEKOU_API_KEY")
        if not args.api_key:
            raise ValueError("API Key not provided. Please set JIEKOU_API_KEY environment variable or use --api_key argument.")

    in_dir = Path(args.dir)
    
    # Auto-generate output directory name
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(f"translations_gemini_direct_{args.lang}")
    
    out_dir.mkdir(parents=True, exist_ok=True)

    # Raw data output directory
    raw_dir = None
    if args.save_full:
        raw_dir = Path(args.raw_dir) if args.raw_dir else (out_dir / "raw")
        raw_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # --- Collect and Filter Tasks ---
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
    all_images = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])

    if not all_images:
        print(f"No supported images found in directory '{in_dir}'.")
        return

    # Get configuration for current language
    lang_cfg = LANG_CONFIG[args.lang]
    target_ext = lang_cfg["ext"]

    tasks_to_process = []
    skipped_count = 0
    for img_path in all_images:
        # Use specific language extension
        target_path = out_dir / (img_path.stem + target_ext)
        if target_path.exists() and not args.overwrite:
            skipped_count += 1
        else:
            tasks_to_process.append(img_path)

    if not tasks_to_process:
        print(f"All images have been processed (Target Language: {lang_cfg['name']}).")
        return

    print(f"=== Task Information ===")
    print(f"Target Language: {lang_cfg['name']} ({args.lang})")
    print(f"Output Directory: {out_dir.resolve()}")
    print(f"Total Images: {len(all_images)}")
    print(f"Skipped Existing: {skipped_count}")
    print(f"To Process: {len(tasks_to_process)}")
    print(f"Concurrent Workers: {args.max_workers}")
    print(f"========================")

    # --- Concurrent Processing ---
    success_count = 0
    failed_count = 0
    
    # Use ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(process_image_translation, img_path, client, args): img_path 
            for img_path in tasks_to_process
        }

        # Use tqdm to show progress
        for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(tasks_to_process), desc=f"Translating to {args.lang}"):
            img_path = future_to_path[future]
            try:
                status, result_data = future.result()

                if status == "success":
                    # Save main output: Target language translation
                    target_path = out_dir / (img_path.stem + target_ext)
                    with open(target_path, "w", encoding="utf-8") as f:
                        f.write(result_data["target_md"])

                    # Optional: Save full raw output
                    if args.save_full and raw_dir:
                        raw_path = raw_dir / (img_path.stem + ".full.txt")
                        with open(raw_path, "w", encoding="utf-8") as f:
                            f.write(result_data["full_text"])

                    success_count += 1
                else:
                    # Use tqdm.write to avoid interrupting progress bar
                    tqdm.write(f"[FAIL] {img_path.name}: {result_data['message']}")
                    failed_count += 1
            except Exception as exc:
                tqdm.write(f"[CRITICAL] Uncaught exception while processing {img_path.name}: {exc}")
                failed_count += 1

    # --- Final Report ---
    print("\n--- ✅ Batch Processing Complete! ---")
    print(f"Successful Translations: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Output Location: {out_dir.resolve()}")

if __name__ == "__main__":
    main()