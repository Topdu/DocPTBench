#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import re
import sys
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Tuple

from openai import OpenAI
from tqdm import tqdm

# ==============================================================================
# 1. Language Configuration
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

# ==============================================================================
# 2. PROMPT & PARSING LOGIC
# ==============================================================================

SYSTEM_PROMPT_TEMPLATE = """You are a professional document translator.

Task: Translate the provided Chinese Markdown into high-fidelity {lang_name} Markdown.

Output format (STRICT):
Return EXACTLY one fenced block, and nothing else:
<{lang_tag}>
...the {lang_name} Markdown translation...
</{lang_tag}>

Translation rules:
- You MUST translate all Chinese narrative text into fluent, professional {lang_name}.
- NO Chinese characters should remain in the <{lang_tag}> block, except within code blocks, URLs, or other elements that naturally contain them.
- Preserve structure: headings (##), lists, blockquotes (>), tables, horizontal rules, references, footnotes.
- Keep code blocks (```...```), inline code (`...`), and URLs/emails/IDs unchanged.
- Formulas (LaTeX): Detect and preserve LaTeX formulas like $$...$$ or $...$. Inside formulas, translate ONLY human words within \\text{{...}} (e.g., \\text{{Mean}} -> \\text{{translated_term}}); keep all math symbols, variables, and units intact.
- Tables: Preserve the Markdown table structure. Translate only the narrative text within cells.
- Hyperlinks/Images: For `[text](url)`, translate the visible `text` but keep the `url` unchanged.
- Target Style: {style_note}
- Do not add or omit content. Your translation should be a faithful representation of the source document's structure and meaning.
- Return ONLY the <{lang_tag}> block. Do not include explanations or any other text outside this block.
"""

USER_INSTRUCTION_TEMPLATE = "Translate the given Chinese Markdown into {lang_name}, following all rules specified in the system prompt."

def parse_target_block(text: str, lang_code: str) -> str:
    """
    Dynamically parse the XML block based on language configuration.
    """
    tag = LANG_CONFIG[lang_code]["tag"]
    # Dynamic regex, ignoring case and allowing dot to match newlines
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    
    m = pattern.search(text)
    if m:
        return m.group(1).strip()
    else:
        # Fallback: return the whole text if tags are missing
        return text.strip()

def build_messages_for_text(md_text: str, lang_code: str) -> List[Dict[str, Any]]:
    """
    Build the message body based on the target language.
    """
    cfg = LANG_CONFIG[lang_code]
    
    system_content = SYSTEM_PROMPT_TEMPLATE.format(
        lang_name=cfg["name"],
        lang_tag=cfg["tag"],
        style_note=cfg["style_note"]
    )
    
    user_instruction = USER_INSTRUCTION_TEMPLATE.format(
        lang_name=cfg["name"]
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": md_text + "\n\n" + user_instruction},
    ]

# ==============================================================================
# 3. CORE WORKER FUNCTION
# ==============================================================================

def process_text_translation(md_path: Path, client: OpenAI, args: argparse.Namespace) -> Tuple[str, Dict[str, Any]]:
    """
    Process a single Markdown file. This function is called concurrently.
    """
    try:
        # 1. Read source file
        src_text = md_path.read_text(encoding="utf-8")

        # 2. Build messages (pass current language)
        messages = build_messages_for_text(src_text, args.lang)

        # 3. Call API (non-streaming recommended for concurrency)
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
            return ("error", {"message": f"API for '{md_path.name}' returned empty content."})

        # 4. Parse result
        target_md = parse_target_block(full_text, args.lang)

        # 5. Return success status and data
        return ("success", {
            "target_md": target_md,
            "full_text": full_text
        })

    except Exception as e:
        return ("error", {"message": f"An unexpected error occurred for '{md_path.name}': {e}"})


# ==============================================================================
# 4. MAIN ORCHESTRATION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Concurrent Document Translation Script (Chinese -> Multi-Language)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- Core Arguments ---
    parser.add_argument("--dir", required=True, help="Input directory containing Chinese Markdown files.")
    parser.add_argument("--lang", default="en", choices=["en", "de", "fr", "ru"], help="Target language (default: en).")
    parser.add_argument("--out_dir", default=None, help="Output directory (auto-generated based on language if not specified).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--ext", default=None, help="Output file extension (auto-selected based on language if not specified).")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of concurrent threads.")

    # --- API and Model Arguments ---
    parser.add_argument("--model", default="gpt-4o", help="Model name to use.")
    parser.add_argument("--api_key", type=str, default=None, help="API Key. Defaults to JIEKOU_API_KEY env var.")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1", help="API Base URL.")
    
    parser.add_argument("--max_tokens", type=int, default=16384, help="Max tokens for API response.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for text generation.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling parameter.")

    # --- Optional Output Arguments ---
    parser.add_argument("--save_full", action="store_true", help="Save the full raw API response (.full.txt).")

    args = parser.parse_args()

    # --- 0. Config Initialization ---
    lang_cfg = LANG_CONFIG[args.lang]
    
    # Auto-set output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(f"translations_gpt_text_{args.lang}")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-set extension
    ext = args.ext if args.ext else lang_cfg["ext"]

    # Set raw directory
    raw_dir = None
    if args.save_full:
        raw_dir = out_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Client Initialization ---
    # Priority: Command line argument > Environment variable
    final_api_key = args.api_key or os.getenv("JIEKOU_API_KEY")
    if not final_api_key:
        print("Error: API Key not provided. Please set JIEKOU_API_KEY env var or use --api_key.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=final_api_key, base_url=args.base_url)

    # --- 2. Collect Tasks ---
    in_dir = Path(args.dir)
    all_md_files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in (".md", ".mmd")])

    if not all_md_files:
        print(f"No Markdown files (.md, .mmd) found in '{in_dir}'.")
        return

    tasks_to_process = []
    skipped_count = 0
    for md_path in all_md_files:
        out_path = out_dir / (md_path.stem + ext)
        if out_path.exists() and not args.overwrite:
            skipped_count += 1
        else:
            tasks_to_process.append(md_path)

    if not tasks_to_process:
        print("All files have been processed. Use --overwrite to re-process.")
        return

    print(f"[INFO] Target Language: {lang_cfg['name']} ({args.lang})")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Output Dir: {out_dir.resolve()}")
    print(f"[INFO] Found {len(all_md_files)} files. Skipped {skipped_count}. Processing {len(tasks_to_process)} files.")
    print(f"[INFO] Concurrency: {args.max_workers} threads.")

    # --- 3. Concurrent Processing ---
    success_count = 0
    failed_count = 0
    
    # Use ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(process_text_translation, md_path, client, args): md_path 
            for md_path in tasks_to_process
        }

        # Use tqdm for progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(tasks_to_process), desc="Translating"):
            md_path = future_to_path[future]
            try:
                status, result_data = future.result()

                if status == "success":
                    # Save translation result
                    out_path = out_dir / (md_path.stem + ext)
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(result_data["target_md"])

                    # Save raw output (optional)
                    if args.save_full and raw_dir:
                        raw_path = raw_dir / (md_path.stem + ".full.txt")
                        with open(raw_path, "w", encoding="utf-8") as f:
                            f.write(result_data["full_text"])

                    success_count += 1
                else:
                    tqdm.write(f"[FAIL] {md_path.name}: {result_data['message']}")
                    failed_count += 1
            except Exception as exc:
                tqdm.write(f"[CRITICAL] Uncaught exception processing {md_path.name}: {exc}")
                failed_count += 1

    # --- 4. Final Report ---
    print("\n--- Batch Processing Complete ---")
    print(f"Success: {success_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"Output Directory: {out_dir.resolve()}")

if __name__ == "__main__":
    main()