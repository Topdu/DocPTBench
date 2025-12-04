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
    "zh": {
        "name": "Simplified Chinese",
        "tag": "chinese_md",
        "ext": ".zh.md",
        "style_note": "Fluent, professional Simplified Chinese with standard punctuation."
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

# ==============================================================================
# 2. Prompt & Parsing Logic (Dynamic Multilingual)
# ==============================================================================

# Dynamic System Prompt Template
SYSTEM_PROMPT_TEMPLATE = """You are a professional document translator.

Task: Translate the provided English Markdown into a high-fidelity {lang_name} Markdown.

Output format (STRICT):
Return EXACTLY one fenced block:
<{lang_tag}>
...the {lang_name} Markdown translation...
</{lang_tag}>

Translation rules:
- You MUST translate all English narrative text into fluent {lang_name}.
- Preserve structure: headings (##), lists, blockquotes (>), tables, horizontal rules, references, footnotes.
- Keep code blocks, inline code, LaTeX formulas ($$...$$), URLs, emails, IDs unchanged.
- In formulas ($$...$$), translate ONLY human words inside \\text{{...}} (e.g., \\text{{mean}} → \\text{{translated_term}}); keep math symbols/variables/units intact.
- In tables, preserve the structure; translate only narrative text in cells.
- Keep hyperlinks/images as [text](url): translate visible text, keep URL unchanged.
- Target Style: {style_note}
- No added/omitted content. Return ONLY the <{lang_tag}> block, nothing else.
"""

USER_INSTRUCTION_TEMPLATE = "Translate the given English Markdown into {lang_name} Markdown, following the rules."

def parse_target_block(text: str, lang_code: str) -> str:
    """
    Parses the specific language XML block content from the full model response.
    """
    tag = LANG_CONFIG[lang_code]["tag"]
    # re.S makes . match newlines
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    m = pattern.search(text)
    return m.group(1).strip() if m else text.strip()

def build_messages_for_text(md_text: str, lang_code: str) -> List[Dict[str, Any]]:
    """
    Builds the message body based on the target language.
    """
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

# ==============================================================================
# 3. Core Worker Function (Concurrency)
# ==============================================================================

def process_text_translation(md_path: Path, client: OpenAI, args: argparse.Namespace, lang_code: str) -> Tuple[str, Dict[str, Any]]:
    """
    Worker function for a single Markdown file, called concurrently.
    """
    try:
        # 1. Read source file
        src_text = md_path.read_text(encoding="utf-8")

        # 2. Build request messages
        messages = build_messages_for_text(src_text, lang_code)

        # 3. Call API (Non-streaming)
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
        target_md = parse_target_block(full_text, lang_code)

        # 5. Return success status and data
        return ("success", {
            "target_md": target_md,
            "full_text": full_text
        })

    except Exception as e:
        # Catch all exceptions and return error status
        return ("error", {"message": f"An unexpected error occurred for '{md_path.name}': {e}"})


# ==============================================================================
# 4. Main Orchestration
# ==============================================================================

def main():
    """Main function handling argument parsing, task distribution, and result summary."""
    parser = argparse.ArgumentParser(
        description="Concurrent Document Text Translation Script (EN -> Multi-Lang)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- Core Arguments ---
    parser.add_argument("--dir", required=True, help="Input directory containing English Markdown files.")
    
    parser.add_argument("--lang", default="zh", choices=["zh", "de", "fr", "ru"], 
                        help="Target language: zh (Chinese), de (German), fr (French), ru (Russian). Default: zh")

    parser.add_argument("--out_dir", default=None, help="Output directory (default: auto-generated based on language).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files if True.")
    parser.add_argument("--ext", default=None, help="Output file extension (default: auto-generated based on language).")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of concurrent worker threads.")

    # --- API and Model Arguments ---
    parser.add_argument("--model", default="gemini-2.5-pro", help="Model name to use.")
    parser.add_argument("--api_key", type=str, default=None, help="API Key. Defaults to JIEKOU_API_KEY env var.")
    parser.add_argument("--base_url", type=str, default="https://ai.liaobots.work/v1", help="API Base URL.")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Max tokens for API call.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling parameter.")

    # --- Optional Output Arguments ---
    parser.add_argument("--save_full", action="store_true", help="Save the full raw API response (.full.txt).")

    args = parser.parse_args()

    # --- Check and Initialization ---
    api_key = args.api_key or os.getenv("JIEKOU_API_KEY")
    
    if not api_key:
        print("Error: API Key not provided. Please set JIEKOU_API_KEY env var or use --api_key.", file=sys.stderr)
        sys.exit(1)

    # Get language config
    lang_cfg = LANG_CONFIG[args.lang]

    in_dir = Path(args.dir)
    
    # Auto-determine output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(f"translations_gpt5_text_{args.lang}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine extension
    ext = args.ext if args.ext else lang_cfg["ext"]
    
    raw_dir = None
    if args.save_full:
        raw_dir = out_dir / "raw" 
        raw_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    # --- Collect and Filter Tasks ---
    all_md_files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in (".md", ".mmd")])

    if not all_md_files:
        print(f"No Markdown files (.md, .mmd) found in '{in_dir}'.")
        return

    tasks_to_process = []
    skipped_count = 0
    for md_path in all_md_files:
        target_path = out_dir / (md_path.stem + ext)
        if target_path.exists() and not args.overwrite:
            skipped_count += 1
        else:
            tasks_to_process.append(md_path)

    if not tasks_to_process:
        print("All files have been processed. Use --overwrite to re-process.")
        return

    print(f"Target Language: {lang_cfg['name']} ({args.lang})")
    print(f"Found {len(all_md_files)} Markdown files.")
    print(f"Skipped {skipped_count} existing files.")
    print(f"Processing {len(tasks_to_process)} new files with {args.max_workers} concurrent threads...")
    print(f"Output Directory: {out_dir.resolve()}")

    # --- Concurrent Processing ---
    success_count = 0
    failed_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_path = {
            executor.submit(process_text_translation, md_path, client, args, args.lang): md_path 
            for md_path in tasks_to_process
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(tasks_to_process), desc="Translating Files"):
            md_path = future_to_path[future]
            try:
                status, result_data = future.result()

                if status == "success":
                    # Save main output: Target language translation
                    target_path = out_dir / (md_path.stem + ext)
                    with open(target_path, "w", encoding="utf-8") as f:
                        f.write(result_data["target_md"])

                    # Optional: Save full raw output
                    if args.save_full and raw_dir:
                        raw_path = raw_dir / (md_path.stem + ".full.txt")
                        with open(raw_path, "w", encoding="utf-8") as f:
                            f.write(result_data["full_text"])

                    success_count += 1
                else:
                    tqdm.write(f"[FAIL] {md_path.name}: {result_data['message']}")
                    failed_count += 1
            except Exception as exc:
                tqdm.write(f"[CRITICAL] Uncaught exception while processing {md_path.name}: {exc}")
                failed_count += 1

    # --- Final Report ---
    print("\n--- ✅ Batch Processing Complete! ---")
    print(f"Language: {lang_cfg['name']}")
    print(f"Successful: {success_count} files")
    print(f"Skipped (Existing): {skipped_count} files")
    print(f"Failed: {failed_count} files")
    print(f"Translations saved to: {out_dir.resolve()}")
    if raw_dir and args.save_full:
        print(f"Full raw outputs saved to: {raw_dir.resolve()}")

if __name__ == "__main__":
    main()