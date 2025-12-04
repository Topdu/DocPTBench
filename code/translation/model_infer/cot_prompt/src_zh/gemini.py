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
    Returns: (target_md, chinese_md)
    """
    tag = LANG_CONFIG[lang_code]["tag"]
    
    # Dynamic Regex for the target language block
    target_re = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    zh_re = re.compile(r"<chinese_md>\s*(.*?)\s*</chinese_md>", re.S | re.I)

    target_md, zh_md = "", ""

    # 1. Extract Target Language Block
    m1 = target_re.search(text)
    if m1:
        target_md = m1.group(1).strip()
    else:
        # Fallback: Try to find start tag only
        m2 = re.search(rf"<{tag}>\s*(.*)", text, re.S | re.I)
        if m2:
            target_md = m2.group(1).strip()
        else:
            # Fallback: If Chinese block ends, assume rest is target
            m3 = re.search(r"</chinese_md>\s*(.*)", text, re.S | re.I)
            if m3:
                target_md = m3.group(1).strip()

    # 2. Extract Chinese Block
    m_zh = zh_re.search(text)
    if m_zh:
        zh_md = m_zh.group(1).strip()

    # 3. Ultimate Fallback
    if not target_md and not zh_md:
        zh_md = text.strip()

    return target_md, zh_md

# ==============================================================================
# 3. IMAGE PROCESSING (Pillow)
# ==============================================================================

def image_path_to_data_url_fallback(image_path: Path) -> str:
    """Fallback raw encoding."""
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type:
        mime_type = "application/octet-stream"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime_type};base64,{b64}"

def preprocess_and_encode_image(image_path: Path, max_dim: int = 2048, quality: int = 85) -> str:
    """
    Resizes and compresses image to optimize for Vision API (Token usage & Latency).
    """
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize if too large
            if max(img.size) > max_dim:
                img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            img_bytes = buffer.getvalue()

            b64_string = base64.b64encode(img_bytes).decode("ascii")
            return f"data:image/jpeg;base64,{b64_string}"
    except Exception as e:
        # tqdm.write might not be initialized yet if called outside loop, but safe here
        print(f"[WARN] Pillow failed for {image_path.name}: {e}. Using fallback.")
        return image_path_to_data_url_fallback(image_path)

def build_messages(data_url: str, sys_prompt: str, user_instruction: str) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                {"type": "text", "text": user_instruction},
            ],
        },
    ]

# ==============================================================================
# 4. WORKER FUNCTION
# ==============================================================================

def process_image_translation(
    image_path: Path, 
    client: OpenAI, 
    args: argparse.Namespace,
    sys_prompt: str,
    user_instruction: str
) -> Tuple[str, Dict[str, Any]]:
    """
    Worker function for concurrent execution.
    """
    try:
        # 1. Preprocess & Encode
        data_url = preprocess_and_encode_image(image_path)

        # 2. Build Messages
        messages = build_messages(data_url, sys_prompt, user_instruction)

        # 3. Call API
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
            return ("error", {"message": f"API returned empty content."})

        # 4. Parse Result (Dynamic based on lang)
        target_md, zh_md = parse_response(full_text, args.lang)

        return ("success", {
            "target_md": target_md,
            "zh_md": zh_md,
            "full_text": full_text
        })

    except Exception as e:
        return ("error", {"message": str(e)})

# ==============================================================================
# 5. MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gemini Concurrent Document Translation (CN -> EN/DE/FR/RU)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- Core ---
    parser.add_argument("--dir", required=True, help="Input directory containing images")
    parser.add_argument("--lang", default="en", choices=["en", "de", "fr", "ru"], 
                        help="Target language: en (English), de (German), fr (French), ru (Russian)")
    parser.add_argument("--out_dir", default=None, help="Output directory (default: translations_<lang>)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--max_workers", type=int, default=8, help="Concurrency level")

    # --- Model/API ---
    parser.add_argument("--model", default="gemini-2.5-pro", help="Model ID")
    parser.add_argument("--api_key", type=str, default=None, help="API Key (or set JIEKOU_API_KEY)")
    parser.add_argument("--base_url", type=str, default="https://ai.liaobots.work/v1", help="API Base URL")
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    # --- Optional Outputs ---
    parser.add_argument("--save_chinese", action="store_true", help="Save Chinese OCR text")
    parser.add_argument("--save_full", action="store_true", help="Save raw API response")

    args = parser.parse_args()

    # 1. Setup Language & Paths
    lang_cfg = LANG_CONFIG[args.lang]
    target_ext = lang_cfg["ext"]
    
    in_dir = Path(args.dir)
    default_out_name = f"translations_gemini_{args.lang}"
    out_dir = Path(args.out_dir) if args.out_dir else Path(default_out_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    zh_dir = out_dir / "zh" if args.save_chinese else None
    if zh_dir: zh_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = out_dir / "raw" if args.save_full else None
    if raw_dir: raw_dir.mkdir(parents=True, exist_ok=True)

    # 2. Initialize Client
    api_key = args.api_key or os.getenv("JIEKOU_API_KEY")
    if not api_key:
        # Fallback to check ARK_API_KEY just in case user mixes env vars, or raise error
        api_key = os.getenv("ARK_API_KEY") 
    
    if not api_key:
        raise ValueError("API Key not found. Set JIEKOU_API_KEY env var or use --api_key.")

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    # 3. Prepare Tasks
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
    all_images = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in valid_exts])
    
    if not all_images:
        print(f"No images found in {in_dir}.")
        return

    tasks = []
    skipped = 0
    for img in all_images:
        out_path = out_dir / (img.stem + target_ext)
        if out_path.exists() and not args.overwrite:
            skipped += 1
        else:
            tasks.append(img)

    print(f"Target: {lang_cfg['name']} | Model: {args.model}")
    print(f"Found {len(all_images)} images. Processing {len(tasks)} (Skipped {skipped}).")
    print(f"Concurrency: {args.max_workers} workers")

    # 4. Generate Prompts (Once)
    sys_prompt, user_instruction = get_prompts(args.lang)

    # 5. Run Concurrently
    success_count = 0
    fail_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit tasks
        future_to_path = {
            executor.submit(process_image_translation, img, client, args, sys_prompt, user_instruction): img 
            for img in tasks
        }

        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(tasks), desc="Translating"):
            img_path = future_to_path[future]
            try:
                status, res = future.result()

                if status == "success":
                    # Save Target
                    out_path = out_dir / (img_path.stem + target_ext)
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(res["target_md"])
                    
                    # Save Chinese
                    if zh_dir and res["zh_md"]:
                        zh_path = zh_dir / (img_path.stem + ".md")
                        with open(zh_path, "w", encoding="utf-8") as f:
                            f.write(res["zh_md"])
                    
                    # Save Raw
                    if raw_dir:
                        raw_path = raw_dir / (img_path.stem + ".full.txt")
                        with open(raw_path, "w", encoding="utf-8") as f:
                            f.write(res["full_text"])
                    
                    success_count += 1
                else:
                    tqdm.write(f"[FAIL] {img_path.name}: {res['message']}")
                    fail_count += 1

            except Exception as e:
                tqdm.write(f"[CRITICAL] {img_path.name}: {e}")
                fail_count += 1

    print(f"\n--- Done! Success: {success_count}, Failed: {fail_count}, Skipped: {skipped} ---")
    print(f"Output: {out_dir.resolve()}")

if __name__ == "__main__":
    main()