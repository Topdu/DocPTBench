import os
import argparse
import base64
import mimetypes
import re
from pathlib import Path
from typing import Tuple, List, Dict, Any

from openai import OpenAI

# ==========================================
# Language Configuration
# ==========================================

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

# ==========================================
# Prompt Templates
# ==========================================
# We use placeholders like __LANG_NAME__ to avoid conflicts with LaTeX/Code braces.

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

# ==========================================
# Helper Functions
# ==========================================

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
    Returns: (translated_md, chinese_md)
    """
    tag = LANG_CONFIG[lang_code]["tag"]
    
    # Dynamic Regex for the target language block
    # Matches <tag> content </tag>
    target_re = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    zh_re = re.compile(r"<chinese_md>\s*(.*?)\s*</chinese_md>", re.S | re.I)

    target_md, zh_md = "", ""

    # 1. Extract Target Language Block
    m1 = target_re.search(text)
    if m1:
        target_md = m1.group(1).strip()
    else:
        # Fallback: Try to find start tag only (if end tag is missing/truncated)
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

    # 3. Ultimate Fallback: If parsing failed, return whole text as target
    if not target_md and not zh_md:
        zh_md = text.strip()

    return target_md, zh_md


def guess_mime(path: Path) -> str:
    m, _ = mimetypes.guess_type(str(path))
    if not m:
        ext = path.suffix.lower()
        if ext in {".jpg", ".jpeg"}: m = "image/jpeg"
        elif ext == ".png": m = "image/png"
        elif ext == ".webp": m = "image/webp"
        elif ext == ".bmp": m = "image/bmp"
        elif ext == ".gif": m = "image/gif"
        else: m = "application/octet-stream"
    return m


def image_path_to_data_url(image_path: Path) -> str:
    mime = guess_mime(image_path)
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_messages(data_url: str, sys_prompt: str, user_instruction: str) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                        "detail": "high"
                    }
                },
                {"type": "text", "text": user_instruction},
            ],
        },
    ]


def call_vision_api(client: OpenAI, model: str, messages: List[Dict[str, Any]], stream: bool,
                    max_tokens: int, temperature: float, top_p: float) -> str:
    if stream:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        out = []
        for chunk in resp:
            delta = getattr(chunk.choices[0].delta, "content", None)
            if delta:
                out.append(delta)
        return "".join(out)
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return "".join(choice.message.content or "" for choice in resp.choices)


def main():
    parser = argparse.ArgumentParser(description="Multi-Language Document Translation (CN -> EN/DE/FR/RU)")
    
    # Core Arguments
    parser.add_argument("--dir", required=True, help="Input directory containing images")
    parser.add_argument("--lang", default="en", choices=["en", "de", "fr", "ru"], 
                        help="Target language: en (English), de (German), fr (French), ru (Russian)")
    
    # Output Configuration
    parser.add_argument("--out_dir", default=None, help="Output directory (default: translations_<lang>)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    
    # Model Parameters
    parser.add_argument("--model", default="doubao-seed-1-6-vision-250815", help="Model endpoint ID")
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--stream", action="store_true", help="Enable streaming response")
    
    # API Configuration
    parser.add_argument("--api_key", type=str, default=None, help="API Key (overrides env var ARK_API_KEY)")
    parser.add_argument("--base_url", type=str, default="https://ark.cn-beijing.volces.com/api/v3", help="API Base URL")

    # Optional Saving
    parser.add_argument("--save_chinese", action="store_true", help="Save reconstructed Chinese OCR text")
    parser.add_argument("--save_full", action="store_true", help="Save raw full API response")

    args = parser.parse_args()

    # 1. Setup Language Config
    lang_cfg = LANG_CONFIG[args.lang]
    target_ext = lang_cfg["ext"]
    print(f"[INFO] Target Language: {lang_cfg['name']} (Tag: <{lang_cfg['tag']}>)")

    # 2. Setup Directories
    in_dir = Path(args.dir)
    default_out_name = f"translations_{args.lang}"
    out_dir = Path(args.out_dir) if args.out_dir else Path(default_out_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    zh_dir = out_dir / "zh" if args.save_chinese else None
    if zh_dir: zh_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = out_dir / "raw" if args.save_full else None
    if raw_dir: raw_dir.mkdir(parents=True, exist_ok=True)

    # 3. Collect Images
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
    image_paths = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in valid_exts])
    
    if not image_paths:
        print(f"[WARN] No images found in {in_dir}. Supported: {valid_exts}")
        return

    # 4. Initialize Client
    api_key = args.api_key or os.getenv("ARK_API_KEY")
    if not api_key:
        raise RuntimeError("API Key not found. Set ARK_API_KEY env var or use --api_key.")
    
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    # 5. Generate Prompts for this language
    sys_prompt, user_instruction = get_prompts(args.lang)

    print(f"[INFO] Processing {len(image_paths)} images -> {out_dir}")

    for img in image_paths:
        out_path = out_dir / (img.stem + target_ext)
        
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {img.name} -> Exists")
            continue

        try:
            data_url = image_path_to_data_url(img)
            messages = build_messages(data_url, sys_prompt, user_instruction)

            full_text = call_vision_api(
                client=client,
                model=args.model,
                messages=messages,
                stream=args.stream,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        except Exception as e:
            print(f"[ERROR] {img.name}: {e}")
            continue

        # Parse based on current language
        target_md, zh_md = parse_response(full_text, args.lang)

        # Save Translation
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(target_md)

        # Save Chinese (Optional)
        zh_log = ""
        if zh_dir and zh_md:
            zh_path = zh_dir / (img.stem + "_zh.md")
            with open(zh_path, "w", encoding="utf-8") as f:
                f.write(zh_md)
            zh_log = " (+zh)"

        # Save Raw (Optional)
        if raw_dir:
            raw_path = raw_dir / (img.stem + ".full.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(full_text)

        print(f"[OK] {img.name} -> {out_path.name}{zh_log}")

if __name__ == "__main__":
    main()