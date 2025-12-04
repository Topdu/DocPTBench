import os
import argparse
import base64
import mimetypes
import re
from pathlib import Path
from typing import Tuple, List, Dict, Any

from openai import OpenAI

# ==============================================================================
# 1. Multi-language Configuration
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

# ==============================================================================
# 2. Dynamic Prompt Templates (English Source -> Target)
# ==============================================================================

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

# ==============================================================================
# 3. Helper Functions
# ==============================================================================

def get_prompts(lang_code: str) -> Tuple[str, str]:
    cfg = LANG_CONFIG[lang_code]
    # Use .format for interpolation
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
    """
    Returns (target_md, en_md).
    Note: The prompt requests <english_md> first, followed by <target_md>.
    """
    tag = LANG_CONFIG[lang_code]["tag"]
    
    # Dynamic regex
    # Match target language block
    target_re = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.S | re.I)
    # Match English source block
    en_re = re.compile(r"<english_md>\s*(.*?)\s*</english_md>", re.S | re.I)

    target_md, en_md = "", ""

    # 1. Extract Target Language (Translation)
    m_target = target_re.search(text)
    if m_target:
        target_md = m_target.group(1).strip()
    else:
        # Fallback: If closing tag is missing, try matching everything after the opening tag
        m_target_loose = re.search(rf"<{tag}>\s*(.*)", text, re.S | re.I)
        if m_target_loose:
            target_md = m_target_loose.group(1).strip()
        else:
            # Secondary Fallback: If tags are missing, check content after </english_md>
            m_after_en = re.search(r"</english_md>\s*(.*)", text, re.S | re.I)
            if m_after_en:
                target_md = m_after_en.group(1).strip()

    # 2. Extract English Source (OCR)
    m_en = en_re.search(text)
    if m_en:
        en_md = m_en.group(1).strip()
    
    # 3. Extreme Fallback
    # If parsing failed but text exists, assume it's English OCR (since source is English)
    if not en_md and not target_md:
        en_md = text.strip()

    return target_md, en_md


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


def build_messages(data_url: str, sys_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
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
                {"type": "text", "text": user_prompt},
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

# ==============================================================================
# 4. Main Execution Flow
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Per-image doc translation (EN Source -> Multi-Target) with Doubao Vision")
    
    # Basic arguments
    parser.add_argument("--dir", required=True, help="Input image directory (English documents)")
    parser.add_argument("--out_dir", default=None, help="Output directory (default: translations_doubao_{lang})")
    
    # Language selection
    parser.add_argument("--lang", default="zh", choices=["zh", "de", "fr", "ru"], 
                        help="Target language: zh (Chinese), de (German), fr (French), ru (Russian)")
    
    # API and Model arguments
    parser.add_argument("--model", default="doubao-seed-1-6-vision-250815", help="Model name")
    parser.add_argument("--api_key", type=str, default=None, help="API Key (or set via env var ARK_API_KEY)")
    parser.add_argument("--base_url", type=str, default="https://ark.cn-beijing.volces.com/api/v3", help="Base URL")
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--stream", action="store_true", help="Enable stream decoding")
    
    # File control
    parser.add_argument("--overwrite", action="store_true", help="Allow overwrite of existing files")
    parser.add_argument("--save_english", action="store_true", help="Save OCR English source (.en.md)")
    parser.add_argument("--save_full", action="store_true", help="Save full raw output (.full.txt)")
    
    args = parser.parse_args()

    # 1. Configuration Initialization
    lang_cfg = LANG_CONFIG[args.lang]
    target_ext = lang_cfg["ext"]
    
    in_dir = Path(args.dir)
    
    # Default output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(f"translations_doubao_{args.lang}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auxiliary directories
    en_dir = out_dir / "en_ocr" if args.save_english else None
    if en_dir: en_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = out_dir / "raw" if args.save_full else None
    if raw_dir: raw_dir.mkdir(parents=True, exist_ok=True)

    # 2. Scan Images
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
    image_paths = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])
    if not image_paths:
        raise ValueError(f"No images found in directory: {in_dir}")

    # 3. Initialize Client
    api_key = args.api_key or os.getenv("ARK_API_KEY")
    if not api_key:
        raise RuntimeError("Please set ARK_API_KEY environment variable or pass via --api_key.")
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    print(f"[INFO] Target Language: {lang_cfg['name']} ({args.lang})")
    print(f"[INFO] Found {len(image_paths)} images. Output: {out_dir.resolve()}")
    print(f"[INFO] Model: {args.model}")

    # 4. Processing Loop
    for img in image_paths:
        target_path = out_dir / (img.stem + target_ext)
        
        if target_path.exists() and not args.overwrite:
            print(f"[SKIP] {target_path.name} exists.")
            continue

        # Prepare Prompt
        sys_prompt, user_prompt = get_prompts(args.lang)
        data_url = image_path_to_data_url(img)
        messages = build_messages(data_url, sys_prompt, user_prompt)

        try:
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
            print(f"[ERR] {img.name}: {e}")
            continue

        # Parse Results
        target_md, en_md = parse_dual_blocks(full_text, args.lang)

        # Save Target Translation
        if target_md:
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(target_md)
        else:
            print(f"[WARN] {img.name}: No translation block found.")

        # Optional: Save English OCR
        en_msg = ""
        if args.save_english and en_dir and en_md:
            en_path = en_dir / (img.stem + ".en.md")
            with open(en_path, "w", encoding="utf-8") as f:
                f.write(en_md)
            en_msg = f" (+en)"

        # Optional: Save Full Output
        if raw_dir:
            raw_path = raw_dir / (img.stem + ".full.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(full_text)

        print(f"[OK] {img.name} → {target_path.name}{en_msg}"
              + (f" (+raw)" if raw_dir else ""))

if __name__ == "__main__":
    main()