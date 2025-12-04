#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import argparse
from collections import defaultdict
from tqdm import tqdm
from sacrebleu import sentence_bleu
from sacrebleu.metrics import CHRF
from zss import simple_distance, Node

# ========================== Dependency Checks ==========================
try:
    import jieba
    _jieba_installed = True
except (ImportError, ModuleNotFoundError):
    _jieba_installed = False
    print("[WARNING] Chinese tokenizer 'jieba' not installed. Run 'pip install jieba' if evaluating Chinese tasks.")

try:
    from nltk.translate.meteor_score import meteor_score as _meteor_score
except (ImportError, ModuleNotFoundError):
    _meteor_score = None
    print("[WARNING] NLTK or its METEOR component not installed. METEOR metric calculation will be skipped.")
# =======================================================================


# ========================== User Configuration ==========================
EVALUATION_CONFIG = [
    {
        "model_name": "",
        "category": "Open Source Models",
        "gt_dir": "/gt/markdown_ch",
        "pred_dir": "/pred/markdown_ch",
        "is_chinese": True,
    }
]
# =======================================================================


# ========================== Core Functions ==========================

def remove_markdown_links_and_images(text):
    """
    Removes all forms of Markdown links and images from the text.
    Handles inline, reference-style, and reference definitions.
    """
    # 1. Inline links/images: [text](url) or ![alt](src)
    text = re.sub(r'!?[.*?]\(.*?\)', '', text)
    # 2. Reference-style links/images: [text][id] or ![alt][id]
    text = re.sub(r'!?[.*?][.*?]', '', text)
    # 3. Reference definitions: [id]: url "title" (must be at the start of a line)
    text = re.sub(r'^\s*[.+?]:.*$', '', text, flags=re.MULTILINE)
    return text

def get_tokenized_text(file_path, is_chinese=False):
    """
    Reads a file, cleans it of links/images, and returns a tokenized text string.
    """
    split_lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        # Clean the line before processing
        cleaned_line = remove_markdown_links_and_images(line)
        line_strip = cleaned_line.strip()
        
        if not line_strip:
            continue

        if is_chinese:
            if not _jieba_installed:
                if 'jieba_warning_shown' not in globals():
                    tqdm.write("[WARNING] 'is_chinese' is True but jieba is not installed. BLEU/METEOR scores will be inaccurate.")
                    globals()['jieba_warning_shown'] = False
                tokenized_line = ' '.join(line_strip.split())
            else:
                tokenized_line = ' '.join(jieba.cut(line_strip))
        else:
            tokenized_line = ' '.join(line_strip.split())
        
        split_lines.append(tokenized_line)
        
    return ' \n\n '.join(split_lines)

def pre_clean(text):
    text = re.sub(r'<bos>|<eos>|<pad>|<unk>', '', text)
    text = re.sub(r'\s##(\S)', r'\1', text)
    text = re.sub(r'\\\s', r'\\', text)
    text = re.sub(r'\s\*\s\*\s', r'**', text)
    text = re.sub(r'{\s', r'{', text)
    text = re.sub(r'\s}', r'}', text)
    text = re.sub(r'\\begin\s', r'\\begin', text)
    text = re.sub(r'\\end\s', r'\\end', text)
    text = re.sub(r'\\end{table}', r'\\end{table} \n\n', text)
    text = text.replace('\n', ' ')
    return text

def metric_post_process(text):
    return pre_clean(text)

def get_tree(mmd_file_path):
    tree = (Node('ROOT').addkid(Node('TITLE')))
    with open(mmd_file_path, 'r', encoding='utf-8') as f:
        mmd_lines = f.readlines()
    lines = []
    for line in mmd_lines:
        # Clean the line before processing for STEDS
        cleaned_line = remove_markdown_links_and_images(line)
        line = pre_clean(cleaned_line)
        if line.strip() != '':
            lines.append(line.strip())
            
    last_title = ''
    for line in lines:
        if line.startswith('#'):
            child = tree.get('ROOT')
            line = line.replace('#', '')
            child.addkid(Node(line))
            last_title = line
        else:
            if last_title == '':
                child = tree.get('TITLE')
                child.addkid(Node(line))
            else:
                child = tree.get(last_title)
                child.addkid(Node(line))
    return tree

def STEDS(pred_tree, ref_tree):
    def my_distance(pred, ref):
        return 1 if len(pred.split()) == 0 or len(ref.split()) == 0 else 0
    
    num_nodes = max(len(list(pred_tree.iter())), len(list(ref_tree.iter())))
    if num_nodes == 0:
        return 1.0
    
    total_distance = simple_distance(pred_tree, ref_tree, label_dist=my_distance)
    return 1 - total_distance / num_nodes

def run_single_evaluation(pred_dir, ref_dir, split_json_path, is_chinese=False):
    """
    Performs evaluation on a single model's predictions against a reference directory.
    """
    try:
        with open(split_json_path, 'r', encoding='utf-8') as f:
            test_name_list = json.load(f)['test_name_list']
    except FileNotFoundError:
        print(f"[ERROR] Split JSON file '{split_json_path}' not found.")
        return None

    pred_tok_list, ref_tok_list = [], []
    pred_char_list, ref_char_list = [], []
    steds_scores = []
    meteor_vals = []
 
    n_used = 0
    for name in tqdm(test_name_list, desc=f"  -> Evaluating files", leave=False):
        pred_path = os.path.join(pred_dir, name + '.md')
        ref_path = os.path.join(ref_dir, name + ".md")
        if not (os.path.isfile(pred_path)):
            pred_path = os.path.join(pred_dir, name + '_ch.md')
        if not (os.path.isfile(pred_path) and os.path.isfile(ref_path)):
            continue
        
        # For BLEU and METEOR (token-based)
        pred_tok = metric_post_process(get_tokenized_text(pred_path, is_chinese=is_chinese))
        ref_tok = metric_post_process(get_tokenized_text(ref_path, is_chinese=is_chinese))
        pred_tok_list.append(pred_tok)
        ref_tok_list.append(ref_tok)

        # For chrF (character-based)
        with open(pred_path, 'r', encoding='utf-8') as f_pred, open(ref_path, 'r', encoding='utf-8') as f_ref:
            pred_char_content = f_pred.read()
            ref_char_content = f_ref.read()
        
        # Clean full content before chrF calculation
        cleaned_pred_char = remove_markdown_links_and_images(pred_char_content)
        cleaned_ref_char = remove_markdown_links_and_images(ref_char_content)
        
        pred_char_list.append(metric_post_process(cleaned_pred_char))
        ref_char_list.append(metric_post_process(cleaned_ref_char))

        # For STEDS (tree-based, already handled inside get_tree)
        steds_scores.append(STEDS(get_tree(pred_path), get_tree(ref_path)))

        if _meteor_score:
            hyp_tokens = [t for t in pred_tok.split() if t]
            ref_tokens = [t for t in ref_tok.split() if t]
            if hyp_tokens and ref_tokens:
                meteor_vals.append(_meteor_score([ref_tokens], hyp_tokens) * 100.0)

        n_used += 1

    if n_used == 0:
        print(f"\n[WARNING] No valid evaluation pairs found between '{os.path.basename(pred_dir)}' and '{os.path.basename(ref_dir)}'.")
        return None

    bleu_scores = [sentence_bleu(hyp, [ref]).score for hyp, ref in zip(pred_tok_list, ref_tok_list)]
    bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    
    chrf_metric = CHRF()
    chrf_scores = [chrf_metric.sentence_score(hyp, [ref]).score for hyp, ref in zip(pred_char_list, ref_char_list)]
    chrf_score = sum(chrf_scores) / len(chrf_scores) if chrf_scores else 0.0

    meteor_score = (sum(meteor_vals) / len(meteor_vals)) if meteor_vals else None
    steds_score = (sum(steds_scores) / len(steds_scores)) * 100.0 if steds_scores else 0.0

    return {
        'bleu': bleu_score,
        'chrf': chrf_score,
        'meteor': meteor_score,
        'steds': steds_score,
    }

# ========================== Report Generation ==========================
def generate_markdown_table(title, results_data):
    """Generates a Markdown table based on the provided data."""
    has_meteor = any(res['scores']['meteor'] is not None for res in results_data)
    
    header = f"| {title:<32} |      BLEU |      chrF |"
    separator = f"| {'-'*32} | --------: | --------: |"
    if has_meteor:
        header += "    METEOR |"
        separator += " --------: |"
    header += "     STEDS |"
    separator += " --------: |"
    
    print(header)
    print(separator)
    
    for res in sorted(results_data, key=lambda x: x['model_name']):
        scores = res['scores']
        row = f"| **{res['model_name']}**{' '*(32-len(res['model_name'])-2)} |"
        
        row += f" {scores.get('bleu', 0.0):>7.2f} |"
        row += f" {scores.get('chrf', 0.0):>7.2f} |"
        if has_meteor:
            meteor_str = f"{scores.get('meteor', 0.0):>7.2f}" if scores.get('meteor') is not None else "N/A".center(8)
            row += f" {meteor_str} |"
        row += f" {scores.get('steds', 0.0):>7.2f} |"
        
        print(row)

def generate_full_report(all_results):
    """Organizes and prints Markdown tables for all evaluation results by category."""
    if not all_results:
        print("\n[INFO] No valid evaluation results collected. Cannot generate report.")
        return
        
    categorized_results = defaultdict(list)
    for res in all_results:
        categorized_results[res['category']].append(res)
        
    categories_order = ["Open Source Models", "Closed Source Models"]
    for category in categories_order:
        if category in categorized_results:
            print("\n")
            generate_markdown_table(category, categorized_results[category])
    
    for category, results in categorized_results.items():
        if category not in categories_order:
            print("\n")
            generate_markdown_table(category, results)

# ========================== Main Execution ==========================
def main():
    parser = argparse.ArgumentParser(description="Integrated evaluation and report generation script based on internal configuration.")
    parser.add_argument('--split_json_path', type=str, required=False, help="Path to the JSON file containing test filenames.")
    args = parser.parse_args()

    all_evaluation_results = []
    print(f"Found {len(EVALUATION_CONFIG)} model configurations to evaluate.")
    
    for model_config in tqdm(EVALUATION_CONFIG, desc="Overall Progress"):
        model_name = model_config["model_name"]
        category = model_config["category"]
        pred_dir = model_config["pred_dir"]
        gt_dir = model_config["gt_dir"]
        is_chinese = model_config.get("is_chinese", False)

        tqdm.write(f"\n--- Processing Model: {model_name} (Chinese Mode: {'Yes' if is_chinese else 'No'}) ---")

        if not os.path.isdir(pred_dir) or not os.path.isdir(gt_dir):
            tqdm.write(f"[WARNING] Skipping '{model_name}' because directory does not exist: \n  Pred: '{pred_dir}'\n  GT:   '{gt_dir}'")
            continue
        
        scores = run_single_evaluation(pred_dir, gt_dir, args.split_json_path, is_chinese=is_chinese)
        
        if scores:
            all_evaluation_results.append({
                "model_name": model_name,
                "category": category,
                "scores": scores
            })
            tqdm.write(f"--- Model {model_name} evaluation finished ---")

    print("\n" + "="*80)
    print("All evaluations completed. Generating final report...")
    print("="*80)
    generate_full_report(all_evaluation_results)
    print("\nReport generation finished.")


if __name__ == '__main__':
    main()