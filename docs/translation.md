# Translation Benchmark

In this walkthrough, we will demonstrate how to perform document translation using various Multimodal Large Language Models (MLLMs) and evaluate the results within the DocPTBench framework.

## 1. Environment Setup

```bash
# Create environment
conda create -n docptbench_trans python=3.10
conda activate docptbench_trans

# 1. Install Model SDKs & Inference Dependencies
pip install openai dashscope google-generativeai volcengine Pillow tqdm

# 2. Install Evaluation Metrics Dependencies
# sacrebleu: for BLEU/chrF
# zss: for STEDS (Tree Edit Distance)
# jieba: for Chinese text segmentation
# nltk: for METEOR score
pip install sacrebleu zss jieba nltk
```



## 2. Data Preparation


```bash
# from huggingface download DocPTBench dataset
huggingface-cli download topdu/DocPTBench --local-dir ./DocPTBench --repo-type dataset
# from modelscope download DocPTBench datasets
# modelscope download --dataset topdktu/DocPTBench --local_dir ./DocPTBench
```

**Directory Structure:**

```
DocPTBench/                   <-- Root of the cloned repo
├── DocPTBench_combined.json  <-- Ground Truth
├── images/                      <-- Image folder
│   ├── images_synreal/         # 981 synthesized images (imitating real-world scenarios)
│   ├── images_synreal_dewarp/  # 981 unwarped synthesized images
│   ├── images_pic_synreal/     # 400 photos shot in the real world
│   └── images_pic_dewarp/      # 400 unwarped photos
├── translation_gt/
|   ├── src_en/                 # En-Zh/De/Fr/Ru ground truth annotations
│   └── src_zh/                 # Zh-En/De/Fr/Ru ground truth annotations
```

## 3. Inference (Generating .md Files)

The inference code is located in `code/translation/model_infer`. We provide three different prompting strategies for translation:

1.  **Simple Prompt (`simple_prompt`)**: Direct style (Image + "Translate this document").
2.  **Chain-of-Thought (`cot_prompt`)**: Two-step CoT reasoning (Image + "Extract text first, then translate").
3.  **Text-Only (`text`)**: Translation based on ground-truth OCR text (Text + "Translate this text").

### Directory Structure

The scripts are organized by source language:
*   `src_en`: Translating **English** documents to Chinese.
*   `src_zh`: Translating **Chinese** documents to English.

```text
code/translation/model_infer/
├── cot_prompt/
│   ├── src_en/  <-- Scripts for En->Zh/De/Fr/Ru (CoT)
│   └── src_zh/  <-- Scripts for Zh->En/De/Fr/Ru (CoT)
├── simple_prompt/
│   ├── src_en/  <-- Scripts for En->Zh/De/Fr/Ru (Simple)
│   └── src_zh/  <-- Scripts for Zh->En/De/Fr/Ru (Simple)
└── text/
    ├── src_en/  <-- Scripts for En->Zh/De/Fr/Ru (TextOnly)
    └── src_zh/  <-- Scripts for Zh->En/De/Fr/Ru (TextOnly)
```

### Running Inference

Choose the model script corresponding to your desired strategy and language direction. Below is a reference example for running **Qwen-VL-Max** using the `simple_prompt` strategy for English-to-Chinese translation.

**Example Script Usage:**

```bash
# Run Qwen-VL-Max (Simple Prompt, En->Zh)
python code/translation/model_infer/simple_prompt/src_en/Qwen_VL_max.py --dir /path/to/input_images/ --lang zh --out_dir /path/to/output_folder/ --api_key xxxx
```


## 4. Evaluation

After generating the translation results, we use BLEU, METEOR, chrF and STEDS metrics to evaluate the performance against the ground truth.

The evaluation script and ground truth annotations are located in `code/translation/evaluate_metric`.

### Ground Truth Files
*   `split_ch_en.json`: Ground truth filenames for **Chinese -> English** translation.
*   `split_en_ch.json`: Ground truth filenames for **English -> Chinese** translation.

### Running Evaluation

Use `evaluate.py` to calculate scores. You need to specify the path to your model's output directory and the corresponding ground truth directory in evaluate.py.

**Example Script Usage:**

```bash
python code/translation/evaluate_metric/evaluate.py \
    --split_json_path code/translation/evaluate_metric/split_ch_en.json
```

### Output
The script will print the evaluation metrics (BLEU, METEOR, chrF, STEDS) to the console.