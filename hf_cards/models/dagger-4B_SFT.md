---
library_name: transformers
license: gemma
license_link: https://ai.google.dev/gemma/terms
pipeline_tag: text-generation
tags:
- math
- reasoning
- computational-graph
- bangla
- low-resource
- distractor-aware
- sft
- small-model
base_model:
- google/gemma-3-4b-it
language:
- bn
- en
datasets:
- dipta007/dagger
- dipta007/DistractMath-Bn
---

# DAGGER-4B-SFT

<p align="center">
  <a href="https://arxiv.org/abs/2601.06853">
    <img src="https://img.shields.io/badge/%F0%9F%94%A5_Accepted_at-EMNLP_2026_(Findings)_%F0%9F%94%A5-b12a00?style=for-the-badge&labelColor=ffb300" alt="Accepted at EMNLP 2026 (Findings)">
  </a>
</p>

[![EMNLP 2026 (Findings)](https://img.shields.io/badge/EMNLP%202026-Findings-blue)](https://arxiv.org/abs/2601.06853)
[![Paper](https://img.shields.io/badge/arXiv-2601.06853-red)](https://arxiv.org/abs/2601.06853)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://dipta007.github.io/DAGGER/)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/dipta007/DistractMath-Bn)
[![Models](https://img.shields.io/badge/HuggingFace-Models-orange)](https://huggingface.co/collections/dipta007/dagger-emnlp-2026-findings)
[![GitHub](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/dipta007/dagger)

## Model Description

**DAGGER-4B-SFT** is a supervised fine-tuned 4B model for computational graph generation. This model serves as initialization for GRPO training and as a lightweight baseline.

## Model Overview

| Attribute | Value |
|-----------|-------|
| Base Model | Gemma-3-4B-Instruct |
| Training | Supervised Fine-Tuning |
| Parameters | 4B |
| LoRA Rank | 64 |

## Performance

| Dataset | Original | +Distractor | Drop |
|---------|----------|-------------|------|
| MGSM | 40.4 | 25.1 | 15.3 |
| MSVAMP | 65.0 | 42.4 | 22.7 |
| **Weighted Avg** | - | - | **44.3** |

### Improvement from GRPO

| Model | Weighted Avg |
|-------|--------------|
| dagger-4B_SFT | 44.3 |
| dagger-4B_SFT_GRPO | **47.3** (+3.0) |

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "dipta007/dagger-4B_SFT"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

USER_PROMPT_TEMPLATE = """You are an expert Bengali Math Reasoner. Your task is to solve mathematical problems by constructing a "Computational Graph".

### Graph Rules:
- `id`: Unique identifier (e.g., "n1", "n2").
- `val`: The raw number extracted from text (for input nodes).
- `op`: The operation (`add`, `sub`, `mul`, `div`, `round`, `sqrt`, `floor`, `sum`, `mean`). Use `const` for input numbers.
- `args`: List of input node IDs.
- `distractor`: Boolean (`true` / `false`). Set to `true` if the node is NOT used in the final calculation path.
- `label`: Label for the node.

### Available Operations:
- Input: `const` (Use this for all numbers found in text or constants).
- Arithmetic: `add`, `sub`, `mul`, `div`, `abs` (absolute difference).
- Logic/Stats: `sum`, `mean`, `min` (minimum), `max` (maximum).
- Rounding: `round` (nearest int), `floor` (round down), `ceil` (round up).
- Advanced: `sqrt`, `pow`, `mod` (remainder), `gcd`, `lcm`.
- Output: `identity` ("final_result" points to the answer node)

Only output a JSON graph representing the solution, nothing else. Nodes must be topologically sorted, and there must be exactly one "final_result" node that represents the final answer. One example is provided below.

### Example:
Question:
মিনার কাছে ১২২১৯৫ টা কলম আছে। রাজুর কাছে ২৫০৮৪ টা কলম আছে। মিনা রাজুর কাছে ১১২৬ টি কলম চাইল। রাজু ১০০০ টি কলম দিতে রাজি হল, কিন্তু পরে আর দিলেনা। প্রতিটি কলমের দাম ৪৫.৬ টাকা। মিনা যদি কলমগুলো বিক্রি করতে চায়, সে কত টাকা পাবে?

Output:
```json
{{
  "nodes": [
    {{"id": "n1", "op": "const", "val": 122195, "distractor": false, "label": "মিনার কলম"}},
    {{"id": "n2", "op": "const", "val": 25084, "distractor": true, "label": "রাজুর কলম"}},
    {{"id": "n3", "op": "const", "val": 1126, "distractor": true, "label": "মিনা রাজুর কাছে চাইল"}},
    {{"id": "n4", "op": "const", "val": 1000, "distractor": true, "label": "রাজু দিতে রাজি হল"}},
    {{"id": "n5", "op": "const", "val": 45.6, "distractor": false, "label": "প্রতিটি কলমের দাম"}},
    {{"id": "total_money", "op": "mul", "args": ["n1", "n5"], "distractor": false, "label": "মিনার মোট টাকা"}},
    {{"id": "final_result", "op": "identity", "args": ["total_money"], "distractor": false, "label": "চূড়ান্ত উত্তর"}}
  ]
}}```

### Your Task:

Question:
{question}

Output:
"""

question = "রজারের 5টি টেনিস বল আছে। সে আরও 2 ক্যান টেনিস বল কিনেছে। প্রতিটি ক্যানে 3টি করে টেনিস বল আছে। তার কাছে এখন কতগুলি টেনিস বল আছে?"
prompt = USER_PROMPT_TEMPLATE.format(question=question)

messages = [
  {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# Generate
outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.7, top_p=0.8)
response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

print(response)
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| LoRA Rank / Alpha | 64 / 128 |
| Global Batch Size | 256 |
| Epochs | 4 |
| Learning Rate | 1e-5 → 1e-6 |
| Precision | BF16 |

## When to Use This Model

- **GRPO initialization**: Starting point for policy optimization
- **Lightweight baseline**: When 12B models are too large
- **Ablation studies**: Comparing SFT vs. GRPO contributions

## Related Models

| Model | Training | Weighted Avg |
|-------|----------|--------------|
| **dagger-4B_SFT** | SFT | 44.3 |
| [dagger-4B_SFT_GRPO](https://huggingface.co/dipta007/dagger-4B_SFT_GRPO) | SFT → GRPO | 47.3 |
| [dagger-4B_GRPO](https://huggingface.co/dipta007/dagger-4B_GRPO) | Base → GRPO | 32.5 |

## License and Data Provenance

Model weights are released under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

**Training data is not fully permissive.** Part of the SFT data and all GRPO prompts come
from `numina-math-cot-bn`, which is **CC BY-NC-SA 4.0 (NonCommercial, ShareAlike)**. For
commercial use, re-derive that portion from the Apache-2.0 upstream
[AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT).

## Citation

```bibtex
@inproceedings{nazi2026dagger,
  title={{\dag}DAGGER: Distractor-Aware Graph Generation for Executable Reasoning in Math Problems},
  author={Zabir Al Nazi and Shubhashis Roy Dipta and Sudipta Kar},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2026},
  year={2026},
  eprint={2601.06853},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2601.06853},
}
```
