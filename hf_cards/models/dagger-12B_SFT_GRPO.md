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
base_model:
- google/gemma-3-12b-it
language:
- bn
- en
datasets:
- dipta007/dagger
- dipta007/DistractMath-Bn
model-index:
- name: dagger-12B_SFT_GRPO
  results:
  - task:
      type: question-answering
      name: Math Word Problems
    dataset:
      name: MGSM-BN
      type: mgsm
    metrics:
    - type: accuracy
      value: 78.4
      name: Original Accuracy
    - type: accuracy
      value: 64.0
      name: Distractor Accuracy
  - task:
      type: question-answering
      name: Math Word Problems
    dataset:
      name: MSVAMP-BN
      type: msvamp
    metrics:
    - type: accuracy
      value: 78.8
      name: Original Accuracy
    - type: accuracy
      value: 66.8
      name: Distractor Accuracy
---

# DAGGER-12B-SFT-GRPO

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

## Highlights

**DAGGER-12B-SFT-GRPO** is our best-performing model for distractor-aware mathematical reasoning in Bangla. Key features:

- **89% fewer tokens** than reasoning models while achieving comparable accuracy
- **Robust to distractors**: Only 12.0-14.4 point accuracy drop under distractor augmentation (vs. 14-20 for reasoning models, up to 41 for standard CoT)
- **Executable outputs**: Generates computational graphs that can be deterministically executed
- **Explicit distractor modeling**: Identifies irrelevant information as distractor nodes

## Model Overview

| Attribute | Value |
|-----------|-------|
| Base Model | Gemma-3-12B-Instruct |
| Training | SFT → GRPO |
| Parameters | 12B |
| LoRA Rank | 64 |
| Max Sequence Length | 4096 |
| Output Format | JSON Computational Graph |

## Performance

### Accuracy Comparison

| Model | MGSM | MSVAMP | MGSM (+D) | MSVAMP (+D) | Weighted Avg | Tokens |
|-------|------|--------|-----------|-------------|--------------|--------|
| Qwen 3-8B (Reasoning) | 88.0 | 81.1 | 70.5 | 66.9 | 71.4 | 3,128 |
| **DAGGER-12B (Ours)** | **78.4** | **78.8** | **64.0** | **66.8** | **69.4** | **359** |
| Gemma 3-12B (CoT) | 76.8 | 72.3 | 54.3 | 48.7 | 55.7 | 599 |

(+D) = with distractors

### Error Rate by Distractor Type

The paper reports error rates as ranges across all evaluated model categories, not as a
single value per model:

| Distractor Type | Error Rate (across all models) |
|-----------------|--------------------------------|
| Related Entity (RED) | 46 - 94% |
| Orthogonal Attribute (OAD) | 24 - 81% |
| Null-Effect Event (NEED) | 27 - 86% |

RED is the most disruptive category. For this model's own robustness see the accuracy
drop above (12.0 - 14.4 points).

## Output Format

The model generates computational graphs in JSON format:

```json
{
  "nodes": [
    {"id": "n1", "op": "const", "val": 122195, "distractor": false, "label": "মিনার কলম"},
    {"id": "n2", "op": "const", "val": 25084, "distractor": true, "label": "রাজুর কলম"},
    {"id": "n3", "op": "const", "val": 45.6, "distractor": false, "label": "প্রতিটি কলমের দাম"},
    {"id": "total", "op": "mul", "args": ["n1", "n3"], "distractor": false, "label": "মোট টাকা"},
    {"id": "final_result", "op": "identity", "args": ["total"], "distractor": false}
  ]
}
```

**Supported Operations**: `const`, `add`, `sub`, `mul`, `div`, `abs`, `sum`, `mean`, `min`, `max`, `floor`, `ceil`, `round`, `sqrt`, `pow`, `mod`, `gcd`, `lcm`, `identity`

## Quickstart

### Using Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "dipta007/dagger-12B_SFT_GRPO"

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

### Using vLLM

```bash
vllm serve dipta007/dagger-12B_SFT_GRPO --max-model-len 4096
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="dipta007/dagger-12B_SFT_GRPO",
    messages=[
        {"role": "system", "content": "You are an expert Bangla Math Reasoner..."},
        {"role": "user", "content": "মিনার কাছে ১০০টি কলম আছে..."}
    ],
    max_tokens=1024
)
```

### Graph Execution

```python
import json

def execute_graph(graph_json):
    """Execute a computational graph and return the final result."""
    nodes = {n["id"]: n for n in graph_json["nodes"]}
    cache = {}

    def compute(node_id):
        if node_id in cache:
            return cache[node_id]

        node = nodes[node_id]
        op = node["op"]

        if op == "const":
            result = node["val"]
        elif op == "add":
            result = sum(compute(arg) if isinstance(arg, str) else arg for arg in node["args"])
        elif op == "sub":
            args = [compute(arg) if isinstance(arg, str) else arg for arg in node["args"]]
            result = args[0] - args[1]
        elif op == "mul":
            result = 1
            for arg in node["args"]:
                result *= compute(arg) if isinstance(arg, str) else arg
        elif op == "div":
            args = [compute(arg) if isinstance(arg, str) else arg for arg in node["args"]]
            result = args[0] / args[1]
        elif op == "identity":
            result = compute(node["args"][0])
        # ... add other operations

        cache[node_id] = result
        return result

    return compute("final_result")

# Parse and execute
graph = json.loads(response)
answer = execute_graph(graph)
print(f"Answer: {answer}")
```

## Training Details

### Stage 1: Supervised Fine-Tuning (SFT)

| Parameter | Value |
|-----------|-------|
| Base Model | Gemma-3-12B-Instruct |
| LoRA Rank / Alpha | 64 / 128 |
| Global Batch Size | 256 |
| Epochs | 4 |
| Learning Rate | 1e-5 → 1e-6 (cosine) |
| Training Data | 3,000 examples |

### Stage 2: Group Relative Policy Optimization (GRPO)

| Parameter | Value |
|-----------|-------|
| Base Model | SFT Checkpoint |
| LoRA Rank / Alpha | 64 / 128 |
| Global Batch Size | 32 |
| Generations per Prompt | 8 |
| Epochs | 4 |
| Loss Type | BNPO |
| β / ε / ε_high | 0.0 / 0.2 / 0.28 |

**Reward Function:**
```
R(g, y) = 0.5 * I_fmt + 0.5 * I_exec + I_acc(exec(g), y)
```
- `I_fmt`: Valid JSON format (+0.5)
- `I_exec`: Successful execution (+0.5)
- `I_acc`: Correct answer (+1.0)

## Best Practices

1. **Temperature**: Use `temperature=0.7` with `top_p=0.8` for best results
2. **Max Tokens**: 1024 tokens is sufficient for most problems
3. **Prompt**: Send the full graph instructions in the **user** message with no system turn, exactly as in the Quickstart. That is the format the model was trained on
4. **Post-processing**: Parse JSON and execute graph for final numeric answer

## Limitations

- Designed for arithmetic word problems; may not generalize to algebra, geometry, or calculus
- Primarily trained on Bangla; English performance not evaluated
- Requires JSON parsing and graph execution for final answers
- 4B variant shows lower performance, suggesting capacity requirements

## Related Models

| Model | Training | Weighted Avg |
|-------|----------|--------------|
| [dagger-12B_SFT_GRPO](https://huggingface.co/dipta007/dagger-12B_SFT_GRPO) | SFT → GRPO | **69.4** |
| [dagger-12B_SFT](https://huggingface.co/dipta007/dagger-12B_SFT) | SFT only | 66.7 |
| [dagger-12B_GRPO](https://huggingface.co/dipta007/dagger-12B_GRPO) | Base → GRPO | 61.5 |
| [dagger-4B_SFT_GRPO](https://huggingface.co/dipta007/dagger-4B_SFT_GRPO) | SFT → GRPO | 47.3 |

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

## Acknowledgments

- [Google Gemma](https://ai.google.dev/gemma) for the base model
- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- [TRL](https://github.com/huggingface/trl) for GRPO implementation
