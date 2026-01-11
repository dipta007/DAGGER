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

<a href="https://arxiv.org/abs/XXXX.XXXXX" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b" style="display: inline-block; vertical-align: middle;"/>
</a>
<a href="https://github.com/dipta007/dagger" target="_blank">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Code-black" style="display: inline-block; vertical-align: middle;"/>
</a>
<a href="https://huggingface.co/datasets/dipta007/DistractMath-Bn" target="_blank">
    <img alt="Dataset" src="https://img.shields.io/badge/Dataset-DistractMath--BN-green" style="display: inline-block; vertical-align: middle;"/>
</a>

## Highlights

**DAGGER-12B-SFT-GRPO** is our best-performing model for distractor-aware mathematical reasoning in Bangla. Key features:

- **89% fewer tokens** than reasoning models while achieving comparable accuracy
- **Robust to distractors**: Only 12-14 point accuracy drop under distractor augmentation (vs. 14-20 for reasoning models, 18-41 for standard CoT)
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

### Robustness (Accuracy Drop)

| Distractor Type | Error Rate |
|-----------------|------------|
| Related Entity (RED) | 36% |
| Orthogonal Attribute (OAD) | 34% |
| Null-Effect Event (NEED) | 33% |

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

**Supported Operations**: `const`, `add`, `sub`, `mul`, `div`, `sum`, `mean`, `min`, `max`, `floor`, `ceil`, `round`, `sqrt`, `pow`, `mod`, `gcd`, `lcm`, `identity`

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

# Prepare input
question = "মিনার কাছে ১০০টি কলম আছে। প্রতিটি কলমের দাম ৫ টাকা। মিনা সব কলম বিক্রি করলে কত টাকা পাবে?"

messages = [
    {"role": "system", "content": "You are an expert Bangla Math Reasoner. Solve by constructing a Computational Graph in JSON format."},
    {"role": "user", "content": question}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

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
3. **System Prompt**: Include the graph generation instructions in system message
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
| [dagger-12B_GRPO](https://huggingface.co/dipta007/dagger-12B_GRPO) | Base → GRPO | 69.4 |
| [dagger-4B_SFT_GRPO](https://huggingface.co/dipta007/dagger-4B_SFT_GRPO) | SFT → GRPO | 47.3 |

## Citation

```bibtex
will be updated
```

## Acknowledgments

- [Google Gemma](https://ai.google.dev/gemma) for the base model
- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- [TRL](https://github.com/huggingface/trl) for GRPO implementation
