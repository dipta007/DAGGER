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

<a href="https://arxiv.org/abs/XXXX.XXXXX" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b" style="display: inline-block; vertical-align: middle;"/>
</a>
<a href="https://github.com/dipta007/dagger" target="_blank">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Code-black" style="display: inline-block; vertical-align: middle;"/>
</a>

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

question = "মিনার কাছে ১০০টি কলম আছে। প্রতিটি কলমের দাম ৫ টাকা।"

messages = [
    {"role": "system", "content": "You are an expert Bangla Math Reasoner. Solve by constructing a Computational Graph."},
    {"role": "user", "content": question}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=1024)
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
| [dagger-4B_GRPO](https://huggingface.co/dipta007/dagger-4B_GRPO) | Base → GRPO | 29.3 |

## Citation

```bibtex
will be updated
```
