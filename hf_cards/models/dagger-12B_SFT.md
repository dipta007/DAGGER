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
base_model:
- google/gemma-3-12b-it
language:
- bn
- en
datasets:
- dipta007/dagger
- dipta007/DistractMath-Bn
---

# DAGGER-12B-SFT

<a href="https://arxiv.org/abs/XXXX.XXXXX" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b" style="display: inline-block; vertical-align: middle;"/>
</a>
<a href="https://github.com/dipta007/dagger" target="_blank">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Code-black" style="display: inline-block; vertical-align: middle;"/>
</a>

## Model Description

**DAGGER-12B-SFT** is a supervised fine-tuned model for computational graph generation in Bangla mathematical reasoning. This is the SFT-only variant, serving as both a standalone model and initialization for GRPO training.

## Highlights

- **SFT-only training** on 3,000 verified computational graph examples
- **Strong baseline performance** for distractor-aware reasoning
- **Foundation for GRPO**: Used as initialization for [dagger-12B_SFT_GRPO](https://huggingface.co/dipta007/dagger-12B_SFT_GRPO)
- **Efficient inference**: ~400 tokens per problem

## Model Overview

| Attribute | Value |
|-----------|-------|
| Base Model | Gemma-3-12B-Instruct |
| Training | Supervised Fine-Tuning |
| Parameters | 12B |
| LoRA Rank | 64 |
| Max Sequence Length | 4096 |

## Performance

| Dataset | Original | +Distractor | Drop |
|---------|----------|-------------|------|
| MGSM | 70.0 | 56.8 | 13.2 |
| MSVAMP | 76.8 | 65.4 | 11.5 |
| **Weighted Avg** | - | - | **66.7** |

### Comparison with GRPO

| Model | Weighted Avg Accuracy |
|-------|----------------------|
| dagger-12B_SFT | 66.7 |
| dagger-12B_SFT_GRPO | **69.4** (+2.7) |

GRPO provides +2.7 points improvement over SFT alone.

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "dipta007/dagger-12B_SFT"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

question = "মিনার কাছে ১০০টি কলম আছে। প্রতিটি কলমের দাম ৫ টাকা। মিনা সব কলম বিক্রি করলে কত টাকা পাবে?"

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
| Optimizer | AdamW |
| Weight Decay | 0.001 |
| Precision | BF16 |

## When to Use This Model

- **As a baseline**: Compare against GRPO-enhanced variants
- **For GRPO initialization**: Use as starting point for policy optimization
- **Resource-constrained settings**: When GRPO training is not feasible
- **Research**: Studying the effect of SFT vs. GRPO on graph generation

## Related Models

| Model | Training | Performance |
|-------|----------|-------------|
| **dagger-12B_SFT** | SFT | 66.7 |
| [dagger-12B_SFT_GRPO](https://huggingface.co/dipta007/dagger-12B_SFT_GRPO) | SFT → GRPO | **69.4** |
| [dagger-12B_GRPO](https://huggingface.co/dipta007/dagger-12B_GRPO) | Base → GRPO | 69.4 |

## Citation

```bibtex
will be updated
```
