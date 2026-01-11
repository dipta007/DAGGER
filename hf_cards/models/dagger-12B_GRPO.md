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
- grpo
- reinforcement-learning
base_model:
- google/gemma-3-12b-it
language:
- bn
- en
datasets:
- dipta007/dagger
- dipta007/DistractMath-Bn
---

# DAGGER-12B-GRPO

<a href="https://arxiv.org/abs/XXXX.XXXXX" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b" style="display: inline-block; vertical-align: middle;"/>
</a>
<a href="https://github.com/dipta007/dagger" target="_blank">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Code-black" style="display: inline-block; vertical-align: middle;"/>
</a>

## Model Description

**DAGGER-12B-GRPO** is trained with Group Relative Policy Optimization (GRPO) directly from the base Gemma-3-12B model, **without SFT initialization**. This model demonstrates that GRPO alone can learn computational graph generation, though SFT initialization provides better distractor robustness.

## Highlights

- **Base → GRPO training** (no SFT phase)
- **Executable reward signal**: Learns from format, execution, and correctness rewards
- **Ablation model**: Demonstrates contribution of SFT initialization

## Model Overview

| Attribute | Value |
|-----------|-------|
| Base Model | Gemma-3-12B-Instruct |
| Training | GRPO (from base) |
| Parameters | 12B |
| LoRA Rank | 64 |

## Performance

| Dataset | Original | +Distractor | Drop |
|---------|----------|-------------|------|
| MGSM | 67.6 | 48.4 | 19.2 |
| MSVAMP | 75.0 | 59.6 | 15.4 |

### Ablation: Effect of SFT Initialization

| Initialization | MGSM (+D) | MSVAMP (+D) |
|---------------|-----------|-------------|
| Base → GRPO | 48.4 | 59.6 |
| **SFT → GRPO** | **64.0** (+15.6) | **66.8** (+7.2) |

**Key Finding**: SFT initialization provides crucial scaffolding that stabilizes GRPO learning and improves distractor robustness by +7-16 points.

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "dipta007/dagger-12B_GRPO"

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
| Base Model | Gemma-3-12B-Instruct (no SFT) |
| LoRA Rank / Alpha | 64 / 128 |
| Global Batch Size | 32 |
| Generations per Prompt | 8 |
| Loss Type | BNPO |
| β / ε / ε_high | 0.0 / 0.2 / 0.28 |

**Reward Function:**
- Valid JSON: +0.5
- Successful execution: +0.5
- Correct answer: +1.0

## When to Use This Model

- **Ablation studies**: Understanding contribution of SFT vs. GRPO
- **GRPO-only scenarios**: When SFT data is unavailable
- **Research**: Studying policy optimization for structured generation

## Related Models

| Model | Training | MGSM (+D) | MSVAMP (+D) |
|-------|----------|-----------|-------------|
| **dagger-12B_GRPO** | Base → GRPO | 48.4 | 59.6 |
| [dagger-12B_SFT_GRPO](https://huggingface.co/dipta007/dagger-12B_SFT_GRPO) | SFT → GRPO | **64.0** | **66.8** |
| [dagger-12B_SFT](https://huggingface.co/dipta007/dagger-12B_SFT) | SFT only | 56.8 | 65.4 |

## Citation

```bibtex
will be updated
```
