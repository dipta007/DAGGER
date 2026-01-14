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

# DAGGER-4B-GRPO

<a href="https://arxiv.org/abs/XXXX.XXXXX" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b" style="display: inline-block; vertical-align: middle;"/>
</a>
<a href="https://github.com/dipta007/dagger" target="_blank">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Code-black" style="display: inline-block; vertical-align: middle;"/>
</a>

## Model Description

**DAGGER-4B-GRPO** is trained with GRPO directly from the base Gemma-3-4B model without SFT initialization. This ablation model demonstrates the critical importance of SFT initialization for smaller models.

## Model Overview

| Attribute | Value |
|-----------|-------|
| Base Model | Gemma-3-4B-Instruct |
| Training | GRPO (from base) |
| Parameters | 4B |
| LoRA Rank | 64 |

## Performance

| Dataset | Original | +Distractor |
|---------|----------|-------------|
| MGSM | 29.2 | 13.1 |
| MSVAMP | 57.1 | 29.3 |

### Critical Finding: SFT Initialization Effect

| Initialization | MGSM | MGSM (+D) | MSVAMP (+D) |
|---------------|------|-----------|-------------|
| Base → GRPO | 29.2 | 13.1 | 29.3 |
| **SFT → GRPO** | **54.8** | **31.4** | **42.9** |

**Key Insight**: For 4B models, GRPO without SFT struggles to learn reliable graph generation. SFT provides essential scaffolding:
- **+25.6 points** on MGSM
- **+18.3 points** on MGSM (+Distractor)
- **+13.6 points** on MSVAMP (+Distractor)

This effect is more pronounced in smaller models than in 12B variants.

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "dipta007/dagger-4B_GRPO"

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
| Base Model | Gemma-3-4B-Instruct (no SFT) |
| LoRA Rank / Alpha | 64 / 128 |
| Global Batch Size | 32 |
| Generations per Prompt | 8 |
| Loss Type | BNPO |

## When to Use This Model

- **Ablation studies**: Understanding SFT contribution for smaller models
- **Research**: Studying capacity requirements for GRPO-only training
- **NOT recommended for production**: Use dagger-4B_SFT_GRPO instead

## Limitations

- **Low accuracy**: Struggles to generate valid computational graphs
- **High failure rate**: Often produces malformed JSON or incorrect structures
- **Poor distractor handling**: Collapses to 13.1% on augmented MGSM

## Recommendation

For 4B models, always use SFT initialization before GRPO:
- [dagger-4B_SFT_GRPO](https://huggingface.co/dipta007/dagger-4B_SFT_GRPO) provides +18 points improvement

## Related Models

| Model | Training | MGSM (+D) |
|-------|----------|-----------|
| **dagger-4B_GRPO** | Base → GRPO | 13.1 |
| [dagger-4B_SFT](https://huggingface.co/dipta007/dagger-4B_SFT) | SFT | 25.1 |
| [dagger-4B_SFT_GRPO](https://huggingface.co/dipta007/dagger-4B_SFT_GRPO) | SFT → GRPO | **31.4** |

## Citation

```bibtex
will be updated
```
