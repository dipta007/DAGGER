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

# DAGGER-4B-SFT-GRPO

<a href="https://arxiv.org/abs/XXXX.XXXXX" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b" style="display: inline-block; vertical-align: middle;"/>
</a>
<a href="https://github.com/dipta007/dagger" target="_blank">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Code-black" style="display: inline-block; vertical-align: middle;"/>
</a>

## Model Description

**DAGGER-4B-SFT-GRPO** is the smaller variant of DAGGER, trained with SFT followed by GRPO on Gemma-3-4B. While showing lower performance than the 12B variant, it demonstrates that the DAGGER framework can work with smaller models.

## Highlights

- **Lightweight**: 4B parameters for resource-constrained deployment
- **SFT → GRPO training**: Full training pipeline
- **Improved over baselines**: Still outperforms CoT on distractor robustness
- **Capacity study**: Demonstrates model size requirements for graph generation

## Model Overview

| Attribute | Value |
|-----------|-------|
| Base Model | Gemma-3-4B-Instruct |
| Training | SFT → GRPO |
| Parameters | 4B |
| LoRA Rank | 64 |

## Performance

| Dataset | Original | +Distractor | Drop |
|---------|----------|-------------|------|
| MGSM | 54.8 | 31.4 | 23.4 |
| MSVAMP | 70.3 | 42.9 | 27.4 |
| **Weighted Avg** | - | - | **47.3** |

### Comparison with 12B Variant

| Model | Params | Weighted Avg |
|-------|--------|--------------|
| dagger-4B_SFT_GRPO | 4B | 47.3 |
| dagger-12B_SFT_GRPO | 12B | **69.4** (+22.1) |

**Key Finding**: The 12B model provides +22 points improvement, suggesting a capacity threshold for effective computational graph generation.

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "dipta007/dagger-4B_SFT_GRPO"

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

Same as 12B variant:

| Parameter | Value |
|-----------|-------|
| LoRA Rank / Alpha | 64 / 128 |
| SFT Batch Size | 256 |
| GRPO Batch Size | 32 |
| Generations per Prompt | 8 |
| Epochs | 4 |

## When to Use This Model

- **Resource-constrained deployment**: When 12B is too large
- **Capacity studies**: Research on model size vs. performance
- **Edge deployment**: Smaller memory footprint
- **Prototyping**: Faster iteration during development

## Limitations

- **Lower accuracy**: 22 points below 12B variant
- **Reduced robustness**: Larger accuracy drop under distractors
- **Capacity constraints**: May struggle with complex multi-step problems

## Related Models

| Model | Size | Weighted Avg |
|-------|------|--------------|
| **dagger-4B_SFT_GRPO** | 4B | 47.3 |
| [dagger-4B_SFT](https://huggingface.co/dipta007/dagger-4B_SFT) | 4B | 44.3 |
| [dagger-12B_SFT_GRPO](https://huggingface.co/dipta007/dagger-12B_SFT_GRPO) | 12B | **69.4** |

## Citation

```bibtex
will be updated
```
