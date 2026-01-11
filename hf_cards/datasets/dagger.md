---
license: cc-by-nc-sa-4.0
task_categories:
- question-answering
- text-generation
language:
- bn
tags:
- math
- reasoning
- computational-graph
- sft
- grpo
- training-data
pretty_name: DAGGER Training Data
size_categories:
- 1K<n<10K
dataset_info:
- config_name: grpo
  features:
  - name: prompt
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: solution
    dtype: float64
  - name: question
    dtype: string
  splits:
  - name: train
    num_bytes: 11816956
    num_examples: 3000
  download_size: 2077239
  dataset_size: 11816956
- config_name: sft
  features:
  - name: conversations
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: response
    dtype: string
  - name: english_answer
    dtype: float64
  splits:
  - name: train
    num_bytes: 17504451
    num_examples: 3000
  - name: val
    num_bytes: 2842631
    num_examples: 481
  download_size: 6738412
  dataset_size: 20347082
configs:
- config_name: grpo
  data_files:
  - split: train
    path: grpo/train-*
- config_name: sft
  data_files:
  - split: train
    path: sft/train-*
  - split: val
    path: sft/val-*
---

# DAGGER Training Dataset

<a href="https://arxiv.org/abs/XXXX.XXXXX" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b" style="display: inline-block; vertical-align: middle;"/>
</a>
<a href="https://github.com/dipta007/dagger" target="_blank">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Code-black" style="display: inline-block; vertical-align: middle;"/>
</a>

## Dataset Description

Training data for **DAGGER** (Distractor-Aware Graph Generation for Executable Reasoning) models. This dataset contains Bangla mathematical word problems paired with computational graph solutions, formatted for both SFT and GRPO training pipelines.

### Highlights

- **3,000 training examples** with verified computational graphs
- **Two training configs**: SFT (with validation) and GRPO formats
- **GPT-4.1 generated** graph annotations with execution verification
- **100% execution success rate**: All graphs verified to produce correct answers

## Dataset Statistics

| Config | Split | # Examples | Description |
|--------|-------|-----------|-------------|
| `sft` | train | 3,000 | Supervised fine-tuning data |
| `sft` | val | 481 | Validation set for SFT |
| `grpo` | train | 3,000 | GRPO training prompts |

## Data Sources

| Source | # Examples | Description |
|--------|-----------|-------------|
| SOMADHAN | 1,500 | Bangla math word problems |
| NuminaMath-CoT-BN | 1,500 | Translated mathematical reasoning data |

## Data Format

### SFT Config

For supervised fine-tuning with conversation format:

| Field | Type | Description |
|-------|------|-------------|
| `conversations` | list[{role, content}] | Multi-turn conversation (system, user, assistant) |
| `response` | string | The computational graph JSON |
| `english_answer` | float64 | Ground truth numerical answer |

```json
{
  "conversations": [
    {"role": "system", "content": "You are an expert Bengali Math Reasoner..."},
    {"role": "user", "content": "মিনার কাছে ১০০টি কলম আছে..."},
    {"role": "assistant", "content": "{\"nodes\": [...]}"}
  ],
  "response": "{\"nodes\": [...]}",
  "english_answer": 500.0
}
```

### GRPO Config

For Group Relative Policy Optimization training:

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | list[{role, content}] | System + user messages only |
| `question` | string | The math question |
| `solution` | float64 | Ground truth numerical answer |

```json
{
  "prompt": [
    {"role": "system", "content": "You are an expert Bengali Math Reasoner..."},
    {"role": "user", "content": "মিনার কাছে ১০০টি কলম আছে..."}
  ],
  "question": "মিনার কাছে ১০০টি কলম আছে...",
  "solution": 500.0
}
```

## Computational Graph Schema

The model outputs JSON graphs representing mathematical operations:

```json
{
  "nodes": [
    {"id": "n1", "op": "const", "val": 100, "distractor": false, "label": "মিনার কলম"},
    {"id": "n2", "op": "const", "val": 5, "distractor": false, "label": "কলমের দাম"},
    {"id": "total", "op": "mul", "args": ["n1", "n2"], "distractor": false, "label": "মোট টাকা"},
    {"id": "final_result", "op": "identity", "args": ["total"], "distractor": false}
  ]
}
```

**Supported Operations:**
- **Input**: `const` (constants/numbers from text)
- **Arithmetic**: `add`, `sub`, `mul`, `div`, `abs`
- **Aggregation**: `sum`, `mean`, `min`, `max`
- **Rounding**: `round`, `floor`, `ceil`
- **Advanced**: `sqrt`, `pow`, `mod`, `gcd`, `lcm`
- **Output**: `identity` (final result pointer)

## Usage

### Loading the Dataset

```python
from datasets import load_dataset

# Load SFT config
sft_train = load_dataset("dipta007/dagger", "sft", split="train")
sft_val = load_dataset("dipta007/dagger", "sft", split="val")
print(f"SFT train: {len(sft_train)}, val: {len(sft_val)}")  # 3000, 481

# Load GRPO config
grpo_train = load_dataset("dipta007/dagger", "grpo", split="train")
print(f"GRPO train: {len(grpo_train)}")  # 3000
```

### Training with TRL (SFT)

```python
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

dataset = load_dataset("dipta007/dagger", "sft")

model = AutoModelForCausalLM.from_pretrained("unsloth/Gemma-3-12b")
tokenizer = AutoTokenizer.from_pretrained("unsloth/Gemma-3-12b")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    # ... additional config
)

trainer.train()
```

### Training with TRL (GRPO)

```python
from trl import GRPOTrainer
from datasets import load_dataset

dataset = load_dataset("dipta007/dagger", "grpo", split="train")

# GRPO uses prompts and generates multiple completions
# Reward is computed based on graph execution correctness
trainer = GRPOTrainer(
    model=model,
    train_dataset=dataset,
    reward_funcs=[format_reward, execution_reward, correctness_reward],
    # ... additional config
)

trainer.train()
```

## Quality Verification

All training examples are verified through:

1. **JSON Schema Validation**: Valid computational graph structure
2. **Topological Execution**: Graph executes without cycles or errors
3. **Answer Verification**: Executed result matches ground truth

## Intended Use

- Supervised fine-tuning of computational graph generation models
- GRPO training with execution-based rewards
- Research on structured mathematical reasoning

## Related Resources

- **Evaluation Benchmark**: [dipta007/DistractMath-Bn](https://huggingface.co/datasets/dipta007/DistractMath-Bn)
- **Best Model**: [dipta007/dagger-12B_SFT_GRPO](https://huggingface.co/dipta007/dagger-12B_SFT_GRPO)

## Citation

```bibtex
will be updated
```
