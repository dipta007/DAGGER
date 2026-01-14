# DAGGER: Distractor-Aware Graph Generation for Executable Reasoning in Math Problems

[![arXiv](https://img.shields.io/badge/arXiv-2601.06853-b31b1b)](https://arxiv.org/abs/2601.06853)
[![Dataset](https://img.shields.io/badge/Dataset-DistractMath--BN-green)](https://huggingface.co/datasets/dipta007/DistractMath-Bn)
[![Model](https://img.shields.io/badge/Model-DAGGER-orange)](https://huggingface.co/collections/dipta007/dagger)

Official implementation of **DAGGER** (Distractor-Aware Graph Generation for Executable Reasoning).

## Key Results

| Model | Weighted Avg. Accuracy | Tokens | Accuracy Drop |
|-------|----------------------|--------|---------------|
| Qwen 3-8B (Reasoning) | 71.4% | 3,128 | 14.2-19.9 pts |
| **DAGGER-12B (Ours)** | **69.4%** | **359** | **12.0-14.4 pts** |
| Standard CoT (Best) | 55.7% | 599 | 18.1-40.7 pts |

**DAGGER achieves comparable accuracy to reasoning models while using 89% fewer tokens.**

## Installation

### Requirements
- Python 3.12+
- CUDA 12.1+ (for GPU training)

### Setup

```bash
# Clone the repository
git clone https://github.com/dipta007/dagger.git
cd dagger

# Install dependencies using uv (recommended)
uv sync
```

## Dataset: DistractMath-BN

**DistractMath-BN** is a distractor-augmented benchmark derived from the Bangla subset of MGSM (250 problems) and MSVAMP (1000 problems). It systematically probes reasoning robustness by introducing three classes of cognitive interference.

To illustrate these distractors, consider the following **Original Problem**:

> **Original Problem:** Julia played tag with 18 children on Monday. On Tuesday she played with 10 children. How many more children did she play with on Monday?

Each problem in the benchmark is augmented with distractors as defined below:

| Distractor Type | Definition | Example Augmentation |
|----------------|------------|----------------------|
| **RED**<br>(Related Entity) | Numerical information about the *same* object type as the target, but associated with different entities or contexts. | *"Her younger sister played hide-and-seek with **12** children on Wednesday."* |
| **OAD**<br>(Orthogonal Attribute) | Supplementary properties (e.g., time, price) measured in dimensions orthogonal to the attribute being queried. | *"It took **1 hour** to play on Monday... It took **45 minutes** to play on Tuesday."* |
| **NEED**<br>(Null-Effect Event) | Descriptions of events (planned, negated, or hypothetical) that have zero net impact on the final result. | *"Julia thought about inviting **5** more children... but she didn't call them later."* |

**Dataset Statistics:**
- MGSM (+Distractor): 738 verified problems
- MSVAMP (+Distractor): 2,947 verified problems
- Average distractors per problem: 2.7

Download from HuggingFace:
```python
from datasets import load_dataset
dataset = load_dataset("dipta007/DistractMath-Bn")
```

## Computational Graph Framework

DAGGER reformulates math word problems as directed acyclic graphs (DAGs):

```json
{
  "nodes": [
    {"id": "n1", "op": "const", "val": 122195, "distractor": false, "label": "Mina's pens"},
    {"id": "n2", "op": "const", "val": 25084, "distractor": true, "label": "Raju's pens"},
    {"id": "n3", "op": "const", "val": 45.6, "distractor": false, "label": "Price per pen"},
    {"id": "total", "op": "mul", "args": ["n1", "n3"], "distractor": false, "label": "Total money"},
    {"id": "final_result", "op": "identity", "args": ["total"], "distractor": false}
  ]
}
```

**Supported Operations:** `add`, `sub`, `mul`, `div`, `sum`, `mean`, `min`, `max`, `floor`, `ceil`, `round`, `sqrt`, `pow`, `mod`, `gcd`, `lcm`, `const`, `identity`

## Usage

### Inference

```bash
python src/inference.py \
  -c ./checkpoints/sft/my_model \
  -q "আপনার গণিত সমস্যা এখানে"
```

Or use our pretrained models:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("dipta007/dagger-12B_SFT_GRPO")
tokenizer = AutoTokenizer.from_pretrained("dipta007/dagger-12B_SFT_GRPO")
```

### Training

#### Supervised Fine-Tuning (SFT)

```bash
python src/train/sft/sft.py \
  --model_name "unsloth/Gemma-3-12b" \
  --dataset_path "./data" \
  --run_name "sft_gemma12b_v1" \
  --lora_rank 64
```

**SFT Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| LoRA Rank / Alpha | 64 / 128 |
| Global Batch Size | 256 |
| Epochs | 4 |
| Learning Rate | 1e-5 → 1e-6 |
| Max Sequence Length | 4096 |

#### Group Relative Policy Optimization (GRPO)

```bash
python src/train/grpo/grpo.py \
  --model_name "./checkpoints/sft/sft_gemma12b_v1" \
  --dataset_path "./data" \
  --run_name "grpo_gemma12b_v1" \
  --num_generations 8 \
  --beta 0.0
```

**GRPO Reward Function:**
```
R(g, y) = 0.5 * I_fmt + 0.5 * I_exec + I_acc(exec(g), y)
```
- `I_fmt`: Valid JSON schema (+0.5)
- `I_exec`: Successful graph execution (+0.5)
- `I_acc`: Correct numerical answer (+1.0)

### Evaluation

```bash
# Evaluate on MGSM
python src/eval/eval.py \
  --model_path ./checkpoints/grpo/grpo_gemma12b_v1 \
  --dataset mgsm \
  --split test

# Evaluate on augmented dataset
python src/eval/eval.py \
  --model_path ./checkpoints/grpo/grpo_gemma12b_v1 \
  --dataset mgsm_augmented \
  --split test
```

## Model Zoo

| Model | MGSM | MSVAMP | MGSM (+D) | MSVAMP (+D) | Weighted Avg |
|-------|------|--------|-----------|-------------|--------------|
| [dagger-4B_SFT](https://huggingface.co/dipta007/dagger-4B_SFT) | 40.4 | 65.0 | 25.1 | 42.4 | 44.3 |
| [dagger-4B_GRPO](https://huggingface.co/dipta007/dagger-4B_GRPO) | 54.8 | 70.3 | 31.4 | 42.9 | 47.3 |
| [dagger-4B_SFT_GRPO](https://huggingface.co/dipta007/dagger-4B_SFT_GRPO) | 54.8 | 70.3 | 31.4 | 42.9 | 47.3 |
| [dagger-12B_SFT](https://huggingface.co/dipta007/dagger-12B_SFT) | 70.0 | 76.8 | 56.8 | 65.4 | 66.7 |
| [dagger-12B_GRPO](https://huggingface.co/dipta007/dagger-12B_GRPO) | 78.4 | 78.8 | 64.0 | 66.8 | 69.4 |
| [dagger-12B_SFT_GRPO](https://huggingface.co/dipta007/dagger-12B_SFT_GRPO) | **78.4** | **78.8** | **64.0** | **66.8** | **69.4** |

(+D) = with distractors

## Project Structure

```
dagger/
├── data/                       # Training data (train.json, val.json, test.json)
├── src/
│   ├── inference.py           # Main inference script
│   ├── prompt.py              # Prompt templates
│   ├── eval/
│   │   ├── helpers.py         # Graph parsing & execution
│   │   ├── eval.py            # Batch evaluation
│   │   └── mgsm_eval.py       # MGSM benchmark evaluation
│   ├── train/
│   │   ├── sft/
│   │   │   └── sft.py         # Supervised fine-tuning
│   │   └── grpo/
│   │       ├── grpo.py        # GRPO training
│   │       └── grpo_rewards.py # Reward functions
│   └── hf_utils/
│       ├── push_model.py      # Upload models to HuggingFace
│       └── push_data.py       # Upload datasets to HuggingFace
├── evaluation/                 # Benchmark evaluation scripts
└── gpt_evaluation/            # GPT baseline evaluation
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{nazi2026dagdaggerdistractorawaregraphgeneration,
      title={{\dag}DAGGER: Distractor-Aware Graph Generation for Executable Reasoning in Math Problems}, 
      author={Zabir Al Nazi and Shubhashis Roy Dipta and Sudipta Kar},
      year={2026},
      eprint={2601.06853},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.06853}, 
}
```

## Acknowledgments

This work builds upon several open-source projects:
- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- [TRL](https://github.com/huggingface/trl) for GRPO training
- [vLLM](https://github.com/vllm-project/vllm) for fast inference
- [MGSM](https://github.com/google-research/url-nlp/tree/main/mgsm) and [MSVAMP](https://github.com/arkilpatel/SVAMP) for base datasets

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
