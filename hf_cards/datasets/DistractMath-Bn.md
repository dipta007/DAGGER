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
- distractors
- multilingual
- low-resource
- benchmark
pretty_name: DistractMath-BN
size_categories:
- 1K<n<10K
dataset_info:
- config_name: mgsm
  features:
  - name: row_index
    dtype: int64
  - name: original_question
    dtype: string
  - name: modified_question
    dtype: string
  - name: ground_truth
    dtype: int64
  - name: augmentation_type
    dtype: string
  - name: added_sentences
    dtype: string
  - name: justification
    dtype: string
  - name: answers_match
    dtype: bool
  splits:
  - name: train
    num_bytes: 1909836
    num_examples: 738
  download_size: 475318
  dataset_size: 1909836
- config_name: msvamp
  features:
  - name: row_index
    dtype: int64
  - name: original_question
    dtype: string
  - name: modified_question
    dtype: string
  - name: ground_truth
    dtype: int64
  - name: augmentation_type
    dtype: string
  - name: added_sentences
    dtype: string
  - name: justification
    dtype: string
  - name: answers_match
    dtype: bool
  splits:
  - name: train
    num_bytes: 5955289
    num_examples: 2947
  download_size: 1231557
  dataset_size: 5955289
configs:
- config_name: mgsm
  data_files:
  - split: train
    path: mgsm/train-*
- config_name: msvamp
  data_files:
  - split: train
    path: msvamp/train-*
---

# DistractMath-BN

<a href="https://arxiv.org/abs/XXXX.XXXXX" target="_blank">
</a>
<a href="https://github.com/dipta007/dagger" target="_blank">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Code-black" style="display: inline-block; vertical-align: middle;"/>
</a>

## Dataset Description

**DistractMath-BN** is a distractor-augmented benchmark for evaluating mathematical reasoning robustness in Bangla (Bengali). It is derived from the Bangla subsets of MGSM and MSVAMP, systematically augmented with semantically coherent but computationally irrelevant information.

### Highlights

- **3,685 distractor-augmented problems** across MGSM and MSVAMP
- **Three distractor categories** targeting different cognitive interference mechanisms
- **Two-stage quality assurance**: GPT-4.1 validation + native Bangla speaker review
- **Answer-preserving augmentations**: All distractors maintain the original correct answer

## Distractor Types

| Type | Abbreviation | Description | Example |
|------|--------------|-------------|---------|
| **Related Entity Distractor** | RED | Numerical info about same object type but different entities | "তার বোন বুধবার ১২ জন ছেলেমেয়ের সঙ্গে লুকোচুরি খেলেছিল।" |
| **Orthogonal Attribute Distractor** | OAD | Properties in different dimensions than queried attribute | "সোমবার খেলতে ১ ঘণ্টা সময় লেগেছিল।" |
| **Null-Effect Event Distractor** | NEED | Actions with zero net impact (planned but not executed) | "রাজু ১০০০ টি দিতে রাজি হল, কিন্তু পরে আর দিলনা।" |

## Dataset Statistics

| Config | Split | # Examples | Description |
|--------|-------|-----------|-------------|
| `mgsm` | train | 738 | Distractor-augmented MGSM-BN problems |
| `msvamp` | train | 2,947 | Distractor-augmented MSVAMP-BN problems |

**Total: 3,685 examples**

## Data Format

Each example contains:

| Field | Type | Description |
|-------|------|-------------|
| `row_index` | int64 | Index of the original problem |
| `original_question` | string | Original math problem without distractors |
| `modified_question` | string | Problem with distractors inserted |
| `ground_truth` | int64 | Correct numerical answer |
| `augmentation_type` | string | Distractor type: RED, OAD, or NEED |
| `added_sentences` | string | The distractor sentences that were added |
| `justification` | string | Explanation of why the distractor is irrelevant |
| `answers_match` | bool | Verification that answer is preserved |

### Example

```json
{
  "row_index": 42,
  "original_question": "জিনের কাছে 30টি ললিপপ আছে। জিন 2টি ললিপপ খেয়েছে। অবশিষ্ট ললিপপগুলো দিয়ে, জিন একটি ব্যাগের মধ্যে 2টি করে ললিপপ ভরতে চায়। তাহলে জিন কতগুলো ব্যাগ ভর্তি করতে পারবে?",
  "modified_question": "জিনের কাছে 30টি ললিপপ আছে। দোকানে আরও ৫০টি ললিপপ বিক্রি হচ্ছিল। তার ছোটবোনের কাছে ১৮টি ললিপপ আছে। জিন 2টি ললিপপ খেয়েছে। জিনের বন্ধু মিমি প্রতিদিন ৩টি ললিপপ খায়। অবশিষ্ট ললিপপগুলো দিয়ে, জিন একটি ব্যাগের মধ্যে 2টি করে ললিপপ ভরতে চায়। তাহলে জিন কতগুলো ব্যাগ ভর্তি করতে পারবে?",
  "ground_truth": 14,
  "augmentation_type": "RED",
  "added_sentences": "দোকানে আরও ৫০টি ললিপপ বিক্রি হচ্ছিল। তার ছোটবোনের কাছে ১৮টি ললিপপ আছে। জিনের বন্ধু মিমি প্রতিদিন ৩টি ললিপপ খায়।",
  "justification": "These sentences mention other people's lollipops and unrelated shop inventory, which don't affect Jin's calculation.",
  "answers_match": true
}
```

## Usage

```python
from datasets import load_dataset

# Load MGSM config
mgsm = load_dataset("dipta007/DistractMath-Bn", "mgsm", split="train")
print(f"MGSM examples: {len(mgsm)}")  # 738

# Load MSVAMP config
msvamp = load_dataset("dipta007/DistractMath-Bn", "msvamp", split="train")
print(f"MSVAMP examples: {len(msvamp)}")  # 2947

# Filter by augmentation type
red_problems = mgsm.filter(lambda x: x["augmentation_type"] == "RED")
oad_problems = mgsm.filter(lambda x: x["augmentation_type"] == "OAD")
need_problems = mgsm.filter(lambda x: x["augmentation_type"] == "NEED")

# Access fields
for example in mgsm.select(range(3)):
    print(f"Type: {example['augmentation_type']}")
    print(f"Original: {example['original_question'][:100]}...")
    print(f"Modified: {example['modified_question'][:100]}...")
    print(f"Answer: {example['ground_truth']}")
    print("---")
```

## Quality Assurance

Each augmented problem underwent a two-stage verification pipeline:

1. **Automated Validation**: GPT-4.1 confirms answer preservation (`answers_match` field)
2. **Human Expert Review**: Native Bangla speaker verifies semantic coherence

Only samples passing both stages are included in the final dataset.

## Benchmark Results

Performance degradation under distractors (accuracy drop in percentage points):

| Model Category | MGSM Drop | MSVAMP Drop |
|---------------|-----------|-------------|
| Standard LLMs (CoT) | 3.2 - 28.5 | 18.1 - 40.7 |
| Reasoning Models | 17.5 - 22.5 | 14.2 - 23.6 |
| DAGGER (Ours) | **12.7 - 14.4** | **11.5 - 13.1** |

## Intended Use

- Evaluating mathematical reasoning robustness under irrelevant context
- Benchmarking distractor-aware reasoning models
- Research on low-resource language mathematical understanding
- Studying failure modes of chain-of-thought reasoning

## Limitations

- Limited to arithmetic word problems; does not cover geometry, algebra, or calculus
- Only three distractor categories; non-numeric distractors not addressed
- Bangla language only; cross-lingual transfer not evaluated
- Quality assurance relied on single human annotator

## Citation

```bibtex
will be updated
```

## Acknowledgments

This dataset builds upon:
- [MGSM](https://github.com/google-research/url-nlp/tree/main/mgsm) (Shi et al., 2023)
- [MSVAMP](https://github.com/arkilpatel/SVAMP) (Patel et al., 2021)
