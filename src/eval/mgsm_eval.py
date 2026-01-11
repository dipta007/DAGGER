"""
MGSM Evaluation Script
Evaluates SFT model on Bengali MGSM dataset
"""

import json
import os
import shutil
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import wandb
from src.eval.helpers import execute_graph


def generate_graphs_batch(model: LLM, tokenizer, questions: list, max_new_tokens: int = 1024) -> list:
    """Generate graphs for multiple questions using batched inference"""
    system_prompt = """
You are an expert Bengali Math Reasoner. Your task is to solve mathematical problems by constructing a "Computational Graph".

Follow these steps:
1. **Analyze:** Identify all numerical entities and implicit variables in the text.
2. **Filter:** Determine which variables are necessary for the solution and which are "distractors" (irrelevant information).
3. **Graph:** Output a JSON graph representing the solution. Nodes must be topologically sorted, and there must be a "final_result" node that represents the final answer.

**Graph Rules:**
- **"id"**: Unique identifier (e.g., "n1", "n2").
- **"val"**: The raw number extracted from text (for input nodes).
- **"op"**: The operation (`add`, `sub`, `mul`, `div`, `round`, `sqrt`, `floor`, `sum`, `mean`, `ratio_split`). Use `const` for input numbers.
- **"args"**: List of input node IDs.
- **"distractor"**: Boolean (`true` / `false`). Set to `true` if the node is NOT used in the final calculation path.

**Available Operations:**
- Input: `const` (Use this for all numbers found in text or constants).
- Arithmetic: `add`, `sub`, `mul`, `div`, `abs` (absolute difference).
- Logic/Stats: `sum`, `mean`, `min` (minimum), `max` (maximum).
- Rounding: `round` (nearest int), `floor` (round down), `ceil` (round up).
- Advanced: `sqrt`, `pow`, `mod` (remainder), `gcd`, `lcm`.
- Output: `identity` ("final_result" points to the answer node)

**Output Format:**

Graph:
```json
{
  "nodes": [ ... ]
}
```
"""

    # Build prompts for all questions
    prompts = []
    for question in questions:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    # Generate for all inputs at once
    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.95,
        max_tokens=max_new_tokens,
        repetition_penalty=1.15,
    )
    outputs = model.generate(
        prompts=prompts,
        sampling_params=sampling_params,
    )
    return [output.outputs[0].text for output in outputs]


def main(base_model: str, lora_path: str, batch_size: int = 8):
    BASE_MODEL = base_model
    LORA_PATH = lora_path
    CHECKPOINT_FILE = f"{LORA_PATH}/mgsm_checkpoint.json"
    RESULTS_FILE = f"{LORA_PATH}/mgsm_results.txt"

    print("=" * 80)
    print("MGSM EVALUATION")
    print("=" * 80)

    print("\n[0/4] Merging model...")
    print(f"Base model: {BASE_MODEL}")
    print(f"LoRA adapters: {LORA_PATH}")

    tmp_dir = "./mgsm_eval_tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    os.system(f"uv run src/eval/merge.py --lora_path {LORA_PATH} --tmp_dir {tmp_dir}")

    # Load tokenizer from base model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, fix_mistral_regex=True)

    print("\n[1/4] Loading model...")
    print(f"Merged model: {tmp_dir}")
    # Load vLLM model with LoRA support
    model = LLM(
        model=tmp_dir,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    print("✓ Model loaded")

    print("\n[2/4] Loading MGSM dataset...")
    dataset = load_dataset("jbross-ibm-research/mgsm", "bn")
    test_data = dataset["test"]
    print(f"✓ Loaded {len(test_data)} samples")

    checkpoint_data = []
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
        print(f"✓ Resuming from checkpoint ({len(checkpoint_data)} completed)")

    print("\n[3/4] Generating and evaluating graphs...")
    # Collect questions for batch
    already_evaluated = set(entry["index"] for entry in checkpoint_data)
    batch_questions = [item["question"] for i, item in enumerate(test_data) if i not in already_evaluated]
    batch_ground_truths = [float(item["answer_number"]) for i, item in enumerate(test_data) if i not in already_evaluated]

    # Generate graphs for entire batch
    batch_graph_outputs = generate_graphs_batch(model, tokenizer, batch_questions)

    # Process each result in the batch
    for i, (question, ground_truth, graph_output) in enumerate(zip(batch_questions, batch_ground_truths, batch_graph_outputs)):
        result = execute_graph(graph_output)

        if isinstance(result, float):
            predicted = result
            execution_success = True
            error_msg = None
        else:
            predicted = None
            execution_success = False
            error_msg = result

        correct = False
        if predicted is not None and abs(predicted - ground_truth) < 0.01:
            correct = True

        entry = {
            "index": len(already_evaluated) + i,
            "question": question,
            "ground_truth": ground_truth,
            "graph_output": graph_output,
            "predicted": predicted,
            "execution_success": execution_success,
            "error_msg": error_msg,
            "correct": correct,
        }

        checkpoint_data.append(entry)

        # Save checkpoint after each batch
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

    print("\n[4/4] Calculating results...")

    total = len(checkpoint_data)
    correct_count = sum(1 for entry in checkpoint_data if entry["correct"])
    execution_success_count = sum(1 for entry in checkpoint_data if entry["execution_success"])

    accuracy = (correct_count / total * 100) if total > 0 else 0
    execution_rate = (execution_success_count / total * 100) if total > 0 else 0

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total samples: {total}")
    print(f"Execution success: {execution_success_count}/{total} ({execution_rate:.2f}%)")
    print(f"Correct answers: {correct_count}/{total} ({accuracy:.2f}%)")
    print("=" * 80)

    exp_name = lora_path.split("/")[-2]
    ckpt_num = int(lora_path.split("/")[-1].split("-")[-1])
    print(f"Experiment name: {exp_name}")
    print(f"Checkpoint number: {ckpt_num}")
    run = wandb.init(project="math2gcot-train", entity="collab-srd", name=exp_name, id=exp_name, resume="auto")
    run.define_metric("eval/accuracy", step_metric="checkpoint_number")
    run.define_metric("eval/execution_rate", step_metric="checkpoint_number")
    run.log({"eval/accuracy": accuracy, "eval/execution_rate": execution_rate, "checkpoint_number": ckpt_num})
    run.finish()

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("MGSM EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Base model: {BASE_MODEL}\n")
        f.write(f"LoRA adapters: {LORA_PATH}\n")
        f.write(f"Total samples: {total}\n")
        f.write(f"Execution success: {execution_success_count}/{total} ({execution_rate:.2f}%)\n")
        f.write(f"Correct answers: {correct_count}/{total} ({accuracy:.2f}%)\n")
        f.write("=" * 80 + "\n")

    print(f"\n✓ Results saved to {RESULTS_FILE}")

    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    root_dir = "/umbc/ada/ferraro/users/sroydip1/collab/math2gcot/checkpoints/initial_lora_b0.0_gemma-3-12b-it_v7"
    dirs = os.listdir(root_dir)
    for dir in dirs:
        if dir.startswith("checkpoint-"):
            main(base_model="unsloth/gemma-3-12b-it", lora_path=f"{root_dir}/{dir}")

    root_dir = "/umbc/ada/ferraro/users/sroydip1/collab/math2gcot/checkpoints/initial_lora_b0.0_gemma-3-4b-it_v7"
    dirs = os.listdir(root_dir)
    for dir in dirs:
        if dir.startswith("checkpoint-"):
            main(base_model="unsloth/gemma-3-4b-it", lora_path=f"{root_dir}/{dir}")
