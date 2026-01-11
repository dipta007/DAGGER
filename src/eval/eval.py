import json
import math
import os
import shutil
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import wandb
from src.eval.helpers import execute_graph
from src.prompt import USER_PROMPT_TEMPLATE


def generate_graphs_batch(model: LLM, tokenizer, questions: list, max_new_tokens: int = 2048) -> list:
    prompts = []
    for question in questions:
        messages = [{"role": "user", "content": USER_PROMPT_TEMPLATE.format(question=question)}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = model.generate(prompts=prompts, sampling_params=sampling_params)
    return [output.outputs[0].text for output in outputs]


def get_dataset(dataset: str):
    """
    return list of dicts with "question" and "answer" and "index"
        question: str in bengali
        answer: float in english
        index: int
    """
    if dataset == "mgsm":
        raw_data = load_dataset("jbross-ibm-research/mgsm", "bn")
        raw_data = raw_data["test"]
        data = []
        for i, item in enumerate(raw_data):
            data.append(
                {
                    "question": item["question"],
                    "answer": item["answer_number"],
                    "index": i,
                }
            )
        return data
    elif dataset == "msvamp":
        raw_data = load_dataset("Mathoctopus/MSVAMP", "bn")
        raw_data = raw_data["test"]
        data = []
        for i, item in enumerate(raw_data):
            data.append(
                {
                    "question": item["m_query"],
                    "answer": item["response"],
                    "index": i,
                }
            )
        return data
    elif dataset == "dmgsm":
        raw_data = pd.read_csv("distractor_augmented_datasets/augmented_mgsm.csv")
        data = []
        for i, item in raw_data.iterrows():
            data.append(
                {
                    "question": item["modified_question"],
                    "answer": item["ground_truth"],
                    "index": i,
                }
            )
        return data
    elif dataset == "dmsvamp":
        raw_data = pd.read_csv("distractor_augmented_datasets/augmented_msvamp.csv")
        data = []
        for i, item in raw_data.iterrows():
            data.append(
                {
                    "question": item["modified_question"],
                    "answer": item["ground_truth"],
                    "index": i,
                }
            )
        return data
    else:
        raise ValueError(f"Invalid dataset: {dataset}")


def test_on_dataset(model: LLM, tokenizer: AutoTokenizer, dataset: str, lora_path: str, wandb_run: wandb.Run):
    data_checkpoint_file = f"{lora_path}/{dataset}_checkpoint_0.0.json"
    data_results_file = f"{lora_path}/{dataset}_results_0.0.json"

    print("\n[3/5] Loading dataset...")
    print(f"Dataset: {dataset}")
    test_data = get_dataset(dataset)
    print(f"✓ Loaded {len(test_data)} samples")

    checkpoint_data = []
    if Path(data_checkpoint_file).exists():
        with open(data_checkpoint_file, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
        print(f"✓ Resuming from checkpoint ({len(checkpoint_data)} completed)")

    print("\n[4/5] Generating and evaluating graphs...")
    already_evaluated = set(entry["index"] for entry in checkpoint_data)

    batch_indices = [item["index"] for item in test_data if item["index"] not in already_evaluated]
    batch_questions = [item["question"] for item in test_data if item["index"] not in already_evaluated]
    batch_ground_truths = [float(item["answer"]) for item in test_data if item["index"] not in already_evaluated]

    batch_graph_outputs = generate_graphs_batch(model, tokenizer, batch_questions)

    for index, question, ground_truth, graph_output in zip(batch_indices, batch_questions, batch_ground_truths, batch_graph_outputs):
        result = execute_graph(graph_output)

        if isinstance(result, (float, int)):
            predicted = result
            execution_success = True
            error_msg = None
        else:
            predicted = None
            execution_success = False
            error_msg = result

        correct = False
        if predicted is not None and math.isclose(predicted, ground_truth, rel_tol=1e-6):
            correct = True

        entry = {
            "index": index,
            "question": question,
            "ground_truth": ground_truth,
            "graph_output": graph_output,
            "predicted": predicted,
            "execution_success": execution_success,
            "error_msg": error_msg,
            "correct": correct,
        }

        checkpoint_data.append(entry)

        with open(data_checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

    print("\n[5/5] Calculating results...")

    total = len(checkpoint_data)
    correct_count = sum(1 for entry in checkpoint_data if entry["correct"])
    execution_success_count = sum(1 for entry in checkpoint_data if entry["execution_success"])
    accuracy = (correct_count / total * 100) if total > 0 else 0
    execution_rate = (execution_success_count / total * 100) if total > 0 else 0

    ckpt_num = int(lora_path.split("/")[-1].split("-")[-1])
    wandb_run.define_metric(f"eval/{dataset}/accuracy_0.0", step_metric="checkpoint_number")
    wandb_run.define_metric(f"eval/{dataset}/execution_rate_0.0", step_metric="checkpoint_number")
    wandb_run.log({f"eval/{dataset}/accuracy_0.0": accuracy, f"eval/{dataset}/execution_rate_0.0": execution_rate, "checkpoint_number": ckpt_num})

    with open(data_results_file, "w", encoding="utf-8") as f:
        results = {
            "lora_path": lora_path,
            "total": total,
            "execution_success_count": execution_success_count,
            "execution_rate": execution_rate,
            "correct_count": correct_count,
            "accuracy": accuracy,
            "dataset": dataset,
        }
        json.dump(results, f, ensure_ascii=False, indent=2)
        print("\nResults:")
        print("=" * 80)
        print(json.dumps(results, ensure_ascii=False, indent=2))
        print("=" * 80)

    print(f"\n✓ Results saved to {data_results_file}")


def main(base_model: str, lora_path: str, datasets: list):
    print("=" * 80)
    print(f"Datasets: {datasets}")
    print("=" * 80)

    print("\n[1/5] Merging model...")
    print(f"Base model: {base_model}")
    print(f"LoRA adapters: {lora_path}")

    tmp_dir = f"./mgsm_eval_tmp/{lora_path.split('/')[-2]}_{lora_path.split('/')[-1]}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)
    full_merge_script_path = os.path.join(os.path.dirname(__file__), "merge.py")
    cmd = f"uv run {full_merge_script_path} --lora_path {lora_path} --tmp_dir {tmp_dir}"
    print(f"Running command: {cmd}")
    exit_code = os.system(cmd)
    assert exit_code == 0, "Failed to merge model"
    print("✓ Model merged")

    # Load tokenizer from base model
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print("\n[2/5] Loading model...")
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

    exp_name = lora_path.split("/")[-2]
    ckpt_num = int(lora_path.split("/")[-1].split("-")[-1])
    print(f"Experiment name: {exp_name}")
    print(f"Checkpoint number: {ckpt_num}")
    print("Saving results to WandB...")
    wandb_run = wandb.init(project="math2gcot-train", entity="collab-srd", name=f"eval_{exp_name}", id=f"eval_{exp_name}", resume="auto", tags=["grpo_v13"])
    for dataset in datasets:
        test_on_dataset(model, tokenizer, dataset, lora_path, wandb_run)
    wandb_run.finish()

    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None, choices=["mgsm", "msvamp", "dmgsm", "dmsvamp"])
    parser.add_argument("--run-all", action="store_true", default=False)
    args = parser.parse_args()

    if args.run_all:
        datasets = ["mgsm", "msvamp", "dmgsm", "dmsvamp"]
        ckpt_dirs = os.listdir(args.lora_path)
        ckpt_dirs = [dir for dir in ckpt_dirs if dir.startswith("checkpoint-")]
        ckpt_dirs.sort(key=lambda x: int(x.split("-")[-1]))
        assert len(ckpt_dirs) > 0, "No checkpoint directories found"
        args.base_model = json.load(open(f"{args.lora_path}/{ckpt_dirs[0]}/adapter_config.json", "r"))["base_model_name_or_path"]
        print(f"Running evaluation for {len(ckpt_dirs)} checkpoints from {args.lora_path}")
        print(f"Base model: {args.base_model}")
        for ckpt_dir in ckpt_dirs:
            main(base_model=args.base_model, lora_path=f"{args.lora_path}/{ckpt_dir}", datasets=datasets)
    else:
        assert args.dataset is not None, "Dataset is required"
        args.base_model = json.load(open(f"{args.lora_path}/adapter_config.json", "r"))["base_model_name_or_path"]
        main(base_model=args.base_model, lora_path=args.lora_path, datasets=[args.dataset])
