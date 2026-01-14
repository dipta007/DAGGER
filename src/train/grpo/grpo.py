import json
import os

# os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
# os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

import unsloth
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastModel
from vllm import SamplingParams

from src.prompt import USER_PROMPT_TEMPLATE
from src.train.grpo.grpo_rewards import response_one_reward_rules_all

RANDOM_SEED = 42
NPROC_PER_NODE = torch.cuda.device_count()
DEVICE_IDS = list(range(NPROC_PER_NODE))

GLOBAL_BATCH_SIZE = 32
BATCH_SIZE = 8
GRAD_ACC_STEPS = GLOBAL_BATCH_SIZE / (BATCH_SIZE * NPROC_PER_NODE)
if GLOBAL_BATCH_SIZE % (BATCH_SIZE * NPROC_PER_NODE) != 0:
    raise ValueError(f"Global batch size {GLOBAL_BATCH_SIZE} is not divisible by {BATCH_SIZE * NPROC_PER_NODE}")
GRAD_ACC_STEPS = int(GRAD_ACC_STEPS)


def get_dataset(args):
    def _get_split(split):
        with open(f"{args.dataset_path}/{split}.json", "r") as reader:
            dataset = json.load(reader)

        data = []
        for item in dataset:
            prompt = [{"role": "user", "content": USER_PROMPT_TEMPLATE.format(question=item["question"].strip())}]
            d = {
                "prompt": prompt,
                "solution": item["english_answer"],
                "question": item["question"].strip(),
            }
            data.append(d)
        dataset_ds = Dataset.from_list(data)
        return dataset_ds

    return {
        "train": _get_split("numina_3000"),
        "val": _get_split("numina_eval_300"),
    }


def main(args):
    output_path = f"./checkpoints/grpo/{args.run_name}"

    dataset_ds = get_dataset(args)

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_8bit=False,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=0.7,
    )

    model = FastModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.0,
        use_gradient_checkpointing="unsloth",
        random_state=RANDOM_SEED,
        use_rslora=args.use_rslora,
        finetune_vision_layers=False,  # Turn off for just text!
        finetune_language_layers=True,  # Should leave on!
        finetune_attention_modules=True,  # Attention good for GRPO
        finetune_mlp_modules=True,  # Should leave on always!
    )

    vllm_sampling_params = SamplingParams(
        top_p=0.95,
        top_k=64,
        seed=RANDOM_SEED,
    )

    training_args = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        temperature=1.0,
        learning_rate=1e-5,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": 1e-6},
        max_grad_norm=1.0,
        optim="adamw_torch",
        warmup_steps=20,
        weight_decay=0.01,
        logging_steps=1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        num_generations=args.num_generations,
        max_prompt_length=1024,
        max_completion_length=1024,
        num_train_epochs=4,
        save_steps=50,
        report_to="wandb",
        log_completions=True,
        num_completions_to_print=1,
        output_dir=output_path,
        epsilon=0.2,
        epsilon_high=0.28,
        loss_type="bnpo",
        beta=args.beta,
        mask_truncated_completions=True,
        shuffle_dataset=True,
        bf16=True,
        fp16=False,
        seed=RANDOM_SEED,
        # For optional training + evaluation
        # fp16_full_eval=True,
        # per_device_eval_batch_size=BATCH_SIZE,
        # eval_accumulation_steps=GRAD_ACC_STEPS,
        # eval_strategy="steps",
        # eval_steps=50,
    )

    # For optional training + evaluation
    # new_dataset = dataset.train_test_split(test_size = 0.01)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            response_one_reward_rules_all,
        ],
        args=training_args,
        train_dataset=dataset_ds["train"],
        # For optional training + evaluation
        # train_dataset = new_dataset["train"],
        # eval_dataset=dataset_ds["val"],
    )

    resume_from_checkpoint = False
    if os.path.exists(output_path) and len([f for f in os.listdir(output_path) if f.startswith("checkpoint-")]) > 0:
        resume_from_checkpoint = True
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


def print_hyperparameters(args):
    print("========================================")
    print("Global Variables:")
    print("========================================")
    print(f"DEVICE_IDS: {DEVICE_IDS}")
    print(f"GLOBAL_BATCH_SIZE: {GLOBAL_BATCH_SIZE}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"GRAD_ACC_STEPS: {GRAD_ACC_STEPS}")
    print(f"RANDOM_SEED: {RANDOM_SEED}")
    print("========================================")
    print("Arguments:")
    print("========================================")
    print(json.dumps(args.__dict__, indent=4))
    print("========================================")
    print("WANDB Variables:")
    print("========================================")
    for key, value in os.environ.items():
        if key.startswith("WANDB_"):
            print(f"{key}: {value}")
    print("========================================")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", "-d", type=str, default="grpo_data")
    parser.add_argument("--model_name", "-m", type=str, required=True)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--use_rslora", action="store_true", default=False, help="Use RSLORA instead of LoRA")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--loss_type", type=str, default="bnpo", choices=["bnpo", "dapo"])
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--run_name", "-r", type=str, required=True)
    parser.add_argument("--version", type=str, default="v8")
    args = parser.parse_args()
    args.model_id = args.model_name.split("/")[-1]
    args.run_name = f"grpo_{args.run_name}_{'rslora' if args.use_rslora else 'lora'}_b{args.beta}_{args.model_id}_{args.version}"

    # set wandb environment variables
    os.environ["WANDB_RUN_ID"] = args.run_name
    os.environ["WANDB_RESUME"] = "auto"
    os.environ["WANDB_ENTITY"] = "collab-srd"
    os.environ["WANDB_PROJECT"] = "math2gcot-train"
    os.environ["WANDB_NAME"] = args.run_name
    os.environ["WANDB_TAGS"] = f"grpo_{args.version},{'rslora' if args.use_rslora else 'lora'},unsloth,{args.model_id}"

    # run the training
    print_hyperparameters(args)
    main(args)


"""
# ! 12b
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run src/train/grpo/grpo.py -m checkpoints/sft/sft_data_v1_rslora_gemma-3-12b-it_v6/sft_rslora_gemma-3-12b-it_v6_merged -r data_v2_bnpo --beta 0.0 --loss_type bnpo --version v12
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run src/train/grpo/grpo.py -m checkpoints/sft/sft_data_v1_rslora_gemma-3-12b-it_v6/sft_rslora_gemma-3-12b-it_v6_merged -r data_v2_dapo --beta 0.0 --loss_type dapo --version v12
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run src/train/grpo/grpo.py -m checkpoints/sft/sft_data_v1_rslora_gemma-3-12b-it_v6/sft_rslora_gemma-3-12b-it_v6_merged -r data_v2_bnpo --beta 0.0 --loss_type bnpo --use_rslora --version v12
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run src/train/grpo/grpo.py -m checkpoints/sft/sft_data_v1_rslora_gemma-3-12b-it_v6/sft_rslora_gemma-3-12b-it_v6_merged -r data_v2_bnpo --beta 0.1 --loss_type bnpo --version v12

# ! 4b
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run src/train/grpo/grpo.py -m checkpoints/sft/sft_data_v1_rslora_gemma-3-4b-it_v6/sft_rslora_gemma-3-4b-it_v6_merged -r data_v2_bnpo --beta 0.0 --loss_type bnpo --version v12
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run src/train/grpo/grpo.py -m checkpoints/sft/sft_data_v1_rslora_gemma-3-4b-it_v6/sft_rslora_gemma-3-4b-it_v6_merged -r data_v2_dapo --beta 0.0 --loss_type dapo --version v12
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. uv run src/train/grpo/grpo.py -m checkpoints/sft/sft_data_v1_rslora_gemma-3-4b-it_v6/sft_rslora_gemma-3-4b-it_v6_merged -r data_v2_bnpo --beta 0.0 --loss_type bnpo --use_rslora --version v12
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. uv run src/train/grpo/grpo.py -m checkpoints/sft/sft_data_v1_rslora_gemma-3-4b-it_v6/sft_rslora_gemma-3-4b-it_v6_merged -r data_v2_bnpo --beta 0.1 --loss_type bnpo --version v12


CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run src/train/grpo/grpo.py -m checkpoints/sft/sft_data_v1_rslora_gemma-3-4b-it_v6/sft_rslora_gemma-3-4b-it_v6_merged -r data_v2_bnpo --beta 0.0 --loss_type bnpo --version v13

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run src/train/grpo/grpo.py -m checkpoints/sft/sft_data_v1_rslora_gemma-3-12b-it_v6/sft_rslora_gemma-3-12b-it_v6_merged -r data_v2_bnpo --beta 0.0 --loss_type bnpo --version v13

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run src/train/grpo/grpo.py -m unsloth/gemma-3-4b-it -r data_v2_bnpo --beta 0.0 --loss_type bnpo --version v13
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run src/train/grpo/grpo.py -m unsloth/gemma-3-12b-it -r data_v2_bnpo --beta 0.0 --loss_type bnpo --version v13
"""
