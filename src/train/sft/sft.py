import unsloth
import argparse
import json
import os

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

from src.prompt import USER_PROMPT_TEMPLATE, OUTPUT_TEMPLATE

RANDOM_SEED = 42
NPROC_PER_NODE = torch.cuda.device_count()
DEVICE_IDS = list(range(NPROC_PER_NODE))

GLOBAL_BATCH_SIZE = 256
BATCH_SIZE = 16
GRAD_ACC_STEPS = GLOBAL_BATCH_SIZE / (BATCH_SIZE * NPROC_PER_NODE)
if GLOBAL_BATCH_SIZE % (BATCH_SIZE * NPROC_PER_NODE) != 0:
    raise ValueError(f"Global batch size {GLOBAL_BATCH_SIZE} is not divisible by {BATCH_SIZE * NPROC_PER_NODE}")
GRAD_ACC_STEPS = int(GRAD_ACC_STEPS)


def get_dataset(args, tokenizer):
    def _get_split(split):
        with open(f"{args.dataset_path}/{split}_raw.json", "r") as reader:
            dataset = json.load(reader)

        data = []
        for item in dataset:
            prompt = [
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(question=item["question"].strip())},
                {"role": "assistant", "content": OUTPUT_TEMPLATE.format(output=item["gpt_raw_response"])},
            ]
            if item.get("english_answer", None) == None:
                gt = item.get("ground_truth")
            else:
                gt = item.get("english_answer")

            data.append(
                {
                    "conversations": prompt,
                    "response": item["gpt_raw_response"],
                    "english_answer": gt,
                }
            )
        return Dataset.from_list(data)

    def _formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix("<bos>") for convo in convos]
        del examples["conversations"]
        examples["text"] = texts
        return examples

    train_dataset = _get_split("train").map(_formatting_prompts_func, batched=True)
    val_dataset = _get_split("val").map(_formatting_prompts_func, batched=True)

    return {
        "train": train_dataset,
        "val": val_dataset,
    }


def main(args):
    output_path = f"./checkpoints/sft/{args.run_name}"

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_8bit=False,
        load_in_4bit=False,
    )
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-3",
    )

    dataset = get_dataset(args, tokenizer)
    train_dataset = dataset["train"]
    val_dataset = dataset["val"]
    print("========================================")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(train_dataset[0])
    print(val_dataset[0])
    print("========================================")

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

    training_args = SFTConfig(
        output_dir=output_path,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        num_train_epochs=4,
        learning_rate=1e-5,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": 1e-6},
        warmup_ratio=0,
        weight_decay=0.001,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        report_to="wandb",
        logging_steps=1,
        eval_steps=2,
        save_steps=2,
        eval_strategy="steps",
        save_strategy="steps",
        seed=RANDOM_SEED,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        eval_dataset=val_dataset,
        max_seq_length=args.max_seq_length,
        args=training_args,
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    print("========================================")
    print("Train Dataset:")
    print("========================================")
    print(trainer.train_dataset[0])
    print("========================================")
    print(tokenizer.decode(trainer.train_dataset[0]["input_ids"]))
    print("========================================")
    print(tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[0]["labels"]]).replace(tokenizer.pad_token, " "))
    print("========================================")
    print("Eval Dataset:")
    print("========================================")
    print(trainer.eval_dataset[0])
    print("========================================")
    print(tokenizer.decode(trainer.eval_dataset[0]["input_ids"]))
    print("========================================")
    print(tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.eval_dataset[0]["labels"]]).replace(tokenizer.pad_token, " "))
    print("========================================")

    resume_from_checkpoint = False
    if os.path.exists(output_path) and len([f for f in os.listdir(output_path) if f.startswith("checkpoint-")]) > 0:
        resume_from_checkpoint = True

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    best_model = trainer.model
    best_model.save_pretrained(f"{output_path}/best_model")
    tokenizer.save_pretrained(f"{output_path}/best_model")


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", "-d", type=str, default="sft_data")
    parser.add_argument("--model_name", "-m", type=str, required=True)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--run_name", "-r", type=str, required=True)
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--use_rslora", action="store_true", default=False, help="Use RSLORA instead of LoRA")

    args = parser.parse_args()
    args.model_id = args.model_name.split("/")[-1]
    args.run_name = f"sft_{args.run_name}_{'rslora' if args.use_rslora else 'lora'}_{args.model_id}_{args.version}"

    # set wandb environment variables
    os.environ["WANDB_RUN_ID"] = args.run_name
    os.environ["WANDB_RESUME"] = "auto"
    os.environ["WANDB_ENTITY"] = "collab-srd"
    os.environ["WANDB_PROJECT"] = "math2gcot-train"
    os.environ["WANDB_NAME"] = args.run_name
    os.environ["WANDB_TAGS"] = f"sft_{args.version},{'rslora' if args.use_rslora else 'lora'},unsloth,{args.model_id},{args.version}"

    # run the training
    print_hyperparameters(args)
    main(args)

# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run src/train/sft/sft.py -m unsloth/gemma-3-4b-it -r data_v1 --use_rslora --version v6
# CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run src/train/sft/sft.py -m unsloth/gemma-3-12b-it -r data_v1 --use_rslora --version v6

# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run src/train/sft/sft.py -m unsloth/gemma-3-4b-it -r data_v1 --version v6
# CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run src/train/sft/sft.py -m unsloth/gemma-3-12b-it -r data_v1 --version v6
