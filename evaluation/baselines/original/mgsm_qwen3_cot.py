# ============================================================
# MGSM Bangla Evaluation with reasoning model
# - checkpointing
# - token usage stats
# ============================================================

### Setup

import os, glob, pathlib, getpass, time
from tqdm import tqdm

# configuration
API_KEY = ""  # api key
VERBOSE = False


from datasets import load_dataset
import re
import json
import pandas as pd
from tqdm import tqdm
import os
from unsloth import FastLanguageModel
import torch

model_name = "unsloth/Qwen3-8B"   # full 16-bit Qwen3-8B via Unsloth
model_name = "unsloth/Qwen3-4B"   # full 16-bit Qwen3-4B via Unsloth

max_seq_length = 8192
dtype = torch.bfloat16
load_in_4bit = False              # keep full model

print(f"Loading model {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)
print("Model loaded")


def llm_infer(prompt: str, max_new_tokens: int = 7168):
    try:
        messages = [{"role": "user", "content": prompt}]
        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # by default on
        )
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
        in_tok = inputs["input_ids"].shape[-1]
        
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        answer = tokenizer.decode(
            output_ids[0][in_tok:],
            skip_special_tokens=True
        ).strip()
        out_tok = output_ids.shape[-1] - in_tok
        
        return answer, in_tok, out_tok
    except Exception as e:
        print(f"LLM error: {e}")
        return "", 0, 0

def extract_last_number(text: str):
    # 1) Try to extract from \boxed{...}
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        nums = re.findall(r"-?\d+", boxed[-1])
        if nums:
            return int(nums[-1])

    # 2) Fallback: last integer anywhere in the text
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None

def build_prompt(question: str):
    instr = (
        "Please reason step by step in Bangla, "
        "and put your final numeric answer within \\boxed{}.\n\n"
    )
    return instr + "প্রশ্ন: " + question.strip()

def run_mgsm_bn(
    split="test",
    max_samples=None,
    checkpoint="mgsm_qwen3_4b_checkpoint.json",
    out_json="mgsm_qwen3_4b_detailed.json",
    out_csv="mgsm_qwen3_4b_summary.csv",
    max_new_tokens=7168
):
    print("Loading dataset...")
    ds = load_dataset("juletxara/mgsm", "bn", split=split)
    if max_samples:
        ds = ds.select(range(max_samples))
    print(f"Total: {len(ds)}")

    results = []
    done = set()
    if os.path.exists(checkpoint):
        try:
            ckpt = pd.read_json(checkpoint, orient="records")
            results = ckpt.to_dict("records")
            done = set(ckpt["id"].tolist())
            print(f"Resuming - already completed {len(done)}")
        except:
            print(f"Could not load checkpoint, starting fresh")

    for i, ex in enumerate(tqdm(ds)):
        if i in done:
            continue

        prompt = build_prompt(ex["question"])
        answer, in_tok, out_tok = llm_infer(prompt, max_new_tokens=max_new_tokens)

        gold = ex["answer_number"]
        pred = extract_last_number(answer)
        correct = (pred == gold) if pred is not None else False

        row = {
            "id": i,
            "question": ex["question"],
            "gold": int(gold),
            "pred": pred,
            "correct": correct,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "prompt": prompt,
            "output": answer,
        }

        results.append(row)
        done.add(i)

        pd.DataFrame(results).to_json(checkpoint, orient="records", indent=2, force_ascii=False)

    df = pd.DataFrame(results)
    df.to_json(out_json, orient="records", indent=2, force_ascii=False)
    df[["id", "gold", "pred", "correct", "input_tokens", "output_tokens"]].to_csv(out_csv, index=False)

    print(f"\nSaved detailed: {out_json}")
    print(f"Saved summary: {out_csv}")
    print(f"Accuracy: {df['correct'].mean()*100:.2f}%")

    return df

if __name__ == "__main__":
    run_mgsm_bn(split="test")