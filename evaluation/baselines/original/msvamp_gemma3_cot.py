# ============================================================
# MSVAMP Bangla Evaluation with Few-shot CoT from https://arxiv.org/pdf/2210.03057 (native)
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

model_name = "google/gemma-3-4b-it"
model_name = "google/gemma-3-12b-it"

max_seq_length = 4096
dtype = torch.bfloat16
load_in_4bit = False

print(f"Loading model {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)
print("Model loaded")

def llm_infer(prompt: str, max_new_tokens: int = 512):
    try:
        # IMPORTANT: For Gemma3Processor, use keyword `text=...`
        # Otherwise the string might be interpreted as `images`.
        encoded = tokenizer(
            text=prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_length,
        )

        # Move tensors to device
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

        in_tok = encoded["input_ids"].shape[-1]

        # Get a valid eos_token_id (some processors don't expose it)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is None:
            eos_id = getattr(model.config, "eos_token_id", None)

        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_id,
        )

        # Decode only the *new* tokens
        answer = tokenizer.decode(
            output_ids[0][in_tok:],
            skip_special_tokens=True,
        ).strip()

        out_tok = output_ids.shape[-1] - in_tok

        return answer, in_tok, out_tok

    except Exception as e:
        import traceback
        print(f"LLM error: {e}")
        traceback.print_exc()
        return "", 0, 0

MGSM_BN_FEWSHOT = """প্রশ্ন: রজারের 5টি টেনিস বল আছে। সে আরও 2 ক্যান টেনিস বল কিনেছে। প্রতিটি ক্যানে 3টি করে টেনিস বল আছে। তার কাছে এখন কতগুলি টেনিস বল আছে?
ধাপে ধাপে উত্তর: রজারের প্রথমে 5টি বল ছিল। 2টি ক্যানের প্রতিটিতে 3টে টেনিস বল মানে 6টি টেনিস বল। 5 + 6 = 11। উত্তর হল 11।

প্রশ্ন: সার্ভার কক্ষে নয়টি কম্পিউটার ছিল। সোমবার থেকে বৃহস্পতিবার প্রতিদিন আরও পাঁচটি করে কম্পিউটার স্থাপন করা হয়েছিল। সার্ভার কক্ষে এখন কতগুলি কম্পিউটার আছে?
ধাপে ধাপে উত্তর: সোমবার থেকে বৃহস্পতিবার 4দিন হয়। প্রতিদিন 5টি করে কম্পিউটার যোগ করা হয়েছে। যার অর্থ মোট 4 * 5 = 20টি কম্পিউটার যোগ করা হয়েছে। শুরুতে 9টি কম্পিউটার ছিল, তাই এখন 9 + 20 = 29টি কম্পিউটার রয়েছে। উত্তর হল 29।

প্রশ্ন: লিয়ার 32টি চকোলেট ছিল এবং তার বোনের ছিল 42টি। যদি তারা 35টি খেয়ে থাকে, তাহলে তাদের কাছে মোট কতগুলি অবিশিষ্ট আছে?
ধাপে ধাপে উত্তর: লিয়ার 32টি চকোলেট ছিল এবং লিয়ার বোনের ছিল 42টি। যার অর্থ শুরুতে 32 + 42 = 74টি চকোলেট ছিল। 35টি খাওয়া হয়ে গেছে। তাই তাদের কাছে মোট 74 - 35 = 39টি চকোলট আছে। উত্তর হল 39।

প্রশ্ন: শনের পাঁচটি খেলনা আছে। ক্রিসমাস উপলক্ষে সে তার মাতা ও পিতা উভয়ের থেকে দুটি করে খেলনা পেয়েছে। তার কাছে এখন কতগুলি খেলনা আছে?
ধাপে ধাপে উত্তর: তার কাছে 5টি খেলনা আছে। সে তার মাতার থেকে 2টি খেলনা পেয়েছিল অতএব, এরপর তার 5 + 2 = 7টি খেলনা হয়েছে। তারপর সে তার পিতার থেকে 2টি খেলনা পেয়েছিল, তাই তার মোট 7 + 2 = 9 টি খেলনা হয়েছে। উত্তর হল 9।

প্রশ্ন: মাইকেলের 58টি গলফ বল ছিল। মঙ্গলবার, সে 23টি গলফ বল হারিয়েছিল। বুধবার, সে আরও 2টি বল হারিয়েছিল। বুধবারের শেষে তার কাছে কয়টি গলফ বল ছিল?
ধাপে ধাপে উত্তর: শুরুতে মাইকেলের কাছে 58টি গলফ বল ছিল এবং সে 23টি বল হারিয়েছিল, তাই তার 58 - 23 = 35টি বল আছে। আরও 2টি বল হারানোর পর, তার এখন 35 - 2 = 33টি বল আছে। উত্তর হল 33।
"""

def extract_last_number(text: str):
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None

def build_prompt(question: str):
    return MGSM_BN_FEWSHOT.strip() + "\n\n" + "প্রশ্ন: " + question.strip() + "\nধাপে ধাপে উত্তর:"

def run_msvamp_bn(
    split="test",
    max_samples=None,
    checkpoint="msvamp_gemma3_12b_checkpoint.json",
    out_json="msvamp_gemma3_12b_detailed.json",
    out_csv="msvamp_gemma3_12b_summary.csv",
    max_new_tokens=2048
):
    print("Loading dataset...")
    ds = load_dataset("Mathoctopus/MSVAMP", "bn", split=split)
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

        prompt = build_prompt(ex["m_query"])
        answer, in_tok, out_tok = llm_infer(prompt, max_new_tokens=max_new_tokens)

        gold_raw = ex["response"]
        gold_num = extract_last_number(str(gold_raw))
        pred = extract_last_number(answer)
        
        correct = (
            pred is not None
            and gold_num is not None
            and pred == gold_num
        )

        row = {
            "id": i,
            "question": ex["m_query"],
            "gold": float(gold_num) if gold_num is not None else None,
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
    run_msvamp_bn(split="test")