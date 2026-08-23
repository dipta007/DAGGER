import os
import json
import csv
import time
import re
from pathlib import Path
from tqdm import tqdm


from openai import OpenAI

# ===============================
# API key
# ===============================
OPENAI_API_KEY = ""
client = OpenAI(api_key=OPENAI_API_KEY)

# ===============================
# Configuration
# ===============================
DATA_DIR = Path("../distractor_augmented_datasets")
CSV_FILES = ["augmented_msvamp.csv"]
MODEL_NAME = "gpt-4.1"

CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

SUMMARY_TXT = Path("./cot_gpt41_results_aug_msvamp.txt")

# ===============================
# Few-shot Bangla CoT prompt
# ===============================
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

# ===============================
# Helpers
# ===============================
def extract_last_number(text: str):
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None

def build_prompt(question: str):
    return (
        MGSM_BN_FEWSHOT.strip()
        + "\n\n"
        + "প্রশ্ন: "
        + question.strip()
        + "\nধাপে ধাপে উত্তর:"
    )

# ===============================
# Main loop
# ===============================
for csv_name in CSV_FILES:
    csv_path = DATA_DIR / csv_name
    checkpoint_path = CHECKPOINT_DIR / f"{csv_name.replace('.csv', '')}_gpt41_cot.json"

    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        processed = {r["row_index"] for r in results}
    else:
        results = []
        processed = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total_tokens = 0
    correct = 0

    for row in tqdm(rows):
        idx = int(row["row_index"])
        if idx in processed:
            continue

        question = row["modified_question"]
        gt = int(row["ground_truth"])

        prompt = build_prompt(question)

        try:
            response = client.responses.create(
                model=MODEL_NAME,
                input=prompt,
                temperature=0.0,
                max_output_tokens=512,
            )

            output_text = response.output_text
            pred = extract_last_number(output_text)
            is_correct = pred == gt

            out_tokens = response.usage.output_tokens if response.usage else 0

            record = {
                "row_index": idx,
                "augmentation_type": row["augmentation_type"],
                "prompt": prompt,
                "full_response": output_text,
                "extracted_answer": pred,
                "ground_truth": gt,
                "correct": is_correct,
                "output_tokens": out_tokens,
            }

            results.append(record)
            total_tokens += out_tokens
            correct += int(is_correct)

            with open(checkpoint_path, "w", encoding="utf-8") as cf:
                json.dump(results, cf, ensure_ascii=False, indent=2)

            time.sleep(0.5)

        except Exception as e:
            print(f"[ERROR] row {idx}: {e}")
            time.sleep(2)

    n = len(results)
    acc = correct / n if n else 0.0
    avg_tokens = total_tokens / n if n else 0.0

    with open(SUMMARY_TXT, "a", encoding="utf-8") as sf:
        sf.write("=" * 80 + "\n")
        sf.write(f"FILE: {csv_name}\n")
        sf.write(f"MODEL: {MODEL_NAME}\n")
        sf.write(f"TOTAL: {n}\n")
        sf.write(f"ACCURACY: {acc:.4f}\n")
        sf.write(f"AVG OUTPUT TOKENS: {avg_tokens:.2f}\n\n")

print("Finished.")
