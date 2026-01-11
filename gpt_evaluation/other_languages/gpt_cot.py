import os
import json
import csv
import time
import re
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI

# ===============================
# API key
# ===============================
OPENAI_API_KEY = ""
client = OpenAI(api_key=OPENAI_API_KEY)

# ===============================
# Configuration
# ===============================






MODEL_NAME = "gpt-4.1"

CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

SUMMARY_TXT = Path("./cot_gpt41_te_results_aug_msvamp.txt")


ds = load_dataset("jbross-ibm-research/mgsm", "te")["test"]
rows = list(ds)
dataset_name = "mgsm_te_test"
checkpoint_path = CHECKPOINT_DIR / f"{dataset_name}_gpt41_cot.json"

# ===============================
# Few-shot Thai CoT prompt
# ===============================
MGSM_TH_FEWSHOT = """
โจทย์: โรเจอร์มีลูกเทนนิส 5 ลูก เขาซื้อลูกเทนนิสเพิ่มอีก 2 กระป๋อง โดยแต่ละกระป๋องมีลูกเทนนิส 3 ลูก ตอนนี้เขามีลูกเทนนิสกี่ลูก
คำตอบทีละขั้นตอน: โรเจอร์เริ่มต้นด้วยลูกเทนนิส 5 ลูก จากนั้นเขาซื้อเพิ่มอีก 2 กระป๋อง โดยแต่ละกระป๋องมีลูกเทนนิส 3 ลูก ดังนั้นลูกเทนนิสที่ซื้อเพิ่มมีจำนวน 2 × 3 = 6 ลูก รวมทั้งหมดเป็น 5 + 6 = 11 ลูก คำตอบคือ 11

โจทย์: มีคอมพิวเตอร์เก้าเครื่องในห้องเซิร์ฟเวอร์ โดยตั้งแต่วันจันทร์ถึงวันพฤหัสบดีมีคอมพิวเตอร์ติดตั้งเพิ่มอีกวันละห้าเครื่อง ตอนนี้มีคอมพิวเตอร์ในห้องเซิร์ฟเวอร์กี่เครื่อง
คำตอบทีละขั้นตอน: ตั้งแต่วันจันทร์ถึงวันพฤหัสบดีมีทั้งหมด 4 วัน โดยแต่ละวันมีการติดตั้งคอมพิวเตอร์เพิ่ม 5 เครื่อง ดังนั้นมีคอมพิวเตอร์เพิ่มทั้งหมด 4 × 5 = 20 เครื่อง เดิมมีอยู่ 9 เครื่อง ดังนั้นรวมเป็น 9 + 20 = 29 เครื่อง คำตอบคือ 29

โจทย์: ลีอามีช็อกโกแลตอยู่ 32 ชิ้น และน้องสาวมีช็อกโกแลตอยู่ 42 ชิ้น หากทั้งสองคนทานช็อกโกแลตไปแล้ว 35 ชิ้น จะเหลือช็อกโกแลตทั้งหมดกี่ชิ้น
คำตอบทีละขั้นตอน: เริ่มต้นมีช็อกโกแลตทั้งหมด 32 + 42 = 74 ชิ้น หลังจากทานไป 35 ชิ้น จะเหลือ 74 − 35 = 39 ชิ้น คำตอบคือ 39

โจทย์: ชอว์นมีของเล่นห้าชิ้น ในวันคริสต์มาส เขาได้รับของเล่นจากแม่และพ่อคนละสองชิ้น ตอนนี้ชอว์นมีของเล่นกี่ชิ้น
คำตอบทีละขั้นตอน: ชอว์นเริ่มต้นด้วยของเล่น 5 ชิ้น เขาได้รับของเล่นจากแม่เพิ่ม 2 ชิ้น ทำให้มี 5 + 2 = 7 ชิ้น จากนั้นได้รับจากพ่ออีก 2 ชิ้น รวมเป็น 7 + 2 = 9 ชิ้น คำตอบคือ 9

โจทย์: ไมเคิลมีลูกกอล์ฟ 58 ลูก ในวันอังคารเขาทำลูกกอล์ฟหายไป 23 ลูก และในวันพุธทำหายอีก 2 ลูก สิ้นสุดวันพุธไมเคิลเหลือลูกกอล์ฟกี่ลูก
คำตอบทีละขั้นตอน: เริ่มต้นไมเคิลมีลูกกอล์ฟ 58 ลูก หลังจากทำหายไป 23 ลูก จะเหลือ 58 − 23 = 35 ลูก จากนั้นทำหายอีก 2 ลูก เหลือ 35 − 2 = 33 ลูก คำตอบคือ 33

"""

# Telegu
MGSM_TE_FEWSHOT = """
ప్రశ్న: రోజర్ వద్ద 5 టెన్నిస్ బంతులు ఉన్నాయి. అతడు మరో 2 క్యాన్‌ల టెన్నిస్ బంతులు కొనుగోలు చేశాడు. ప్రతి క్యాన్‌లో 3 టెన్నిస్ బంతులున్నాయి. ఇప్పుడు అతడి వద్ద ఎన్ని టెన్నిస్ బంతులు ఉన్నాయి?
దశల వారీ సమాధానం: రోజర్ ప్రారంభంలో 5 టెన్నిస్ బంతులు కలిగి ఉన్నాడు. తర్వాత అతడు 2 క్యాన్‌లు కొనుగోలు చేశాడు, ఒక్కో క్యాన్‌లో 3 బంతులు ఉన్నాయి. కాబట్టి అదనంగా వచ్చిన బంతులు 2 × 3 = 6. మొత్తం బంతులు 5 + 6 = 11. సమాధానం 11.

ప్రశ్న: సర్వర్ రూమ్‌లో తొమ్మిది కంప్యూటర్‌లు ఉన్నాయి. సోమవారం నుంచి గురువారం వరకు ప్రతిరోజూ మరో ఐదు కంప్యూటర్‌లు ఇన్‌స్టాల్ చేయబడ్డాయి. సర్వర్ రూమ్‌లో ఇప్పుడు ఎన్ని కంప్యూటర్‌లు ఉన్నాయి?
దశల వారీ సమాధానం: సోమవారం నుంచి గురువారం వరకు మొత్తం 4 రోజులు ఉన్నాయి. ప్రతిరోజూ 5 కంప్యూటర్‌లు చొప్పున చేర్చారు, కాబట్టి అదనంగా 4 × 5 = 20 కంప్యూటర్‌లు వచ్చాయి. మొదట 9 కంప్యూటర్‌లు ఉన్నాయి. మొత్తం 9 + 20 = 29. సమాధానం 29.

ప్రశ్న: లీలా వద్ద 32 చాక్లెట్‌లు ఉన్నాయి మరియు ఆమె సోదరి వద్ద 42 చాక్లెట్‌లు ఉన్నాయి. వారు మొత్తం 35 చాక్లెట్‌లు తిన్నారు. ఇప్పుడు మొత్తం మీద ఎన్ని చాక్లెట్‌లు మిగిలి ఉన్నాయి?
దశల వారీ సమాధానం: మొదట మొత్తం చాక్లెట్‌లు 32 + 42 = 74. వారు 35 తిన్నారు కాబట్టి మిగిలినవి 74 − 35 = 39. సమాధానం 39.

ప్రశ్న: షాన్ వద్ద 5 బొమ్మలు ఉన్నాయి. క్రిస్మస్ రోజున అతడు తన అమ్మ నుంచి 2 బొమ్మలు మరియు నాన్న నుంచి 2 బొమ్మలు పొందాడు. ఇప్పుడు అతడి వద్ద ఎన్ని బొమ్మలు ఉన్నాయి?
దశల వారీ సమాధానం: షాన్ మొదట 5 బొమ్మలు కలిగి ఉన్నాడు. అమ్మ నుంచి 2 బొమ్మలు రావడంతో 5 + 2 = 7 అయ్యాయి. తర్వాత నాన్న నుంచి మరో 2 బొమ్మలు రావడంతో 7 + 2 = 9 అయ్యాయి. సమాధానం 9.

ప్రశ్న: మైకెల్ వద్ద 58 గోల్ఫ్ బంతులు ఉన్నాయి. మంగళవారం అతడు 23 గోల్ఫ్ బంతులు కోల్పోయాడు, బుధవారం మరో 2 కోల్పోయాడు. బుధవారం చివరికి అతడి వద్ద ఎన్ని గోల్ఫ్ బంతులు ఉన్నాయి?
దశల వారీ సమాధానం: మొదట మైకెల్ వద్ద 58 బంతులు ఉన్నాయి. మంగళవారం 23 కోల్పోయిన తర్వాత 58 − 23 = 35 మిగిలాయి. బుధవారం మరో 2 కోల్పోయిన తర్వాత 35 − 2 = 33 మిగిలాయి. సమాధానం 33.
"""


# ===============================
# Helpers
# ===============================
def extract_last_number(text: str):
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None

# Thai
def build_prompt(question: str):
    return (
        MGSM_TH_FEWSHOT.strip()
        + "\n\n"
        + "โจทย์: "
        + question.strip()
        + "\nคำตอบทีละขั้นตอน:"
    )

# Telegu
def build_prompt(question: str):
    return (
        MGSM_TE_FEWSHOT.strip()
        + "\n\n"
        + "ప్రశ్న: "
        + question.strip()
        + "\nదశల వారీ సమాధానం:"
    )


if checkpoint_path.exists():
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    processed = {int(r["row_index"]) for r in results if "row_index" in r}
else:
    results = []
    processed = set()


total_tokens = 0
correct = 0

for idx, row in enumerate(tqdm(rows), start=0):
    if idx in processed:
        continue

    question = row["question"]
    gt = int(row["answer_number"])

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
            "augmentation_type": "no_aug",
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
correct_total = sum(1 for r in results if r.get("correct") is True)
tokens_total = sum(int(r.get("output_tokens", 0) or 0) for r in results)

acc = correct_total / n if n else 0.0
avg_tokens = tokens_total / n if n else 0.0

with open(SUMMARY_TXT, "a", encoding="utf-8") as sf:
    sf.write("=" * 80 + "\n")
    sf.write(f"FILE: {dataset_name}\n")
    sf.write(f"MODEL: {MODEL_NAME}\n")
    sf.write(f"TOTAL: {n}\n")
    sf.write(f"ACCURACY: {acc:.4f}\n")
    sf.write(f"AVG OUTPUT TOKENS: {avg_tokens:.2f}\n\n")

print("Finished.")
