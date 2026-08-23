#!/usr/bin/env python3
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from tqdm import tqdm
from openai import OpenAI

OPENAI_API_KEY = ""
client = OpenAI(api_key=OPENAI_API_KEY)

DATA_DIR = Path("../distractor_augmented_datasets")
CSV_FILES = ["augmented_mgsm.csv", "augmented_msvamp.csv"]

MODEL_NAME = "gpt-4.1"
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 512

CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

SUMMARY_TXT = Path("./pot_gpt41_results.txt")

MGSM_BN_POT_FEWSHOT = """প্রশ্ন: রজারের 5টি টেনিস বল আছে। সে আরও 2 ক্যান টেনিস বল কিনেছে। প্রতিটি ক্যানে 3টি করে টেনিস বল আছে। তার কাছে এখন কতগুলি টেনিস বল আছে?
# Python code, return ans
initial_balls = 5
num_cans = 2
balls_per_can = 3
bought_balls = num_cans * balls_per_can
ans = initial_balls + bought_balls

প্রশ্ন: সার্ভার কক্ষে নয়টি কম্পিউটার ছিল। সোমবার থেকে বৃহস্পতিবার প্রতিদিন আরও পাঁচটি করে কম্পিউটার স্থাপন করা হয়েছিল। সার্ভার কক্ষে এখন কতগুলি কম্পিউটার আছে?
# Python code, return ans
initial_computers = 9
days = 4
added_per_day = 5
added_total = days * added_per_day
ans = initial_computers + added_total

প্রশ্ন: লিয়ার 32টি চকোলেট ছিল এবং তার বোনের ছিল 42টি। যদি তারা 35টি খেয়ে থাকে, তাহলে তাদের কাছে মোট কতগুলি অবিশিষ্ট আছে?
# Python code, return ans
leah_chocolates = 32
sister_chocolates = 42
eaten = 35
total_initial = leah_chocolates + sister_chocolates
ans = total_initial - eaten
"""


def build_user_prompt(question: str) -> str:
    return (
        MGSM_BN_POT_FEWSHOT.strip()
        + "\n\n"
        + "প্রশ্ন: "
        + question.strip()
        + "\n# Python code, return ans\n"
    )


def parse_ground_truth(x: str) -> Union[int, float, str]:
    s = (x or "").strip()
    if s == "":
        return ""
    try:
        if any(ch in s for ch in [".", "e", "E"]):
            v = float(s)
            if v.is_integer():
                return int(v)
            return v
        return int(s)
    except Exception:
        return s


def safe_float_equal(
    pred: Union[int, float],
    gt: Union[int, float, str],
    tol: float = 1e-6,
) -> bool:
    try:
        p = float(pred)
        g = float(gt)
    except Exception:
        return False

    if abs(p - round(p)) < tol and abs(g - round(g)) < tol:
        return int(round(p)) == int(round(g))

    return abs(p - g) <= tol


def load_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_checkpoint(path: Path) -> Tuple[List[Dict[str, Any]], Set[int]]:
    if not path.exists():
        return [], set()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    processed: Set[int] = set()
    if isinstance(data, list):
        for r in data:
            if isinstance(r, dict) and "row_index" in r:
                try:
                    processed.add(int(r["row_index"]))
                except Exception:
                    pass
    return (data if isinstance(data, list) else []), processed


def save_checkpoint(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def call_model(prompt: str) -> Tuple[str, int]:
    resp = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    out_text = resp.output_text or ""
    out_tokens = resp.usage.output_tokens if resp.usage else 0
    return out_text, int(out_tokens)


def extract_python_code(text: str) -> str:
    t = (text or "").strip()

    if "```" in t:
        parts = t.split("```")
        for i in range(1, len(parts), 2):
            block = parts[i].strip()
            if block.startswith("python"):
                block = block[len("python") :].strip()
            if "ans" in block:
                return block

    lines = t.splitlines()
    code_lines = []
    started = False
    for ln in lines:
        if ln.strip().startswith("# Python code"):
            started = True
            continue
        if started:
            code_lines.append(ln)
    if code_lines:
        return "\n".join(code_lines).strip()

    return t


def execute_python_answer(code: str) -> Union[float, str]:
    code_str = (code or "").strip()
    if not code_str:
        return "Empty code."

    env: Dict[str, Any] = {}

    safe_builtins: Dict[str, Any] = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "range": range,
        "len": len,
        "int": int,
        "float": float,
    }
    env["__builtins__"] = safe_builtins

    try:
        exec(code_str, env, env)
    except Exception as e:
        return f"Code execution failed: {e}"

    if "ans" not in env:
        return "Missing variable 'ans'."

    ans_val = env["ans"]
    if isinstance(ans_val, bool):
        return float(int(ans_val))
    if isinstance(ans_val, (int, float)):
        return float(ans_val)

    try:
        return float(ans_val)
    except Exception:
        return f"'ans' is not numeric: {type(ans_val)}"


def compute_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_total = 0
    n_exec_ok = 0
    n_correct = 0
    tok_sum = 0

    for r in records:
        if not isinstance(r, dict):
            continue
        if "model_output_raw" not in r:
            continue

        n_total += 1
        if r.get("exec_ok") is True:
            n_exec_ok += 1
        if r.get("correct") is True:
            n_correct += 1
        tok_sum += int(r.get("output_tokens", 0) or 0)

    exec_rate = (n_exec_ok / n_total) if n_total else 0.0
    acc = (n_correct / n_total) if n_total else 0.0
    avg_out_tok = (tok_sum / n_total) if n_total else 0.0

    return {
        "total": n_total,
        "exec_success": n_exec_ok,
        "exec_rate": exec_rate,
        "accuracy": acc,
        "avg_output_tokens": avg_out_tok,
    }


def main() -> None:
    for csv_name in CSV_FILES:
        csv_path = DATA_DIR / csv_name
        if not csv_path.exists():
            print(f"Missing CSV: {csv_path}")
            continue

        checkpoint_path = CHECKPOINT_DIR / f"{csv_name.replace('.csv', '')}_{MODEL_NAME}_pot.json"
        records, processed = load_checkpoint(checkpoint_path)

        rows = load_csv_rows(csv_path)

        for row in tqdm(rows, desc=csv_name):
            idx = int(row["row_index"])
            if idx in processed:
                continue

            question = (row.get("modified_question") or "").strip()
            aug_type = (row.get("augmentation_type") or "").strip()
            gt = parse_ground_truth(row.get("ground_truth", ""))

            prompt = build_user_prompt(question)

            record: Dict[str, Any] = {
                "row_index": idx,
                "augmentation_type": aug_type,
                "prompt": prompt,
                "ground_truth": gt,
            }

            try:
                output_text, out_tokens = call_model(prompt)
                record["model_output_raw"] = output_text
                record["output_tokens"] = out_tokens

                code = extract_python_code(output_text)
                record["extracted_code"] = code

                exec_result = execute_python_answer(code)
                record["exec_result"] = exec_result

                if isinstance(exec_result, (int, float)):
                    record["exec_ok"] = True
                    record["correct"] = safe_float_equal(exec_result, gt)
                else:
                    record["exec_ok"] = False
                    record["correct"] = False

            except Exception as e:
                record["error"] = str(e)
                record["exec_ok"] = False
                record["correct"] = False

            records.append(record)
            processed.add(idx)
            save_checkpoint(checkpoint_path, records)
            time.sleep(0.5)

        summary = compute_summary(records)
        with SUMMARY_TXT.open("a+", encoding="utf-8") as sf:
            sf.write("=" * 80 + "\n")
            sf.write(f"FILE: {csv_name}\n")
            sf.write(f"MODEL: {MODEL_NAME}\n")
            sf.write(f"TOTAL: {summary['total']}\n")
            sf.write(f"EXEC_SUCCESS: {summary['exec_success']}\n")
            sf.write(f"EXEC_RATE: {summary['exec_rate']:.4f}\n")
            sf.write(f"ACCURACY: {summary['accuracy']:.4f}\n")
            sf.write(f"AVG OUTPUT TOKENS: {summary['avg_output_tokens']:.2f}\n\n")

    print("Finished.")


if __name__ == "__main__":
    main()
