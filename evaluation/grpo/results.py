#!/usr/bin/env python3
import os
import re
import json
import glob
from statistics import mean

# -----------------------
# Hardcoded folders
# -----------------------
FOLDERS = ["original", "augmented"]

# Map result filename model code → checkpoint model name
MODEL_MAP = {
    "4b": "gemma3_4b",
    "12b": "gemma3_12b",
}

RESULT_RE = re.compile(r"^(?P<model>[^_]+)_(?P<ds>[^_]+)_results_0\.0\.json$")
CKPT_RE   = re.compile(r"^(?P<ds>[^_]+)_(?P<model>gemma3_[^_]+)_0\.0_checkpoint\.json$")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return mean(vals) if vals else None


def build_token_index(folder):
    """
    (dataset, model) → (checkpoint_filename, avg_output_tokens)
    """
    index = {}
    for path in glob.glob(os.path.join(folder, "*_checkpoint.json")):
        fname = os.path.basename(path)
        m = CKPT_RE.match(fname)
        if not m:
            continue

        dataset = m.group("ds").lower()
        model = m.group("model").lower()

        data = load_json(path)
        tokens = [
            item["output_token"]
            for item in data
            if isinstance(item, dict) and isinstance(item.get("output_token"), (int, float))
        ]

        index[(dataset, model)] = (fname, safe_mean(tokens))

    return index


def collect_results(folder):
    token_index = build_token_index(folder)
    rows = []

    for path in glob.glob(os.path.join(folder, "*_results_0.0.json")):
        res_fname = os.path.basename(path)
        m = RESULT_RE.match(res_fname)
        if not m:
            continue

        model_code = m.group("model").lower()
        dataset = m.group("ds").lower()
        model_name = MODEL_MAP.get(model_code, model_code)

        data = load_json(path)
        accuracy = data.get("accuracy")
        accuracy = float(accuracy) if isinstance(accuracy, (int, float)) else None

        ckpt_fname, avg_tokens = token_index.get(
            (dataset, model_name), (None, None)
        )

        display_fname = ckpt_fname if ckpt_fname else res_fname

        rows.append(
            (display_fname, model_name, dataset, accuracy, avg_tokens)
        )

    # Stable ordering
    rows.sort(key=lambda x: (x[2], x[1]))
    return rows


def print_table(title, rows):
    print(f"\n{title}")
    print("=" * 120)
    print(
        f"{'Filename':45} {'Model':18} {'Dataset':10} "
        f"{'Accuracy (%)':14} {'Avg Output Tokens'}"
    )
    print("-" * 120)

    for fname, model, ds, acc, tok in rows:
        acc_str = f"{acc:.2f}" if acc is not None else "NA"
        tok_str = f"{tok:.2f}" if tok is not None else "NA"

        print(
            f"{fname:45} {model:18} {ds:10} "
            f"{acc_str:14} {tok_str}"
        )


def write_tsv(filename, rows):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Filename\tModel\tDataset\tAccuracy(%)\tAvgOutputTokens\n")
        for r in rows:
            f.write(
                f"{r[0]}\t{r[1]}\t{r[2]}\t"
                f"{'' if r[3] is None else f'{r[3]:.4f}'}\t"
                f"{'' if r[4] is None else f'{r[4]:.4f}'}\n"
            )


# -----------------------
# Main
# -----------------------
for folder in FOLDERS:
    rows = collect_results(folder)
    print_table(folder.upper(), rows)
    write_tsv(f"summary_{folder}.tsv", rows)
