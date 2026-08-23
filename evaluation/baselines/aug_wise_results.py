#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict

AUG_LABELS = {
    "RED": "RED (Related Entity Distractor)",
    "CDD": "OAD (Orthogonal Attribute Distractor)",
    "NED": "NEED (Null-Effect Event Distractor)",
}

def safe_load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_filename(filename: str):
    """
    Expected: <dataset>_<model>_checkpoint.json
    Example:  dmgsm_gemma3_4b_0.0_checkpoint.json
    """
    name = filename.replace("_checkpoint.json", "")
    parts = name.split("_")
    if len(parts) >= 2:
        dataset = parts[0]
        model = "_".join(parts[1:])
    else:
        dataset = parts[0]
        model = "unknown"
    return dataset, model

def fmt_pct(x):
    return f"{x:6.2f}%"

def compute_augtype_stats(entries):
    """
    Returns dict:
      stats[aug_type] = {total, correct, wrong, acc}
      plus a TOTAL row in stats["ALL"]
    """
    counts = defaultdict(lambda: {"total": 0, "correct": 0, "wrong": 0})

    for e in entries:
        aug = str(e.get("aug_type", "UNKNOWN")).strip().upper()
        is_correct = bool(e.get("correct", False))

        counts[aug]["total"] += 1
        if is_correct:
            counts[aug]["correct"] += 1
        else:
            counts[aug]["wrong"] += 1

    # Also compute ALL
    all_total = sum(v["total"] for v in counts.values())
    all_correct = sum(v["correct"] for v in counts.values())
    all_wrong = sum(v["wrong"] for v in counts.values())
    counts["ALL"] = {"total": all_total, "correct": all_correct, "wrong": all_wrong}

    # Add accuracy
    stats = {}
    for aug, v in counts.items():
        total = v["total"]
        acc = (v["correct"] / total * 100.0) if total else 0.0
        stats[aug] = {**v, "acc": acc}

    return stats

def write_block(f, title, stats):
    f.write(title + "\n")
    f.write("-" * len(title) + "\n")
    f.write(f"{'AugType':<6} {'Total':>7} {'Correct':>9} {'Wrong':>7} {'Accuracy':>10}   Description\n")
    f.write("-" * 90 + "\n")

    # Print in fixed order
    order = ["RED", "CDD", "NED"]
    for aug in order:
        row = stats.get(aug, {"total": 0, "correct": 0, "wrong": 0, "acc": 0.0})
        f.write(
            f"{aug:<6} {row['total']:>7} {row['correct']:>9} {row['wrong']:>7} {fmt_pct(row['acc']):>10}   "
            f"{AUG_LABELS.get(aug, '')}\n"
        )

    # Unknown types (if any)
    extra = [k for k in stats.keys() if k not in set(order + ["ALL"])]
    for aug in sorted(extra):
        row = stats[aug]
        f.write(
            f"{aug:<6} {row['total']:>7} {row['correct']:>9} {row['wrong']:>7} {fmt_pct(row['acc']):>10}   (unexpected)\n"
        )

    # Total
    all_row = stats.get("ALL", {"total": 0, "correct": 0, "wrong": 0, "acc": 0.0})
    f.write("-" * 90 + "\n")
    f.write(
        f"{'ALL':<6} {all_row['total']:>7} {all_row['correct']:>9} {all_row['wrong']:>7} {fmt_pct(all_row['acc']):>10}\n"
    )
    f.write("\n")

def main():
    augmented_dir = Path("augmented")
    json_files = sorted(augmented_dir.glob("*_checkpoint.json"))

    # Per-file (model, dataset) blocks
    per_file_blocks = []

    # Combined per model across datasets
    per_model_entries = defaultdict(list)

    for jf in json_files:
        dataset, model = parse_filename(jf.name)
        entries = safe_load_json(jf)
        per_file_blocks.append((jf.name, dataset, model, compute_augtype_stats(entries)))
        per_model_entries[model].extend(entries)

    out_path = Path("augmented_augtype_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Augmented Dataset Results by Augmentation Type (aug_type)\n")
        f.write("=" * 70 + "\n\n")

        # 1) Per (model, dataset) = per file
        f.write("A) Per model × dataset (from each *_checkpoint.json)\n")
        f.write("=" * 70 + "\n\n")
        for filename, dataset, model, stats in per_file_blocks:
            title = f"[FILE] {filename}   |   dataset={dataset}   |   model={model}"
            write_block(f, title, stats)

        # 2) Combined across datasets per model
        f.write("B) Combined across datasets per model (all files aggregated)\n")
        f.write("=" * 70 + "\n\n")
        for model in sorted(per_model_entries.keys()):
            stats = compute_augtype_stats(per_model_entries[model])
            title = f"[MODEL AGG] model={model}   |   (all datasets combined)"
            write_block(f, title, stats)

    print(f"Saved: {out_path.resolve()}")

if __name__ == "__main__":
    main()
