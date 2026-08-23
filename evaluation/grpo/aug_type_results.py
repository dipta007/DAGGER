#!/usr/bin/env python3
import json
import csv
from pathlib import Path
from collections import defaultdict

# Map CSV augmentation_type -> your canonical 3 types
# Adjust these keys if your CSV uses different strings.
AUG_CANON = {
    "RED": "RED",
    "CDD": "CDD",
    "NED": "NED",
}

AUG_LABELS = {
    "RED": "RED (Related Entity Distractor)",
    "CDD": "CDD (OAD / Orthogonal Attribute Distractor)",
    "NED": "NED (NEED / Null-Effect Event Distractor)",
}

DATASET_TO_CSV = {
    # JSON dataset prefix -> CSV filename
    "dmgsm": "augmented_mgsm.csv",
    "mgsm": "augmented_mgsm.csv",
    "dmsvamp": "augmented_msvamp.csv",
    "msvamp": "augmented_msvamp.csv",
}

CSV_ROOT = Path("../../distractor_augmented_datasets")
AUGMENTED_JSON_DIR = Path("augmented")
OUT_PATH = Path("augmented_augtype_results.txt")


def parse_filename(filename: str):
    """
    Expected: <dataset>_<model>_checkpoint.json
    Example: dmgsm_gemma3_4b_0.0_checkpoint.json
    """
    name = filename.replace("_checkpoint.json", "")
    parts = name.split("_")
    dataset = parts[0] if parts else "unknown"
    model = "_".join(parts[1:]) if len(parts) >= 2 else "unknown"
    return dataset, model


def load_json_list(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_augtypes_from_csv(csv_path: Path):
    """
    Returns a list aug_types where aug_types[i] is the augmentation_type of row i.
    """
    aug_types = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "augmentation_type" not in reader.fieldnames:
            raise ValueError(f"'augmentation_type' column not found in {csv_path}")
        for row in reader:
            raw = (row.get("augmentation_type") or "").strip()
            aug_types.append(raw)
    return aug_types


def canon_augtype(raw: str):
    """
    Normalize the CSV augmentation_type to one of {RED, CDD, NED} when possible.
    If it already matches, keep it; otherwise try common normalization.
    """
    if raw is None:
        return "UNKNOWN"
    s = str(raw).strip().upper()

    # Direct hit
    if s in AUG_CANON:
        return AUG_CANON[s]

    # Some common variants (edit if your CSV uses these)
    if s in {"RELATED_ENTITY_DISTRACTOR", "RELATED ENTITY DISTRACTOR"}:
        return "RED"
    if s in {"ORTHOGONAL_ATTRIBUTE_DISTRACTOR", "OAD", "CDD"}:
        return "CDD"
    if s in {"NULL_EFFECT_EVENT_DISTRACTOR", "NULL EFFECT EVENT DISTRACTOR", "NEED", "NED"}:
        return "NED"

    return s  # keep as-is (will show up as unexpected)


def compute_stats(entries, augtype_list, dataset_tag):
    """
    Compute per-aug_type totals using JSON order alignment with CSV augmentation_type list.
    Uses entry['correct'] for correctness.
    """
    counts = defaultdict(lambda: {"total": 0, "correct": 0, "wrong": 0})

    n_json = len(entries)
    n_csv = len(augtype_list)

    if n_json > n_csv:
        raise ValueError(
            f"{dataset_tag}: JSON has {n_json} entries but CSV has only {n_csv} rows. "
            f"Order mapping would be out-of-range."
        )

    for i, e in enumerate(entries):
        aug_raw = augtype_list[i]
        aug = canon_augtype(aug_raw)

        is_correct = bool(e.get("correct", False))
        counts[aug]["total"] += 1
        if is_correct:
            counts[aug]["correct"] += 1
        else:
            counts[aug]["wrong"] += 1

    # Add ALL
    all_total = sum(v["total"] for v in counts.values())
    all_correct = sum(v["correct"] for v in counts.values())
    all_wrong = sum(v["wrong"] for v in counts.values())
    counts["ALL"] = {"total": all_total, "correct": all_correct, "wrong": all_wrong}

    # Add accuracy
    stats = {}
    for k, v in counts.items():
        total = v["total"]
        acc = (v["correct"] / total * 100.0) if total else 0.0
        stats[k] = {**v, "acc": acc}

    return stats, n_json, n_csv


def fmt_pct(x):
    return f"{x:6.2f}%"


def write_block(f, title, stats):
    f.write(title + "\n")
    f.write("-" * len(title) + "\n")
    f.write(f"{'AugType':<8} {'Total':>7} {'Correct':>9} {'Wrong':>7} {'Accuracy':>10}   Description\n")
    f.write("-" * 92 + "\n")

    order = ["RED", "CDD", "NED"]
    for aug in order:
        row = stats.get(aug, {"total": 0, "correct": 0, "wrong": 0, "acc": 0.0})
        f.write(
            f"{aug:<8} {row['total']:>7} {row['correct']:>9} {row['wrong']:>7} {fmt_pct(row['acc']):>10}   "
            f"{AUG_LABELS.get(aug, '')}\n"
        )

    extra = [k for k in stats.keys() if k not in set(order + ["ALL"])]
    for aug in sorted(extra):
        row = stats[aug]
        f.write(
            f"{aug:<8} {row['total']:>7} {row['correct']:>9} {row['wrong']:>7} {fmt_pct(row['acc']):>10}   (unexpected)\n"
        )

    all_row = stats.get("ALL", {"total": 0, "correct": 0, "wrong": 0, "acc": 0.0})
    f.write("-" * 92 + "\n")
    f.write(
        f"{'ALL':<8} {all_row['total']:>7} {all_row['correct']:>9} {all_row['wrong']:>7} {fmt_pct(all_row['acc']):>10}\n"
    )
    f.write("\n")


def main():
    # Preload CSV augmentation type lists
    csv_augtypes = {}
    for ds_key, csv_name in set(DATASET_TO_CSV.items()):
        csv_path = CSV_ROOT / csv_name
        if csv_path.exists():
            csv_augtypes[csv_name] = load_augtypes_from_csv(csv_path)

    json_files = sorted(AUGMENTED_JSON_DIR.glob("*checkpoint.json"))

    per_file_blocks = []
    per_model_agg_counts = defaultdict(list)  # model -> list of (entries, augtype_list, dataset_tag)

    for jf in json_files:
        dataset_prefix, model = parse_filename(jf.name)
        csv_name = DATASET_TO_CSV.get(dataset_prefix.lower())

        if not csv_name:
            raise ValueError(
                f"Cannot map dataset '{dataset_prefix}' from filename '{jf.name}' to a CSV. "
                f"Add it to DATASET_TO_CSV."
            )

        if csv_name not in csv_augtypes:
            raise FileNotFoundError(f"CSV not found or not loaded: {CSV_ROOT / csv_name}")

        entries = load_json_list(jf)
        augtype_list = csv_augtypes[csv_name]

        stats, n_json, n_csv = compute_stats(entries, augtype_list, dataset_tag=dataset_prefix)

        per_file_blocks.append((jf.name, dataset_prefix, model, stats, n_json, n_csv, csv_name))
        per_model_agg_counts[model].append((entries, augtype_list, dataset_prefix))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("Augmented Results by Augmentation Type (aug_type from CSV order mapping)\n")
        f.write("=" * 78 + "\n\n")

        f.write("A) Per model × dataset (each *_checkpoint.json)\n")
        f.write("=" * 78 + "\n\n")
        for filename, dataset, model, stats, n_json, n_csv, csv_name in per_file_blocks:
            title = (
                f"[FILE] {filename}   |   dataset={dataset}   |   model={model}\n"
                f"       CSV={csv_name}   |   mapped_by_row_order: JSON={n_json} rows, CSV={n_csv} rows"
            )
            write_block(f, title, stats)

        f.write("B) Combined across datasets per model (all files aggregated)\n")
        f.write("=" * 78 + "\n\n")
        for model in sorted(per_model_agg_counts.keys()):
            # Aggregate counts by iterating all entries and mapping using each file's augtype_list
            agg_counts = defaultdict(lambda: {"total": 0, "correct": 0, "wrong": 0})
            total_json = 0

            for entries, augtype_list, dataset_tag in per_model_agg_counts[model]:
                if len(entries) > len(augtype_list):
                    raise ValueError(
                        f"Model {model}, dataset {dataset_tag}: JSON larger than CSV; cannot map safely."
                    )
                total_json += len(entries)
                for i, e in enumerate(entries):
                    aug = canon_augtype(augtype_list[i])
                    is_correct = bool(e.get("correct", False))
                    agg_counts[aug]["total"] += 1
                    if is_correct:
                        agg_counts[aug]["correct"] += 1
                    else:
                        agg_counts[aug]["wrong"] += 1

            # Build stats with ALL + accuracy
            all_total = sum(v["total"] for v in agg_counts.values())
            all_correct = sum(v["correct"] for v in agg_counts.values())
            all_wrong = sum(v["wrong"] for v in agg_counts.values())
            agg_counts["ALL"] = {"total": all_total, "correct": all_correct, "wrong": all_wrong}

            agg_stats = {}
            for k, v in agg_counts.items():
                total = v["total"]
                acc = (v["correct"] / total * 100.0) if total else 0.0
                agg_stats[k] = {**v, "acc": acc}

            title = f"[MODEL AGG] model={model}   |   all datasets combined   |   total_json_rows={total_json}"
            write_block(f, title, agg_stats)

    print(f"Saved: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
