import json
import jsonlines
from datasets import Dataset

from src.prompt import USER_PROMPT_TEMPLATE, OUTPUT_TEMPLATE


def _get_sft_split(split):
    with open(f"sft_data/{split}_raw.json", "r") as reader:
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


def _get_grpo_split():
    with open("grpo_data/numina_3000.json", "r") as reader:
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


def main():
    sft_train = _get_sft_split("train")
    sft_val = _get_sft_split("val")
    sft_train.push_to_hub("dipta007/math2gcot", "sft", split="train")
    sft_val.push_to_hub("dipta007/math2gcot", "sft", split="val")

    grpo_train = _get_grpo_split()
    grpo_train.push_to_hub("dipta007/math2gcot", "grpo", split="train")


if __name__ == "__main__":
    main()
