from unsloth import FastModel


def merge_model(lora_path: str, tmp_dir: str, max_seq_length: int = 4096) -> str:
    model, tokenizer = FastModel.from_pretrained(
        model_name=lora_path,
        max_seq_length=max_seq_length,
        load_in_4bit=False,  # Must be False for 16bit merge
    )
    model.save_pretrained_merged(tmp_dir, tokenizer, save_method="merged_16bit")
    return tmp_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--tmp_dir", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    args = parser.parse_args()
    merge_model(args.lora_path, args.tmp_dir, args.max_seq_length)

# uv run src/eval/merge.py --lora_path checkpoints/sft/sft_data_v1_rslora_gemma-3-4b-it_v3/best_model --tmp_dir checkpoints/sft/sft_data_v1_rslora_gemma-3-4b-it_v3/sft_gemma_4b_it
# uv run src/eval/merge.py --lora_path checkpoints/sft/sft_data_v1_rslora_gemma-3-12b-it_v3/best_model --tmp_dir checkpoints/sft/sft_data_v1_rslora_gemma-3-12b-it_v3/sft_gemma_12b_it
