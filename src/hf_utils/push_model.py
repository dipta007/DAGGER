import os
import shutil
from unsloth import FastModel
from huggingface_hub import HfApi


def merge_model(lora_path: str, repo_id: str, max_seq_length: int = 4096, local_dir: str = None) -> str:
    model, tokenizer = FastModel.from_pretrained(
        model_name=lora_path,
        max_seq_length=max_seq_length,
        load_in_4bit=False,  # Must be False for 16bit merge
    )

    # Use temp directory on same filesystem to avoid disk space issues
    if local_dir is None:
        # local_dir = os.path.join(os.path.dirname(lora_path), f"_merged_upload_{os.path.basename(lora_path)}")
        local_dir = f"./tmp/{repo_id}"

    print(f"Saving merged model to: {local_dir}")
    model.save_pretrained_merged(local_dir, tokenizer, save_method="merged_16bit")

    print(f"Uploading to HuggingFace: {repo_id}")
    api = HfApi()
    api.create_repo(repo_id, private=True, exist_ok=True)
    api.upload_folder(folder_path=local_dir, repo_id=repo_id)

    # Clean up local merged model
    print(f"Cleaning up: {local_dir}")
    shutil.rmtree(local_dir)

    return repo_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    args = parser.parse_args()
    merge_model(args.lora_path, args.repo_id, args.max_seq_length)
