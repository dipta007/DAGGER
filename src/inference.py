#!/usr/bin/env python3
"""
Inference script for math2gcot models.
Takes a checkpoint path and question as arguments.
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from unsloth import FastLanguageModel
from src.eval.helpers import execute_graph
from src.prompt import USER_PROMPT_TEMPLATE


def load_model(checkpoint_path: str):
    """Load model and tokenizer from checkpoint path."""
    print(f"Loading model from: {checkpoint_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=4096,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded successfully!")
    return model, tokenizer


def generate_graph(model, tokenizer, question: str, max_new_tokens: int = 1024) -> str:
    """Generate computational graph for the given question."""
    messages = [
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(question=question)},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # tokenizer.padding_side = "right"

    inputs = tokenizer(text=prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )

    # Decode only the new tokens
    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response


def run_inference(checkpoint_path: str, question: str, verbose: bool = False):
    """Run inference on a single question."""
    model, tokenizer = load_model(checkpoint_path)

    print("\n" + "=" * 60)
    print("Question:")
    print(question)
    print("=" * 60)

    for i in range(20):
        print("\n\n" + "=" * 60)
        print(f"Iteration {i + 1}")
        print("-" * 60)
        # Generate graph
        graph_output = generate_graph(model, tokenizer, question)

        if verbose:
            print("\nGenerated Graph:")
            print("-" * 60)
            print(graph_output)
            print("-" * 60)

        # Execute graph
        result = execute_graph(graph_output)

        print("\nResult:")
        if isinstance(result, float):
            # Format nicely - show as int if whole number
            if result == int(result):
                print(f"  Answer: {int(result)}")
            else:
                print(f"  Answer: {result}")
        else:
            print(f"  Error: {result}")

        print("\n" + "=" * 60)
        print("=" * 60)
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a math question using a trained model checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/inference.py --checkpoint checkpoints/my_model --question "রহিম ৫টি আপেল কিনলো। সে ২টি আপেল খেয়ে ফেললো। তার কাছে কয়টি আপেল আছে?"

  python src/inference.py -c checkpoints/my_model -q "A farmer has 10 apples. He sells 3. How many remain?" -v
        """,
    )

    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (LoRA adapter or merged model)",
    )

    parser.add_argument(
        "-q",
        "--question",
        type=str,
        required=True,
        help="The math question to solve",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show the generated graph output",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate (default: 1024)",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )

    args = parser.parse_args()

    # Validate checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint path does not exist: {args.checkpoint}")
        sys.exit(1)

    # Run inference
    result = run_inference(
        checkpoint_path=str(checkpoint_path),
        question=args.question,
        verbose=args.verbose,
    )

    if args.json:
        print("\nJSON Output:")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
