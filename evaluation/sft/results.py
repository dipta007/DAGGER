import json
import math
import re
import os
from pathlib import Path
from typing import Dict, Union, Optional, List, Tuple
from collections import defaultdict


# ============================================================================
# GRAPH EXECUTION FUNCTIONS (from provided code)
# ============================================================================

def parse_graph_json(graph_json: str) -> Optional[Dict]:
    """
    Parse JSON from various formats with aggressive cleaning.
    Handles: quotes, garbage, corruption, incomplete JSON, etc.
    """
    if not graph_json or not isinstance(graph_json, str):
        return None

    text = graph_json.strip()

    # Remove wrapping quotes
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]

    # Extract from markdown code blocks
    if "```json" in text:
        try:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                text = text[start:end].strip()
        except:
            pass
    elif text.count("```") >= 2:
        try:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                extracted = text[start:end].strip()
                if extracted.startswith("json"):
                    extracted = extracted[4:].strip()
                text = extracted
        except:
            pass

    # Remove garbage suffixes
    if "} ] }" in text:
        idx = text.rfind("} ] }")
        text = text[: idx + 5]
    elif "}\n]" in text and "}" in text[text.rfind("}\n]") :]:
        idx = text.rfind("}\n]")
        remaining = text[idx + 3 :]
        brace_idx = remaining.find("}")
        if brace_idx != -1:
            text = text[: idx + 3 + brace_idx + 1]
    elif "] }" in text:
        idx = text.rfind("] }")
        text = text[: idx + 3]

    # Fix number corruptions
    text = re.sub(r"(\d+)wo\b", r"\1", text)
    text = re.sub(r":\s*(\d+)[a-zA-Z]+\s*([,}])", r": \1\2", text)
    text = re.sub(r":\s*(\d+\.?\d*)[a-zA-Z]+\s*([,}])", r": \1\2", text)

    # Fix common JSON errors
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)
    text = re.sub(r"}\s*{", "},{", text)

    if "'" in text and '"' not in text:
        text = text.replace("'", '"')

    # Try parsing in multiple ways
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "nodes" in data:
            return data
    except json.JSONDecodeError:
        pass

    try:
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            json_str = text[start:end]
            data = json.loads(json_str)
            if isinstance(data, dict) and "nodes" in data:
                return data
    except:
        pass

    try:
        if '"id"' in text and '"op"' in text:
            if "[" in text and "]" in text:
                start = text.find("[")
                end = text.rfind("]") + 1
                nodes_str = text[start:end]
                json_str = f'{{"nodes": {nodes_str}}}'
                data = json.loads(json_str)
                if isinstance(data, dict) and "nodes" in data:
                    return data
    except:
        pass

    return None


def validate_and_clean_graph(graph_data: Dict) -> Optional[Dict]:
    """
    Validate and clean parsed graph data.
    Fixes common issues with node values and structure.
    """
    if not graph_data or "nodes" not in graph_data:
        return None

    nodes = graph_data["nodes"]
    if not isinstance(nodes, list) or len(nodes) == 0:
        return None

    cleaned_nodes = []

    for node in nodes:
        if not isinstance(node, dict):
            continue

        if "id" not in node or "op" not in node:
            continue

        cleaned_node = {"id": str(node["id"]), "op": str(node["op"]).lower().strip()}

        if "val" in node:
            val = node["val"]
            if isinstance(val, str):
                match = re.match(r"^(\d+\.?\d*)", val)
                if match:
                    try:
                        cleaned_node["val"] = float(match.group(1))
                    except:
                        cleaned_node["val"] = val
                else:
                    cleaned_node["val"] = val
            else:
                cleaned_node["val"] = val

        if "args" in node:
            args = node["args"]
            if isinstance(args, list):
                cleaned_args = []
                for arg in args:
                    if isinstance(arg, str):
                        cleaned_args.append(arg.strip())
                    else:
                        cleaned_args.append(arg)
                cleaned_node["args"] = cleaned_args
            else:
                cleaned_node["args"] = args

        for key in ["distractor", "label"]:
            if key in node:
                cleaned_node[key] = node[key]

        cleaned_nodes.append(cleaned_node)

    return {"nodes": cleaned_nodes}


def find_final_result_node(nodes: Dict[str, Dict]) -> Optional[str]:
    """Find the final result node in the graph"""
    if "final_result" in nodes:
        return "final_result"

    for node_id in ["final", "result", "answer", "output", "final_answer"]:
        if node_id in nodes:
            return node_id

    non_distractor_nodes = [node_id for node_id, node in nodes.items() if not node.get("distractor", False)]

    if non_distractor_nodes:
        referenced_nodes = set()
        for node in nodes.values():
            if "args" in node:
                for arg in node["args"]:
                    if isinstance(arg, str) and arg in nodes:
                        referenced_nodes.add(arg)

        leaf_nodes = [node_id for node_id in non_distractor_nodes if node_id not in referenced_nodes]

        if leaf_nodes:
            return leaf_nodes[-1]

        return non_distractor_nodes[-1]

    if nodes:
        return list(nodes.keys())[-1]

    return None


def execute_graph(graph_json: str) -> Union[float, str]:
    """
    Execute computational graph with robust parsing and error handling.
    """
    try:
        graph_data = parse_graph_json(graph_json)
        if not graph_data:
            return "The graph JSON is malformed or missing."

        graph_data = validate_and_clean_graph(graph_data)
        if not graph_data:
            return "Graph structure is invalid after cleaning."

        if "nodes" not in graph_data:
            return "Graph structure is invalid. Missing 'nodes' key."

        if not graph_data["nodes"]:
            return "Graph has no nodes. The nodes array is empty."

        nodes = {node["id"]: node for node in graph_data["nodes"]}
        computed = {}
        visited = set()

        final_node_id = find_final_result_node(nodes)
        if not final_node_id:
            return "Could not identify the final result node."

        def normalize_operation(op: str) -> str:
            op_lower = op.lower().strip()
            op_map = {
                "const": "const", "constant": "const", "value": "const",
                "add": "add", "addition": "add", "plus": "add",
                "sum": "sum",
                "sub": "sub", "subtract": "sub", "minus": "sub", "subtraction": "sub",
                "mul": "mul", "multiply": "mul", "mult": "mul", "product": "mul", "times": "mul",
                "div": "div", "divide": "div", "division": "div",
                "mean": "mean", "average": "mean", "avg": "mean",
                "min": "min", "minimum": "min",
                "max": "max", "maximum": "max",
                "sqrt": "sqrt", "square_root": "sqrt",
                "pow": "pow", "power": "pow", "exponent": "pow",
                "round": "round", "rounding": "round",
                "floor": "floor",
                "ceil": "ceil", "ceiling": "ceil",
                "abs": "abs", "absolute": "abs",
                "mod": "mod", "modulo": "mod", "remainder": "mod",
                "gcd": "gcd",
                "lcm": "lcm",
                "identity": "identity", "id": "identity", "pass": "identity",
            }
            return op_map.get(op_lower, op_lower)

        def resolve_arg(arg: Union[str, int, float]) -> float:
            if isinstance(arg, (int, float)):
                return float(arg)

            arg_str = str(arg).strip()

            if arg_str in nodes:
                return compute_node(arg_str)

            try:
                return float(arg_str)
            except ValueError:
                raise ValueError(f"Argument '{arg}' is neither a valid node ID nor a numeric value")

        def compute_node(node_id: str) -> float:
            if node_id in computed:
                return computed[node_id]

            if node_id in visited:
                raise ValueError(f"Circular dependency detected at node '{node_id}'")

            if node_id not in nodes:
                raise ValueError(f"Node '{node_id}' not found in graph")

            visited.add(node_id)
            node = nodes[node_id]

            if "op" not in node:
                raise ValueError(f"Node '{node_id}' is missing required 'op' field")

            op = normalize_operation(node["op"])

            try:
                if op == "const":
                    if "val" not in node:
                        raise ValueError(f"Const node '{node_id}' is missing required 'val' field")
                    result = float(node["val"])

                elif op == "identity":
                    if not node.get("args"):
                        raise ValueError(f"Identity node '{node_id}' missing args")
                    result = resolve_arg(node["args"][0])

                elif op == "add":
                    args = node.get("args", [])
                    if len(args) < 2:
                        raise ValueError(f"Add operation needs at least 2 arguments")
                    result = sum(resolve_arg(arg) for arg in args)

                elif op == "sub":
                    args = node.get("args", [])
                    if len(args) != 2:
                        raise ValueError(f"Subtract operation needs exactly 2 arguments")
                    result = resolve_arg(args[0]) - resolve_arg(args[1])

                elif op == "mul":
                    args = node.get("args", [])
                    if len(args) < 2:
                        raise ValueError(f"Multiply operation needs at least 2 arguments")
                    result = 1
                    for arg in args:
                        result *= resolve_arg(arg)

                elif op == "div":
                    args = node.get("args", [])
                    if len(args) != 2:
                        raise ValueError(f"Divide operation needs exactly 2 arguments")
                    dividend = resolve_arg(args[0])
                    divisor = resolve_arg(args[1])
                    if abs(divisor) < 1e-10:
                        raise ValueError(f"Division by zero")
                    result = dividend / divisor

                elif op == "sum":
                    args = node.get("args", [])
                    if not args:
                        raise ValueError(f"Sum operation needs at least 1 argument")
                    result = sum(resolve_arg(arg) for arg in args)

                elif op == "mean":
                    args = node.get("args", [])
                    if not args:
                        raise ValueError(f"Mean operation needs at least 1 argument")
                    vals = [resolve_arg(arg) for arg in args]
                    result = sum(vals) / len(vals)

                elif op == "min":
                    args = node.get("args", [])
                    if not args:
                        raise ValueError(f"Min operation needs at least 1 argument")
                    result = min(resolve_arg(arg) for arg in args)

                elif op == "max":
                    args = node.get("args", [])
                    if not args:
                        raise ValueError(f"Max operation needs at least 1 argument")
                    result = max(resolve_arg(arg) for arg in args)

                elif op == "sqrt":
                    args = node.get("args", [])
                    if not args:
                        raise ValueError(f"Sqrt operation needs 1 argument")
                    val = resolve_arg(args[0])
                    if val < 0:
                        raise ValueError(f"Cannot compute square root of negative number")
                    result = math.sqrt(val)

                elif op == "pow":
                    args = node.get("args", [])
                    if len(args) != 2:
                        raise ValueError(f"Pow operation needs exactly 2 arguments")
                    base = resolve_arg(args[0])
                    exponent = resolve_arg(args[1])
                    result = base**exponent

                elif op == "round":
                    args = node.get("args", [])
                    if not args:
                        raise ValueError(f"Round operation needs 1 argument")
                    result = round(resolve_arg(args[0]))

                elif op == "floor":
                    args = node.get("args", [])
                    if not args:
                        raise ValueError(f"Floor operation needs 1 argument")
                    result = math.floor(resolve_arg(args[0]))

                elif op == "ceil":
                    args = node.get("args", [])
                    if not args:
                        raise ValueError(f"Ceil operation needs 1 argument")
                    result = math.ceil(resolve_arg(args[0]))

                elif op == "abs":
                    args = node.get("args", [])
                    if len(args) == 1:
                        result = abs(resolve_arg(args[0]))
                    elif len(args) == 2:
                        result = abs(resolve_arg(args[0]) - resolve_arg(args[1]))
                    else:
                        raise ValueError(f"Abs operation needs 1 or 2 arguments")

                elif op == "mod":
                    args = node.get("args", [])
                    if len(args) != 2:
                        raise ValueError(f"Mod operation needs exactly 2 arguments")
                    dividend = resolve_arg(args[0])
                    divisor = resolve_arg(args[1])
                    if abs(divisor) < 1e-10:
                        raise ValueError(f"Modulo by zero")
                    result = dividend % divisor

                elif op == "gcd":
                    args = node.get("args", [])
                    if len(args) < 2:
                        raise ValueError(f"GCD operation needs at least 2 arguments")
                    vals = [int(resolve_arg(arg)) for arg in args]
                    result = vals[0]
                    for val in vals[1:]:
                        result = math.gcd(result, val)

                elif op == "lcm":
                    args = node.get("args", [])
                    if len(args) < 2:
                        raise ValueError(f"LCM operation needs at least 2 arguments")
                    vals = [int(resolve_arg(arg)) for arg in args]
                    result = vals[0]
                    for val in vals[1:]:
                        result = abs(result * val) // math.gcd(result, val)

                else:
                    raise ValueError(f"Unknown operation '{op}' at node '{node_id}'")

                computed[node_id] = result
                visited.remove(node_id)
                return result

            except Exception as e:
                visited.remove(node_id)
                raise

        result = compute_node(final_node_id)
        return result

    except Exception as e:
        return f"Graph execution failed: {str(e)}"


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def extract_model_dataset_from_filename(filename: str) -> Tuple[str, str]:
    """
    Extract model and dataset names from filename.
    Format: {dataset}_{model}_checkpoint.json
    
    Examples:
    - mgsm_gemma3_4b_checkpoint.json -> dataset: mgsm, model: gemma3_4b
    - msvamp_gemma3_12b_checkpoint.json -> dataset: msvamp, model: gemma3_12b
    """
    name = filename.replace('.json', '')
    
    if name.endswith('_checkpoint'):
        name = name[:-11]
    
    parts = name.split('_')
    
    if len(parts) >= 2:
        dataset = parts[0]
        model = '_'.join(parts[1:])
        return dataset, model
    
    return "unknown", "unknown"


def evaluate_checkpoints(checkpoint_dir: str = "./checkpoints") -> Dict:
    """
    Evaluate all checkpoint files in the specified directory.
    
    Returns:
        Dictionary with results per model/dataset combination
    """
    results = defaultdict(lambda: {
        'correct': 0,
        'total': 0,
        'total_output_tokens': 0,
        'filename': '',
        'errors': 0
    })
    
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print(f"Error: Directory '{checkpoint_dir}' does not exist")
        return results
    
    checkpoint_files = list(checkpoint_path.glob("*_checkpoint.json"))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in '{checkpoint_dir}'")
        return results
    
    print(f"Found {len(checkpoint_files)} checkpoint files\n")
    
    for checkpoint_file in checkpoint_files:
        filename = checkpoint_file.name
        dataset, model = extract_model_dataset_from_filename(filename)
        
        key = (filename, model, dataset)
        results[key]['filename'] = filename
        
        print(f"Processing: {filename}")
        print(f"  Dataset: {dataset}, Model: {model}")
        
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                print(f"  Warning: Expected list, got {type(data)}")
                continue
            
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                
                gold = float(entry.get('ground_truth'))
                output_json = entry.get('graph_output', '')
                output_tokens = entry.get('output_token', 0)
                
                if gold is None:
                    continue
                
                results[key]['total'] += 1
                results[key]['total_output_tokens'] += output_tokens
                
                try:
                    computed_result = execute_graph(output_json)
                    
                    if isinstance(computed_result, (int, float)):
                        # Compare with gold (with tolerance for floating point)
                        if abs(computed_result - gold) < 1e-6:
                            results[key]['correct'] += 1
                    else:
                        # Execution returned error message
                        results[key]['errors'] += 1
                        
                except Exception as e:
                    results[key]['errors'] += 1
            
            total = results[key]['total']
            correct = results[key]['correct']
            errors = results[key]['errors']
            accuracy = (correct / total * 100) if total > 0 else 0
            
            print(f"  Processed {total} entries: {correct} correct, {errors} errors ({accuracy:.2f}% accuracy)\n")
            
        except Exception as e:
            print(f"  Error reading file: {e}\n")
            continue
    
    return results


def write_results_to_file(results: Dict, output_file: str = "evaluation_results.txt"):
    """
    Write evaluation results to a formatted text file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write(f"{'Filename':<50} {'Model':<20} {'Dataset':<15} {'Accuracy (%)':<15} {'Avg Output Tokens':<20}\n")
        f.write("=" * 120 + "\n")
        
        # Sort by filename
        sorted_results = sorted(results.items(), key=lambda x: x[0][0])
        
        for (filename, model, dataset), stats in sorted_results:
            total = stats['total']
            correct = stats['correct']
            total_tokens = stats['total_output_tokens']
            
            if total > 0:
                accuracy = (correct / total) * 100
                avg_tokens = total_tokens / total
            else:
                accuracy = 0.0
                avg_tokens = 0.0
            
            f.write(f"{filename:<50} {model:<20} {dataset:<15} {accuracy:<15.2f} {avg_tokens:<20.2f}\n")
        
        # Write summary
        f.write("\n" + "=" * 120 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 120 + "\n")
        
        # Group by dataset
        dataset_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'errors': 0})
        for (filename, model, dataset), stats in results.items():
            dataset_stats[dataset]['correct'] += stats['correct']
            dataset_stats[dataset]['total'] += stats['total']
            dataset_stats[dataset]['errors'] += stats['errors']
        
        f.write("\nAccuracy by Dataset:\n")
        for dataset, stats in sorted(dataset_stats.items()):
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                f.write(f"  {dataset:<15}: {accuracy:>6.2f}% ({stats['correct']}/{stats['total']}, {stats['errors']} errors)\n")
        
        # Group by model
        model_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'errors': 0})
        for (filename, model, dataset), stats in results.items():
            model_stats[model]['correct'] += stats['correct']
            model_stats[model]['total'] += stats['total']
            model_stats[model]['errors'] += stats['errors']
        
        f.write("\nAccuracy by Model:\n")
        for model, stats in sorted(model_stats.items()):
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                f.write(f"  {model:<20}: {accuracy:>6.2f}% ({stats['correct']}/{stats['total']}, {stats['errors']} errors)\n")
    
    print(f"\nResults written to {output_file}")


def main():
    """
    Main evaluation function
    """
    # Hardcoded folder path (relative to current directory)
    checkpoint_dir = "original"
    results_file = "original_results.txt"
    
    print(f"\nReading checkpoints from: {checkpoint_dir}\n")
    
    # Evaluate all checkpoints
    results = evaluate_checkpoints(checkpoint_dir)
    
    if not results:
        print("\nNo results to write. Exiting.")
        return
    
    # Write results to file
    write_results_to_file(results, results_file)


if __name__ == "__main__":
    main()