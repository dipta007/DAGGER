#!/usr/bin/env python3
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import json
import math
import re
from typing import Dict, Optional, Union

import math
from typing import Union


CHECKPOINT_DIR = Path("./checkpoints")
SUMMARY_TXT = Path("./graph_exec_baseline_results_reexec.txt")


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def save_json(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)




def parse_graph_json(graph_json: str) -> Optional[Dict]:
    """
    Parse JSON from various formats with aggressive cleaning.
    Handles: quotes, garbage, corruption, incomplete JSON, etc.
    """
    if not graph_json or not isinstance(graph_json, str):
        return None

    text = graph_json.strip()

    # ========================================================================
    # STEP 1: REMOVE WRAPPING QUOTES
    # ========================================================================
    # Handle: "{ "nodes": [...] }"
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    # Handle: '{ "nodes": [...] }'
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]

    # ========================================================================
    # STEP 2: EXTRACT FROM MARKDOWN CODE BLOCKS
    # ========================================================================

    # Try ```json ... ```
    if "```json" in text:
        try:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                text = text[start:end].strip()
        except:
            pass

    # Try ```\n { ... } \n```
    elif text.count("```") >= 2:
        try:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                extracted = text[start:end].strip()
                # Remove language identifier if present
                if extracted.startswith("json"):
                    extracted = extracted[4:].strip()
                text = extracted
        except:
            pass

    # ========================================================================
    # STEP 3: REMOVE GARBAGE SUFFIXES
    # ========================================================================

    # Remove everything after final JSON closing
    # Pattern: } ] } followed by garbage
    if "} ] }" in text:
        idx = text.rfind("} ] }")
        text = text[: idx + 5]

    # Pattern: }\n] followed by garbage
    elif "}\n]" in text and "}" in text[text.rfind("}\n]") :]:
        idx = text.rfind("}\n]")
        # Find the closing } after ]
        remaining = text[idx + 3 :]
        brace_idx = remaining.find("}")
        if brace_idx != -1:
            text = text[: idx + 3 + brace_idx + 1]

    # Pattern: ] } followed by garbage (different spacing)
    elif "] }" in text:
        idx = text.rfind("] }")
        text = text[: idx + 3]

    # Remove common garbage patterns
    # Chinese characters, special symbols, etc.
    garbage_patterns = [
        r"精彩播报.*$",  # Chinese
        r"DERP.*$",  # Placeholder text
        r"\u4e00-\u9fff.*$",  # Chinese unicode range
    ]
    for pattern in garbage_patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL)

    # ========================================================================
    # STEP 4: FIX NUMBER CORRUPTIONS
    # ========================================================================

    # Fix: "val": 2wo, → "val": 2,
    text = re.sub(r"(\d+)wo\b", r"\1", text)

    # Fix: "val": 123abc, → "val": 123,
    text = re.sub(r":\s*(\d+)[a-zA-Z]+\s*([,}])", r": \1\2", text)

    # Fix: "val": 12.5xyz → "val": 12.5
    text = re.sub(r":\s*(\d+\.?\d*)[a-zA-Z]+\s*([,}])", r": \1\2", text)

    # ========================================================================
    # STEP 5: FIX COMMON JSON ERRORS
    # ========================================================================

    # Fix trailing commas: {..., } → {...}
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)

    # Fix missing commas between objects: }{  → },{
    text = re.sub(r"}\s*{", "},{", text)

    # Fix single quotes: {'id': 'n1'} → {"id": "n1"}
    # Be careful not to break legitimate single quotes in strings
    if "'" in text and '"' not in text:
        text = text.replace("'", '"')


    text = re.sub(
        r'(\]\s*)\}\s*,\s*"(distractor|label)"',
        r'\1, "\2"',
        text
    )

    # ========================================================================
    # STEP 6: TRY PARSING IN MULTIPLE WAYS
    # ========================================================================

    # Try 1: Standard JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "nodes" in data:
            return data
    except json.JSONDecodeError:
        pass

    # Try 2: Extract JSON object from text
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

    # Try 3: Repair incomplete JSON (missing wrapper)
    # Handle: [ {"id": "n1", ...}, ... ] without outer wrapper
    try:
        if '"id"' in text and '"op"' in text:
            # Find the array
            if "[" in text and "]" in text:
                start = text.find("[")
                end = text.rfind("]") + 1
                nodes_str = text[start:end]

                # Wrap it properly
                json_str = f'{{"nodes": {nodes_str}}}'
                data = json.loads(json_str)
                if isinstance(data, dict) and "nodes" in data:
                    return data
    except:
        pass

    # Try 4: Extract nodes array even if malformed
    try:
        # Look for "nodes": [ ... ]
        if '"nodes"' in text or "'nodes'" in text:
            # Find the start of the array
            nodes_match = re.search(r'["\']nodes["\']\s*:\s*\[', text)
            if nodes_match:
                start = nodes_match.end() - 1  # Include the [
                # Find matching ]
                bracket_count = 0
                end = start
                for i in range(start, len(text)):
                    if text[i] == "[":
                        bracket_count += 1
                    elif text[i] == "]":
                        bracket_count -= 1
                        if bracket_count == 0:
                            end = i + 1
                            break

                if end > start:
                    nodes_str = text[start:end]
                    json_str = f'{{"nodes": {nodes_str}}}'
                    data = json.loads(json_str)
                    if isinstance(data, dict) and "nodes" in data:
                        return data
    except:
        pass

    # Try 5: Extract individual node objects
    try:
        # Find all {...} objects that look like nodes
        node_objects = re.findall(r'\{\s*"id"\s*:\s*"[^"]+"\s*,\s*"op"\s*:\s*"[^"]+".+?\}', text, re.DOTALL)
        # node_objects = re.findall(r'\{\s*"id"\s*:\s*"[^"]+"\s*,\s*"op"\s*:\s*"[^"]+".+?\}', text, re.DOTALL)

        if node_objects:
            # Try to parse each as valid JSON
            valid_nodes = []
            for obj_str in node_objects:
                try:
                    node = json.loads(obj_str)
                    if "id" in node and "op" in node:
                        valid_nodes.append(node)
                except:
                    pass

            if valid_nodes:
                return {"nodes": valid_nodes}
    except:
        pass

    return None


def validate_and_clean_graph(graph_data: Dict) -> Optional[Dict]:
    """
    Validate and clean parsed graph data.
    """
    if not graph_data or "nodes" not in graph_data:
        return None

    nodes = graph_data["nodes"]
    if not isinstance(nodes, list) or not nodes:
        return None

    cleaned_nodes: List[Dict[str, Any]] = []
    inline_nodes_all: List[Dict[str, Any]] = [] 

    def clean_one_node(node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(node, dict):
            return None
        if "id" not in node or "op" not in node:
            return None

        cleaned_node: Dict[str, Any] = {
            "id": str(node["id"]).strip(),
            "op": str(node["op"]).lower().strip(),
        }

        # Clean value field
        if "val" in node:
            val = node["val"]
            if isinstance(val, str):
                s = val.strip()
                m = re.match(r"^\s*(-?\d+)\s*/\s*(\d+)\s*$", s)
                if m:
                    num = float(m.group(1))
                    den = float(m.group(2))
                    cleaned_node["val"] = (num / den) if den != 0 else val
                else:
                    match = re.match(r"^(-?\d+\.?\d*)", s)
                    if match:
                        try:
                            cleaned_node["val"] = float(match.group(1))
                        except Exception:
                            cleaned_node["val"] = val
                    else:
                        cleaned_node["val"] = val
            else:
                cleaned_node["val"] = val

        # Clean args field + collect inline nodes
        if "args" in node:
            args = node["args"]
            if isinstance(args, list):
                cleaned_args = []
                for arg in args:
                    if isinstance(arg, str):
                        cleaned_args.append(arg.strip())
                    elif isinstance(arg, dict):
                        # inline node or id wrapper
                        arg_id = arg.get("id")
                        if arg_id is not None:
                            arg_id = str(arg_id).strip()
                            cleaned_args.append(arg_id)

                            # if it's a full inline node (has op), collect it
                            if "op" in arg:
                                tmp = dict(arg)
                                tmp["id"] = arg_id
                                inline_nodes_all.append(tmp)
                        else:
                            cleaned_args.append(arg)
                    else:
                        cleaned_args.append(arg)
                cleaned_node["args"] = cleaned_args
            else:
                cleaned_node["args"] = args

        # Copy other fields
        for key in ["distractor", "label"]:
            if key in node:
                cleaned_node[key] = node[key]

        return cleaned_node

    # First pass: clean top-level nodes
    for node in nodes:
        cn = clean_one_node(node)
        if cn is not None:
            cleaned_nodes.append(cn)

    # Second pass: add inline nodes (dedup)
    existing_ids = {n["id"] for n in cleaned_nodes if isinstance(n, dict) and "id" in n}

    # Keep iterating because inline nodes can themselves contain inline args (rare but happens)
    queue = list(inline_nodes_all)
    while queue:
        in_node = queue.pop(0)
        cn = clean_one_node(in_node)
        if cn is None:
            continue
        if cn["id"] in existing_ids:
            continue
        cleaned_nodes.append(cn)
        existing_ids.add(cn["id"])


        for extra in inline_nodes_all:
            if isinstance(extra, dict) and "id" in extra and str(extra["id"]).strip() not in existing_ids:
                # avoid blowing up: only enqueue if not already present
                if extra not in queue:
                    queue.append(extra)

    return {"nodes": cleaned_nodes}



def find_final_result_node(nodes: Dict[str, Dict]) -> Optional[str]:
    """Find the final result node in the graph"""
    if "final_result" in nodes:
        return "final_result"

    # Check common final node names
    for node_id in ["final", "result", "answer", "output", "final_answer"]:
        if node_id in nodes:
            return node_id

    # Find non-distractor nodes
    non_distractor_nodes = [node_id for node_id, node in nodes.items() if not node.get("distractor", False)]

    if non_distractor_nodes:
        # Find leaf nodes (not referenced by others)
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

    # Return last node
    if nodes:
        return list(nodes.keys())[-1]

    return None


def execute_graph(graph_json: str) -> Union[float, str]:
    """
    Execute computational graph with robust parsing and error handling.
    """
    try:
        # Parse with aggressive cleaning
        graph_data = parse_graph_json(graph_json)
        if not graph_data:
            return "The graph JSON is malformed or missing."

        # Validate and clean
        graph_data = validate_and_clean_graph(graph_data)
        if not graph_data:
            return "Graph structure is invalid after cleaning."

        if "nodes" not in graph_data:
            return "Graph structure is invalid. Missing 'nodes' key."

        if not graph_data["nodes"]:
            return "Graph has no nodes. The nodes array is empty."

        # Build node lookup
        nodes = {node["id"]: node for node in graph_data["nodes"]}
        computed = {}
        visited = set()

        # Find final result node
        final_node_id = find_final_result_node(nodes)
        if not final_node_id:
            return "Could not identify the final result node."

        def normalize_operation(op: str) -> str:
            """Normalize operation names"""
            op_lower = op.lower().strip()
            op_map = {
                "const": "const",
                "constant": "const",
                "value": "const",
                "add": "add",
                "addition": "add",
                "plus": "add",
                "sum": "sum",
                "sub": "sub",
                "subtract": "sub",
                "minus": "sub",
                "subtraction": "sub",
                "mul": "mul",
                "multiply": "mul",
                "mult": "mul",
                "product": "mul",
                "times": "mul",
                "div": "div",
                "divide": "div",
                "division": "div",
                "mean": "mean",
                "average": "mean",
                "avg": "mean",
                "min": "min",
                "minimum": "min",
                "max": "max",
                "maximum": "max",
                "sqrt": "sqrt",
                "square_root": "sqrt",
                "pow": "pow",
                "power": "pow",
                "exponent": "pow",
                "round": "round",
                "rounding": "round",
                "floor": "floor",
                "ceil": "ceil",
                "ceiling": "ceil",
                "abs": "abs",
                "absolute": "abs",
                "mod": "mod",
                "modulo": "mod",
                "remainder": "mod",
                "gcd": "gcd",
                "lcm": "lcm",
                "identity": "identity",
                "id": "identity",
                "pass": "identity",
            }
            return op_map.get(op_lower, op_lower)

        def resolve_arg(arg: Union[str, int, float, dict]) -> float:
            if isinstance(arg, (int, float)):
                return float(arg)

            if isinstance(arg, dict):
                # If it's an inline const node (or got parsed as dict), evaluate directly
                op = str(arg.get("op", "")).lower().strip()
                if op == "const" and "val" in arg:
                    return float(arg["val"])
                if "id" in arg:
                    return compute_node(str(arg["id"]).strip())
                raise ValueError(f"Argument dict not resolvable: {arg}")

            arg_str = str(arg).strip()

            if arg_str in nodes:
                return compute_node(arg_str)

            try:
                return float(arg_str)
            except ValueError:
                raise ValueError(f"Argument '{arg}' is neither a valid node ID nor a numeric value")

        def compute_node(node_id: str) -> float:
            """Compute node value recursively"""
            # Check cache
            if node_id in computed:
                return computed[node_id]

            # Check for circular dependency
            if node_id in visited:
                raise ValueError(f"Circular dependency detected at node '{node_id}'")

            # Check node exists
            if node_id not in nodes:
                raise ValueError(f"Node '{node_id}' not found in graph")

            visited.add(node_id)
            node = nodes[node_id]

            # Check for operation
            if "op" not in node:
                raise ValueError(f"Node '{node_id}' is missing required 'op' field")

            op = normalize_operation(node["op"])

            try:
                # Execute operation
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
                    if abs(divisor) < 1e-10:  # Near-zero check
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

                # Cache result
                computed[node_id] = result
                visited.remove(node_id)
                return result

            except Exception as e:
                visited.remove(node_id)
                raise

        # Compute final result
        result = compute_node(final_node_id)
        return math.fabs(result)

    except Exception as e:
        return f"Graph execution failed: {str(e)}"


def parse_ground_truth_value(gt: Any) -> Union[int, float, str]:
    if gt is None:
        return ""
    if isinstance(gt, (int, float)):
        return gt
    s = str(gt).strip()
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


def safe_float_equal(pred: Union[int, float], gt: Union[int, float, str], tol: float = 1e-6) -> bool:
    try:
        p = float(pred)
        g = float(gt)
    except Exception:
        return False

    if abs(p - round(p)) < tol and abs(g - round(g)) < tol:
        return int(round(p)) == int(round(g))

    return abs(p - g) <= tol


def compute_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_total = 0
    n_exec_ok = 0
    n_correct = 0
    tok_sum = 0
    n_tok = 0

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

        if "output_tokens" in r:
            tok_sum += int(r.get("output_tokens", 0) or 0)
            n_tok += 1

    exec_rate = (n_exec_ok / n_total) if n_total else 0.0
    acc = (n_correct / n_total) if n_total else 0.0
    avg_tokens = (tok_sum / n_tok) if n_tok else 0.0

    return {
        "total": n_total,
        "exec_success": n_exec_ok,
        "exec_rate": exec_rate,
        "accuracy": acc,
        "avg_output_tokens": avg_tokens,
    }


def main() -> None:
    ckpt_files = sorted(CHECKPOINT_DIR.glob("*_graph.json"))
    if not ckpt_files:
        print(f"No checkpoint files found in {CHECKPOINT_DIR}")
        return

    with SUMMARY_TXT.open("w", encoding="utf-8") as sf:
        for ckpt_path in ckpt_files:
            records = load_json(ckpt_path)

            changed = False
            for r in records:
                if not isinstance(r, dict):
                    continue
                if "model_output_raw" not in r:
                    continue

                output_text = r.get("model_output_raw") or ""
                gt = parse_ground_truth_value(r.get("ground_truth"))

                exec_result = execute_graph(output_text)  
                r["exec_result"] = exec_result

                if isinstance(exec_result, (int, float)):
                    r["exec_ok"] = True
                    r["correct"] = safe_float_equal(exec_result, gt)
                else:
                    r["exec_ok"] = False
                    r["correct"] = False

                changed = True

            if changed:
                save_json(ckpt_path, records)

            summary = compute_summary(records)
            sf.write("=" * 80 + "\n")
            sf.write(f"CHECKPOINT: {ckpt_path.name}\n")
            sf.write(f"TOTAL: {summary['total']}\n")
            sf.write(f"EXEC_SUCCESS: {summary['exec_success']}\n")
            sf.write(f"EXEC_RATE: {summary['exec_rate']:.4f}\n")
            sf.write(f"ACCURACY: {summary['accuracy']:.4f}\n")
            sf.write(f"AVG OUTPUT TOKENS: {summary['avg_output_tokens']:.2f}\n\n")

            time.sleep(0.1)

    print(f"Wrote: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
