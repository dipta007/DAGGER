import json
import math
import re
from typing import Dict, Optional, Union


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

    # Try 5: Last resort - extract individual node objects
    try:
        # Find all {...} objects that look like nodes
        node_objects = re.findall(r'\{\s*"id"\s*:\s*"[^"]+"\s*,\s*"op"\s*:\s*"[^"]+".+?\}', text, re.DOTALL)

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

        # Must have id and op
        if "id" not in node or "op" not in node:
            continue

        cleaned_node = {"id": str(node["id"]), "op": str(node["op"]).lower().strip()}

        # Clean value field
        if "val" in node:
            val = node["val"]
            # Handle numeric values with text: "2wo" → 2
            if isinstance(val, str):
                # Extract numeric part
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

        # Clean args field
        if "args" in node:
            args = node["args"]
            if isinstance(args, list):
                cleaned_args = []
                for arg in args:
                    # Clean string args (node IDs)
                    if isinstance(arg, str):
                        cleaned_args.append(arg.strip())
                    else:
                        cleaned_args.append(arg)
                cleaned_node["args"] = cleaned_args
            else:
                cleaned_node["args"] = args

        # Copy other fields
        for key in ["distractor", "label"]:
            if key in node:
                cleaned_node[key] = node[key]

        cleaned_nodes.append(cleaned_node)

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

    # Last resort: return last node
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

        def resolve_arg(arg: Union[str, int, float]) -> float:
            """Resolve argument to numeric value"""
            if isinstance(arg, (int, float)):
                return float(arg)

            arg_str = str(arg).strip()

            # Try as node ID
            if arg_str in nodes:
                return compute_node(arg_str)

            # Try as numeric value
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
        return result

    except Exception as e:
        return f"Graph execution failed: {str(e)}"
