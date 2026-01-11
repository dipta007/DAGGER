import json
import math
from typing import Dict, Optional, Union

# Module-level constant - created once instead of on every execute_graph call
_OP_MAP = {
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


def _normalize_operation(op: str) -> str:
    """Normalize operation name to canonical form."""
    op_lower = op.lower().strip()
    return _OP_MAP.get(op_lower, op_lower)


def find_final_result_node(nodes: Dict[str, Dict]) -> Optional[str]:
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


def parse_graph_json(graph_json: str) -> Optional[Dict]:
    try:
        return json.loads(graph_json)
    except json.JSONDecodeError:
        pass

    try:
        if "```json" in graph_json:
            start = graph_json.find("```json") + 7
            end = graph_json.find("```", start)
            json_str = graph_json[start:end].strip()
            return json.loads(json_str)
    except:
        pass

    try:
        if "```" in graph_json:
            start = graph_json.find("```") + 3
            end = graph_json.find("```", start)
            json_str = graph_json[start:end].strip()
            if json_str.startswith("json"):
                json_str = json_str[4:].strip()
            return json.loads(json_str)
    except:
        pass

    try:
        if "{" in graph_json and "}" in graph_json:
            start = graph_json.find("{")
            end = graph_json.rfind("}") + 1
            json_str = graph_json[start:end].strip()
            return json.loads(json_str)
    except:
        pass

    return None


def execute_graph_from_data(graph_data: Dict) -> Union[float, None]:
    """Execute graph from already-parsed data to avoid double parsing."""
    try:
        if not graph_data or "nodes" not in graph_data or not graph_data["nodes"]:
            return None

        nodes = {node["id"]: node for node in graph_data["nodes"]}
        computed = {}
        visiting = set()

        final_node_id = find_final_result_node(nodes)
        if not final_node_id:
            return None

        def resolve_arg(arg: Union[str, int, float]) -> float:
            if isinstance(arg, (int, float)):
                return float(arg)

            arg_str = str(arg)

            if arg_str in nodes:
                return compute_node(arg_str)

            try:
                return float(arg_str)
            except ValueError:
                raise ValueError(f"Argument '{arg}' is neither a valid node ID nor a numeric value")

        def compute_node(node_id: str) -> float:
            if node_id in computed:
                return computed[node_id]

            if node_id in visiting:
                raise ValueError(f"Circular dependency detected at node '{node_id}'")

            if node_id not in nodes:
                raise ValueError(f"Node '{node_id}' not found in graph")

            visiting.add(node_id)
            node = nodes[node_id]

            if "op" not in node:
                raise ValueError(f"Node '{node_id}' is missing required 'op' field")

            op = _normalize_operation(node["op"])

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
                        raise ValueError(f"Add operation in node '{node_id}' needs at least 2 arguments, got {len(args)}")
                    result = sum(resolve_arg(arg) for arg in args)

                elif op == "sub":
                    args = node.get("args", [])
                    if len(args) != 2:
                        raise ValueError(f"Subtract operation in node '{node_id}' needs exactly 2 arguments, got {len(args)}")
                    result = resolve_arg(args[0]) - resolve_arg(args[1])

                elif op == "mul":
                    args = node.get("args", [])
                    if len(args) < 2:
                        raise ValueError(f"Multiply operation in node '{node_id}' needs at least 2 arguments, got {len(args)}")
                    result = 1
                    for arg in args:
                        result *= resolve_arg(arg)

                elif op == "div":
                    args = node.get("args", [])
                    if len(args) != 2:
                        raise ValueError(f"Divide operation in node '{node_id}' needs exactly 2 arguments, got {len(args)}")
                    dividend = resolve_arg(args[0])
                    divisor = resolve_arg(args[1])
                    if divisor == 0:
                        raise ValueError(f"Division by zero in node '{node_id}'. Check if divisor is computed correctly")
                    result = dividend / divisor

                elif op == "sum":
                    args = node.get("args", [])
                    if not args:
                        raise ValueError(f"Sum operation in node '{node_id}' needs at least 1 argument")
                    result = sum(resolve_arg(arg) for arg in args)

                elif op == "mean":
                    args = node.get("args", [])
                    if not args:
                        raise ValueError(f"Mean operation in node '{node_id}' needs at least 1 argument")
                    vals = [resolve_arg(arg) for arg in args]
                    result = sum(vals) / len(vals)

                elif op == "min":
                    args = node.get("args", [])
                    if not args:
                        raise ValueError(f"Min operation in node '{node_id}' needs at least 1 argument")
                    result = min(resolve_arg(arg) for arg in args)

                elif op == "max":
                    args = node.get("args", [])
                    if not args:
                        raise ValueError(f"Max operation in node '{node_id}' needs at least 1 argument")
                    result = max(resolve_arg(arg) for arg in args)

                elif op == "sqrt":
                    args = node.get("args", [])
                    if len(args) != 1:
                        raise ValueError(f"Sqrt operation in node '{node_id}' needs exactly 1 argument, got {len(args)}")
                    val = resolve_arg(args[0])
                    if val < 0:
                        raise ValueError(f"Cannot compute square root of negative number in node '{node_id}'")
                    result = math.sqrt(val)

                elif op == "pow":
                    args = node.get("args", [])
                    if len(args) != 2:
                        raise ValueError(f"Power operation in node '{node_id}' needs exactly 2 arguments, got {len(args)}")
                    base = resolve_arg(args[0])
                    exponent = resolve_arg(args[1])
                    result = base**exponent

                elif op == "round":
                    args = node.get("args", [])
                    if len(args) != 1:
                        raise ValueError(f"Round operation in node '{node_id}' needs exactly 1 argument, got {len(args)}")
                    result = round(resolve_arg(args[0]))

                elif op == "floor":
                    args = node.get("args", [])
                    if len(args) != 1:
                        raise ValueError(f"Floor operation in node '{node_id}' needs exactly 1 argument, got {len(args)}")
                    result = math.floor(resolve_arg(args[0]))

                elif op == "ceil":
                    args = node.get("args", [])
                    if len(args) != 1:
                        raise ValueError(f"Ceil operation in node '{node_id}' needs exactly 1 argument, got {len(args)}")
                    result = math.ceil(resolve_arg(args[0]))

                elif op == "abs":
                    args = node.get("args", [])
                    if len(args) == 1:
                        result = abs(resolve_arg(args[0]))
                    elif len(args) == 2:
                        result = abs(resolve_arg(args[0]) - resolve_arg(args[1]))
                    else:
                        raise ValueError(f"Abs operation in node '{node_id}' needs 1 or 2 arguments, got {len(args)}")

                elif op == "mod":
                    args = node.get("args", [])
                    if len(args) != 2:
                        raise ValueError(f"Mod operation in node '{node_id}' needs exactly 2 arguments, got {len(args)}")
                    dividend = resolve_arg(args[0])
                    divisor = resolve_arg(args[1])
                    if divisor == 0:
                        raise ValueError(f"Modulo by zero in node '{node_id}'")
                    result = dividend % divisor

                elif op == "gcd":
                    args = node.get("args", [])
                    if len(args) < 2:
                        raise ValueError(f"GCD operation in node '{node_id}' needs at least 2 arguments, got {len(args)}")
                    vals = [int(resolve_arg(arg)) for arg in args]
                    result = vals[0]
                    for val in vals[1:]:
                        result = math.gcd(result, val)

                elif op == "lcm":
                    args = node.get("args", [])
                    if len(args) < 2:
                        raise ValueError(f"LCM operation in node '{node_id}' needs at least 2 arguments, got {len(args)}")
                    vals = [int(resolve_arg(arg)) for arg in args]
                    result = vals[0]
                    for val in vals[1:]:
                        result = abs(result * val) // math.gcd(result, val)

                else:
                    raise ValueError(f"Unknown operation '{op}' in node '{node_id}'. Check if the operation name is correct")

                computed[node_id] = result
                visiting.discard(node_id)
                return result

            except Exception as e:
                visiting.discard(node_id)
                print(f"Error computing node '{node_id}': {e}")
                return None

        result = compute_node(final_node_id)
        return result

    except Exception:
        return None


def execute_graph(graph_json: str) -> Union[float, None]:
    """Wrapper that parses JSON then executes. Use execute_graph_from_data if already parsed."""
    graph_data = parse_graph_json(graph_json)
    return execute_graph_from_data(graph_data)


def correct_format(completion: str) -> str:
    """Correct the format of the completion."""
    try:
        completion = completion.strip()
        assert completion.startswith("```json"), "Completion must start with ```json"
        assert completion.endswith("```"), "Completion must end with ```"
        completion = completion.removeprefix("```json").removesuffix("```").strip()
        g = json.loads(completion)
        g = g["nodes"]
        assert isinstance(g, list), "Nodes must be a list"
        assert all(isinstance(node, dict) for node in g), "Nodes must be a list of dictionaries"
        assert len(g) >= 3, "Graph must have at least 3 nodes"

        final_result_nodes = [node for node in g if node["id"] == "final_result"]
        assert len(final_result_nodes) == 1, "There must be exactly one final_result node"

        return True
    except:
        return False


def is_correct(gt, pred) -> bool:
    def _convert_to_english(text) -> str:
        text = str(text)
        # convert numbers to english
        bn_to_en = {
            "১": "1",
            "২": "2",
            "৩": "3",
            "৪": "4",
            "৫": "5",
            "৬": "6",
            "৭": "7",
            "৮": "8",
            "৯": "9",
            "০": "0",
        }

        text = "".join([bn_to_en.get(char, char) for char in text])
        return text

    try:
        gt = _convert_to_english(gt)
        pred = _convert_to_english(pred)
    except Exception as e:
        print(f"Error converting to english: {e}")
        return False
    return math.isclose(float(gt), float(pred), rel_tol=1e-6)


def response_one_reward_rules_all(completions, **kwargs):
    """Optimized reward calculation - parses JSON once per completion."""
    rewards = []
    for completion, solution in zip(completions, kwargs["solution"]):
        completion = completion[0]["content"]
        reward = 0.0

        # Parse once and reuse
        if correct_format(completion):
            graph_data = parse_graph_json(completion)
            if graph_data is not None:
                # Valid JSON format - give 0.5 reward
                reward = 0.5
                gen_result = execute_graph_from_data(graph_data)
                if gen_result is not None:
                    # Valid execution - give 0.5 reward
                    reward += 0.5
                    if gen_result == solution or is_correct(solution, gen_result):
                        # Correct answer - give 1.0 reward
                        reward += 1.0

        rewards.append(reward)

    return rewards
