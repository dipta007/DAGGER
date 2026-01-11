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
from datasets import load_dataset

from tqdm import tqdm
from openai import OpenAI



OPENAI_API_KEY = ""
client = OpenAI(api_key=OPENAI_API_KEY)



MODEL_NAME = "gpt-4.1"
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 1024

CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

DATASET_NAME = "mgsm_te_test"
CHECKPOINT_PATH = CHECKPOINT_DIR / f"{DATASET_NAME}_{MODEL_NAME}_graph.json"
SUMMARY_TXT = Path("./graph_exec_baseline_results_mgsm_te.txt")

# Thai
USER_PROMPT_TEMPLATE_THAI = """You are an expert Thai Math Reasoner. Your task is to solve mathematical problems by constructing a "Computational Graph".

### Graph Rules:
- `id`: Unique identifier (e.g., "n1", "n2").
- `val`: The raw number extracted from text (for input nodes).
- `op`: The operation (`add`, `sub`, `mul`, `div`, `round`, `sqrt`, `floor`, `sum`, `mean`). Use `const` for input numbers.
- `args`: List of input node IDs.
- `distractor`: Boolean (`true` / `false`). Set to `true` if the node is NOT used in the final calculation path.
- `label`: Label for the node in Thai.

### Available Operations:
- Input: `const` (Use this for all numbers found in text or constants).
- Arithmetic: `add`, `sub`, `mul`, `div`, `abs` (absolute difference).
- Logic/Stats: `sum`, `mean`, `min` (minimum), `max` (maximum).
- Rounding: `round` (nearest int), `floor` (round down), `ceil` (round up).
- Advanced: `sqrt`, `pow`, `mod` (remainder), `gcd`, `lcm`.
- Output: `identity` ("final_result" points to the answer node)

Only output a JSON graph representing the solution, nothing else. Nodes must be topologically sorted, and there must be exactly one "final_result" node that represents the final answer.

### Example 1:
Question:
สะพานไม้สามารถรับน้ำหนักได้ไม่เกิน 5000 ปอนด์ รถบรรทุกส่งของที่บรรทุกกล่องที่เหมือนกันหนักกล่องละ 15 ปอนด์จะผ่านสะพานนี้ แต่ละกล่องมีหนังสือ 20 เล่ม น้ำหนักรวมของคนขับและรถบรรทุกเปล่าคือ 3755 ปอนด์ รถบรรทุกเปล่าใช้น้ำมัน 60 ลิตรต่อระยะทาง 100 กิโลเมตร สามารถบรรทุกกล่องได้มากที่สุดกี่กล่องโดยไม่เกินขีดจำกัดน้ำหนักของสะพาน

Output:
```json
{{
  "nodes": [
    {{"id": "n1", "op": "const", "val": 5000, "distractor": false, "label": "น้ำหนักสูงสุดของสะพาน"}},
    {{"id": "n2", "op": "const", "val": 3755, "distractor": false, "label": "น้ำหนักรถและคนขับ"}},
    {{"id": "n3", "op": "const", "val": 15, "distractor": false, "label": "น้ำหนักต่อกล่อง"}},
    {{"id": "n4", "op": "const", "val": 20, "distractor": true, "label": "จำนวนหนังสือต่อกล่อง"}},
    {{"id": "n5", "op": "const", "val": 60, "distractor": true, "label": "น้ำมัน (ลิตร)"}},
    {{"id": "n6", "op": "const", "val": 100, "distractor": true, "label": "ระยะทาง (กม.)"}},
    {{"id": "available_weight", "op": "sub", "args": ["n1", "n2"], "distractor": false, "label": "น้ำหนักที่เหลือสำหรับกล่อง"}},
    {{"id": "max_boxes_raw", "op": "div", "args": ["available_weight", "n3"], "distractor": false, "label": "จำนวนกล่องสูงสุด (ทศนิยม)"}},
    {{"id": "max_boxes", "op": "floor", "args": ["max_boxes_raw"], "distractor": false, "label": "จำนวนกล่องสูงสุด (จำนวนเต็ม)"}},
    {{"id": "final_result", "op": "identity", "args": ["max_boxes"], "distractor": false, "label": "คำตอบสุดท้าย"}}
  ]
}}
```

### Example 2:
Question:
ลีอามีช็อกโกแลตอยู่ 32 ชิ้น และน้องสาวมีช็อกโกแลตอยู่ 42 ชิ้น หากทั้งสองคนทานช็อกโกแลตไปแล้ว 35 ชิ้น จะเหลือช็อกโกแลตทั้งหมดกี่ชิ้น

Output:
```json
{{
  "nodes": [
    {{"id": "n1", "op": "const", "val": 32, "distractor": false, "label": "ช็อกโกแลตของลีอา"}},
    {{"id": "n2", "op": "const", "val": 42, "distractor": false, "label": "ช็อกโกแลตของน้องสาว"}},
    {{"id": "n3", "op": "add", "args": ["n1", "n2"], "distractor": false, "label": "ช็อกโกแลตรวม"}},
    {{"id": "n4", "op": "const", "val": 35, "distractor": false, "label": "ช็อกโกแลตที่ทาน"}},
    {{"id": "remaining", "op": "sub", "args": ["n3", "n4"], "distractor": false, "label": "ช็อกโกแลตที่เหลือ"}},
    {{"id": "final_result", "op": "identity", "args": ["remaining"], "distractor": false, "label": "คำตอบสุดท้าย"}}
  ]
}}
```

### Example 3:
Question:
ไมเคิลมีลูกกอล์ฟ 58 ลูก ในวันอังคารเขาทำลูกกอล์ฟหายไป 23 ลูก และในวันพุธทำหายอีก 2 ลูก สิ้นสุดวันพุธไมเคิลเหลือลูกกอล์ฟกี่ลูก

Output:
```json
{{
  "nodes": [
    {{"id": "n1", "op": "const", "val": 58, "distractor": false, "label": "ลูกกอล์ฟเริ่มต้น"}},
    {{"id": "n2", "op": "const", "val": 23, "distractor": false, "label": "หายไปวันอังคาร"}},
    {{"id": "n3", "op": "sub", "args": ["n1", "n2"], "distractor": false, "label": "เหลือหลังวันอังคาร"}},
    {{"id": "n4", "op": "const", "val": 2, "distractor": false, "label": "หายไปวันพุธ"}},
    {{"id": "remaining", "op": "sub", "args": ["n3", "n4"], "distractor": false, "label": "เหลือหลังวันพุธ"}},
    {{"id": "final_result", "op": "identity", "args": ["remaining"], "distractor": false, "label": "คำตอบสุดท้าย"}}
  ]
}}
```

### Your Task:
Question:
{question}

Output:
"""


# Telegu
USER_PROMPT_TEMPLATE_TELUGU = """You are an expert Telugu Math Reasoner. Your task is to solve mathematical problems by constructing a "Computational Graph".

### Graph Rules:
- `id`: Unique identifier (e.g., "n1", "n2").
- `val`: The raw number extracted from text (for input nodes).
- `op`: The operation (`add`, `sub`, `mul`, `div`, `round`, `sqrt`, `floor`, `sum`, `mean`). Use `const` for input numbers.
- `args`: List of input node IDs.
- `distractor`: Boolean (`true` / `false`). Set to `true` if the node is NOT used in the final calculation path.
- `label`: Label for the node in Telugu.

### Available Operations:
- Input: `const` (Use this for all numbers found in text or constants).
- Arithmetic: `add`, `sub`, `mul`, `div`, `abs` (absolute difference).
- Logic/Stats: `sum`, `mean`, `min` (minimum), `max` (maximum).
- Rounding: `round` (nearest int), `floor` (round down), `ceil` (round up).
- Advanced: `sqrt`, `pow`, `mod` (remainder), `gcd`, `lcm`.
- Output: `identity` ("final_result" points to the answer node)

Only output a JSON graph representing the solution, nothing else. Nodes must be topologically sorted, and there must be exactly one "final_result" node that represents the final answer.

### Example 1:
Question:
ఒక చెక్క వంతెన 5000 పౌండ్ల కంటే ఎక్కువ బరువు మోయలేదు. ఆ వంతెన మీదుగా 15 పౌండ్ల బరువున్న ఒకే రకమైన పెట్టెలతో నిండిన ఒక డెలివరీ ట్రక్ వెళ్తుంది. ప్రతి పెట్టెలో 20 పుస్తకాలు ఉన్నాయి. డ్రైవర్ మరియు ఖాలీ ట్రక్ కలిపి 3755 పౌండ్ల బరువు ఉంది. ఖాలీ ట్రక్ 60 లీటర్ల ఇంధనంతో 100 కిలోమీటర్లు వెళ్లగలదు. వంతెన బరువు పరిమితిని దాటకుండా గరిష్టంగా ఎన్ని పెట్టెలు ట్రక్‌లో ఎక్కించవచ్చు?

Output:
```json
{{
  "nodes": [
    {{"id": "n1", "op": "const", "val": 5000, "distractor": false, "label": "వంతెన గరిష్ట బరువు"}},
    {{"id": "n2", "op": "const", "val": 3755, "distractor": false, "label": "ట్రక్ మరియు డ్రైవర్ బరువు"}},
    {{"id": "n3", "op": "const", "val": 15, "distractor": false, "label": "ప్రతి పెట్టె బరువు"}},
    {{"id": "n4", "op": "const", "val": 20, "distractor": true, "label": "ప్రతి పెట్టెలో పుస్తకాలు"}},
    {{"id": "n5", "op": "const", "val": 60, "distractor": true, "label": "ఇంధనం (లీటర్లు)"}},
    {{"id": "n6", "op": "const", "val": 100, "distractor": true, "label": "దూరం (కి.మీ.)"}},
    {{"id": "available_weight", "op": "sub", "args": ["n1", "n2"], "distractor": false, "label": "పెట్టెల కోసం అందుబాటులో ఉన్న బరువు"}},
    {{"id": "max_boxes_raw", "op": "div", "args": ["available_weight", "n3"], "distractor": false, "label": "గరిష్ట పెట్టెలు (దశాంశం)"}},
    {{"id": "max_boxes", "op": "floor", "args": ["max_boxes_raw"], "distractor": false, "label": "గరిష్ట పెట్టెలు (పూర్ణాంకం)"}},
    {{"id": "final_result", "op": "identity", "args": ["max_boxes"], "distractor": false, "label": "చివరి సమాధానం"}}
  ]
}}
```

### Example 2:
Question:
లీలా వద్ద 32 చాక్లెట్‌లు ఉన్నాయి మరియు ఆమె సోదరి వద్ద 42 చాక్లెట్‌లు ఉన్నాయి. వారు మొత్తం 35 చాక్లెట్‌లు తిన్నారు. ఇప్పుడు మొత్తం మీద ఎన్ని చాక్లెట్‌లు మిగిలి ఉన్నాయి?

Output:
```json
{{
  "nodes": [
    {{"id": "n1", "op": "const", "val": 32, "distractor": false, "label": "లీలా చాక్లెట్‌లు"}},
    {{"id": "n2", "op": "const", "val": 42, "distractor": false, "label": "సోదరి చాక్లెట్‌లు"}},
    {{"id": "n3", "op": "add", "args": ["n1", "n2"], "distractor": false, "label": "మొత్తం చాక్లెట్‌లు"}},
    {{"id": "n4", "op": "const", "val": 35, "distractor": false, "label": "తిన్న చాక్లెట్‌లు"}},
    {{"id": "remaining", "op": "sub", "args": ["n3", "n4"], "distractor": false, "label": "మిగిలిన చాక్లెట్‌లు"}},
    {{"id": "final_result", "op": "identity", "args": ["remaining"], "distractor": false, "label": "చివరి సమాధానం"}}
  ]
}}
```

### Example 3:
Question:
మైకెల్ వద్ద 58 గోల్ఫ్ బంతులు ఉన్నాయి. మంగళవారం అతడు 23 గోల్ఫ్ బంతులు కోల్పోయాడు, బుధవారం మరో 2 కోల్పోయాడు. బుధవారం చివరికి అతడి వద్ద ఎన్ని గోల్ఫ్ బంతులు ఉన్నాయి?

Output:
```json
{{
  "nodes": [
    {{"id": "n1", "op": "const", "val": 58, "distractor": false, "label": "ప్రారంభ గోల్ఫ్ బంతులు"}},
    {{"id": "n2", "op": "const", "val": 23, "distractor": false, "label": "మంగళవారం కోల్పోయినవి"}},
    {{"id": "n3", "op": "sub", "args": ["n1", "n2"], "distractor": false, "label": "మంగళవారం తర్వాత మిగిలినవి"}},
    {{"id": "n4", "op": "const", "val": 2, "distractor": false, "label": "బుధవారం కోల్పోయినవి"}},
    {{"id": "remaining", "op": "sub", "args": ["n3", "n4"], "distractor": false, "label": "బుధవారం తర్వాత మిగిలినవి"}},
    {{"id": "final_result", "op": "identity", "args": ["remaining"], "distractor": false, "label": "చివరి సమాధానం"}}
  ]
}}
```

### Your Task:
Question:
{question}

Output:
"""

OUTPUT_TEMPLATE = """\
```json
{output}
```
"""




def safe_float_equal(
    pred: Union[int, float],
    gt: Union[int, float, str],
    tol: float = 1e-6,
) -> bool:
    """
    Numeric equality check for executed graph output vs ground truth.

    - Exact match for integers
    - Tolerant match for floats (to handle minor FP noise)
    - Returns False if ground truth is non-numeric
    """
    try:
        p = float(pred)
        g = float(gt)
    except Exception:
        return False

    # Treat near-integers as integers
    if abs(p - round(p)) < tol and abs(g - round(g)) < tol:
        return int(round(p)) == int(round(g))

    return abs(p - g) <= tol

def build_user_prompt(question: str) -> str:
    return USER_PROMPT_TEMPLATE_TELUGU.format(
        question=question.strip()
    )




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

    # Try 5: Extract individual node objects
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

        def resolve_arg(arg: Union[str, int, float]) -> float:
            """Resolve argument to numeric value"""
            if isinstance(arg, (int, float)):
                return float(arg)
            if isinstance(arg, dict):
                if "id" in arg:
                    return compute_node(str(arg["id"]).strip())
                # raise ValueError(f"Argument dict missing 'id': {arg}")

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
        return math.fabs(result)

    except Exception as e:
        return f"Graph execution failed: {str(e)}"

def load_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_checkpoint(path: Path) -> Tuple[List[Dict[str, Any]], Set[int]]:
    if not path.exists():
        return [], set()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    processed: Set[int] = set()
    for r in data:
        if isinstance(r, dict) and "row_index" in r:
            try:
                processed.add(int(r["row_index"]))
            except Exception:
                pass
    return data, processed


def save_checkpoint(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def call_model(prompt: str) -> Tuple[str, int]:
    resp = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    out_text = resp.output_text or ""
    out_tokens = resp.usage.output_tokens if resp.usage else 0
    return out_text, int(out_tokens)


def parse_ground_truth(x: Any) -> Union[int, float, str]:
    s = "" if x is None else str(x)
    s = s.strip().replace(",", "")
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



def compute_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_total = 0
    n_exec_ok = 0
    n_correct = 0
    tok_sum = 0

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
        tok_sum += int(r.get("output_tokens", 0) or 0)

    exec_rate = (n_exec_ok / n_total) if n_total else 0.0
    acc = (n_correct / n_total) if n_total else 0.0
    avg_out_tok = (tok_sum / n_total) if n_total else 0.0

    return {
        "total": n_total,
        "exec_success": n_exec_ok,
        "exec_rate": exec_rate,
        "accuracy": acc,
        "avg_output_tokens": avg_out_tok,
    }


# ===============================
# Main
# ===============================
def main() -> None:
    # Load Thai / Telegu MGSM
    ds = load_dataset("jbross-ibm-research/mgsm", "te", split="test")
    rows = list(ds)

    records, processed = load_checkpoint(CHECKPOINT_PATH)

    for i, ex in enumerate(tqdm(rows, desc="mgsm_te_test")):
        if i in processed:
            continue

        question = str(ex.get("question", "")).strip()
        gt_raw = ex.get("answer_number", "")

        gt = parse_ground_truth(gt_raw)
        prompt = build_user_prompt(question)

        record: Dict[str, Any] = {
            "row_index": i,
            "augmentation_type": "none",
            "prompt": prompt,
            "ground_truth": gt,
            "question": question,
        }

        try:
            output_text, out_tokens = call_model(prompt)
            record["model_output_raw"] = output_text
            record["output_tokens"] = out_tokens

            exec_result = execute_graph(output_text)
            record["exec_result"] = exec_result

            if isinstance(exec_result, (int, float)):
                record["exec_ok"] = True
                record["correct"] = safe_float_equal(exec_result, gt)
            else:
                record["exec_ok"] = False
                record["correct"] = False

        except Exception as e:
            record["error"] = str(e)
            record["exec_ok"] = False
            record["correct"] = False

        records.append(record)
        processed.add(i)
        save_checkpoint(CHECKPOINT_PATH, records)
        time.sleep(0.1)

    summary = compute_summary(records)
    with SUMMARY_TXT.open("a", encoding="utf-8") as sf:
        sf.write("=" * 80 + "\n")
        sf.write(f"DATASET: {DATASET_NAME}\n")
        sf.write(f"MODEL: {MODEL_NAME}\n")
        sf.write(f"TOTAL: {summary['total']}\n")
        sf.write(f"EXEC_SUCCESS: {summary['exec_success']}\n")
        sf.write(f"EXEC_RATE: {summary['exec_rate']:.4f}\n")
        sf.write(f"ACCURACY: {summary['accuracy']:.4f}\n")
        sf.write(f"AVG OUTPUT TOKENS: {summary['avg_output_tokens']:.2f}\n\n")

    print("Finished.")


if __name__ == "__main__":
    main()