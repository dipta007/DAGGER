import json
import os
import re
from pathlib import Path
from datasets import load_dataset


def extract_last_number(text):
    answer_pattern = r'উত্তর হল\s+([\d\$,\.]+)'
    match = re.search(answer_pattern, text)
    
    if match:
        num_str = match.group(1)
        num_str = num_str.replace('$', '').replace(',', '')
        try:
            num = float(num_str)
            if num == int(num):
                return int(num)
            return num
        except:
            pass
    
    patterns = [
        r'-?\d+\.?\d*',
        r'-?\d+,\d+(?:,\d+)*(?:\.\d+)?'
    ]
    
    numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            clean_num = match.replace(',', '')
            try:
                num = float(clean_num)
                numbers.append(num)
            except:
                pass
    
    if numbers:
        last_num = numbers[-1]
        try:
            if last_num == int(last_num):
                return int(last_num)
            return last_num
        except:
            return 0
    return 0


def convert_response_to_number(response_str):
    if isinstance(response_str, (int, float)):
        return float(response_str)
    
    response_str = str(response_str).replace('$', '').replace(',', '').strip()
    
    try:
        num = float(response_str)
        if num == int(num):
            return int(num)
        return num
    except:
        return 0


def process_json_file(filepath, dataset_name):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ground_truth_map = {}
    if dataset_name.lower() == 'msvamp':
        try:
            ds = load_dataset("Mathoctopus/MSVAMP", "bn")
            test_data = ds['test']
            
            for item in test_data:
                m_query = item['m_query']
                response = convert_response_to_number(item['response'])
                ground_truth_map[m_query] = response
        except Exception as e:
            print(f"Error loading MSVAMP dataset: {e}")
    
    total_correct = 0
    total_entries = len(data)
    total_output_tokens = 0
    
    for entry in data:
        if dataset_name.lower() == 'msvamp' and ground_truth_map:
            question = entry.get('question', '')
            if question in ground_truth_map:
                entry['gold'] = ground_truth_map[question]
        
        output_text = entry.get('output', '')
        pred = extract_last_number(output_text)
        entry['pred'] = pred
        
        gold = entry.get('gold', 0)
        if pred == gold:
            entry['correct'] = True
            total_correct += 1
        else:
            entry['correct'] = False
        
        total_output_tokens += entry.get('output_tokens', 0)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    accuracy = (total_correct / total_entries * 100) if total_entries > 0 else 0
    avg_output_tokens = total_output_tokens / total_entries if total_entries > 0 else 0
    
    return accuracy, avg_output_tokens


def parse_filename(filename):
    name_without_ext = filename.replace('_checkpoint.json', '')
    parts = name_without_ext.split('_')
    
    if len(parts) >= 2:
        dataset = parts[0]
        model = '_'.join(parts[1:])
    else:
        dataset = parts[0]
        model = 'unknown'
    
    return dataset, model


def main():
    current_dir = Path('augmented')
    json_files = list(current_dir.glob('*_checkpoint.json'))
    
    results = []
    
    for json_file in json_files:
        filename = json_file.name
        dataset, model = parse_filename(filename)
        
        accuracy, avg_output_tokens = process_json_file(json_file, dataset)
        
        results.append({
            'filename': filename,
            'model': model,
            'dataset': dataset,
            'accuracy': accuracy,
            'avg_output_tokens': avg_output_tokens
        })
    
    with open('augmented_results.txt', 'w', encoding='utf-8') as f:
        f.write(f"{'Filename':<50} {'Model':<20} {'Dataset':<15} {'Accuracy (%)':<15} {'Avg Output Tokens':<20}\n")
        f.write('=' * 120 + '\n')
        
        for result in results:
            f.write(f"{result['filename']:<50} {result['model']:<20} {result['dataset']:<15} {result['accuracy']:<15.2f} {result['avg_output_tokens']:<20.2f}\n")


if __name__ == '__main__':
    main()