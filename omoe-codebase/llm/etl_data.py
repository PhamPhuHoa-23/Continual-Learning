import re

def parse_data_from_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    
    lines = data.strip().split('\n')
    records = []
    experts = set()

    for line in lines:
        parts = line.strip().split(' | ')
        record = {}
        for part in parts:
            if ':' not in part:
                continue  # skip malformed parts
            key, value = part.split(':', 1)
            value = value.strip()

            try:
                # Normalize value (remove trailing dot without decimals like "2048.")
                normalized = re.sub(r'\.(?!\d)', '', value)
                number = float(normalized)
                if number.is_integer():
                    number = int(number)
            except ValueError:
                number = value  # fallback, just in case
                
            if key == 'ID':
                record['ID'] = int(number)
            elif key == 'Correct':
                record['Correct'] = number
            else:
                record[key] = number
                experts.add(key)
        
        records.append(record)
    
    return records, sorted(experts)

def calculate_accuracy(records, experts):
    accuracy = {expert: {'correct': 0, 'total': 0} for expert in experts}
    
    for record in records:
        correct_answer = record['Correct']
        for expert in experts:
            if expert in record:
                accuracy[expert]['total'] += 1
                if str(record[expert]).lower() == str(correct_answer).lower():
                    accuracy[expert]['correct'] += 1
    
    results = {}
    for expert in experts:
        if accuracy[expert]['total'] > 0:
            results[expert] = accuracy[expert]['correct'] / accuracy[expert]['total']
        else:
            results[expert] = 0.0
    
    return results

