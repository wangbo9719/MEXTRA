import json

def read_jsonl(data_path):
    data = []
    with open(data_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def read_json(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    return data

def save_json(data, data_path):
    json.dump(data, open(data_path, "w"))