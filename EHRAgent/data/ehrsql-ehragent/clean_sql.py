import sqlite3
import json
import jsonlines

DATASET = "mimic_iii"

def sql_interpreter(command):
    con = sqlite3.connect("/<YOUR_DATASET_PATH>/{}/{}.db".format(DATASET, DATASET))
    cur = con.cursor()
    try:
        results = cur.execute(command).fetchall()
    except Exception as e:
        results = e
    return results

file_path = "/<YOUR_DATASET_PATH>/{}/valid.json".format(DATASET)
# read from json file
with open(file_path, 'r') as f:
    contents = json.load(f)

new_contents = []
for item in contents:
    try:
        answer = sql_interpreter(item['query'])
        answer = [str(i[0]) for i in answer]
        item['answer'] = answer
        if len(answer) != 0:
            new_contents.append(item)
    except:
        continue

print(len(new_contents))

output_file_path = "/<YOUR_OUTPUT_DATA_PATH>/{}/valid_filtered_20240104.json".format(DATASET)
with open(output_file_path, 'w') as f:
    json.dump(new_contents, f)

output_file_path = "/<YOUR_OUTPUT_DATA_PATH>/{}/valid_filtered_20240104.jsonl".format(DATASET)
# write the new contents to jsonlines file
with jsonlines.open(output_file_path, mode='w') as writer:
    writer.write_all(new_contents)