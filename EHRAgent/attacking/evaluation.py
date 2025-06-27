from EHRAgent.help import *
import os
import random
from sentence_transformers import SentenceTransformer, util
import torch

import matplotlib.pyplot as plt
from openai import OpenAI
from tqdm import tqdm
# from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer

OPENAI_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.openai.com/v1"


def count_tokens(text, model="gpt-3.5-turbo-0125"):
    client = OpenAI(api_key=OPENAI_KEY, base_url=BASE_URL)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": text}],
        max_tokens=1
    )

    input_tokens = response.usage.prompt_tokens
    return input_tokens

def eval_dir():

    token_length_list = []

    for memory_size in [200]:

        path = "/egr/research-dselab/wangbo9/workspace/EhrAgent/running/final_results/gpt4o_general_edit_k4_q30_m{}/extraction_result.json".format(memory_size)
        data = json.load(open(path, 'r'))
        print("Memory size {}".format(memory_size))
        print(len(data))


        for count in [30]:
            retrieved_set = set()
            best_retrieved_set = set()
            success_number = 0
            half_success_number = 0
            for sample in data[:count]:

                flag = True
                half_flag = False
                cur_retrieved_queries = [example['question'] for example in sample['retrieved_examples']]

                if count == 30:
                    token_length = count_tokens(sample['first_log'][2])
                    token_length_list.append(token_length)

                for query in cur_retrieved_queries:
                    best_retrieved_set.add(query.lower())
                    if query.lower() in sample['first_output'].lower() or query.lower() in sample[
                        'second_output'].lower() or query in sample['third_output'].lower():
                        retrieved_set.add(query.lower())
                        half_flag = half_flag or True
                    else:
                        flag = flag and False

                if flag:
                    success_number += 1
                if half_flag:
                    half_success_number += 1


            print("n={}, RN={}, EN={}, CER={:.2f}, AER={:.2f}, EE={:.2f}".format(count, len(best_retrieved_set), len(retrieved_set), success_number/count, half_success_number/count, len(retrieved_set)/(4*count)))

        print("-----------------------------")

    print(sum(token_length_list)/30)


eval_dir()
