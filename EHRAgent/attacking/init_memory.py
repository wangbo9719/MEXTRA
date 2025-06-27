import json
import random
import numpy as np
import argparse
import autogen
from EHRAgent.ehragent.toolset_high import *
from EHRAgent.ehragent.medagent import MedAgent
from EHRAgent.ehragent.config import openai_config, llm_config_list
import time
from tqdm import tqdm

from EHRAgent.ehragent.prompts_mimic import EHRAgent_4Shots_Knowledge
from EHRAgent.help import *

data_path = "XXX/EhrAgent/data/ehrsql-ehragent/mimic_iii/valid_preprocessed.jsonl"

def init_memory():
    samples_list = read_jsonl(data_path)


    random.shuffle(samples_list)


    memory_size_list = [50, 100, 200, 300, 400, 500]

    for curr_size in memory_size_list:
        querys = []
        for sample in samples_list[:curr_size]:
            querys.append(sample['question'])

        json.dump(querys, open('XXX/EhrAgent/running/memory_split/'+str(curr_size) +'.json', 'w'))

def judge(pred, ans):
    old_flag = True
    if not ans in pred:
        old_flag = False
    if "True" in pred:
        pred = pred.replace("True", "1")
    else:
        pred = pred.replace("False", "0")
    if ans == "False" or ans == "false":
        ans = "0"
    if ans == "True" or ans == "true":
        ans = "1"
    if ans == "No" or ans == "no":
        ans = "0"
    if ans == "Yes" or ans == "yes":
        ans = "1"
    if ans == "None" or ans == "none":
        ans = "0"
    if ", " in ans:
        ans = ans.split(', ')
    if ans[-2:] == ".0":
        ans = ans[:-2]
    if not type(ans) == list:
        ans = [ans]
    new_flag = True
    for i in range(len(ans)):
        if not ans[i] in pred:
            new_flag = False
            break
    return (old_flag or new_flag)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def run_memory_solution():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="test4o")
    parser.add_argument("--num_questions", type=int, default=-1)
    parser.add_argument("--dataset", type=str, default="mimic_iii")
    parser.add_argument("--data_path", type=str, default="XXX/EhrAgent/data/ehrsql-ehragent/mimic_iii/valid_preprocessed.json")
    parser.add_argument("--logs_path", type=str, default="XXX/EhrAgent/attacking/test_logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_shots", type=int, default=4)  # retrieval depth k
    parser.add_argument("--retrieve_method", type=str, default="edit_distance", help="edit_distance or cosine")  # similarity scoring function

    parser.add_argument("--memory_path", type=str,  # raw user query
                        default="XXX/EhrAgent/running/memory_split/500.json")


    args = parser.parse_args()
    set_seed(args.seed)


    config_list = [openai_config(args.llm)]
    llm_config = llm_config_list(args.seed, config_list)

    chatbot = autogen.agentchat.AssistantAgent(
        name="chatbot",
        system_message="For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done. Save the answers to the questions in the variable 'answer'. Please only generate the code.",
        llm_config=llm_config,
    )

    user_proxy = MedAgent(
        name="user_proxy",
        is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        max_consecutive_auto_reply=2,
        code_execution_config={"work_dir": "coding", "use_docker": False},
        config_list=config_list,
        retrieve_method=args.retrieve_method,
    )


    # register the functions
    user_proxy.register_function(
        function_map={
            "python": run_code
        }
    )

    user_proxy.register_dataset(args.dataset)



    with open(args.memory_path, 'r') as f:
        contents = json.load(f)  # list


    start_time = time.time()
    if args.num_questions == -1:
        args.num_questions = len(contents)
    long_term_memory = []
    init_memory = EHRAgent_4Shots_Knowledge
    init_memory = init_memory.split('\n\n')
    for i in range(len(init_memory)):
        item = init_memory[i]
        item = item.split('Question:')[-1]
        question = item.split('\nKnowledge:\n')[0]
        item = item.split('\nKnowledge:\n')[-1]
        knowledge = item.split('\nSolution:')[0]
        code = item.split('\nSolution:')[-1]
        new_item = {"question": question, "knowledge": knowledge, "code": code}
        long_term_memory.append(new_item)


    file_path = "{}/{}/".format(args.logs_path, args.num_shots) + "init_memory_results.txt"


    answer = 'none'
    true_count = 0
    for i in tqdm(range(0, args.num_questions), desc="Memory solution Generation"):

        tries = 0
        max_tries = 3
        question = contents[i]
        while tries < max_tries:
            try:
                try:
                    user_proxy.update_memory(args.num_shots, long_term_memory)
                    user_proxy.initiate_chat(
                        chatbot,
                        message=question,
                    )
                    logs = user_proxy._oai_messages

                    logs_string = []
                    logs_string.append(str(question))
                    logs_string.append(str(answer))
                    for agent in list(logs.keys()):
                        for j in range(len(logs[agent])):
                            if logs[agent][j]['content'] != None:
                                logs_string.append(logs[agent][j]['content'])
                            else:
                                argums = logs[agent][j]['function_call']['arguments']
                                if type(argums) == dict and 'cell' in argums.keys():
                                    logs_string.append(argums['cell'])
                                else:
                                    logs_string.append(argums)
                except Exception as e:
                    logs_string = [str(e)]
                print(logs_string)
                file_directory = file_path
                if type(answer) == list:
                    answer = ', '.join(answer)
                logs_string.append("Ground-Truth Answer ---> "+answer)
                logs_string = '\n----------------------------------------------------------\n'.join(logs_string)
                if '"cell": "' in logs_string:
                    last_code_start = logs_string.rfind('"cell": "')
                    last_code_end = logs_string.rfind('"\n}')
                    last_code = logs_string[last_code_start+9:last_code_end]
                else:
                    last_code_end = logs_string.rfind('Solution:')
                prediction_end = logs_string.rfind('TERMINATE')
                prediction = logs_string[last_code_end:prediction_end]
                result = judge(prediction, answer)
                true_count += 1 if result == True else 0

                result = True  # save the result whatever the solution is correct or not
                if result:
                    new_item = {"question": question, "knowledge": user_proxy.knowledge, "code": user_proxy.code}
                    long_term_memory.append(new_item)

                break
            except Exception as e:
                print(f"Execution failed on attempt {tries + 1}: {e}")
                tries += 1
                if tries == max_tries:
                    print("Max retries reached. Execution failed.")
                    raise e

        save_json(long_term_memory, "XXX/EhrAgent/running/memory_split/500_solution.json")



        logs_string += "\n\n\nReturn results --> \n" + prediction
        with open(file_directory, 'w') as f:
            f.write(logs_string)


    end_time = time.time()
    print("Time elapsed: ", end_time - start_time)



if __name__ == '__main__':

    init_memory()