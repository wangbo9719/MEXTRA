import os
import json
import random
import numpy as np
import argparse
import autogen
import sys


from EHRAgent.ehragent.toolset_high import *
from EHRAgent.ehragent.medagent import MedAgent
from EHRAgent.ehragent.config import openai_config, llm_config_list
import time
from EHRAgent.help import *
from tqdm import tqdm


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_questions", type=int, default=-1)
    parser.add_argument("--dataset", type=str, default="mimic_iii")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--data_path", type=str, default="XXX/EhrAgent/running/queries/xxx", # TODO: Revise it for your running
                        help="python list in json file, the path of generated attacking prompts")
    parser.add_argument("--memory_path", type=str, default="XXX/EhrAgent/running/memory_split/500_solution.json",  # TODO: Revise it for your running
                        help="json file, the path of memory data")

    parser.add_argument("--memory_size", type=int, default=500) # m
    parser.add_argument("--num_shots", type=int, default=4) # retrieval depth k
    parser.add_argument("--retrieve_method", type=str, default="edit_distance", help="edit_distance or cosine")
    parser.add_argument("--llm", type=str, default="gpt-4o")

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
        max_consecutive_auto_reply=1,
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

    # read from json file
    with open(args.data_path, 'r') as f:
        contents = json.load(f)

    start_time = time.time()
    if args.num_questions == -1:
        args.num_questions = len(contents)

    init_memory = json.load(open(args.memory_path, 'r'))


    answer = "gold ansuwer is none"


    # for memory_size in [50, 100, 200, 300, 400,500]:
    for memory_size in [200]:
        long_term_memory = init_memory[:memory_size]
        output_dir = "XXX/EhrAgent/running/final_results/gpt4o_general_edit_k4_q30_m" + str(memory_size) + "/"


        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        stealed_memory = []
        for i in tqdm(range(0, args.num_questions), desc="questions"):

            question = contents[i]

            current_result = {'idx': str(i), 'question': question}

            retries = 0
            max_retries = 3

            while retries < max_retries:
                logs_string = []
                try:
                    user_proxy.update_memory(args.num_shots, long_term_memory)
                    user_proxy.initiate_chat(
                        chatbot,
                        message=question,
                    )
                    logs = user_proxy._oai_messages


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
                    logs_string.append(str(e))
                # print(logs_string)
                logs_string.append("Ground-Truth Answer ---> "+answer)


                if retries == 0:

                    current_result['retrieved_examples'] = user_proxy.retrieved_examples

                    current_result['first_successful_flag'] = user_proxy.successful_flag
                    current_result['first_output'] = user_proxy.output
                    current_result['first_code'] = user_proxy.code
                    current_result['first_knowledge'] = user_proxy.knowledge
                    current_result['first_log'] = logs_string

                    current_result['second_successful_flag'] = 'none'
                    current_result['second_output'] = 'none'
                    current_result['second_code'] = 'none'
                    current_result['second_knowledge'] = 'none'
                    current_result['second_log'] = 'none'

                    current_result['third_successful_flag'] = 'none'
                    current_result['third_output'] = 'none'
                    current_result['third_code'] = 'none'
                    current_result['third_knowledge'] = 'none'
                    current_result['third_log'] = 'none'

                elif retries == 1:
                    current_result['second_successful_flag'] = user_proxy.successful_flag
                    current_result['second_output'] = user_proxy.output
                    current_result['second_code'] = user_proxy.code
                    current_result['second_knowledge'] = user_proxy.knowledge
                    current_result['second_log'] = logs_string
                elif retries == 2:
                    current_result['third_successful_flag'] = user_proxy.successful_flag
                    current_result['third_output'] = user_proxy.output
                    current_result['third_code'] = user_proxy.code
                    current_result['third_knowledge'] = user_proxy.knowledge
                    current_result['third_log'] = logs_string

                retries += 1
                if retries == max_retries:
                    stealed_memory.append(current_result)


            save_json(stealed_memory, output_dir + "/extraction_result.json")


    end_time = time.time()
    print("Time elapsed: ", end_time - start_time)

def re_func(raw_str):
    import re
    return re.sub(u"([^\u0041-\u005a\u0061-\u007a])", "", raw_str)
if __name__ == "__main__":
    main()