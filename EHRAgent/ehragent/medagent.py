import time
from typing import Dict, List, Optional, Union, Callable, Literal, Optional, Union
import logging
import asyncio
import openai
import json
from openai import OpenAI, AzureOpenAI
from autogen.agentchat import Agent, UserProxyAgent, ConversableAgent
from termcolor import colored
import Levenshtein
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class MedAgent(UserProxyAgent):
    def __init__(
        self,
        name: str,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "ALWAYS",
        function_map: Optional[Dict[str, Callable]] = None,
        code_execution_config: Optional[Union[Dict, Literal[False]]] = None,
        default_auto_reply: Optional[Union[str, Dict, None]] = "",
        llm_config: Optional[Union[Dict, Literal[False]]] = False,
        system_message: Optional[Union[str, List]] = "",
        config_list: Optional[List[Dict]] = None,
        retrieve_method: Optional[str] = "edit_distance"
    ):
        super().__init__(
            name=name,
            system_message=system_message,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            function_map=function_map,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply,
        )
        self.config_list = config_list
        self.question = ''
        self.code = ''
        self.knowledge = ''
        self.retrieved_examples = []
        self.output = ''
        self.successful_flag = False
        self.retrieve_method = retrieve_method
        
    def retrieve_knowledge(self, config, query):
        # import prompt
        if self.dataset == 'mimic_iii':
            from ehragent.prompts_mimic import RetrKnowledge
        else:
            from ehragent.prompts_eicu import RetrKnowledge
        # Returns the related information to the given query.
        patience = 2
        sleep_time = 30
        # openai.api_type = config["api_type"]
        # openai.api_base = config["base_url"]
        # openai.api_version = config["api_version"]
        openai.api_key = config["api_key"]
        engine = config["model"]
        query_message = RetrKnowledge.format(question=query)
        messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},
                    {"role":"user","content": query_message}]
        # client = AzureOpenAI(
        #     api_key=config["api_key"],
        #     azure_endpoint=config["base_url"],
        #     api_version=config["api_version"],
        # )
        client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"]
        )
        while patience > 0:
            patience -= 1
            try:
                response = client.chat.completions.create(
                    model=engine,
                    messages = messages,
                    temperature=0,
                    max_tokens=800,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None)
                prediction = response.choices[0].message.content.strip()
                if prediction != "" and prediction != None:
                    return prediction
            except Exception as e:
                print(e)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        return "Fail to retrieve related knowledge, please try again later."

    def retrieve_examples(self, query):
        similarity_dist = {}
        txt_edit_dist = {}

        if self.retrieve_method == "edit_distance":
            for i in range(len(self.memory)):
                question = self.memory[i]["question"]
                similarity_dist[i] = Levenshtein.distance(query, question)
                txt_edit_dist[i] = [Levenshtein.distance(query, question), question]
            similarity_dist = sorted(similarity_dist.items(), key=lambda x: x[1], reverse=False)
            txt_edit_dist = sorted(txt_edit_dist.items(), key=lambda x: x[1][0], reverse=False)
        elif self.retrieve_method == "cosine":
            def mean_pooling(model_output, attention_mask):
                token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                          min=1e-9)

            def get_neg_cosine_similarity(cosine_scores, query, questions):

                cosine_scores = cosine_scores.tolist()
                similarity_dist = {_i:-c for _i, c in enumerate(cosine_scores)}
                similarity_dist = sorted(similarity_dist.items(), key=lambda x: x[1], reverse=False)
                txt_neg_cosine_dist = {_i: [-cosine_scores[_i], questions[_i]] for _i in range(len(cosine_scores))}
                txt_neg_cosine_dist = sorted(txt_neg_cosine_dist.items(), key=lambda x: x[1][0], reverse=False)
                return similarity_dist, txt_neg_cosine_dist

            # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


            questions = [self.memory[i]["question"] for i in range(len(self.memory))] + [query]
            encoded_questions_query = tokenizer(questions, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                encoded_output = model(**encoded_questions_query)
            encoded_embeddings = mean_pooling(encoded_output, encoded_questions_query['attention_mask'])
            encoded_embeddings = F.normalize(encoded_embeddings, p=2, dim=1)
            questions_embeddings, query_embedding = encoded_embeddings[:-1], encoded_embeddings[-1].unsqueeze(0)

            similarities = torch.mm(query_embedding, questions_embeddings.T).squeeze()
            similarity_dist, txt_edit_dist = get_neg_cosine_similarity(similarities, query, questions)



        print("######################################################")
        print("Query: " + query)
        idx_list = []
        for t in txt_edit_dist:
            print(t)
            idx_list.append(t[0])
        print(idx_list[:4])
        print("######################################################")



        selected_indexes = [similarity_dist[i][0] for i in range(min(self.num_shots, len(similarity_dist)))]
        examples = []
        raw_examples = []
        for i in selected_indexes:
            template = "Question: {}\nKnowledge:\n{}\nSolution:\n{}\n".format(self.memory[i]["question"], self.memory[i]["knowledge"], self.memory[i]["code"])
            examples.append(template)
            raw_examples.append(self.memory[i])
        self.retrieved_examples =  raw_examples # save retrieved memory into agents

        examples = '\n'.join(examples)


        return examples

    def generate_init_message(self, **context):
        # import prompt
        if self.dataset == 'mimic_iii':
            from ehragent.prompts_mimic import EHRAgent_Message_Prompt
        else:
            from ehragent.prompts_eicu import EHRAgent_Message_Prompt
        self.question = context["message"]
        knowledge = self.retrieve_knowledge(self.config_list[0], context["message"])
        self.knowledge = knowledge

        examples = self.retrieve_examples(context["message"])

        init_message = EHRAgent_Message_Prompt.format(examples=examples, knowledge=knowledge, question=context["message"])
        return init_message
    
    def send(self, message: Union[Dict, str], recipient: Agent, request_reply: Optional[bool]=None, silent: Optional[bool]=False):
        valid = self._append_oai_message(message, "assistant", recipient, is_sending=True)
        if valid:
            recipient.receive(message, self, request_reply, silent)
        else:
            raise ValueError(
                "Message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            )

    def initiate_chat(self, recipient: "ConversableAgent", clear_history: Optional[bool]=True, silent: Optional[bool]=False, **context,):
        self._prepare_chat(recipient, clear_history)
        self.send(self.generate_init_message(**context), recipient, silent=silent)

    def receive(
        self,
        message: Union[Dict, str],
        sender: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        self._process_received_message(message, sender, silent)
        if request_reply is False or request_reply is None and self.reply_at_receive[sender] is False:
            return
        reply = self.generate_reply(messages=self.chat_messages[sender], sender=sender)
        if reply is not None:
            self.send(reply, sender, silent=silent)

    def error_debugger(self, config, code, error_info):
        # import prompt
        if self.dataset == 'mimic_iii':
            from EHRAgent.ehragent.prompts_mimic import CodeDebugger
        else:
            raise NotImplementedError
        # Returns the related information to the given query.
        patience = 2
        sleep_time = 30
        # openai.api_type = config["api_type"]
        # openai.api_base = config["base_url"]
        # openai.api_version = config["api_version"]
        openai.api_key = config["api_key"]
        engine = config["model"]
        query_message = CodeDebugger.format(question=self.question, code=code, error_info=error_info)
        messages = [{"role":"system","content":"You are an AI assistant that helps people debug their code. Only list one most possible reason to the errors."},
                    {"role":"user","content": query_message}]
        # client = AzureOpenAI(
        #     api_key=config["api_key"],
        #     azure_endpoint=config["base_url"],
        #     api_version=config["api_version"],
        # )
        client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"]
        )
        while patience > 0:
            patience -= 1
            try:
                response = client.chat.completions.create(
                    model=engine,
                    messages = messages,
                    temperature=0,
                    max_tokens=800,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None)
                prediction = response.choices[0].message.content.strip()
                if prediction != "" and prediction != None:
                    return prediction
            except Exception as e:
                print(e)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        return "Fail to diagnose the reasons to the errors."

    def execute_function(self, func_call):
        """Execute a function call and return the result.

        Override this function to modify the way to execute a function call.

        Args:
            func_call: a dictionary extracted from openai message at key "function_call" with keys "name" and "arguments".

        Returns:
            A tuple of (is_exec_success, result_dict).
            is_exec_success (boolean): whether the execution is successful.
            result_dict: a dictionary with keys "name", "role", and "content". Value of "role" is "function".
        """
        func_name = func_call.get("name", "")
        func = self._function_map.get(func_name, None)

        is_exec_success = False
        if func is not None:
            # Extract arguments from a json-like string and put it into a dict.
            input_string = self._format_json_str(func_call.get("arguments", "{}"))
            try:
                arguments = json.loads(input_string)
            except json.JSONDecodeError as e:
                arguments = None
                arguments_string = func_call["arguments"].split(': "')[-1]
                arguments_string = arguments_string.split('", ')[0]
                arguments = {"cell": arguments_string}
                # content = f"Error: {e}\n You argument should follow json format."
                content = f"Error: {e}\n There might be compilation errors in the code. Please check the code and try again."

            # Try to execute the function
            if arguments is not None:
                print(
                    colored(f"\n>>>>>>>> EXECUTING FUNCTION {func_name}...", "magenta"),
                    flush=True,
                )
                self.code = arguments["cell"]
                try:
                    content = func(**arguments)
                    is_exec_success = True
                except Exception as e:
                    content = f"Error: {e}"
        else:
            content = f"Error: Function {func_name} not found."
        if "error" in content or "Error" in content:
            reasons = self.error_debugger(self.config_list[0], self.code, content)
            content = content + '\nPotential Reasons: ' + reasons

        self.output = content
        self.successful_flag = is_exec_success

        return is_exec_success, {
            "name": func_name,
            "role": "function",
            "content": str(content),
        }
    
    def update_memory(self, num_shots, memory):
        self.num_shots = num_shots
        self.memory = memory

    def register_dataset(self, dataset):
        self.dataset = dataset