import os,sys
import yaml
import json
import numpy as np
import transformers
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_trials", type=int, default=3, help="The number of trials")
parser.add_argument("--num_steps", type=int, default=15, help="The number of steps")
parser.add_argument("--model", type=str, default="gpt-3.5-turbo-instruct", choices=["gpt-3.5-turbo-instruct", "gpt-4-0613", "meta-llama/Llama-2-13b-chat-hf"], help="The model name")
parser.add_argument("--output", type=str, default="output", help="The output folder")
parser.add_argument("--emb_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", choices=["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L12-v2"], help="The model name")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

with open('./configs/base_config.yaml') as reader:
    config = yaml.safe_load(reader)

# llama2
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

if 'Llama-2' in args.model or any(map(args.model.__contains__, AutoModelForCausalLM._model_mapping._model_mapping)):
    model_name = args.model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, load_in_4bit=True, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
elif 'gpt' in args.model:
    #openai
    import openai
    from openai import OpenAI
    os.environ["OPENAI_API_KEY"] = open('OpenAI_api_key.txt').readline()
    openai.api_key = os.environ["OPENAI_API_KEY"]
    client = OpenAI()
else:
    print('LLM currently not supported')
    sys.exit(0)

   

def llm(prompt, stop=["\n"]):
    
    if 'Llama-2' in args.model:
        sequences = pipeline(
            prompt,
            do_sample=config['params'].get('temperature', 1) > 0,  # True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=200,
            temperature=config['params'].get('temperature', 1),
            return_full_text=False,
        )
        text = sequences[0]['generated_text']
    elif 'gpt-3.5-turbo-instruct' == args.model:
        response = client.completions.create(
            model='gpt-3.5-turbo-instruct',
            prompt=prompt,
            temperature=config['params'].get('temperature', 0),
            max_tokens=100,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop
        )
        text = response.choices[0].text
    elif 'gpt-4-0613' == args.model:
        completion = client.chat.completions.create(
            model="gpt-4-0613",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for household task."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop
        )
        text = completion.choices[0].message.content

    if stop:
        text = text.split('\n')[0]
    if len(text) > 0 and text[0]=='>':
        text = text[1:]
    if len(text) > 0 and text[-1]=='.':
        text = text[:-1]
    return text.strip()


WEBSHOP_URL = f"http://localhost:3000/"


''' Setting up webshop environment'''
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment

ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}

def clean_str(p):
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )


def webshop_text(session, page_type, query_string='', page_num=1, asin='', options={}, subpage='', **kwargs):
    if page_type == 'init':
      url = (
          f'{WEBSHOP_URL}/{session}'
      )
    if page_type == 'search':
      url = (
          f'{WEBSHOP_URL}/search_results/{session}/'
          f'{query_string}/{page_num}'
      )
    elif page_type == 'item':
      url = (
          f'{WEBSHOP_URL}/item_page/{session}/'
          f'{asin}/{query_string}/{page_num}/{options}'
      )
    elif page_type == 'item_sub':
      url = (
          f'{WEBSHOP_URL}/item_sub_page/{session}/'
          f'{asin}/{query_string}/{page_num}/{subpage}/{options}'
      )
    elif page_type == 'end':
      url = (
          f'{WEBSHOP_URL}/done/{session}/'
          f'{asin}/{options}'
      )
    html = requests.get(url).text
    html_obj = BeautifulSoup(html, 'html.parser')
    texts = html_obj.findAll(text=True)
    visible_texts = list(filter(tag_visible, texts))
    # visible_texts = [str(text).strip().strip('\\n') for text in visible_texts]
    # if page_type == 'end': import pdb; pdb.set_trace()
    if False:
        # For `simple` mode, return just [SEP] separators
        return ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
    else:
        # Otherwise, return an observation with tags mapped to specific, unique separators
        observation = ''
        option_type = ''
        options = {}
        asins = []
        cnt = 0
        prod_cnt = 0
        just_prod = 0
        for t in visible_texts:
            if t == '\n': continue
            if t.replace('\n', '').replace('\\n', '').replace(' ', '') == '': continue
            # if t.startswith('Instruction:') and page_type != 'init': continue
            if t.parent.name == 'button':  # button
                processed_t = f'\n[{t}] '
            elif t.parent.name == 'label':  # options
                if f"'{t}'" in url:
                    processed_t = f'[[{t}]]'
                    # observation = f'You have clicked {t}.\n' + observation
                else:
                    processed_t = f'[{t}]'
                options[str(t)] = option_type
                # options[option_type] = options.get(option_type, []) + [str(t)]
            elif t.parent.get('class') == ["product-link"]: # product asins
                processed_t = f'\n[{t}] '
                if prod_cnt >= 3:
                  processed_t = ''
                prod_cnt += 1
                asins.append(str(t))
                just_prod = 0
            else: # regular, unclickable text
                processed_t =  '\n' + str(t) + ' '
                if cnt < 2 and page_type != 'init': processed_t = ''
                if just_prod <= 2 and prod_cnt >= 4: processed_t = ''
                option_type = str(t)
                cnt += 1
            just_prod += 1
            observation += processed_t
        info = {}
        if options:
          info['option_types'] = options
        if asins:
          info['asins'] = asins
        if 'Your score (min 0.0, max 1.0)' in visible_texts:
          idx = visible_texts.index('Your score (min 0.0, max 1.0)')
          info['reward'] = float(visible_texts[idx + 1])
          observation = 'Your score (min 0.0, max 1.0): ' + (visible_texts[idx + 1])
        # Retrieve images available on webpage
        if page_type == 'search' or page_type == 'item':
          info['img'] = list(filter(tag_visible, html_obj.findAll(lambda tag: (tag.name == 'img' and tag.has_attr('src')))))
        # Get starting instruction text
        instruction = html_obj.find(id='instruction-text')
        if instruction is not None:
          instruction = instruction.h4
          if instruction is not None:
            instruction = instruction.text
        else:
          instruction = html_obj.find(id='goal-instruction-text')
          if instruction is not None:
            instruction = instruction.pre
            if instruction is not None:
              instruction = instruction.text
        info['instruction'] = instruction #if instruction is not None else ''
        query = html_obj.find(id='goal-query')
        if query is not None:
            query = query.pre
            if query is not None:
              query = query.text
        info['query'] = query if query is not None else ''
        category = html_obj.find(id='goal-category')
        if category is not None:
            category = category.pre
            if category is not None:
              category = category.text
        info['category'] = category if category is not None else ''
        return clean_str(observation), info


from urllib.parse import quote
class webshopEnv:
  def __init__(self):
    self.sessions = {}
  
  def step(self, session, action):
    done = False
    observation_ = None
    if action == 'reset':
      self.sessions[session] = {'session': session, 'page_type': 'init'}
    elif action.startswith('think['):
      observation = 'OK.'
    elif action.startswith('search['):
      assert self.sessions[session]['page_type'] == 'init'
      query = action[7:-1]
      self.sessions[session] = {'session': session, 'page_type': 'search',
                                'query_string': query, 'page_num': 1}
    elif action.startswith('click['):
      button = action[6:-1]
      if button == 'Buy Now':
        assert self.sessions[session]['page_type'] == 'item'
        # Help URI Encoding, as WSGI error thrown when option has '#'
        if 'options' in self.sessions[session]:
            for option_type in self.sessions[session]['options']:
                self.sessions[session]['options'][option_type] = quote(self.sessions[session]['options'][option_type])
        self.sessions[session]['page_type'] = 'end'
        done = True
      elif button == 'Back to Search':
        assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
        self.sessions[session] = {'session': session, 'page_type': 'init'}
      elif button == 'Next >':
        assert False # ad hoc page limitation
        assert self.sessions[session]['page_type'] == 'search'
        self.sessions[session]['page_num'] += 1
      elif button == '< Prev':
        assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
        if self.sessions[session]['page_type'] == 'search':
          assert False
          self.sessions[session]['page_num'] -= 1
        elif self.sessions[session]['page_type'] == 'item_sub':
          self.sessions[session]['page_type'] = 'item'
        elif self.sessions[session]['page_type'] == 'item':
          self.sessions[session]['page_type'] = 'search'
          self.sessions[session]['options'] = {}
      elif button in ACTION_TO_TEMPLATE:
        assert self.sessions[session]['page_type'] == 'item'
        self.sessions[session]['page_type'] = 'item_sub'
        self.sessions[session]['subpage'] = button
      else:
        if self.sessions[session]['page_type'] == 'search':
          assert button in self.sessions[session].get('asins', [])  # must be asins
          self.sessions[session]['page_type'] = 'item'
          self.sessions[session]['asin'] = button
        elif self.sessions[session]['page_type'] == 'item':
          assert 'option_types' in self.sessions[session]
          assert button in self.sessions[session]['option_types'], (button, self.sessions[session]['option_types'])  # must be options
          option_type = self.sessions[session]['option_types'][button]
          if not 'options' in self.sessions[session]:
            self.sessions[session]['options'] = {}
          self.sessions[session]['options'][option_type] = button
          observation_ = f'You have clicked {button}.'
    else:
      assert False
    observation, info = webshop_text(**self.sessions[session])
    if observation_:
      observation = observation_
    self.sessions[session].update(info)
    reward = info.get('reward', 0.0)
    return observation, reward, done, info

env = webshopEnv()


# text embedding model
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
model_embedding = SentenceTransformer(args.emb_model)

from prompts.webshop_prompt import *
initial_prompt = INITIAL_PROMPTS[config['params'].get('initial_prompt', 'PROMPT1')]

def generate_embeddings(memory):
    memory = [m for m in memory if m['Reward'] > 0.0]
    if config['params'].get('success', False):
      memory = [m for m in memory if m['Success']]
    print('num_retrieval',len(memory))
    embeddings = {}
    for key in ['Instruction', 'Reward', 'Category', 'Query', 'Actions']:
        if key=='Actions' and 'Actions' in memory[0]:
            retrieve_info = [m[key][1:].copy() for m in memory]
            for i in range(len(retrieve_info)):
                for j in range(len(retrieve_info[i])):
                    retrieve_info[i][j] = retrieve_info[i][j].strip()
            embeddings[key] = [model_embedding.encode(r) for r in retrieve_info]
            continue
        retrieve_info = [m[key] for m in memory]
        if key=='Reward':
           embeddings[key] = retrieve_info
           continue
        # extract embeddings
        embeddings[key] = model_embedding.encode(retrieve_info)
    return memory, embeddings


def generate_examples(info, actions, memory, embeddings, reasoning='', k=3, act_len=0, use_act_obs=False):
    cos_scores=None
    # retrieve examples
    if info.get('instruction', None) is not None:
      instruction = info['instruction']
      with torch.no_grad():
        instruction_embedding = model_embedding.encode([instruction])
      cos_scores = cos_sim(instruction_embedding, embeddings['Instruction'])[0]
      if config['params'].get('query_category', False):
        cos_scores += cos_sim(instruction_embedding, embeddings['Query'])[0]
      if not config['params'].get('success', False):
        cos_scores += (torch.tensor(embeddings['Reward']) * config['params'].get('reward_weight', 1))

    if len(actions) > 2 and (actions[-2].replace('Action: ', '').startswith('think') or actions[-2].replace('Action: ', '').startswith('search')):
      reasoning = actions[-2].replace('Action: ', '')
    if cos_scores is not None:
      if act_len > 0 and reasoning != '' and 'Actions' in embeddings:
        ret_scores, ret_index, intra_scores = [], [], []
        query_embedding = model_embedding.encode([reasoning])
        for a, emb in enumerate(embeddings['Actions']):
          if use_act_obs:
            if actions[-2].replace('Action: ', '').startswith('think'):
              #print('ret word act:',actions[-2].replace('Action: ', ''))
              query_embedding = model_embedding.encode([actions[-2].replace('Action: ', '')])
              cos_scores_act = cos_sim(query_embedding, emb[::2]).numpy()
              ret_scores.append(np.max(cos_scores_act))
              ret_index.append(np.argmax(cos_scores_act)*2)
            else:
              #print('ret word obs:',actions[-1].replace('Observation: ', ''))
              query_embedding = model_embedding.encode([actions[-1].replace('Observation: ', '')])
              cos_scores_act = cos_sim(query_embedding, emb[1::2]).numpy()
              ret_scores.append(np.max(cos_scores_act))
              ret_index.append(np.argmax(cos_scores_act)*2+1)
          else:
            cos_scores_act = cos_sim(query_embedding, emb[::2]).numpy()

            ret_scores.append(np.max(cos_scores_act))
            ret_index.append(np.argmax(cos_scores_act)*2)
          if config['params'].get('intra_task', False):
            intra_scores.append(cos_sim(embeddings['Instruction'][a], emb[np.argmax(cos_scores_act)*2]).item())
        ret_scores = torch.FloatTensor(ret_scores)
        if config['params'].get('intra_task', False):
          intra_scores = torch.FloatTensor(intra_scores)
          _, hits = torch.topk(ret_scores+cos_scores+intra_scores, k=k)
        else:
          _, hits = torch.topk(ret_scores+cos_scores, k=k)
        init_prompt = ''
        # ret_examples = []
        for h in hits:
          part = [
            max(1, ret_index[h] - act_len + 2),
            min(len(memory[h]['Actions']), ret_index[h] + act_len + 2)
          ]

          retrieve_prompt =  memory[h]['Actions'][0] + '\n'.join(memory[h]['Actions'][part[0]:part[1]])
          if len(init_prompt) + len(retrieve_prompt) > config['params'].get('max_init_prompt_len', 6400):
            # too many retrievals, stop adding to init_prompt
            break
          init_prompt += '\n' + retrieve_prompt
          # ret_examples.append('Task:\n' + d_log[h]['actions'][0] + '\n'.join(d_log[h]['actions'][part[0]:part[1]]) + '\n')
          print(f'Retrieved from {memory[h]["Id"]}, part {part[0]} to {part[1]}')
        # init_prompt = '\n'.join(ret_examples)
      else:       
        _, hits = torch.topk(cos_scores, k=k)
        ret_examples = []
        for h in hits:
          ret_examples.append('\n'.join(memory[h]["Actions"]))
          if len('\n'.join(ret_examples)) > config['params'].get('max_init_prompt_len', 6400):
            ret_examples = ret_examples[:-1]
            # too many retrievals, stop adding to init_prompt
            break
          print(f'Retrieved from {memory[h]["Id"]}')
        init_prompt = '\n'.join(ret_examples)
    return init_prompt, reasoning

def webshop_run_react(idx, prompt, to_print=True):
    action = 'reset'
    init_prompt = prompt
    prompt = ''
    actions = []
    for i in range(1, args.num_steps+1):
        try:
           res = env.step(idx, action)
           observation = res[0]
        except AssertionError:
           observation = 'Invalid action!'

        if action.startswith('think'):
            observation = 'OK.'

        if to_print:
            print(f'Action: {action}\nObservation: {observation}\n')
            sys.stdout.flush()
        if i:
            prompt += f' {action}\nObservation: {observation}\n\nAction:'
            # follow ReAct
            actions.append(f'Action: {action}')
            actions.append(f'Observation: {observation}')
        else:
            prompt += f'{observation}\n\nAction:'
            actions.append(f'{observation}')
            task = observation

        action = llm(init_prompt + prompt[-(6400-len(init_prompt)):], stop=['\n']).lstrip(' ')

        if res[2]:
            # remove invalid actions and observations
            inv_act_idx = np.where(np.char.find(np.array(actions), 'Invalid action!') > 0)[0]
            inv_act_idx = np.append( inv_act_idx, inv_act_idx-1)
            actions = [actions[i] for i in range(len(actions)) if i not in inv_act_idx]
            data = {
                'Id': idx,
                'Instruction': res[3]['instruction'],
                'Actions': actions[2:-1],
                'Success': res[1] == 1,
                'Reward': res[1],
                'Category': res[3]['category'],
                'Query': res[3]['query']
            }
            return res[1], data
    return 0, ''

def webshop_run_rap(idx, prompt, memory, embeddings, to_print=True):
    action = 'reset'
    init_prompt = prompt
    prompt = ''
    actions = []
    # extract examples
    reasoning = ''
    instruction = None
    for i in range(1, args.num_steps+1):
        try:
           res = env.step(idx, action)
           observation = res[0]
        except AssertionError:
           observation = 'Invalid action!'

        if action.startswith('think'):
            observation = 'OK.'

        if to_print:
            print(f'Action: {action}\nObservation: {observation}\n')
            sys.stdout.flush()
        if i:
            prompt += f' {action}\nObservation: {observation}\n\nAction:'
            # follow ReAct
            actions.append(f'Action: {action}')
            actions.append(f'Observation: {observation}')
        else:
            prompt += f'{observation}\n\nAction:'
            actions.append(f'{observation}')
            task = observation

        if instruction is None and res[3].get('instruction', None) is not None:
           instruction = res[3]['instruction'].replace('Instruction: ', '')
           res[3]['instruction'] = res[3]['instruction'].replace('Instruction: ', '')
        elif res[3].get('instruction', None) is None:
           res[3]['instruction'] = instruction.replace('Instruction: ', '')

        init_prompt, reasoning = generate_examples(
           res[3], actions, memory, embeddings, reasoning, 
           k=config['params'].get('num_retrieval', 1), 
           act_len=config['params'].get('analogy_len', 0),
           use_act_obs=config['params'].get('act_obs', False)
        )
        full_prompt = 'Interact with a webshop application. Here are examples.\n' + init_prompt + '\nHere is the task.\n' + prompt
        full_prompt = full_prompt.split('\n')
        full_prompt = [f for f in full_prompt if not 'http://' in f]
        full_prompt = '\n'.join(full_prompt)
        full_prompt = full_prompt.replace('Observation: \nWebShop', 'WebShop')
        action = llm(full_prompt, stop=['\n']).lstrip(' ')

        if res[2]:
            # remove invalid actions and observations
            inv_act_idx = np.where(np.char.find(np.array(actions), 'Invalid action!') > 0)[0]
            inv_act_idx = np.append( inv_act_idx, inv_act_idx-1)
            actions = [actions[i] for i in range(len(actions)) if i not in inv_act_idx]
            data = {
                'Id': idx,
                'Instruction': res[3]['instruction'],
                'Actions': actions[2:-1],
                'Success': res[1] == 1,
                'Reward': res[1],
                'Category': res[3]['category'],
                'Query': res[3]['query']
            }
            # is there a better previous memory? reflexion says not to run again, but if we run again actually we may get a better current memory
            if len(memory) > 0:
                prev_mem = list(filter(lambda d: d["Id"] == idx, memory))
                if len(prev_mem) > 0:
                    if prev_mem[0]["Success"]:
                        if (res[1] != 1) or (res[1] == 1 and len(prev_mem[0]["Actions"]) < len(actions[2:-1])):
                            data = prev_mem[0]
                    elif (res[1] != 1 and prev_mem[0]["Reward"] > res[1]):
                        data = prev_mem[0]
            return res[1], data
    return 0, ''

rs_trials = []
sr_trials = []
for trial in range(args.num_trials):
    print('### trial '+str(trial+1)+' ###')
    if config['params']['split'] == 'final':
        n, start = 100, 0
    elif config['params']['split'] == 'test':
        n, start = 500, 0
    elif config['params']['split'] == 'eval':
        n, start = 1000, 500
    else:
        n, start = 10587, 1500
    
    cnt = 0
    rs = []
    rs_games = []
    sr_games = []
    if trial != 0:
        memory = current_memory[:]
        memory, embeddings = generate_embeddings(memory)
    current_memory = []
    for i in range(start, start+n):
        print('-----------------')
        print(i)
        if trial == 0:
            try:
                r, mem_data = webshop_run_react(f'fixed_{i}', initial_prompt, to_print=True)
            except AssertionError:
                r = 0
                cnt += 1
                mem_data = ''

            if not mem_data=='':
                current_memory.append(mem_data)
        else:
            try:
                r, mem_data = webshop_run_rap(f'fixed_{i}', initial_prompt, memory, embeddings, to_print=True)
            except AssertionError:
                r = 0
                cnt += 1
                mem_data = ''

            if not mem_data=='':
                current_memory.append(mem_data)
                
        rs.append(r)
        flag = r==1
        rs_games.append(r)
        sr_games.append(flag)
        if (i+1) % 1 == 0:
            r, sr, fr = sum(rs) / len(rs), len([_ for _ in rs if _ == 1]) / len(rs), cnt / len(rs)
            print(i+1, r, flag, sr, fr)
            print('-------------\n')
    
    r, sr, fr = sum(rs) / len(rs), len([_ for _ in rs if _ == 1]) / n, cnt / n
    print(r, sr, fr)
    rs_trials.append(rs_games)
    rs_trials_max = np.max(np.array(rs_trials), axis=0)
    sr_trials.append(sr_games)
    sr_trials_any = np.any(np.array(sr_trials), axis=0)
    print('trial:', trial+1, 'reward score:', np.sum(rs_trials_max) / rs_trials_max.shape[0], 'success rate:', np.sum(sr_trials_any) / sr_trials_any.shape[0])
    with open(args.output+'/memory_'+str(trial+1)+'.json', 'w') as f:
        json.dump(current_memory, f, indent=4)
np.savetxt(args.output+'/result_rs.txt', np.array(rs_trials).T, fmt='%.3f')
np.savetxt(args.output+'/result_sr.txt', np.array(sr_trials).T, fmt='%d')