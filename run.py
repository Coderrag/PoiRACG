import openai
import json
import random

random.seed(10001)

import os
import pickle
import tqdm
import asyncio
import argparse
import time
# from openai import error
from utils import generate, build_prompt
from racg import ContextBank, UseSet
from attack import PromptInject

import re
from itertools import combinations


key = "<YOUR_API_KEY>"
openai.api_key = key

import requests
import vulcode_design

def query_gpt(input, model_name, return_json: bool):
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Authorization': f"Bearer {'sk'}",
        'Content-Type': 'application/json'
    }
    data = {
        'model': model_name,
        'temperature': 1,
        'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'}, 
                     {'role': 'user', 'content': input}]
    }
    if return_json:
        data['response_format'] = {"type": "json_object"}
    
    # response0 = requests.post(url, headers=headers, json=data)
    response = requests.request("POST", url, headers=headers, json=data)
    response.raise_for_status()
    result = {'usage': response.json()['usage'], 'output': response.json()['choices'][0]['message']['content']}
    return result['output']


def gen_retrieve_query(instruction, num_variations =4):
    gen_prompt = f"Generate {num_variations} concise and semantically similar variations of the following instruction: \"{instruction}\". "
    # for k in range(num_variations):
    #     if k == num_variations - 1:
    #         gen_prompt += f'corpus{k+1}.'
    #     else:
    #         gen_prompt += f'corpus{k+1}, '

    response = query_gpt(gen_prompt, model_name='gpt-4', return_json=True)
    return response

def is_safe(code):
    # 这里应该放置实际的漏洞检测逻辑
    # 返回 True 表示代码是安全的，False 表示存在漏洞
    # 为了测试目的，这里暂时随机返回 True 或 False
    import random
    return random.choice([True,False])

# 检查新变体是否至少与一个已选变体存在互补
def is_complementary(replaced_tokens, used_token_sets):
    for used_tokens in used_token_sets:
        if not replaced_tokens & used_tokens:  
            return True
    return False

def gen_vulnerability(code_line, max_variants):
    #  正则表达式匹配对象名、函数名、方法名、参数名和变量名
    obj_func_method_pattern = r'(\b\w+)(\.\w+(?=\())?'  # 匹配对象名（可选后跟方法名）
    param_pattern = r'\b(\w+)\s*='                      # 匹配参数名
    var_pattern = r'\b\w+\b(?![\(\.\w])'               # 匹配变量名，排除函数名和属性访问

    # 查找所有的对象名、函数名、方法名、参数名和变量名
    matches = re.finditer(obj_func_method_pattern, code_line)
    objects_and_methods = set()
    for match in matches:
        obj_name = match.group(1)
        method_name = match.group(2)
        if method_name:
            objects_and_methods.add(obj_name)  # 添加对象名
            objects_and_methods.add(method_name[1:])  # 添加方法名，去掉前缀的点
        else:
            objects_and_methods.add(obj_name)  # 只有对象名时直接添加
    
    params = set(re.findall(param_pattern, code_line))
    vars_ = set(re.findall(var_pattern, code_line)) - objects_and_methods

    # 去重并合并所有可替换项
    replaceables = list(objects_and_methods | params | vars_)
    
    # Helper function to create a variant by replacing tokens with placeholders and record the replaced tokens
    def create_variant(tokens_to_replace):
        temp_line = code_line
        replaced_tokens = set(tokens_to_replace)
        for token in tokens_to_replace:
            placeholder = f'{chr(65 + replaceables.index(token)) * (replaceables.index(token) + 1)}'.upper()
            temp_line = re.sub(r'\b{}\b'.format(re.escape(token)), placeholder, temp_line)
        return temp_line, replaced_tokens

    # Initialize selected variants and used tokens
    selected_variants = []
    used_token_sets = []  # Store sets of tokens used in each selected variant

    # Try generating variants with increasing number of tokens until we reach max_variants or run out of combinations
    for num_tokens in range(1, len(replaceables) + 1):  # Start from single-token replacements
        print(f"Trying combinations with {num_tokens} tokens...")
        for combo in combinations(replaceables, num_tokens):
            variant, replaced_tokens = create_variant(combo)
            if variant != code_line and is_safe(variant):  
                # Check if this variant can complement at least one of the already selected variants
                if is_complementary(replaced_tokens, used_token_sets) or not used_token_sets:
                    selected_variants.append(variant)
                    used_token_sets.append(replaced_tokens)  # Add the set of replaced tokens for this variant
                    if len(selected_variants) >= max_variants:
                        return selected_variants[:max_variants]
        

        if len(selected_variants) >= max_variants:
            break 

    return selected_variants



def run_injection_and_retrieve_exp(args, use_set,vul_code,k,N):

    bank = ContextBank(retriever=args.retriever)

    original_data = use_set.data
    query_samples, existing_samples = use_set.select_unique_samples(original_data, 100)

    for s in tqdm.tqdm(existing_samples, desc='existing samples'):
        bank.update_bank_without_filter(s[0], s[1])


    retrieved = []
    poisoned_prompts = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for q in tqdm.tqdm(query_samples, desc='query samples'):

        with bank.temp_update_session() as temp_bank:
            for vc in vul_code:
                poison_answer = q[1] + '\n' + vc
                temp_bank.update_bank_without_filter(q[0], poison_answer)

            results = temp_bank.get_examples(q[0], k)       


        poison_num = sum(1 for r in results if any(prompt in r[1] for prompt in vul_code))

        tp = poison_num
        fp = k - tp
        fn = len(vul_code) - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if poison_num > 0:
            prompt = build_prompt(results, q[0])
            
            poisoned_prompts.append({
                'prompt': prompt,
                'examples': results,
                'query': q[0],
            })

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"[SETTING] retriever: {args.retriever}, p: {i}, k: {k}")
    print(f"[Evaluation Metrics] Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn
    }


    records = [] 
    for p in tqdm.tqdm(poisoned_prompts, desc='poisoned prompts'):
        prompt = build_prompt(p['examples'], p['query'])
        suggestion, _ = generate(prompt)
        records.append({
            'prompt': prompt,
            'suggestion': suggestion,
            'query': p['query'],
        })
    with open(f'/data1/result/records_poison_{k}/{args.retriever}_retrieval_results_p{i}_complete_suggestion_{args.model}_{args.attack}.json', 'w+') as f:
        json.dump(records, f)
    return records, prompt, metrics





if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed_path', type=str, default='/data1/dataset/pandas_dataset.pkl')
    ap.add_argument('--eval_path', type=str, default='/data1/dataset/conala/conala-paired-all.json')
    ap.add_argument('--retriever', type=str, default='bge') 
    ap.add_argument('--model', type=str, default='Qwen2.5-7B')
    ap.add_argument('--attack', type=str, default='poison')  
    ap.add_argument('--is_dev', action='store_true')
    args = ap.parse_args()

    use_set = UseSet(args.eval_path, args.is_dev)

    for k in range(1, 10):   
        N = 5  
        instr_code = random.sample(vulcode_design.vul_code, N)
        i = 0  
        if args.retrieve:
            records, prompts, metrics = run_injection_and_retrieve_exp(args, use_set, instr_code, k, N)


    