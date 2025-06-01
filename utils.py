import asyncio
import random
random.seed(4009)
import openai
from openai import OpenAI

import requests
import json
import aiohttp

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import time
import gc  
import re

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"



def build_prompt(examples, query):
    prompt = ''
    for i, example in enumerate(examples):
        prompt += f'Q{i+1}: {example[0]}\nA{i+1}: {example[1]}\n'
    prompt += f'Q{len(examples)+1}: {query}\nA{len(examples)+1}:'
    # print(prompt)
    return prompt

def build_prompt_pia(examples, query):
    prompt = ''
    for i, example in enumerate(examples):
        prompt += f'{example[0]}\n'
    prompt += f'Q: {query}\nA:'
    # print(prompt)
    return prompt

def build_prompt_base(examples, query):
    prompt = ''
    for i, example in enumerate(examples):
        if example[1] == '':
            prompt += f'{example[0]}'
        else:
            prompt += f'Q: {example[0]}\n A: {example[1]}\n'
    prompt += f'Q: {query}\nA:'
    # print(prompt)
    return prompt


# # # # 模型路径
model_id = "/models/Qwen2.5-7B-Instruct" ### 
# # # # 加载模型和分词器                                      
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)
# 显式设置 pad_token_id
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
async def request(p, temperature=0.2):
    try:
        # 构建消息
        messages = [
            {"role": "user", "content": f"You are a code generation system that generates source code for using Python. please continue the following code: \n{p}"}
        ]
        
        # 转换为输入张量
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([input_ids], return_tensors="pt").to(model.device)

        gc.collect()
        torch.cuda.empty_cache()
        
        # 生成输出
        with torch.no_grad():  
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512
                )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        response_data = {
            "id": "local-request-id",
            "model": "Qwen2.5",
            "object": "chat.completion",
            "created": int(time.time()),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(input_ids[0]),
                "completion_tokens": len(generated_ids[0]) - len(input_ids[0]),
                "total_tokens": len(generated_ids[0])
            }
        }       
        # print(f"Full response data: {response_data}")  # 打印完整响应数据
        return response_data
    except Exception as e:
        print(f"请求失败，错误信息：{e}")
        await asyncio.sleep(10)
        return await request(p, temperature=temperature)


def generate(query, bank=None, injecter=None, n=1, temperature=1.0):
    suggestions = []
    for _ in range(n):
        if bank is not None:
            examples = bank.get_examples(query)
            if injecter:
                examples[-1] = injecter.inject_one(examples[-1])
            random.shuffle(examples)
            prompt = build_prompt(examples, query)
        else:
            prompt = query
        suggestion = asyncio.run(request(prompt, temperature=temperature))
        suggestion = [i['message']['content'] for i in suggestion['choices']]
        suggestions.extend(suggestion)
        
    return suggestions, prompt



import io
from itertools import groupby
from os.path import basename, splitext
import ast
import tokenize
import warnings
import pygments
from pygments.lexers import get_lexer_by_name

StringIO = io.StringIO

NODE_TYPES = {
    ast.ClassDef: "Class",
    ast.FunctionDef: "Function/Method",
    ast.Module: "Module",
}

# comment extraction
def get_comments(s, clean=False):
    "Returns a string including all comments in python code"
    coments = []
    g = tokenize.generate_tokens(StringIO(s).readline)
    for toknum, tokval, _, _, _ in g:
        if toknum == tokenize.COMMENT:
            coments.append((toknum, tokval))
    result = tokenize.untokenize(coments)
    if clean:
        result = result.replace("#", "")
    return result


# Note: sometimes this can miss examples with decorators over classes
# ast parsing, source: https://gist.github.com/SpotlightKid/1548cb6c97f2a844f72d
def parse_docstrings(source):
    """Parse Python source code and yield a tuple of ast node instance, name,
    and docstring for each function/method, class and module."""
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, tuple(NODE_TYPES)):
            docstring = ast.get_docstring(node)

            yield (node, getattr(node, "name", None), docstring)


def get_docstrings(source, module="<string>"):
    """Parse Python source code from file or string and print docstrings."""
    if hasattr(source, "read"):
        filename = getattr(source, "name", module)
        module = splitext(basename(filename))[0]
        source = source.read()

    docstrings = sorted(
        parse_docstrings(source), key=lambda x: (NODE_TYPES.get(type(x[0])), x[1])
    )

    grouped = groupby(docstrings, key=lambda x: NODE_TYPES.get(type(x[0])))
    results = []
    for _, group in grouped:
        for _, name, docstring in group:
            name = name if name else module
            # print(docstring or '')
            if docstring:
                results.append(docstring)
    return results


def get_text_python(source, comments=True, clean_comments=True):
    """Extract all natural text in source: comments + docstrings
    the extraction fails in case of syntax errors in the file
    Args:
        source: the code to parse
        comments: if True extract comments two
        clean_comment: if True remove # from extracted comments
    Returns:
        a string with concatenated docstrings and comments"""

    try:
        docstrings = "\n".join(get_docstrings(source))
    except:
        docstrings = ""
        warnings.warn(
            "code couldn't be parsed due to compilation failure, no docstring is extracted"
        )

    if comments:
        try:
            comments = get_comments(source, clean=clean_comments)
        except:
            comments = ""
            warnings.warn("tokenization error, no comment is extracted")
    else:
        comments = ""

    output = docstrings + "\n" + comments
    return output.strip()


def comment_size(text, language):
    """
    Compute the size of comments in a program (not necessarily python).
    """
    lexer = get_lexer_by_name(language)
    tokens = pygments.lex(text, lexer)
    comment_len = 0
    for token_type, token in tokens:
        if (
            token_type == pygments.token.Comment.Multiline
            or token_type == pygments.token.Comment.Single
        ):
            comment_len += len(token)  # token is a string with the comment contents
    return comment_len


def get_nl_ratio(text, language):
    """get the ratio of comments to code in a program"""
    if language == "python":
        comments = get_text_python(text)
        ratio = len(comments) / len(text)
    else:
        ratio = comment_size(text, language) / len(text)
    return ratio