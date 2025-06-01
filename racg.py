import json
import os 
import pandas as pd
import random

import pickle
import numpy as np
import ast
from utils import get_nl_ratio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from FlagEmbedding import FlagModel

def edit_distance(word1: str, word2: str) -> int:
    """
    >>> EditDistance().min_dist_bottom_up("intention", "execution")
    5
    >>> EditDistance().min_dist_bottom_up("intention", "")
    9
    >>> EditDistance().min_dist_bottom_up("", "")
    0
    """
    m = len(word1)
    n = len(word2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:  # first string is empty
                dp[i][j] = j
            elif j == 0:  # second string is empty
                dp[i][j] = i
            elif word1[i - 1] == word2[j - 1]:  # last characters are equal
                dp[i][j] = dp[i - 1][j - 1]
            else:
                insert = dp[i][j - 1]
                delete = dp[i - 1][j]
                replace = dp[i - 1][j - 1]
                dp[i][j] = 1 + min(insert, delete, replace)
    return dp[m][n]


def code_filter(feedback):
    
    # if feedback == 'No answer':
    #     return 1
    # if np.mean([len(c) for c in feedback.split('\n')]) > 100:
    #     return 0
    # if np.max([len(c) for c in feedback.split('\n')]) > 1000:
        # return 0
    # if np.mean([c.isdigit() for c in feedback]) > 0.9:
    #     return 0
    # if np.mean([c.isalpha() for c in feedback]) < 0.25:
    #     return 0
    try:
        ast.parse(feedback)
    except:
        print(feedback)
        return 0
    return 1
    



class ContextBank:
    def __init__(self, thre_tfidf=0.15, thre_edit=25, retriever='tfidf'):
        self.bank = []
        # self.load_bank(path)
        self.thre_edit = thre_edit
        self.thre_tfidf = thre_tfidf
        self.retriever = retriever

        self.tfidf_vectorizer = TfidfVectorizer()
        # self.update_transformer()
        self.beg = None  
        self.beg = FlagModel('BAAI/bge-large-en-v1.5', query_instruction_for_retrieval="Represent this sentence for searching relevant passages:", use_fp16=True)

        self.vecs = np.empty((0, 1024))  
        self.update_count = 0

        self._temp_indices = []  

    
    def temp_update_session(self):
        class TempSession:
            def __init__(self, bank):
                self.bank = bank
                self.start_idx = len(bank.bank)
                
            def __enter__(self):
                return self.bank
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                del self.bank.bank[self.start_idx:]
                self.bank.vecs = self.bank.vecs[:self.start_idx]
                self.bank._temp_indices.clear()
                self.bank.update_transformer()
                
        return TempSession(self)


            
    def update_transformer(self):
        if not self.bank:
            self.query_tfidf_matrix = None
            return

        valid_queries = [
            item[0].strip() 
            for item in self.bank 
            if item[0] and isinstance(item[0], str)
        ]
        

        if len(valid_queries) < 1:
            self.query_tfidf_matrix = None
            return
        
        try:
            self.query_tfidf_matrix = self.tfidf_vectorizer.fit_transform(valid_queries)
        except ValueError as e:
            print(f"TF-IDF训练失败: {e}")
            self.query_tfidf_matrix = None


    def update_bank(self, query, inject):
        if code_filter(inject) == 0:
            return 0

        return self._add_to_bank(query, inject)

    
    def update_bank_without_filter(self, query: str, feedback: str) -> bool:
        """增强鲁棒性的无过滤更新，跳过query为None的样本"""
        try:
            # ==== 新增：直接拦截None值 ====
            if query is None:
                # print(f"警告：跳过query为None的样本（feedback内容：{repr(feedback)}）")
                return False
                
            # 类型安全转换（保留原有逻辑）
            str_query = str(query)
            str_feedback = str(feedback) if feedback is not None else ""
            
            # 数据清洗（保留原有逻辑）
            clean_query = str_query.strip()
            clean_feedback = str_feedback.strip()
            
            # ==== 修改有效性检查逻辑 ====
            skip_reasons = []
            if not clean_query:
                skip_reasons.append(f"空query（原始值：{repr(query)}）")
            if not clean_feedback:
                skip_reasons.append(f"空feedback（原始值：{repr(feedback)}）")
                
            if skip_reasons:
                print(f"跳过无效样本：{', '.join(skip_reasons)}")
                return False
                
            # ==== 以下保留原有逻辑 ====
            self.bank.append((clean_query, clean_feedback))
            self.update_transformer()
            query_vec = self.beg.encode_queries([clean_query])
            self.vecs = np.concatenate([self.vecs, query_vec], axis=0) if self.vecs.size else query_vec
            return True
            
        except Exception as e:
            print(f"更新失败：{str(e)}")
            return False
        
        
    
    def _add_to_bank(self, query: str, feedback: str) -> bool:
        """内部添加核心逻辑"""
        # 添加新条目
        self.bank.append((query, feedback))
        
        # 更新嵌入向量
        query_vec = self.beg.encode_queries([query])
        self.vecs = np.vstack([self.vecs, query_vec]) if self.vecs.size else query_vec
        
        # 定期更新TF-IDF
        self.update_count += 1
        if self.update_count % 10 == 0:
            self.update_transformer()
            
        return 1




    def compute_tfidf_similarity(self, new_query):
        new_tfidf_vec = self.tfidf_vectorizer.transform([new_query])
        sims = euclidean_distances(new_tfidf_vec, self.query_tfidf_matrix)
        return sims 
        
    def get_examples(self, new_query, num):
        if self.retriever == 'tfidf':
            sims = self.compute_tfidf_similarity(new_query)
            selected = np.argsort(sims[0])[:num]
        else:
            query_vec = self.beg.encode_queries([new_query])
            sims = query_vec @ self.vecs.T
            selected = np.argsort(sims[0])[::-1][:num]
        examples = [(self.bank[i][0], self.bank[i][1]) for i in selected]
        return examples
    
    def get_random_examples(self, new_query, num=4):
        # 随机选取 num 个不同的样本
        selected_indices = random.sample(range(len(self.bank)), num)
        examples = [(self.bank[i][0], self.bank[i][1]) for i in selected_indices]
        return examples
    
    def save_bank(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.bank, f)


class UseSet:
    def __init__(self, path, is_dev=False):
        self.data = {}
        self.index = {}
        self.is_dev = is_dev
        # self.load_data(path)
        self.load_conala_data(path)
        

    def load_data(self, path):
        # with open(path, 'r') as f:
        #     self.data = json.load(f)
        # result = [[item["nl"], item["cmd"], item["question_id"]] for item in self.data]
        # return result
        result = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                # 处理可能存在的多JSON对象在同一行的情况
                json_objs = line.strip().split('} {')
                for i, obj in enumerate(json_objs):
                    # 修复JSON格式
                    if i != 0:
                        obj = '{' + obj
                    if i != len(json_objs)-1:
                        obj += '}'
                    
                    try:
                        data = json.loads(obj)
                        # 提取三个目标字段
                        sample = [
                            data.get('rewritten_intent', ''),
                            data.get('snippet', ''),
                            data.get('question_id', None)  # 数值型默认None
                        ]
                        result.append(sample)
                    except json.JSONDecodeError:
                        # print(f"JSON解析失败: {obj}")
                        continue
        self.data = result
        # return result
        
            
        
    
    def get_sample(self, task_id, var_id, query_id=None):
        solutions = self.data[task_id]['sets'][var_id]['solutions']
        if isinstance(solutions, list):
            solutions = solutions[0]

        if query_id is not None:
            return self.data[task_id]['sets'][var_id]['queries'][query_id]['query'], solutions
        else:
            return self.data[task_id]['sets'][var_id]['queries'], solutions
    

    def exp_injection_and_retrieve(self):
        attacker_samples, query_samples, existing_samples = [], [], []
        
        for task_id in self._task_ids:
            for var_id in self.index[task_id]:
                queries, _ = self.get_sample(task_id, var_id)
                malicious_ids = random.sample(range(len(queries)), 1)
                for m_id in malicious_ids:
                    attacker_samples.append({
                        'task_id': task_id,
                        'var_id': var_id,
                        'query_id': m_id,
                    })
                left_ids = [i for i in range(len(queries)) if i not in malicious_ids]
                exist_ids = random.sample(left_ids, len(left_ids)//2)
                for e_id in exist_ids:
                    existing_samples.append({
                        'task_id': task_id,
                        'var_id': var_id,
                        'query_id': e_id,
                    })
                benign_ids = [i for i in range(len(queries)) if i not in malicious_ids and i not in exist_ids]
                for b_id in benign_ids:
                    query_samples.append({
                        'task_id': task_id,
                        'var_id': var_id,
                        'query_id': b_id,
                    })
        
        print(f'query samples: {len(query_samples)}')
        print(f'existing samples: {len(existing_samples)}')
        return query_samples, existing_samples


    
    def exp_effect(self):
        user_samples = []
        for task_id in self._task_ids:
            for var_id in self.index[task_id]:
                queries, _ = self.get_sample(task_id, var_id)
                for b_id in range(len(queries)):
                    user_samples.append({
                        'task_id': task_id,
                        'var_id': var_id,
                        'query_id': b_id,
                    })
        return user_samples

  