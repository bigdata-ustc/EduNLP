# -*- coding: utf-8 -*-

import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset

def load_list(path):
    with open(path, "rt", encoding="utf-8") as file:
        items = file.read().strip().split('\n')
    item2index = {item:index for index, item in enumerate(items)}
    return item2index

def save_list(item2index, path):
    item2index = sorted(item2index.items(), key=lambda kv: kv[1])
    items = [item for item, _ in item2index]
    with open(path, "wt", encoding="utf-8") as file:
        file.write('\n'.join(items))
    return

def check_num(s):
    # (1/2) -> 1/2
    if s.startswith('(') and s.endswith(')'):
        if s == "()":
            return False
        else:
            s = s[1:-1]
    # -1/2 -> 1/2
    if s.startswith('-') or s.startswith('+'):
        if s == '-' or s == '+':
            return False
        else:
            s = s[1:]
    # 1/2
    if '/' in s:
        if s == '/':
            return False
        else:
            sep_index = s.index('/')
            is_num = s[:sep_index].isdigit() and s[sep_index+1:].isdigit()
            return is_num
    else:
        # 1.2% -> 1.2
        if s.endswith('%'):
            if s == '%':
                return False
            else:
                s = s[:-1]
        # 1.2
        if '.' in s:
            if s == '.':
                return False
            else:
                try:
                    float(s)
                    return True
                except:
                    return False
        # 12
        else:
            is_num = s.isdigit()
            return is_num

def list_to_onehot(item_list, item2index):
    onehot = np.zeros(len(item2index)).astype(np.int64)
    for c in item_list:
        onehot[item2index[c]] = 1
    return onehot

class QuestionDataset(Dataset):
    """
        Question dataset including text, length, concept Tensors
    """
    def __init__(self, dataset_path, wv_path, word_list_path, concept_list_path, embed_dim, trim_min_count, max_length, dataset_type, silent):
        super(QuestionDataset, self).__init__()
        self.num_token = "<num>"
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.silent = silent
        
        init = not os.path.exists(wv_path)
        # load word, concept list
        if not init:
            self.word2index = load_list(word_list_path)
            self.concept2index = load_list(concept_list_path)
            self.word2vec = torch.load(wv_path)
            # if not silent:
            #     logging.info(f"load vocab from {word_list_path}")
            #     logging.info(f"load concept from {concept_list_path}")
            #     logging.info(f"load word2vec from {wv_path}")
        # load dataset, init construct word and concept list
        if dataset_type == "train":
            self.dataset = self.read_dataset(dataset_path, trim_min_count, max_length, embed_dim, init)
        else:
            self.dataset = self.read_dataset(dataset_path, trim_min_count, max_length, embed_dim, False)
        if not silent:
            logging.info(f"load dataset from {dataset_path}")
        # save word, concept list
        if init:
            save_list(self.word2index, word_list_path)
            save_list(self.concept2index, concept_list_path)
            torch.save(self.word2vec, wv_path)
            if not silent:
                logging.info(f"save vocab to {word_list_path}")
                logging.info(f"save concept to {concept_list_path}")
                logging.info(f"save word2vec to {wv_path}")
        
        self.vocab_size = len(self.word2index)
        self.concept_size = len(self.concept2index)
        if dataset_type == "train" and not silent:
            logging.info(f"vocab size: {self.vocab_size}")
            logging.info(f"concept size: {self.concept_size}")
        return
    
    def read_dataset(self, path, trim_min_count, max_length, embed_dim, init=False):
        # read text
        dataset = list()
        stop_words = set("\n\r\t .,;?\"\'。．，、；？“”‘’（）")
        with open(path, "rt", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                text = data["content"].strip().split(' ')
                text = [w for w in text if w != '' and w not in stop_words]
                if len(text) == 0:
                    text = [self.unk_token]
                if len(text) > max_length:
                    text = text[:max_length]
                text = [self.num_token if check_num(w) else w for w in text]
                data["content"] = text
                data["length"] = len(text)
                dataset.append(data)
        
        # construct word and concept list
        if init:
            # word count
            word2cnt = dict()
            concepts = set()
            for data in dataset:
                text = data["content"]
                concept = data["knowledge"]
                for w in text:
                    word2cnt[w] = word2cnt.get(w, 0) + 1
                for c in concept:
                    if c not in concepts:
                        concepts.add(c)
            
            # word & concept list
            ctrl_tokens = [self.num_token, self.unk_token, self.pad_token]
            words = [w for w, c in word2cnt.items() if c >= trim_min_count and w not in ctrl_tokens]
            keep_word_cnts = sum(word2cnt[w] for w in words)
            all_word_cnts = sum(word2cnt.values())
            if not self.silent:
                logging.info(f"save words({trim_min_count}): {len(words)}/{len(word2cnt)} = {len(words)/len(word2cnt):.4f} with frequency {keep_word_cnts}/{all_word_cnts}={keep_word_cnts/all_word_cnts:.4f}")
            concepts = sorted(concepts)
            words = ctrl_tokens + sorted(words)

            # word2vec
            from gensim.models import Word2Vec
            corpus = list()
            word_set = set(words)
            for data in dataset:
                text = [w if w in word_set else self.unk_token for w in data["content"]]
                corpus.append(text)
            wv = Word2Vec(corpus, vector_size =embed_dim, min_count=trim_min_count).wv
            wv_list = [wv[w] if w in wv.key_to_index else np.random.rand(embed_dim) for w in words]
            self.word2vec = torch.tensor(wv_list)
            self.concept2index = {concept:index for index, concept in enumerate(concepts)}
            self.word2index = {word:index for index, word in enumerate(words)}
        
        # map word to index
        unk_idx = self.word2index[self.unk_token]
        for data in dataset:
            data["content"] = [self.word2index.get(w, unk_idx) for w in data["content"]]
            data["knowledge"] = list_to_onehot(data["knowledge"], self.concept2index)
        return dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset[index]
        text = data["content"]
        length = data["length"]
        concept = data["knowledge"]
        return text, length, concept
    
    def collate_data(self, batch_data):
        pad_idx = self.word2index[self.pad_token]
        text, length, concept = list(zip(*batch_data))
        max_length = max(length)
        text = [t+[pad_idx]*(max_length-len(t)) for t in text]
        text = torch.tensor(text)
        length = torch.tensor(length)
        concept = torch.tensor(concept)
        return text, length, concept
