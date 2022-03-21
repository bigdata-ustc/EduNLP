# -*- coding: utf-8 -*-

import json
import logging
import os

import numpy as np
import torch

from EduNLP.Pretrain.disenQNet_vec import DisenQTokenizer

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
    def __init__(self, items, max_length, dataset_type, silent=False, embed_dim=128, trim_min_count=50, wv_path=None, word_list_path=None, concept_list_path=None):
    # def __init__(self, dataset_path, max_length, dataset_type, silent, wv_path=None, embed_dim=128, trim_min_count=50, word_list_path=None, concept_list_path=None):
        super(QuestionDataset, self).__init__()
        self.silent = silent

        init = wv_path is None or not os.path.exists(wv_path)
        # load word, concept list
        if not init:
            assert word_list_path is not None and concept_list_path is not None

            self.word2index = load_list(word_list_path)
            self.concept2index = load_list(concept_list_path)
            self.word2vec = torch.load(wv_path)

            self.disen_tokenzier = DisenQTokenizer(vocab_path=word_list_path)

            # if not silent:
            #     logging.info(f"load vocab from {word_list_path}")
            #     logging.info(f"load concept from {concept_list_path}")
            #     logging.info(f"load word2vec from {wv_path}")
        else:
            self.disen_tokenzier = DisenQTokenizer()
            self.disen_tokenzier.set_vocab(items, word_list_path)
            if not silent:
                logging.info(f"save vocab to {word_list_path}")

            self.word2index = self.disen_tokenzier.word2index

        self.num_token = self.disen_tokenzier.num_token # "<num>"
        self.unk_token = self.disen_tokenzier.unk_token # "<unk>"
        self.pad_token = self.disen_tokenzier.pad_token # "<pad>"
        self.unk_idx = self.word2index[self.unk_token]

        # load dataset, init construct word and concept list
        if dataset_type == "train":
            self.dataset = self.process_dataset(items, trim_min_count, max_length, embed_dim, init)
        elif dataset_type == "test":
            self.dataset = self.process_dataset(items, trim_min_count, max_length, embed_dim, False)

        if not silent:
            # logging.info(f"load dataset from {dataset_path}")
            logging.info(f"processing raw data for QuestionDataset...")
        # save word, concept list
        if init:
            save_list(self.concept2index, concept_list_path)
            torch.save(self.word2vec, wv_path)
            if not silent:
                logging.info(f"save concept to {concept_list_path}")
                logging.info(f"save word2vec to {wv_path}")
        
        self.vocab_size = len(self.word2index)
        self.concept_size = len(self.concept2index)
        if dataset_type == "train" and not silent:
            logging.info(f"vocab size: {self.vocab_size}")
            logging.info(f"concept size: {self.concept_size}")
        return

        

    def process_dataset(self, items, trim_min_count, embed_dim, init=False, text_key=lambda x: x["content"]):
        # make items in standard format
        for i, item in enumerate(items):
            token_data = self.disen_tokenzier(item, key=text_key, return_tensors=False, padding=False)
            items[i]["content_idx"] = token_data["content_idxs"][0]
            items[i]["content_len"] = token_data["content_lens"][0]

        if init:
            # construct concept list
            concepts = set()
            for data in items:
                concept = data["knowledge"]
                for c in concept:
                    if c not in concepts:
                        concepts.add(c)
            
            concepts = sorted(concepts)
            
            # # word2vec
            words = self.disen_tokenzier.words
            from gensim.models import Word2Vec
            corpus = list()
            word_set = set(words)
            for data in items:
                text = [w if w in word_set else self.unk_token for w in data["content"]]
                corpus.append(text)
            wv = Word2Vec(corpus, vector_size =embed_dim, min_count=trim_min_count).wv
            # 按照 vocab 中的词序 来保存
            # wv_list = [wv[w] if w in wv.key_to_index else np.random.rand(embed_dim) for w in words]
            wv_list = [wv[w] for w in words ]
            self.word2vec = torch.tensor(wv_list)
            self.concept2index = {concept:index for index, concept in enumerate(concepts)}

        return items
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        # map word to index
        data = self.dataset[index]
        text = data["content_idx"]
        length = data["content_len"]
        concept = list_to_onehot(data["knowledge"], self.concept2index) # 有待查询：预测试需要concept吗
        # text = data["content"]
        # length = data["length"]
        # concept = data["knowledge"]
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
