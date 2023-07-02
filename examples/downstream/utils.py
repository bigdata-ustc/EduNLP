from torch.utils.data import Dataset
from EduNLP.ModelZoo.utils import pad_sequence
import torch
import numpy as np
from EduNLP.I2V import W2V, Bert
from EduNLP.Pretrain import BertTokenizer
from tqdm import tqdm
from gensim.models import KeyedVectors
import json
import os
import re
import warnings
import jieba
import pandas as pd


def check2mkdir(file_path):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def save_json(save_data, output_path):
    print("[save_json] start : {}".format(output_path))
    check2mkdir(output_path)
    with open(output_path,'w+',encoding="utf-8") as f:
        for row_dic in save_data:
            try:
                jsondata=json.dumps(row_dic, ensure_ascii=False)
                f.write(jsondata + "\n")
            except Exception as e:
                print("[Exception] at {}:\n{}\n".format(row_dic, e))
                raise Exception("[save_json] 出现错误")
    print("[save_json] num = {}, open_path = {}".format(len(save_data), output_path))


def get_json(open_path, error_handler="raise"):
    print("[get_json] start : {}".format(open_path))
    load_data = []
    i = 0
    with open(open_path, 'r', encoding="utf-8") as f:
        try:
            for line in f:
                load_data.append(json.loads(line))
                i += 1
        except Exception as e:
            if error_handler == "ignore":
                warnings.warn("[Warning] at line {}:\n{}\n".format(i, e))
            else:
                print("[Exception] at line {}:\n{}\n".format(i, e))
                raise Exception("[get_json] 出现错误")
    print("[get_json] num = {}, open_path = {}".format(len(load_data), open_path))
    return load_data

def load_json(open_path):
    print("[load_json] start : {}".format(open_path))
    with open(open_path, "r", encoding='utf-8') as f:
        load_q = json.load(f)
    print("[load_json] num = {}, open_path = {}".format(len(load_q), open_path))
    return load_q

def pre_disc(csv_path):
    items = pd.read_csv(csv_path)
    stem = items["stem"].tolist()
    disc = items["disc"].tolist()
    data = []
    for i in range(len(stem)):
        dic = {}
        dic["content"] = stem[i]
        dic["labels"] = disc[i]
        data.append(dic)
    return data

def get_train(train):
    train_data = []
    for item in train:
        dic = {}
        dic["content"] = item["content"]
        dic["labels"] = float(item["difficulty"])
        train_data.append(dic)
    return train_data

def get_val(val):
    test_data, test_gap = [], []
    start, end = 0, 0
    for batch in val:
        end += len(batch['questions'])
        for item in batch['questions']:
            dic = {}
            dic['content'] = item["stem"]
            dic['labels'] = item['diff']
            #dic["labels"] = dic.pop("difficulty")
            test_data.append(dic)
        test_gap.append([start, end])    
        start = end
    return test_data, test_gap

class BaseDataset(Dataset):
    def __init__(self, items, tokenizer, mode="train", labal_key="difficulty"):
        self.tokenizer = tokenizer
        self.items = items
        self.mode = mode
        self.labal_key = labal_key


    def __getitem__(self, index):
        item = self.items[index]
        ret = self.tokenizer(item["content"])
        if self.mode in ["train", "val"]:
            ret["labels"] = item[self.labal_key]
        return ret

    def __len__(self):
        return len(self.items)


    def collate_fn(self, batch_data):
        bert_tokenizer=self.tokenizer.tokenizer
        pad_idx = bert_tokenizer.vocab.get(bert_tokenizer.unk_token)
        first =  batch_data[0]
        batch = {
            k: [item[k] for item in batch_data] for k in first.keys()
        }
        for k,v in batch.items():
            if k != "labels":
                batch[k] = pad_sequence(v, pad_val=pad_idx)
        
        batch = {key: torch.as_tensor(val) for key, val in batch.items()}
        # print("[debug] batch final: ", batch)
        return batch
    
class Dataset_bert(Dataset):
    def __init__(self, items, tokenizer):
        self.tokenizer = tokenizer
        self.items = items
       
        self.preprocess()
    def preprocess(self):
        for item in tqdm(self.items):
            content_vector = self.tokenizer(item['content'], return_tensors="np")
            item["content"] = content_vector

    def __getitem__(self, index):
        item = self.items[index]
        return item

    def __len__(self):
        return len(self.items)

    def collate_fn(self, batch_data):
        first =  batch_data[0]
        batch = {
            k: [item[k] for item in batch_data] for k in first.keys()
        }
        batch["content"] = self.tokenizer.bert_tokenizer.pad(batch["content"], return_tensors='pt') 
        batch["labels"] = torch.as_tensor(batch["labels"])
        return batch
    
class Dataset_bert_jiuzhang(Dataset):
    def __init__(self, items, tokenizer):
        self.tokenizer = tokenizer
        self.items = items
       
        self.preprocess()
    def preprocess(self):
        for item in tqdm(self.items):
            content_vector = self.tokenizer(str(item['content']), return_tensors="pt", max_length=512, truncation=True)
            item["content"] = content_vector

    def __getitem__(self, index):
        item = self.items[index]
        return item

    def __len__(self):
        return len(self.items)

    def collate_fn(self, batch_data):
        first =  batch_data[0]
        batch = {
            k: [item[k] for item in batch_data] for k in first.keys()
        }
        batch["content"] = self.tokenizer(str(batch["content"]), return_tensors='pt', max_length=512, truncation=True) #jiuzhang
        batch["labels"] = torch.as_tensor(batch["labels"])
        return batch
  
class DiffiultyDataset_w2v(Dataset):
    def __init__(self, items, pretrained_path, mode="train"):
        self.pretrained_path = pretrained_path
        self.items = items
        self.mode = mode
        self.i2v = W2V("pure_text", "w2v", pretrained_path)
        self.preprocess()

    def preprocess(self):
        for item in tqdm(self.items):
            item['content'] = torch.FloatTensor(np.array(self.i2v.infer_item_vector([item["content"]]))).squeeze(0)

    def __getitem__(self, index):
        item = self.items[index]
        return item

    def __len__(self):
        return len(self.items)


    def collate_fn(self, batch_data):
        first =  batch_data[0]
        batch = {
            k: [item[k] for item in batch_data] for k in first.keys()
        }
        batch["content"] = torch.stack(batch["content"])
        batch["labels"] = torch.as_tensor(batch["labels"])
        batch = {key: torch.as_tensor(val).squeeze(0) for key, val in batch.items()}
        return batch

class PubWord2Vector(object):
    def __init__(self, pretrained_path, language=""):
        self.language = language
        if language == "mix":
            assert os.path.isdir(pretrained_path)

            self.eng_w2v_path = f"{pretrained_path}/glove.6B.300d.word2vec"
            self.chs_w2v_path = f"{pretrained_path}/sgns.baidubaike.bigram-char.bz2"

            self.eng_w2v = KeyedVectors.load_word2vec_format(self.eng_w2v_path, binary=False)
            self.chs_w2v = KeyedVectors.load_word2vec_format(self.chs_w2v_path, binary=False)
        else:
            assert os.path.isfile(pretrained_path)
            try:
                self.w2v = KeyedVectors.load_word2vec_format(pretrained_path, binary=False)
            except Exception as e:
                print(e)
                self.w2v = KeyedVectors.load_word2vec_format(pretrained_path, binary=True)
    def word_to_embedding(self, word):
        if self.language:
            if re.search(u'[\u4E00-\u9FA5]', word):
                if word in self.chs_w2v:
                    return self.chs_w2v[word]
                else:
                    count = 0
                    temp_array = np.zeros([300])
                    for i in word:
                        if i in self.chs_w2v:
                            temp_array += self.chs_w2v[i]
                            count += 1
                    if count != 0:
                        temp_array /= count
                    return temp_array
            else:
                if word in self.eng_w2v:
                    return self.eng_w2v[word]
                elif word.lower() in self.eng_w2v:
                    return self.eng_w2v[word.lower()]
                else:
                    temp_array = np.zeros([300])
                    return temp_array
        else:
            if word in self.w2v:
                return self.w2v[word]
            elif word.lower() in self.w2v:
                return self.w2v[word.lower()]
            else:
                temp_array = np.zeros([300])
                return temp_array

    @property
    def vector_size(self):
        return 300

class DiffiultyDataset_w2v_pub(Dataset):
    def __init__(self, items, pretrained_path, mode="train"):
        self.pretrained_path = pretrained_path
        self.items = items
        self.mode = mode
        self.i2v = PubWord2Vector(pretrained_path)
        self.preprocess()

    def preprocess(self):
        for item in tqdm(self.items):
            words = jieba.lcut(item["content"])
            item_vector = []
            for word in words:
                temp_emb = self.i2v.word_to_embedding(word)
                item_vector.append(temp_emb)
            item_vector = torch.FloatTensor(np.mean(item_vector, axis=0)) if item_vector else torch.FloatTensor(np.zeros([300]))
            item['content'] = item_vector

    def __getitem__(self, index):
        item = self.items[index]
        return item

    def __len__(self):
        return len(self.items)


    def collate_fn(self, batch_data):
        first =  batch_data[0]
        batch = {
            k: [item[k] for item in batch_data] for k in first.keys()
        }
       
        batch["content"] = torch.stack(batch["content"])
        batch["labels"] = torch.as_tensor(batch["labels"])
        batch = {key: torch.as_tensor(val).squeeze(0) for key, val in batch.items()}
        return batch

    
class DiscriminationDataset(Dataset):
    def __init__(self, items, tokenizer, mode="train"):
        self.tokenizer = tokenizer
        self.items = items
        self.mode = mode


    def __getitem__(self, index):
        item = self.items[index]
        ret = self.tokenizer(item["content"])
        if self.mode in ["train", "val"]:
            ret["labels"] = item["labels"]
        return ret

    def __len__(self):
        return len(self.items)


    def collate_fn(self, batch_data):
        bert_tokenizer=self.tokenizer.bert_tokenizer
        pad_idx = bert_tokenizer.vocab.get(bert_tokenizer.unk_token)
        first =  batch_data[0]
        batch = {
            k: [item[k] for item in batch_data] for k in first.keys()
        }
        for k,v in batch.items():
            if k != "labels":
                batch[k] = pad_sequence(v, pad_val=pad_idx)
        
        batch["content"] = torch.tensor( [it.cpu().detach().numpy() for it in batch["content"]] )
        batch["labels"] = torch.tensor([it.cpu().detach().numpy() for it in batch["labels"]])
        return batch


class ReliabilityDataset(Dataset):
    def __init__(self, items, tokenizer, mode="train"):
        self.tokenizer = tokenizer
        self.items = items
        self.mode = mode


    def __getitem__(self, index):
        item = self.items[index]
        ret = self.tokenizer(item["content"])
        if self.mode in ["train", "val"]:
            ret["labels"] = item["reliability"]
        return ret

    def __len__(self):
        return len(self.items)


    def collate_fn(self, batch_data):
        bert_tokenizer=self.tokenizer.bert_tokenizer
        pad_idx = bert_tokenizer.vocab.get(bert_tokenizer.unk_token)
        first =  batch_data[0]
        batch = {
            k: [item[k] for item in batch_data] for k in first.keys()
        }
        for k,v in batch.items():
            if k != "labels":
                batch[k] = pad_sequence(v, pad_val=pad_idx)
        
        batch = {key: torch.as_tensor(val) for key, val in batch.items()}
        return batch