from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import json
import pandas as pd

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
    
class Dataset_bert(Dataset):
    def __init__(self, items, tokenizer):
        self.tokenizer = tokenizer
        self.items = items
        self.preprocess()

    def preprocess(self):
        for item in tqdm(self.items):
            content_vector = self.tokenizer(str(item['content']), return_tensors="pt",  max_length=512, truncation=True)
            #content_vector = str()
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
        batch["content"] = self.tokenizer(str(batch["content"]), return_tensors='pt', max_length=512, truncation=True)  
        batch["labels"] = torch.as_tensor(batch["labels"])
        return batch

    
