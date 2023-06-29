import json
import pickle
import os
import warnings
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