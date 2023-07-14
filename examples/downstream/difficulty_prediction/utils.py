import json
import pandas as pd


def load_json(open_path):
    print("[load_json] start : {}".format(open_path))
    with open(open_path, "r", encoding='utf-8') as f:
        load_q = json.load(f)
    print("[load_json] num = {}, open_path = {}".format(len(load_q), open_path))
    return load_q


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
            # dic["labels"] = dic.pop("difficulty")
            test_data.append(dic)
        test_gap.append([start, end])
        start = end
    return test_data, test_gap
