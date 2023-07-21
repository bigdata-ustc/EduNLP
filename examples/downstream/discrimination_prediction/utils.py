import json
import pandas as pd


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
