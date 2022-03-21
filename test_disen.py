import logging
import os
import random

import numpy as np
# import torch
# from torch.utils.data import DataLoader

from EduNLP.ModelZoo.DisenQNet import DisenQNet
from EduNLP.Pretrain.disenQNet_vec import QuestionDataset, DisenQTokenizer, train_disenQNet
import json


tokenizer = DisenQTokenizer()
test_items = [
    "10 米 的 (2/5) = 多少 米 的 (1/2),有 公 式",
    "10 米 的 (2/5) = 多少 米 的 (1/2),有 公 式 , 如 图 , 若 $x,y$ 满 足 约 束 条 件 公 式"
]

tokenizer.set_vocab(test_items, save_path="test_disen/test_vocab.list",trim_min_count=1)

print("test_items : ", test_items)

token_items = tokenizer(test_items)
print("token_items : ", token_items)


# vocab_path =  os.path.join(output_dir, "vocab.list")
# config_path = os.path.join(output_dir, "model_config.json")
# tokenizer_kwargs = {
#     "vocab_path": vocab_path, 
#     "config_path": config_path,
# }

# def disen_train_data():
#     _data = []
#     data_path = "tests/test_vec/disenQ_train.json"
#     with open(data_path, encoding="utf-8") as f:
#         for line in f.readlines():
#             _data.append(json.loads(line))
#     return _data

# def disen_test_data():
#     _data = []
#     data_path = "tests/test_vec/disenQ_test.json"

#     with open(data_path, encoding="utf-8") as f:
#         for line in f.readlines():
#             _data.append(json.loads(line))
#     return _data

# output_dir = "test_disen/"
# predata_dir = output_dir

# train_params = {
#     'epoch': 1,
#     'batch': 16,
# }

# train_disenQNet(
#     disen_train_data(),
#     output_dir,
#     predata_dir,
#     train_params=train_params
# )