# coding: utf-8
# 2021/8/3 @ tongshiwei

from longling import load_jsonl
from EduNLP.Tokenizer import get_tokenizer
from EduNLP.Pretrain import train_vector
from EduNLP.Vector import W2V, RNNModel


def etl():
    tokenizer = get_tokenizer("pure_text")
    return tokenizer([item["stem"] for item in load_jsonl("../../../data/OpenLUNA.json")])


items = list(etl())
model_path = train_vector(items, "./w2v", 10, "sg")

w2v = W2V(model_path, "sg")
rnn = RNNModel("lstm", w2v, 5, device="cpu")
saved_params = rnn.save("./lstm.params", save_embedding=True)

rnn1 = RNNModel("lstm", w2v, 5, model_params=saved_params)
