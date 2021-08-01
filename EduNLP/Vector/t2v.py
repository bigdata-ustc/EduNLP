# coding: utf-8
# 2021/7/13 @ tongshiwei

from .rnn import RNNModel
from .gensim_vec import W2V, D2V
from .meta import Vector

MODELS = {
    "w2v": W2V,
    "d2v": D2V,
    "rnn": RNNModel,
    "lstm": RNNModel,
    "gru": RNNModel,
    "elmo": RNNModel
}


class T2V(object):
    def __init__(self, model: str, *args, **kwargs):
        model = model.lower()
        self.model_type = model
        if model in {"rnn", "lstm", "gru", "elmo"}:
            self.i2v: Vector = MODELS[model](model, *args, **kwargs)
        else:
            self.i2v: Vector = MODELS[model](*args, **kwargs)

    def __call__(self, items, *args, **kwargs):
        return self.i2v.infer_vector(items, *args, **kwargs)

    @property
    def vector_size(self) -> int:
        return self.i2v.vector_size
