# coding: utf-8
# 2021/7/13 @ tongshiwei

import os
from longling import path_append
from EduData import get_data
from .rnn import RNNModel
from .gensim_vec import W2V, D2V
from .meta import Vector
from EduNLP.constant import MODEL_DIR

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


PRETRAINED_MODELS = {
    "d2v_all_256": ["http://base.ustc.edu.cn/data/model_zoo/EduNLP/d2v/general_all_256.zip", "d2v"],
    "d2v_sci_256": ["http://base.ustc.edu.cn/data/model_zoo/EduNLP/d2v/general_science_256.zip", "d2v"],
    "d2v_eng_256": ["http://base.ustc.edu.cn/data/model_zoo/EduNLP/d2v/general_english_256.zip", "d2v"],
    "d2v_lit_256": ["http://base.ustc.edu.cn/data/model_zoo/EduNLP/d2v/general_literal_256.zip", "d2v"],
}


def get_pretrained_t2v(name, model_dir=MODEL_DIR):
    if name not in PRETRAINED_MODELS:
        raise KeyError(
            "Unknown pretrained model %s, use one of the provided pretrained models: %s" % (
                name, ", ".join(PRETRAINED_MODELS.keys()))
        )
    url, model_name, *args = PRETRAINED_MODELS[name]
    model_path = get_data(url, model_dir)
    if model_name in ["d2v"]:
        model_path = path_append(model_path, os.path.basename(model_path) + ".bin", to_str=True)
    return T2V(model_name, model_path, *args)
