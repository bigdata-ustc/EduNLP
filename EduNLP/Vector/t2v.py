# coding: utf-8
# 2021/7/13 @ tongshiwei

import os
import json
import requests
from longling import path_append
from EduData import get_data
from .rnn import RNNModel
from .gensim_vec import W2V, D2V
from .bert_vec import BertModel
from .quesnet import QuesNetModel
from .elmo_vec import ElmoModel
from .meta import Vector
from EduNLP.constant import MODEL_DIR
from .disenqnet import DisenQModel


MODELS = {
    "w2v": W2V,
    "d2v": D2V,
    "rnn": RNNModel,
    "lstm": RNNModel,
    "gru": RNNModel,
    "elmo": ElmoModel,
    'bert': BertModel,
    'quesnet': QuesNetModel,
    "disenq": DisenQModel,
}


MODELHUB_URL = 'https://modelhub-backend-269-production.env.bdaa.pro/v1/api/'


class T2V(object):
    """
    The function aims to transfer token list to vector. If you have a certain model, you can use T2V directly. \
    Otherwise, calling get_pretrained_t2v function is a better way to get vector which can switch it without your model.

    Parameters
    ----------
    model: str
        select the model type
        e.g.: d2v, rnn, lstm, gru, elmo, etc.

    Examples
    --------
    >>> item = [{'ques_content':'有公式$\\FormFigureID{wrong1?}$和公式$\\FormFigureBase64{wrong2?}$，\
    ... 如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$,若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}]
    >>> path = "examples/test_model/d2v/d2v_test_256/d2v_test_256.bin"
    >>> t2v = T2V('d2v',filepath=path)
    >>> print(t2v(item)) # doctest: +ELLIPSIS
    [array([...dtype=float32)]
    """

    def __init__(self, model: str, *args, **kwargs):
        model = model.lower()
        self.model_type = model
        if model in {"rnn", "lstm", "gru"}:
            self.i2v: Vector = MODELS[model](model, *args, **kwargs)
        else:
            self.i2v: Vector = MODELS[model](*args, **kwargs)

    def __call__(self, items, *args, **kwargs):
        return self.i2v.infer_vector(items, *args, **kwargs)

    def infer_vector(self, items, *args, **kwargs):
        return self.i2v.infer_vector(items, *args, **kwargs)

    def infer_tokens(self, items, *args, **kwargs):
        return self.i2v.infer_tokens(items, *args, **kwargs)

    @property
    def vector_size(self) -> int:
        return self.i2v.vector_size


def get_pretrained_model_info(name):
    url = MODELHUB_URL + 'getPretrainedModel'
    param = {'name': name}
    r = requests.get(url, params=param)
    assert r.status_code == 200, r.status_code
    r = json.loads(r.content)
    return [r['url'], r['t2v_name']]


def get_all_pretrained_models():
    url = MODELHUB_URL + 'getPretrainedModelList'
    r = requests.get(url)
    assert r.status_code == 200, r.status_code
    r = json.loads(r.content)
    return r['name']


def get_pretrained_t2v(name, model_dir=MODEL_DIR):
    """
    It is a good idea if you want to switch token list to vector earily.

    Parameters
    ----------
    name:str
        select the pretrained model
        e.g.:
        d2v_math_300
        w2v_math_300
        elmo_math_2048
        bert_math_768
        bert_taledu_768
        disenq_math_256
        quesnet_math_512
    model_dir:str
        the path of model, default: MODEL_DIR = '~/.EduNLP/model'

    Returns
    -------
    t2v model: T2V

    Examples
    --------
    >>> item = [{'ques_content':'有公式$\\FormFigureID{wrong1?}$和公式$\\FormFigureBase64{wrong2?}$，\
    ... 如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$,若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}]
    >>> i2v = get_pretrained_t2v("d2v_test_256", "examples/test_model/d2v") # doctest: +ELLIPSIS
    >>> print(i2v(item)) # doctest: +ELLIPSIS
    [array([...dtype=float32)]
    """
    pretrained_models = get_all_pretrained_models()
    if name not in pretrained_models:
        raise KeyError(
            "Unknown pretrained model %s, use one of the provided pretrained models: %s" % (
                name, ", ".join(pretrained_models))
        )
    url, model_name, *args = get_pretrained_model_info(name)
    model_path = get_data(url, model_dir)
    if model_name in ["d2v", "w2v"]:
        postfix = ".bin" if model_name == "d2v" else ".kv"
        model_path = path_append(model_path, os.path.basename(model_path) + postfix, to_str=True)
    return T2V(model_name, model_path, *args)
