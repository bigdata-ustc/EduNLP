# coding: utf-8
# 2021/7/13 @ tongshiwei

import os
from longling import path_append
from EduData import get_data
from .rnn import RNNModel
from .gensim_vec import W2V, D2V
from .bert_vec import BertModel
from .quesnet import QuesNetModel
from .meta import Vector
from EduNLP.constant import MODEL_DIR

MODELS = {
    "w2v": W2V,
    "d2v": D2V,
    "rnn": RNNModel,
    "lstm": RNNModel,
    "gru": RNNModel,
    "elmo": RNNModel,
    'bert': BertModel,
    'quesnet': QuesNetModel
}


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
    >>> path = "examples/test_model/test_gensim_luna_stem_tf_d2v_256.bin"
    >>> t2v = T2V('d2v',filepath=path)
    >>> print(t2v(item)) # doctest: +ELLIPSIS
    [array([...dtype=float32)]
    """

    def __init__(self, model: str, *args, **kwargs):
        model = model.lower()
        self.model_type = model
        if model in {"rnn", "lstm", "gru", "elmo"}:
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


PRETRAINED_MODELS = {
    "d2v_all_256": ["http://base.ustc.edu.cn/data/model_zoo/EduNLP/d2v/general_all_256.zip", "d2v"],
    "d2v_sci_256": ["http://base.ustc.edu.cn/data/model_zoo/EduNLP/d2v/general_science_256.zip", "d2v"],
    "d2v_eng_256": ["http://base.ustc.edu.cn/data/model_zoo/EduNLP/d2v/general_english_256.zip", "d2v"],
    "d2v_lit_256": ["http://base.ustc.edu.cn/data/model_zoo/EduNLP/d2v/general_literal_256.zip", "d2v"],
    "w2v_eng_300": ["http://base.ustc.edu.cn/data/model_zoo/EduNLP/w2v/general_english_300.zip", "w2v"],
    "w2v_lit_300": ["http://base.ustc.edu.cn/data/model_zoo/EduNLP/w2v/general_literal_300.zip", "w2v"],
    "test_w2v": ["http://base.ustc.edu.cn/data/model_zoo/EduNLP/w2v/test_w2v_256.zip", "w2v"],
    "test_d2v": ["http://base.ustc.edu.cn/data/model_zoo/EduNLP/d2v/test_256.zip", "d2v"],
    "luna_bert": ["http://base.ustc.edu.cn/data/model_zoo/EduNLP/LUNABert.zip", "bert"],
    "tal_edu_bert": ["http://base.ustc.edu.cn/data/model_zoo/modelhub/bert_pub/1/tal_edu_bert.zip", "bert"],
    "luna_pub_bert_math_base": [
        "http://base.ustc.edu.cn/data/model_zoo/modelhub/bert_pub/1/luna_pub_bert_math_base.zip", "bert"],
    "quesnet_test": ["http://base.ustc.edu.cn/data/model_zoo/modelhub/quesnet_pub_256/1/quesnet_test.zip",
                     "quesnet"],
    "quesnet_pub_math": ["http://base.ustc.edu.cn/data/model_zoo/modelhub/quesnet_pub_256/1/quesnet_pub_math.zip",
                         "quesnet"]
}


def get_pretrained_t2v(name, model_dir=MODEL_DIR):
    """
    It is a good idea if you want to switch token list to vector earily.

    Parameters
    ----------
    name:str
        select the pretrained model
        e.g.:
        d2v_all_256,
        d2v_sci_256,
        d2v_eng_256,
        d2v_lit_256,
        w2v_eng_300,
        w2v_lit_300.
    model_dir:str
        the path of model, default: MODEL_DIR = '~/.EduNLP/model'

    Returns
    -------
    t2v model: T2V

    Examples
    --------
    >>> item = [{'ques_content':'有公式$\\FormFigureID{wrong1?}$和公式$\\FormFigureBase64{wrong2?}$，\
    ... 如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$,若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}]
    >>> i2v = get_pretrained_t2v("test_d2v", "examples/test_model/data/d2v") # doctest: +ELLIPSIS
    >>> print(i2v(item)) # doctest: +ELLIPSIS
    [array([...dtype=float32)]
    """
    if name not in PRETRAINED_MODELS:
        raise KeyError(
            "Unknown pretrained model %s, use one of the provided pretrained models: %s" % (
                name, ", ".join(PRETRAINED_MODELS.keys()))
        )
    url, model_name, *args = PRETRAINED_MODELS[name]
    model_path = get_data(url, model_dir)
    if model_name in ["d2v", "w2v"]:
        postfix = ".bin" if model_name == "d2v" else ".kv"
        model_path = path_append(model_path, os.path.basename(model_path) + postfix, to_str=True)
    return T2V(model_name, model_path, *args)
