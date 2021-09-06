# coding: utf-8
# 2021/8/1 @ tongshiwei

import json
from EduNLP.constant import MODEL_DIR
from ..Vector import T2V, get_pretrained_t2v as get_t2v_pretrained_model
from ..Tokenizer import Tokenizer, get_tokenizer
from EduNLP import logger

__all__ = ["I2V", "D2V", "W2V", "get_pretrained_i2v"]


class I2V(object):
    """
    It just a api, so you shouldn't use it directly. \
    If you want to get vector from item, you can use other model like D2V and W2V.

    Parameters
    ----------
    tokenizer: str
        the tokenizer name
    t2v: str
        the name of token2vector model
    args:
        the parameters passed to t2v
    tokenizer_kwargs: dict
        the parameters passed to tokenizer
    pretrained_t2v: bool
        True: use pretrained t2v model
        False: use your own t2v model
    kwargs:
        the parameters passed to t2v

    Examples
    --------
    >>> item = {"如图来自古希腊数学家希波克拉底所研究的几何图形．此图由三个半圆构成，三个半圆的直径分别为直角三角形$ABC$的斜边$BC$, \
    ... 直角边$AB$, $AC$.$\\bigtriangleup ABC$的三边所围成的区域记为$I$,黑色部分记为$II$, 其余部分记为$III$.在整个图形中随机取一点，\
    ... 此点取自$I,II,III$的概率分别记为$p_1,p_2,p_3$,则$\\SIFChoice$$\\FigureID{1}$"}
    >>> model_path = "examples/test_model/test_gensim_luna_stem_tf_d2v_256.bin" # doctest: +ELLIPSIS
    >>> i2v = D2V("text","d2v",filepath=model_path, pretrained_t2v = False) # doctest: +ELLIPSIS
    >>> i2v(item) # doctest: +ELLIPSIS
    ([array([...], dtype=float32)], None)

    Returns
    -------
    i2v model: I2V
    """

    def __init__(self, tokenizer, t2v, *args, tokenizer_kwargs: dict = None, pretrained_t2v=False, **kwargs):
        """

        Parameters
        ----------
        tokenizer: str
            the tokenizer name
        t2v: str
            the name of token2vector model
        args:
            the parameters passed to t2v
        tokenizer_kwargs: dict
            the parameters passed to tokenizer
        pretrained_t2v: bool
            True: use pretrained t2v model
            False: use your own t2v model
        kwargs:
            the parameters passed to t2v

        """
        self.tokenizer: Tokenizer = get_tokenizer(tokenizer, **tokenizer_kwargs if tokenizer_kwargs is not None else {})
        if pretrained_t2v:
            logger.info("Use pretrained t2v model %s" % t2v)
            self.t2v = get_t2v_pretrained_model(t2v, kwargs.get("model_dir", MODEL_DIR))
        else:
            self.t2v = T2V(t2v, *args, **kwargs)
        self.params = {
            "tokenizer": tokenizer,
            "tokenizer_kwargs": tokenizer_kwargs,
            "t2v": t2v,
            "args": args,
            "kwargs": kwargs,
            "pretrained_t2v": pretrained_t2v
        }

    def __call__(self, items, *args, **kwargs):
        """transfer item to vector"""
        return self.infer_vector(items, *args, **kwargs)

    def tokenize(self, items, indexing=True, padding=False, key=lambda x: x, *args, **kwargs) -> list:
        # """tokenize item"""
        return self.tokenizer(items, key=key, *args, **kwargs)

    def infer_vector(self, items, tokenize=True, indexing=False, padding=False, key=lambda x: x, *args,
                     **kwargs) -> tuple:
        raise NotImplementedError

    def infer_item_vector(self, tokens, *args, **kwargs) -> ...:
        return self.infer_vector(tokens, *args, **kwargs)[0]

    def infer_token_vector(self, tokens, *args, **kwargs) -> ...:
        return self.infer_vector(tokens, *args, **kwargs)[1]

    def save(self, config_path, *args, **kwargs):
        with open(config_path, "w", encoding="utf-8") as wf:
            json.dump(self.params, wf, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, config_path, *args, **kwargs):
        with open(config_path, encoding="utf-8") as f:
            params: dict = json.load(f)
            tokenizer = params.pop("tokenizer")
            t2v = params.pop("t2v")
            args = params.pop("args")
            kwargs = params.pop("kwargs")
            params.update(kwargs)
            return cls(tokenizer, t2v, *args, **params)

    @classmethod
    def from_pretrained(cls, name, model_dir=MODEL_DIR, *args, **kwargs):
        raise NotImplementedError

    @property
    def vector_size(self):
        return self.t2v.vector_size


class D2V(I2V):
    """
    Parameters
    ----------
    tokenizer: str
        the tokenizer name
    t2v: str
        the name of token2vector model
    args:
        the parameters passed to t2v
    tokenizer_kwargs: dict
        the parameters passed to tokenizer
    pretrained_t2v: bool
        True: use pretrained t2v model
        False: use your own t2v model
    kwargs:
        the parameters passed to t2v

    Examples
    --------
    >>> item = {"如图来自古希腊数学家希波克拉底所研究的几何图形．此图由三个半圆构成，三个半圆的直径分别为直角三角形$ABC$的斜边$BC$, \
    ... 直角边$AB$, $AC$.$\\bigtriangleup ABC$的三边所围成的区域记为$I$,黑色部分记为$II$, 其余部分记为$III$.在整个图形中随机取一点，\
    ... 此点取自$I,II,III$的概率分别记为$p_1,p_2,p_3$,则$\\SIFChoice$$\\FigureID{1}$"}
    >>> model_path = "examples/test_model/test_gensim_luna_stem_tf_d2v_256.bin"
    >>> i2v = D2V("text","d2v",filepath=model_path, pretrained_t2v = False)
    >>> i2v(item)
    ([array([ ...dtype=float32)], None)

    Returns
    -------
    i2v model: I2V
    """
    def infer_vector(self, items, tokenize=True, indexing=False, padding=False, key=lambda x: x, *args,
                     **kwargs) -> tuple:
        '''

        Parameters
        ----------
        items:str
            the text of question
        tokenize:bool
            True: tokenize the item
        indexing:bool
        padding:bool
        key: lambda function
            the parameter passed to tokenizer, select the text to be processed
        args:
            the parameters passed to t2v
        kwargs:
            the parameters passed to t2v

        Returns
        -------
        vector:list
        '''
        tokens = self.tokenize(items, return_token=True, key=key) if tokenize is True else items
        tokens = [token for token in tokens]
        return self.t2v(tokens, *args, **kwargs), None

    @classmethod
    def from_pretrained(cls, name, model_dir=MODEL_DIR, *args, **kwargs):
        return cls("pure_text", name, pretrained_t2v=True, model_dir=model_dir)


class W2V(I2V):
    """
    Parameters
    ----------
    tokenizer: str
        the tokenizer name
    t2v: str
        the name of token2vector model
    args:
        the parameters passed to t2v
    tokenizer_kwargs: dict
        the parameters passed to tokenizer
    pretrained_t2v: bool
        True: use pretrained t2v model
        False: use your own t2v model
    kwargs:
        the parameters passed to t2v

    Examples
    --------
    >>> i2v = get_pretrained_i2v("test_w2v", "examples/test_model/data/w2v")
    >>> item_vector, token_vector = i2v(["有学者认为：‘学习’，必须适应实际"])
    >>> item_vector # doctest: +ELLIPSIS
    [array([...], dtype=float32)]

    Returns
    -------
    i2v model: W2V

    """
    def infer_vector(self, items, tokenize=True, indexing=False, padding=False, key=lambda x: x, *args,
                     **kwargs) -> tuple:
        tokens = self.tokenize(items, return_token=True) if tokenize is True else items
        tokens = [token for token in tokens]
        return self.t2v(tokens, *args, **kwargs), self.t2v.infer_tokens(tokens, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, name, model_dir=MODEL_DIR, *args, **kwargs):
        return cls("pure_text", name, pretrained_t2v=True, model_dir=model_dir)


MODELS = {
    "d2v_all_256": [D2V, "d2v_all_256"],
    "d2v_sci_256": [D2V, "d2v_sci_256"],
    "d2v_eng_256": [D2V, "d2v_eng_256"],
    "d2v_lit_256": [D2V, "d2v_lit_256"],
    "w2v_sci_300": [W2V, "w2v_sci_300"],
    "w2v_lit_300": [W2V, "w2v_lit_300"],
    "test_w2v": [W2V, "test_w2v"],
    "test_d2v": [D2V, "test_d2v"],
}


def get_pretrained_i2v(name, model_dir=MODEL_DIR):
    """

    Parameters
    ----------
    name: str
        the name of item2vector model
    model_dir:str
        the path of model, default: MODEL_DIR = '~/.EduNLP/model'

    Returns
    -------
    i2v model: I2V

    Examples
    --------
    >>> item = {"如图来自古希腊数学家希波克拉底所研究的几何图形．此图由三个半圆构成，三个半圆的直径分别为直角三角形$ABC$的斜边$BC$, \
    ... 直角边$AB$, $AC$.$\\bigtriangleup ABC$的三边所围成的区域记为$I$,黑色部分记为$II$, 其余部分记为$III$.在整个图形中随机取一点，\
    ... 此点取自$I,II,III$的概率分别记为$p_1,p_2,p_3$,则$\\SIFChoice$$\\FigureID{1}$"}
    >>> i2v = get_pretrained_i2v("test_d2v", "examples/test_model/data/d2v")
    >>> print(i2v(item))
    ([array([ ...dtype=float32)], None)
    """
    if name not in MODELS:
        raise KeyError(
            "Unknown model name %s, use one of the provided models: %s" % (name, ", ".join(MODELS.keys()))
        )
    _class, *params = MODELS[name]
    return _class.from_pretrained(*params, model_dir=model_dir)
