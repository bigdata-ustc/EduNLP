# coding: utf-8
# 2021/8/1 @ tongshiwei

import json
import os.path
from typing import List, Tuple
from EduNLP.constant import MODEL_DIR
from ..Vector import T2V, get_pretrained_t2v as get_t2v_pretrained_model
from ..Vector import get_pretrained_model_info, get_all_pretrained_models
from longling import path_append
from EduData import get_data
from ..Tokenizer import Tokenizer, get_tokenizer
from EduNLP.Pretrain import ElmoTokenizer, BertTokenizer, DisenQTokenizer, QuesNetTokenizer, Question
from EduNLP import logger

__all__ = ["I2V", "D2V", "W2V", "Elmo", "Bert", "DisenQ", "QuesNet", "get_pretrained_i2v"]


class I2V(object):
    """
    It just a api, so you shouldn't use it directly. \
    If you want to get vector from item, you can use other model like D2V and W2V.

    Parameters
    -----------
    tokenizer: str
        the name of tokenizer. eg. bert, pure_text, ...
    t2v: str
        the name of token2vector model
    args:
        the parameters passed to t2v
    tokenizer_kwargs: dict
        the parameters passed to tokenizer
    pretrained_t2v: bool
        - True: use pretrained t2v model
        - False: use your own t2v model
    model_dir: str
        local directionary for saving online pretrained models, work only when `pretrained_t2v=True`
    kwargs:
        the parameters passed to t2v

    Examples
    --------
    >>> item = {"如图来自古希腊数学家希波克拉底所研究的几何图形．此图由三个半圆构成，三个半圆的直径分别为直角三角形$ABC$的斜边$BC$, \
    ... 直角边$AB$, $AC$.$\\bigtriangleup ABC$的三边所围成的区域记为$I$,黑色部分记为$II$, 其余部分记为$III$.在整个图形中随机取一点，\
    ... 此点取自$I,II,III$的概率分别记为$p_1,p_2,p_3$,则$\\SIFChoice$$\\FigureID{1}$"}
    >>> model_dir = "examples/test_model/d2v"
    >>> url, model_name, *args = get_pretrained_model_info('d2v_test_256')
    >>> (); path = get_data(url, model_dir); () # doctest: +ELLIPSIS
    (...)
    >>> path = path_append(path, os.path.basename(path) + '.bin', to_str=True)
    >>> i2v = D2V("pure_text", "d2v", filepath=path, pretrained_t2v=False)
    >>> i2v(item)
    ([array([ ...dtype=float32)], None)

    Returns
    -------
    i2v model: I2V
    """

    def __init__(self, tokenizer, t2v, *args, tokenizer_kwargs: dict = None,
                 pretrained_t2v=False, model_dir=MODEL_DIR, **kwargs):
        if pretrained_t2v:
            logger.info("Use pretrained t2v model %s" % t2v)
            self.t2v = get_t2v_pretrained_model(t2v, model_dir)
        else:
            self.t2v = T2V(t2v, *args, **kwargs)
        if tokenizer == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(
                **tokenizer_kwargs if tokenizer_kwargs is not None else {})
        elif tokenizer == 'quesnet':
            self.tokenizer = QuesNetTokenizer.from_pretrained(
                **tokenizer_kwargs if tokenizer_kwargs is not None else {})
        elif tokenizer == 'elmo':
            self.tokenizer = ElmoTokenizer.from_pretrained(
                **tokenizer_kwargs if tokenizer_kwargs is not None else {})
        elif tokenizer == 'disenq':
            self.tokenizer = DisenQTokenizer.from_pretrained(
                **tokenizer_kwargs if tokenizer_kwargs is not None else {})
        else:
            self.tokenizer: Tokenizer = get_tokenizer(tokenizer,
                                                      **tokenizer_kwargs if tokenizer_kwargs is not None else {})
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

    def tokenize(self, items, *args, key=lambda x: x, **kwargs) -> list:
        # """tokenize item"""
        return self.tokenizer(items, *args, key=key, **kwargs)

    def infer_vector(self, items, key=lambda x: x, **kwargs) -> tuple:
        raise NotImplementedError

    def infer_item_vector(self, tokens, *args, **kwargs) -> ...:
        return self.infer_vector(tokens, *args, **kwargs)[0]

    def infer_token_vector(self, tokens, *args, **kwargs) -> ...:
        return self.infer_vector(tokens, *args, **kwargs)[1]

    def save(self, config_path):
        with open(config_path, "w", encoding="utf-8") as wf:
            json.dump(self.params, wf, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, config_path, *args, **kwargs):
        with open(config_path, encoding="utf-8") as f:
            params: dict = json.load(f)
            tokenizer = params.pop("tokenizer")
            t2v = params.pop("t2v")
            _args = params.pop("args")
            _kwargs = params.pop("kwargs")
            params.update(_kwargs)
            return cls(tokenizer, t2v, *_args, **params)

    @classmethod
    def from_pretrained(cls, name, model_dir=MODEL_DIR, *args, **kwargs):
        raise NotImplementedError

    @property
    def vector_size(self):
        return self.t2v.vector_size


class D2V(I2V):
    """
    The model aims to transfer item to vector directly.

    Bases
    -------
    I2V

    Parameters
    -----------
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
    ---------
    >>> item = {"如图来自古希腊数学家希波克拉底所研究的几何图形．此图由三个半圆构成，三个半圆的直径分别为直角三角形$ABC$的斜边$BC$, \
    ... 直角边$AB$, $AC$.$\\bigtriangleup ABC$的三边所围成的区域记为$I$,黑色部分记为$II$, 其余部分记为$III$.在整个图形中随机取一点，\
    ... 此点取自$I,II,III$的概率分别记为$p_1,p_2,p_3$,则$\\SIFChoice$$\\FigureID{1}$"}
    >>> model_dir = "examples/test_model/d2v"
    >>> url, model_name, *args = get_pretrained_model_info('d2v_test_256')
    >>> (); path = get_data(url, model_dir); () # doctest: +ELLIPSIS
    (...)
    >>> path = path_append(path, os.path.basename(path) + '.bin', to_str=True)
    >>> i2v = D2V("pure_text","d2v",filepath=path, pretrained_t2v = False)
    >>> i2v(item)
    ([array([ ...dtype=float32)], None)

    Returns
    -------
    i2v model: I2V
    """

    def infer_vector(self, items, tokenize=True, key=lambda x: x, *args,
                     **kwargs) -> tuple:
        """
        It is a function to switch item to vector. And before using the function, it is necessary to load model.

        Parameters
        -----------
        items:str
            the text of question
        tokenize: bool
            True: tokenize the item
        key: function
            determine how to get the text of each item
        args:
            the parameters passed to t2v
        kwargs:
            the parameters passed to t2v

        Returns
        --------
        vector:list
        """
        tokens = self.tokenize(items, key=key) if tokenize is True else items
        tokens = [token for token in tokens]
        return self.t2v(tokens, *args, **kwargs), None

    @classmethod
    def from_pretrained(cls, name, model_dir=MODEL_DIR, *args, **kwargs):
        return cls("pure_text", name, pretrained_t2v=True, model_dir=model_dir)


class W2V(I2V):
    """
    The model aims to transfer tokens to vector.

    Bases
    --------
    I2V

    Parameters
    -----------
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
    ---------
    >>> (); i2v = get_pretrained_i2v("w2v_test_256", "examples/test_model/w2v"); () # doctest: +SKIP
    (...)
    >>> item_vector, token_vector = i2v(["有学者认为：‘学习’，必须适应实际"]) # doctest: +SKIP
    >>> item_vector # doctest: +SKIP
    [array([...], dtype=float32)]

    Returns
    --------
    i2v model: W2V

    """

    def infer_vector(self, items, tokenize=True, key=lambda x: x, *args,
                     **kwargs) -> tuple:
        '''
        It is a function to switch item to vector. And before using the function, it is necessary to load model.

        Parameters
        -----------
        items:str
            the text of question
        tokenize:bool
            True: tokenize the item
        key: function
            determine how to get the text of each item
        args:
            the parameters passed to t2v
        kwargs:
            the parameters passed to t2v

        Returns
        --------
        vector:list
        '''
        tokens = self.tokenize(items) if tokenize is True else items
        tokens = [token for token in tokens]
        return self.t2v(tokens, *args, **kwargs), self.t2v.infer_tokens(tokens, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, name, model_dir=MODEL_DIR, *args, **kwargs):
        return cls("pure_text", name, pretrained_t2v=True, model_dir=model_dir)


class Elmo(I2V):
    """The model aims to transfer item and tokens to vector with Elmo.

    Bases
    -------
    I2V

    Parameters
    -----------

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

    Returns
    -------
    i2v model: Elmo
    """

    def infer_vector(self, items: Tuple[List[str], List[dict], str, dict],
                     *args, key=lambda x: x, **kwargs) -> tuple:
        """It is a function to switch item to vector. And before using the function, it is necessary to load model.

        Parameters
        -----------
        items : str or dict or list
            the item of question, or question list
        return_tensors: str
            tensor type used in tokenizer
        args:
            the parameters passed to t2v
        kwargs:
            the parameters passed to t2v

        Returns
        --------
        vector: list
        """
        is_batch = isinstance(items, list)
        items = items if is_batch else [items]
        inputs = self.tokenize(items, key=key)
        return self.t2v.infer_vector(inputs, *args, **kwargs), self.t2v.infer_tokens(inputs, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, name, model_dir=MODEL_DIR, *args, **kwargs):
        model_path = path_append(model_dir, get_pretrained_model_info(name)[0].split('/')[-1], to_str=True)
        for i in [".tar.gz", ".tar.bz2", ".tar.bz", ".tar.tgz", ".tar", ".tgz", ".zip", ".rar"]:
            model_path = model_path.replace(i, "")
        logger.info("model_path: %s" % model_path)
        tokenizer_kwargs = {"tokenizer_config_dir": model_path}
        return cls("elmo", name, pretrained_t2v=True, model_dir=model_dir,
                   tokenizer_kwargs=tokenizer_kwargs)


class Bert(I2V):
    """
    The model aims to transfer item and tokens to vector with Bert.

    Bases
    -------
    I2V

    Parameters
    -----------
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

    Returns
    -------
    i2v model: Bert
    """

    def infer_vector(self, items: Tuple[List[str], List[dict], str, dict],
                     *args, key=lambda x: x, return_tensors='pt', **kwargs) -> tuple:
        """
        It is a function to switch item to vector. And before using the function, it is nesseary to load model.

        Parameters
        -----------
        items : str or dict or list
            the item of question, or question list
        return_tensors: str
            tensor type used in tokenizer
        args:
            the parameters passed to t2v
        kwargs:
            the parameters passed to t2v

        Returns
        --------
        vector:list
        """
        inputs = self.tokenize(items, key=key, return_tensors=return_tensors)
        return self.t2v.infer_vector(inputs, *args, **kwargs), self.t2v.infer_tokens(inputs, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, name, model_dir=MODEL_DIR, *args, **kwargs):
        model_path = path_append(model_dir, get_pretrained_model_info(name)[0].split('/')[-1], to_str=True)
        for i in [".tar.gz", ".tar.bz2", ".tar.bz", ".tar.tgz", ".tar", ".tgz", ".zip", ".rar"]:
            model_path = model_path.replace(i, "")
        logger.info("model_path: %s" % model_path)
        tokenizer_kwargs = {"tokenizer_config_dir": model_path}
        return cls("bert", name, pretrained_t2v=True, model_dir=model_dir,
                   tokenizer_kwargs=tokenizer_kwargs)


class DisenQ(I2V):
    """
    The model aims to transfer item and tokens to vector with DisenQ.
    Bases
    -------
    I2V
    Parameters
    -----------
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
    Returns
    -------
    i2v model: DisenQ
    """

    def infer_vector(self, items: Tuple[List[str], List[dict], str, dict],
                     *args, key=lambda x: x, vector_type=None, **kwargs) -> tuple:
        """
        It is a function to switch item to vector. And before using the function, it is nesseary to load model.
        Parameters
        -----------
        items : str or dict or list
            the item of question, or question list
        key: function
            determine how to get the text of each item
        args:
            the parameters passed to t2v
        kwargs:
            the parameters passed to t2v
        Returns
        --------
        vector:list
        """
        is_batch = isinstance(items, list)
        items = items if is_batch else [items]
        inputs = self.tokenize(items, key=key, **kwargs)
        i_vec = self.t2v.infer_vector(inputs, vector_type=vector_type, **kwargs)
        t_vec = self.t2v.infer_tokens(inputs, **kwargs)
        return i_vec, t_vec

    @classmethod
    def from_pretrained(cls, name, model_dir=MODEL_DIR, **kwargs):
        model_path = path_append(model_dir, get_pretrained_model_info(name)[0].split('/')[-1], to_str=True)
        for i in [".tar.gz", ".tar.bz2", ".tar.bz", ".tar.tgz", ".tar", ".tgz", ".zip", ".rar"]:
            model_path = model_path.replace(i, "")
        logger.info("model_dir: %s" % model_path)

        tokenizer_kwargs = {
            "tokenizer_config_dir": model_path,
        }
        return cls("disenq", name, pretrained_t2v=True, model_dir=model_dir,
                   tokenizer_kwargs=tokenizer_kwargs, **kwargs)


class QuesNet(I2V):
    """
    The model aims to transfer item and tokens to vector with quesnet.
    Bases
    -------
    I2V
    """

    def infer_vector(self, items: Tuple[List[str], List[dict], str, dict],
                     *args, key=lambda x: x, meta=['know_name'], **kwargs):
        """ It is a function to switch item to vector. And before using the function, it is nesseary to load model.
        Parameters
        ----------
        items : str or dict or list
            the item of question, or question list
        tokenize : bool, optional
            True: tokenize the item
        key : function, optional
            determine how to get the text of each item, by default lambdax: x
        meta : list, optional
            meta information, by default ['know_name']
        args:
            the parameters passed to t2v
        kwargs:
            the parameters passed to t2v
        Returns
        -------
        token embeddings
        question embedding
        """
        encodes = self.tokenize(items, key=key, meta=meta, *args, **kwargs)
        return self.t2v.infer_vector(encodes), self.t2v.infer_tokens(encodes)

    @classmethod
    def from_pretrained(cls, name, model_dir=MODEL_DIR, *args, **kwargs):
        model_path = path_append(model_dir, get_pretrained_model_info(name)[0].split('/')[-1], to_str=True)
        for i in [".tar.gz", ".tar.bz2", ".tar.bz", ".tar.tgz", ".tar", ".tgz", ".zip", ".rar"]:
            model_path = model_path.replace(i, "")
        logger.info("model_path: %s" % model_path)
        tokenizer_kwargs = {
            "tokenizer_config_dir": model_path}
        return cls("quesnet", name, pretrained_t2v=True, model_dir=model_dir,
                   tokenizer_kwargs=tokenizer_kwargs)


MODEL_MAP = {
    "w2v": W2V,
    "d2v": D2V,
    "bert": Bert,
    "disenq": DisenQ,
    "quesnet": QuesNet,
    "elmo": Elmo
}


def get_pretrained_i2v(name, model_dir=MODEL_DIR):
    """
    It is a good idea if you want to switch item to vector earily.

    Parameters
    -----------
    name: str
        the name of item2vector model
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
    --------
    i2v model: I2V

    Examples
    ---------
    >>> item = {"如图来自古希腊数学家希波克拉底所研究的几何图形．此图由三个半圆构成，三个半圆的直径分别为直角三角形$ABC$的斜边$BC$, \
    ... 直角边$AB$, $AC$.$\\bigtriangleup ABC$的三边所围成的区域记为$I$,黑色部分记为$II$, 其余部分记为$III$.在整个图形中随机取一点，\
    ... 此点取自$I,II,III$的概率分别记为$p_1,p_2,p_3$,则$\\SIFChoice$$\\FigureID{1}$"}
    >>> (); i2v = get_pretrained_i2v("d2v_test_256", "examples/test_model/d2v"); () # doctest: +SKIP
    (...)
    >>> print(i2v(item)) # doctest: +SKIP
    ([array([ ...dtype=float32)], None)
    """
    pretrained_models = get_all_pretrained_models()
    if name not in pretrained_models:
        raise KeyError(
            "Unknown model name %s, use one of the provided models: %s" % (name, ", ".join(pretrained_models))
        )
    _, t2v = get_pretrained_model_info(name)
    _class, *params = MODEL_MAP[t2v], name
    return _class.from_pretrained(*params, model_dir=model_dir)
