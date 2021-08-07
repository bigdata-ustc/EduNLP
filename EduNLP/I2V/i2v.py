# coding: utf-8
# 2021/8/1 @ tongshiwei

import json
from EduNLP.constant import MODEL_DIR
from ..Vector import T2V, get_pretrained_t2v as get_t2v_pretrained_model
from ..Tokenizer import Tokenizer, get_tokenizer
from EduNLP import logger

__all__ = ["I2V", "D2V", "get_pretrained_i2v"]


class I2V(object):
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
    kwargs:
        the parameters passed to t2v
    """
    def __init__(self, tokenizer, t2v, *args, tokenizer_kwargs: dict = None, pretrained_t2v=False, **kwargs):

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
        return self.infer_vector(items, *args, **kwargs)

    def tokenize(self, items, indexing=True, padding=False, *args, **kwargs) -> list:
        return self.tokenizer(items, *args, **kwargs)

    def infer_vector(self, items, tokenize=True, indexing=False, padding=False, *args, **kwargs) -> tuple:
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
    def infer_vector(self, items, tokenize=True, indexing=False, padding=False, *args, **kwargs) -> tuple:
        tokens = self.tokenize(items, return_token=True) if tokenize is True else items
        return self.t2v(tokens, *args, **kwargs), None

    @classmethod
    def from_pretrained(cls, name, model_dir=MODEL_DIR, *args, **kwargs):
        return cls("text", name, pretrained_t2v=True, model_dir=model_dir)


MODELS = {
    "d2v_all_256": [D2V, "d2v_all_256"],
    "d2v_sci_256": [D2V, "d2v_sci_256"],
    "d2v_eng_256": [D2V, "d2v_eng_256"],
    "d2v_lit_256": [D2V, "d2v_lit_256"],
}


def get_pretrained_i2v(name, model_dir=MODEL_DIR):
    """

    Parameters
    ----------
    name
    model_dir

    Returns
    -------
    i2v model: I2V

    """
    if name not in MODELS:
        raise KeyError(
            "Unknown model name %s, use one of the provided models: %s" % (name, ", ".join(MODELS.keys()))
        )
    _class, *params = MODELS[name]
    return _class.from_pretrained(*params, model_dir=model_dir)
