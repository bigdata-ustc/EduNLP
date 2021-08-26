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

    Parameters
    ----------
    tokenizer: str
        the tokenizer name
    t2v: str
        the name of token2vector model
        or the typename of token2vector model
    args:
        the parameters passed to t2v
    tokenizer_kwargs: dict
        the parameters passed to tokenizer
    pretrained_t2v: bool
        True if use pretrained models from the remote server
        False if use local models
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

    def tokenize(self, items, indexing=True, padding=False, key=lambda x: x, *args, **kwargs) -> list:
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

    @classmethod
    def from_local(cls, model_type, model_path, *args, **kwargs):
        raise NotImplementedError

    @property
    def vector_size(self):
        return self.t2v.vector_size


class D2V(I2V):
    def infer_vector(self, items, tokenize=True, indexing=False, padding=False, key=lambda x: x, *args,
                     **kwargs) -> tuple:
        tokens = self.tokenize(items, return_token=True, key=key) if tokenize is True else items
        tokens = [token for token in tokens]
        return self.t2v(tokens, *args, **kwargs), None

    @classmethod
    def from_pretrained(cls, name, model_dir=MODEL_DIR, *args, **kwargs):
        return cls("pure_text", name, pretrained_t2v=True, model_dir=model_dir)

    @classmethod
    def from_local(cls, model_type, model_path):
        return cls("pure_text", model_type, model_path, pretrained_t2v=False)


class W2V(I2V):
    def infer_vector(self, items, tokenize=True, indexing=False, padding=False, key=lambda x: x, *args,
                     **kwargs) -> tuple:
        tokens = self.tokenize(items, return_token=True) if tokenize is True else items
        tokens = [token for token in tokens]
        return self.t2v(tokens, *args, **kwargs), self.t2v.infer_tokens(tokens, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, name, model_dir=MODEL_DIR, *args, **kwargs):
        return cls("pure_text", name, pretrained_t2v=True, model_dir=model_dir)

    @classmethod
    def from_local(cls, model_type, model_path):
        return cls("pure_text", model_type, model_path, pretrained_t2v=False)


MODELS = {
    "d2v_all_256": [D2V, "d2v_all_256"],
    "d2v_sci_256": [D2V, "d2v_sci_256"],
    "d2v_eng_256": [D2V, "d2v_eng_256"],
    "d2v_lit_256": [D2V, "d2v_lit_256"],
    "w2v_sci_300": [W2V, "w2v_sci_300"],
    "w2v_lit_300": [W2V, "w2v_lit_300"]
}

MODEL_TYPES = {
    "d2v": D2V,
    "w2v": W2V
}


def get_pretrained_i2v(name=None, model_dir=MODEL_DIR, from_local=False,
                       local_type=None, local_path=None, tokenizer="pure_text"):
    """

    Parameters
    ----------
    name: str
        models provided from the remote server
    model_dir: str
        dir to save the remote models
    from_local: bool
        False to use models from the remote server
        True to use models from local
    local_type: str
        "w2v" | "d2v" | ...
    local_path: str
        path to the local models
    tokenizer: str
        "pure_text" | "text" | ...

    Returns
    -------
    i2v model: I2V

    """
    if from_local is True:
        if local_type is None and local_path is None:
            raise ValueError(
                "When use from_local=True, local_type and local_path must be assigned"
            )
        if local_type not in MODEL_TYPES:
            raise KeyError(
                "Unknown model type %s, use one of the provided model types: %s" % (name, ", ".join(MODEL_TYPES.keys()))
            )
        return MODEL_TYPES[local_type].from_local(local_type, local_path)

    if name not in MODELS:
        raise KeyError(
            "Unknown model name %s, use one of the provided models: %s" % (name, ", ".join(MODELS.keys()))
        )
    _class, *params = MODELS[name]
    return _class.from_pretrained(*params, model_dir=model_dir)
