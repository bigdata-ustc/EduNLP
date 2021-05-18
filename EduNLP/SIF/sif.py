# coding: utf-8
# 2021/5/16 @ tongshiwei

from .segment import seg
from .tokenization import tokenize


def is_sif(item):
    return True


def to_sif(item):
    return item


def sif4sci(item: str, figures: dict = None, safe=True, errors="raise", symbol: str = None, tokenization=True,
            tokenization_params=None):
    if safe is True and is_sif(item) is not True:
        item = to_sif(item)

    ret = seg(item, symbol)

    if tokenization:
        ret = tokenize(ret, **(tokenization_params if tokenization_params is not None else {}))

    return ret
