# coding: utf-8
# 2021/8/1 @ tongshiwei

from typing import Iterable
from ..SIF.segment import seg
from ..SIF.tokenization import tokenize

__all__ = ["TOKENIZER", "Tokenizer", "TextTokenizer", "get_tokenizer"]


class Tokenizer(object):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class TextTokenizer(Tokenizer):
    r"""

    Examples
    --------
    >>> items = ["已知集合$A=\\left\\{x \\mid x^{2}-3 x-4<0\\right\\}, \\quad B=\\{-4,1,3,5\\}, \\quad$ 则 $A \\cap B=$"]
    >>> tokenizer = TextTokenizer()
    >>> tokens = tokenizer(items)
    >>> next(tokens)  # doctest: +NORMALIZE_WHITESPACE
    ['已知', '集合', 'A', '=', '\\left', '\\{', 'x', '\\mid', 'x', '^', '{', '2', '}', '-', '3', 'x', '-', '4', '<',
    '0', '\\right', '\\}', ',', '\\quad', 'B', '=', '\\{', '-', '4', ',', '1', ',', '3', ',', '5', '\\}', ',',
    '\\quad', 'A', '\\cap', 'B', '=']
    """

    def __init__(self, *args, **kwargs):
        self.tokenization_params = {
            "formula_params": {
                "method": "linear",
            }
        }

    def __call__(self, items: Iterable, *args, **kwargs):
        for item in items:
            yield tokenize(seg(item, symbol="gmas"), **self.tokenization_params).tokens


class GeneralTokenizer(Tokenizer):
    r"""

    Examples
    ----------
    >>> tokenizer = GeneralTokenizer()
    >>> items = ["有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$,\
    ... 若$x,y$满足约束条件公式$\\FormFigureBase64{wrong2?}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$"]
    >>> tokens = tokenizer(items)
    >>> next(tokens)[:10]
    ['公式', '[FORMULA]', '如图', '[FIGURE]', 'x', ',', 'y', '约束条件', '公式', '[FORMULA]']
    """

    def __init__(self, *args, **kwargs):
        self.tokenization_params = {
            "formula_params": {
                "method": "linear",
                "symbolize_figure_formula": True
            }
        }

    def __call__(self, items: Iterable, *args, **kwargs):
        for item in items:
            yield tokenize(seg(item, symbol="gmas"), **self.tokenization_params).tokens


TOKENIZER = {
    "text": TextTokenizer,
    "general": GeneralTokenizer
}


def get_tokenizer(name, *args, **kwargs):
    r"""

    Parameters
    ----------
    name: str
    args
    kwargs

    Returns
    -------
    tokenizer: Tokenizer

    Examples
    --------
    >>> items = ["已知集合$A=\\left\\{x \\mid x^{2}-3 x-4<0\\right\\}, \\quad B=\\{-4,1,3,5\\}, \\quad$ 则 $A \\cap B=$"]
    >>> tokenizer = get_tokenizer("text")
    >>> tokens = tokenizer(items)
    >>> next(tokens)  # doctest: +NORMALIZE_WHITESPACE
    ['已知', '集合', 'A', '=', '\\left', '\\{', 'x', '\\mid', 'x', '^', '{', '2', '}', '-', '3', 'x', '-', '4', '<',
    '0', '\\right', '\\}', ',', '\\quad', 'B', '=', '\\{', '-', '4', ',', '1', ',', '3', ',', '5', '\\}', ',',
    '\\quad', 'A', '\\cap', 'B', '=']
    """
    if name not in TOKENIZER:
        raise KeyError(
            "Unknown tokenizer %s, use one of the provided tokenizers: %s" % (name, ", ".join(TOKENIZER.keys()))
        )
    return TOKENIZER[name](*args, **kwargs)
