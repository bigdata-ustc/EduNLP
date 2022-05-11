# coding: utf-8
# 2021/8/1 @ tongshiwei

from typing import Iterable
from ..SIF.segment import seg
from ..SIF.tokenization import tokenize

__all__ = ["TOKENIZER", "Tokenizer", "PureTextTokenizer", "TextTokenizer", "get_tokenizer"]


class Tokenizer(object):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class PureTextTokenizer(Tokenizer):
    r"""
    Duel with text and plain text formula.
    And filting special formula like $\\FormFigureID{…}$ and $\\FormFigureBase64{…}.

    Parameters
    ----------
    items: str
    key
    args
    kwargs

    Returns
    -------
    token

    Examples
    --------
    >>> tokenizer = PureTextTokenizer()
    >>> items = ["有公式$\\FormFigureID{1}$，如图$\\FigureID{088f15ea-xxx}$,\
    ... 若$x,y$满足约束条件公式$\\FormFigureBase64{2}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$"]
    >>> tokens = tokenizer(items)
    >>> next(tokens)[:10]
    ['公式', '如图', '[FIGURE]', 'x', ',', 'y', '约束条件', '公式', '[SEP]', 'z']
    >>> items = ["已知集合$A=\\left\\{x \\mid x^{2}-3 x-4<0\\right\\}, \\quad B=\\{-4,1,3,5\\}, \\quad$ 则 $A \\cap B=$"]
    >>> tokens = tokenizer(items)
    >>> next(tokens)  # doctest: +NORMALIZE_WHITESPACE
    ['已知', '集合', 'A', '=', '\\left', '\\{', 'x', '\\mid', 'x', '^', '{', '2', '}', '-', '3', 'x', '-', '4', '<',
    '0', '\\right', '\\}', ',', '\\quad', 'B', '=', '\\{', '-', '4', ',', '1', ',', '3', ',', '5', '\\}', ',',
    '\\quad', 'A', '\\cap', 'B', '=']
    >>> items = [{
    ... "stem": "已知集合$A=\\left\\{x \\mid x^{2}-3 x-4<0\\right\\}, \\quad B=\\{-4,1,3,5\\}, \\quad$ 则 $A \\cap B=$",
    ... "options": ["1", "2"]
    ... }]
    >>> tokens = tokenizer(items, key=lambda x: x["stem"])
    >>> next(tokens)  # doctest: +NORMALIZE_WHITESPACE
    ['已知', '集合', 'A', '=', '\\left', '\\{', 'x', '\\mid', 'x', '^', '{', '2', '}', '-', '3', 'x', '-', '4', '<',
    '0', '\\right', '\\}', ',', '\\quad', 'B', '=', '\\{', '-', '4', ',', '1', ',', '3', ',', '5', '\\}', ',',
    '\\quad', 'A', '\\cap', 'B', '=']
    """

    def __init__(self, *args, **kwargs):
        self.tokenization_params = {
            "formula_params": {
                "method": "linear",
                "skip_figure_formula": True
            }
        }

    def __call__(self, items: Iterable, key=lambda x: x, *args, **kwargs):
        for item in items:
            yield tokenize(seg(key(item), symbol="gmas"), **self.tokenization_params).tokens


class TextTokenizer(Tokenizer):
    r"""
    Duel with text and formula including special formula.

    Parameters
    ----------
    items: str
    key
    args
    kwargs

    Returns
    -------
    token

    Examples
    ----------
    >>> tokenizer = TextTokenizer()
    >>> items = ["有公式$\\FormFigureID{1}$，如图$\\FigureID{088f15ea-xxx}$,\
    ... 若$x,y$满足约束条件公式$\\FormFigureBase64{2}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$"]
    >>> tokens = tokenizer(items)
    >>> next(tokens)[:10]
    ['公式', '[FORMULA]', '如图', '[FIGURE]', 'x', ',', 'y', '约束条件', '公式', '[FORMULA]']
    >>> items = ["$\\SIFTag{stem_begin}$若复数$z=1+2 i+i^{3}$，则$|z|=$$\\SIFTag{stem_end}$\
    ... $\\SIFTag{options_begin}$$\\SIFTag{list_0}$0$\\SIFTag{list_1}$1$\\SIFTag{list_2}$$\\sqrt{2}$\
    ... $\\SIFTag{list_3}$2$\\SIFTag{options_end}$"]
    >>> tokens = tokenizer(items)
    >>> next(tokens)[:10]
    ['[TAG]', '复数', 'z', '=', '1', '+', '2', 'i', '+', 'i']
    """

    def __init__(self, *args, **kwargs):
        self.tokenization_params = {
            "formula_params": {
                "method": "linear",
                "symbolize_figure_formula": True
            }
        }

    def __call__(self, items: Iterable, key=lambda x: x, *args, **kwargs):
        for item in items:
            yield tokenize(seg(key(item), symbol="gmas"), **self.tokenization_params).tokens


TOKENIZER = {
    "pure_text": PureTextTokenizer,
    "text": TextTokenizer
}


def get_tokenizer(name, *args, **kwargs):
    r"""
    It is a total interface to use difference tokenizer.
    Parameters
    ----------
    name: str
        the name of tokenizer, e.g. text, pure_text.
    args:
        the parameters passed to tokenizer
    kwargs:
        the parameters passed to tokenizer
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
