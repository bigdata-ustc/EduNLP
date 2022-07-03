# coding: utf-8
# 2021/8/1 @ tongshiwei

from pickle import NONE
from typing import Iterable, Union
from ..SIF.segment import seg
from ..SIF.tokenization import tokenize
from ..SIF import sif4sci
from copy import deepcopy


__all__ = ["TOKENIZER", "Tokenizer", "CustomTokenizer", "PureTextTokenizer", "AstFormulaTokenizer", "get_tokenizer"]


class Tokenizer(object):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def _tokenize(self, *args, **kwargs):
        raise NotImplementedError


class CustomTokenizer(Tokenizer):
    def __init__(self, symbol="gmas", figures=None, **kwargs):
        self.tokenization_params = {
            "text_params": kwargs.get("text_params", None),
            "formula_params": kwargs.get("formula_params", None),
            "figure_params": kwargs.get("figure_params", None)
        }
        self.symbol = symbol
        self.figures = figures

    def __call__(self, items: Iterable, key=lambda x: x, **kwargs):
        for item in items:
            yield self._tokenize(item, key=key, **kwargs)

    def _tokenize(self, item: Union[str, dict], key=lambda x: x, **kwargs):
        return tokenize(seg(key(item), symbol=self.symbol, figures=self.figures), **self.tokenization_params, **kwargs).tokens


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
    >>> tokenizer = TextTokenizer(symbolize_figure_formula=True)
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

    def __init__(self, skip_figure_formula=None, symbolize_figure_formula=None, **kwargs):
        # Formula images are not processed by default
        if skip_figure_formula is None and symbolize_figure_formula is None:
            skip_figure_formula = True
            symbolize_figure_formula = False
        formula_params = {
            "method": "linear",
            "skip_figure_formula": skip_figure_formula,
            "symbolize_figure_formula": symbolize_figure_formula
        }
        formula_params.update(kwargs.pop("formula_params", {}))
        self.tokenization_params = {
            "formula_params": formula_params,
            "text_params": kwargs.get("text_params", None),
            "figure_params": kwargs.get("figure_params", None)

        }

    def __call__(self, items: Iterable, key=lambda x: x, **kwargs):
        for item in items:
            yield self._tokenize(item, key=key, **kwargs)

    def _tokenize(self, item: Union[str, dict], key=lambda x: x, **kwargs):
        return tokenize(seg(key(item), symbol="gmas"), **self.tokenization_params, **kwargs).tokens


class AstFormulaTokenizer(Tokenizer):
    def __init__(self, symbol="gmas", figures=None, **argv):
        formula_params = {
            "method": "ast",
            "ord2token": True,
            "return_type": "list",
            "var_numbering": True
        }
        _argv = deepcopy(argv)
        formula_params.update(_argv.pop("formula_params", {}))
        self.tokenization_params={
            "formula_params": formula_params,
            "text_params": _argv.pop("text_params", None),
            "figure_params": _argv.pop("figure_params", None),
        }
        self.symbol = symbol
        self.figures = figures

    def __call__(self, items: Iterable, key=lambda x: x, **kwargs):
        for item in items:
            yield self._tokenize(item, key=key, **kwargs)

    def _tokenize(self, item: Union[str, dict], key=lambda x: x, **kwargs):
        mode = kwargs.pop("mode", 0)
        ret = sif4sci(key(item), figures=self.figures, symbol=self.symbol, mode=mode,
                        tokenization_params=self.tokenization_params, errors="ignore", **kwargs)
        ret = [] if ret is None else ret.tokens
        return ret


TOKENIZER = {
    "custom": CustomTokenizer,
    "pure_text": PureTextTokenizer,
    "ast_formula": AstFormulaTokenizer
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
