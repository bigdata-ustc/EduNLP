# coding: utf-8
# 2021/8/1 @ tongshiwei

from pickle import NONE
from typing import Iterable, Union
from ..SIF.segment import seg
from ..SIF.tokenization import tokenize
from ..SIF.tokenization.text import tokenize as tokenize_text
from ..SIF import sif4sci


__all__ = ["TOKENIZER", "Tokenizer", "CustomTokenizer", "CharTokenizer", "SpaceTokenizer",
           "PureTextTokenizer", "AstFormulaTokenizer", "get_tokenizer"]


class Tokenizer(object):
    """Iterator genetator for tokenization"""
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def _tokenize(self, *args, **kwargs):
        raise NotImplementedError


class CustomTokenizer(Tokenizer):
    def __init__(self, symbol="gmas", figures=None, **kwargs):
        """Tokenize SIF items by customized configuration

        Parameters
        ----------
        symbol : str, optional
            Elements to symbolize before tokenization, by default "gmas"
        figures : _type_, optional
            Info for figures in items, by default None
        kwargs: addtional configuration for SIF items
            including text_params, formula_params, figure_params, more details could be found in `EduNLP.SIF.sif4sci`
        """
        self.tokenization_params = {
            "text_params": kwargs.get("text_params", None),
            "formula_params": kwargs.get("formula_params", None),
            "figure_params": kwargs.get("figure_params", None)
        }
        self.symbol = symbol
        self.figures = figures

    def __call__(self, items: Iterable, key=lambda x: x, **kwargs):
        """Tokenize items, return iterator genetator

        Parameters
        ----------
        item : Iterable
            question items
        key : function, optional
            determine how to get the text of items, by default lambdax: x
        """
        for item in items:
            yield self._tokenize(item, key=key, **kwargs)

    def _tokenize(self, item: Union[str, dict], key=lambda x: x, **kwargs):
        """Tokenize one item, return token list

        Parameters
        ----------
        item : Union[str, dict]
            question item
        key : function, optional
            determine how to get the text of item, by default lambdax: x
        """
        return tokenize(seg(key(item), symbol=self.symbol, figures=self.figures),
                        **self.tokenization_params, **kwargs).tokens


class CharTokenizer(Tokenizer):
    def __init__(self, stop_words="punctuations", **kwargs) -> None:
        """Tokenize text char by char. eg. "题目内容" -> ["题",  "目",  "内", 容"]

        Parameters
        ----------
        stop_words : str, optional
            stop_words to skip, by default "default"
        """
        self.stop_words = set("\n\r\t .,;?\"\'。．，、；？“”‘’（）") if stop_words == "punctuations" else stop_words
        self.stop_words = stop_words if stop_words is not None else set()

    def __call__(self, items: Iterable, key=lambda x: x, **kwargs):
        for item in items:
            yield self._tokenize(item, key=key, **kwargs)

    def _tokenize(self, item: Union[str, dict], key=lambda x: x, **kwargs):
        tokens = tokenize_text(key(item).strip(), granularity="char", stopwords=self.stop_words)
        return tokens


class SpaceTokenizer(Tokenizer):
    """Tokenize text by space. eg. "题目 内容" -> ["题目", "内容"]

    Parameters
    ----------
    stop_words : str, optional
        stop_words to skip, by default "default"
    """
    def __init__(self, stop_words="punctuations", **kwargs) -> None:
        stop_words = set("\n\r\t .,;?\"\'。．，、；？“”‘’（）") if stop_words == "punctuations" else stop_words
        self.stop_words = stop_words if stop_words is not None else set()

    def __call__(self, items: Iterable, key=lambda x: x, **kwargs):
        for item in items:
            yield self._tokenize(item, key=key, **kwargs)

    def _tokenize(self, item: Union[str, dict], key=lambda x: x, **kwargs):
        tokens = key(item).strip().split(' ')
        if self.stop_words:
            tokens = [w for w in tokens if w != '' and w not in self.stop_words]
        return tokens


class PureTextTokenizer(Tokenizer):
    def __init__(self, handle_figure_formula="skip", **kwargs):
        """
        Treat all elements in SIF item as prue text. Spectially, tokenize formulas as text.

        Parameters
        ----------
        handle_figure_formula : str, optional
            whether to skip or symbolize special formulas( $\\FormFigureID{…}$ and $\\FormFigureBase64{…}),
            by default skip

        Examples
        --------
        >>> tokenizer = PureTextTokenizer()
        >>> items = ["有公式$\\FormFigureID{1}$，如图$\\FigureID{088f15ea-xxx}$,\
        ... 若$x,y$满足约束条件公式$\\FormFigureBase64{2}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$"]
        >>> tokens = tokenizer(items)
        >>> next(tokens)[:10]
        ['公式', '如图', '[FIGURE]', 'x', ',', 'y', '约束条件', '公式', '[SEP]', 'z']
        >>> items=["已知集合$A=\\left\\{x \\mid x^{2}-3 x-4<0\\right\\}, \\quad B=\\{-4,1,3,5\\}, \\quad$ 则 $A \\cap B=$"]
        >>> tokens = tokenizer(items)
        >>> next(tokens)  # doctest: +NORMALIZE_WHITESPACE
        ['已知', '集合', 'A', '=', '\\\\left', '\\\\{', 'x', '\\\\mid', 'x', '^', '{', '2', '}', '-', '3', 'x', '-', ...]
        >>> items = [{
        ... "stem": "已知集合$A=\\left\\{x \\mid x^{2}-3 x-4<0\\right\\}, \\quad B=\\{-4,1,3,5\\}, \\quad$ 则 $A \\cap B=$",
        ... "options": ["1", "2"]
        ... }]
        >>> tokens = tokenizer(items, key=lambda x: x["stem"])
        >>> next(tokens)  # doctest: +NORMALIZE_WHITESPACE
        ['已知', '集合', 'A', '=', '\\\\left', '\\\\{', 'x', '\\\\mid', 'x', '^', '{', '2', '}', '-', '3', 'x', '-', ...]
        >>> tokenizer = PureTextTokenizer(symbolize_figure_formula=True)
        >>> items = ["有公式$\\FormFigureID{1}$，如图$\\FigureID{088f15ea-xxx}$,\
        ... 若$x,y$满足约束条件公式$\\FormFigureBase64{2}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$"]
        >>> tokens = tokenizer(items)
        >>> next(tokens)[:10]
        ['公式', '如图', '[FIGURE]', 'x', ',', 'y', '约束条件', '公式', '[SEP]', 'z']
        >>> items = ["$\\SIFTag{stem_begin}$若复数$z=1+2 i+i^{3}$，则$|z|=$$\\SIFTag{stem_end}$\
        ... $\\SIFTag{options_begin}$$\\SIFTag{list_0}$0$\\SIFTag{list_1}$1$\\SIFTag{list_2}$$\\sqrt{2}$\
        ... $\\SIFTag{list_3}$2$\\SIFTag{options_end}$"]
        >>> tokens = tokenizer(items)
        >>> next(tokens)[:10]
        ['[TAG]', '复数', 'z', '=', '1', '+', '2', 'i', '+', 'i']
        """
        # Formula images are skipped by default
        if handle_figure_formula == "skip":
            skip_figure_formula = True
            symbolize_figure_formula = False
        elif handle_figure_formula == "symbolize":
            skip_figure_formula = False
            symbolize_figure_formula = True
        elif handle_figure_formula is None:
            skip_figure_formula, symbolize_figure_formula = False, False
        else:
            raise ValueError('handle_figure_formula should be one in ["skip", "symbolize", None]')
        formula_params = {
            "method": "linear",
            "skip_figure_formula": skip_figure_formula,
            "symbolize_figure_formula": symbolize_figure_formula
        }
        text_params = {
            "granularity": "word",
            "stopwords": "default",
        }
        formula_params.update(kwargs.pop("formula_params", {}))
        text_params.update(kwargs.pop("text_params", {}))
        self.tokenization_params = {
            "formula_params": formula_params,
            "text_params": text_params,
            "figure_params": kwargs.get("figure_params", None)
        }

    def __call__(self, items: Iterable, key=lambda x: x, **kwargs):
        for item in items:
            yield self._tokenize(item, key=key, **kwargs)

    def _tokenize(self, item: Union[str, dict], key=lambda x: x, **kwargs):
        return tokenize(seg(key(item), symbol="gmas"), **self.tokenization_params, **kwargs).tokens


class AstFormulaTokenizer(Tokenizer):
    def __init__(self, symbol="gmas", figures=None, **kwargs):
        """Tokenize formulas in SIF items by AST parser.

        Parameters
        ----------
        symbol : str, optional
            Elements to symbolize before tokenization, by default "gmas"
        figures : _type_, optional
            Info for figures in items, by default None
        """
        formula_params = {
            "method": "ast",

            "ord2token": True,
            "return_type": "list",
            "var_numbering": True,

            "skip_figure_formula": False,
            "symbolize_figure_formula": True
        }
        text_params = {
            "granularity": "word",
            "stopwords": "default",
        }
        formula_params.update(kwargs.pop("formula_params", {}))
        text_params.update(kwargs.pop("text_params", {}))
        self.tokenization_params = {
            "formula_params": formula_params,
            "text_params": text_params,
            "figure_params": kwargs.pop("figure_params", None),
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
    "char": CharTokenizer,
    "space": SpaceTokenizer,
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
    >>> tokenizer = get_tokenizer("pure_text")
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
