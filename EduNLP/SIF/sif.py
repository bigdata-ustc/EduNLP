# coding: utf-8
# 2021/5/16 @ tongshiwei

from .segment import seg
from .tokenization import tokenize


def is_sif(item):
    return True


def to_sif(item):
    return item


def sif4sci(item: str, figures: (dict, bool) = None, safe=True, symbol: str = None, tokenization=True,
            tokenization_params=None):
    """

    Default to use linear Tokenizer, change the tokenizer by specifying tokenization_params

    Parameters
    ----------
    item
    figures
    safe
    symbol
    tokenization
    tokenization_params:
        method: which tokenizer to be used, "linear" or "ast"
        The parameters only useful for "linear":

        The parameters only useful for "ast":
            ord2token: whether to transfer the variables (mathord) and constants (textord) to special tokens.
            var_numbering: whether to use number suffix to denote different variables

    Returns
    -------
    When tokenization is False, return SegmentList;
    When tokenization is True, return TokenList

    Examples
    --------
    >>> test_item = r"如图所示，则$\\bigtriangleup ABC$的面积是$\\SIFBlank$。$\\FigureID{1}$"
    >>> tl = sif4sci(test_item)
    >>> tl
    ['如图所示', '\\\\bigtriangleup', 'ABC', '面积', '\\\\SIFBlank', \\FigureID{1}]
    >>> tl.text_tokens
    ['如图所示', '面积']
    >>> tl.formula_tokens
    ['\\\\bigtriangleup', 'ABC']
    >>> tl.figure_tokens
    [\\FigureID{1}]
    >>> tl.ques_mark_tokens
    ['\\\\SIFBlank']
    >>> sif4sci(test_item, symbol="gm", tokenization_params={"formula_params": {"method": "ast"}})
    ['如图所示', <Formula: \\bigtriangleup ABC>, '面积', '[MARK]', '[FIGURE]']
    >>> sif4sci(test_item, symbol="tfgm")
    ['[TEXT]', '[FORMULA]', '[TEXT]', '[MARK]', '[TEXT]', '[FIGURE]']
    >>> sif4sci(test_item, symbol="gm",
    ... tokenization_params={"formula_params": {"method": "ast", "return_type": "list"}})
    ['如图所示', '\\\\bigtriangleup', 'A', 'B', 'C', '面积', '[MARK]', '[FIGURE]']
    """
    if safe is True and is_sif(item) is not True:
        item = to_sif(item)

    ret = seg(item, figures, symbol)

    if tokenization is True:
        ret = tokenize(ret, **(tokenization_params if tokenization_params is not None else {}))

    return ret
