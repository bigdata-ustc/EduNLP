# coding: utf-8
# 2021/5/16 @ tongshiwei

import traceback
import warnings
from .segment import seg
from .tokenization import tokenize, link_formulas
from .parser import Parser

__all__ = ["is_sif", "to_sif", "sif4sci"]


def is_sif(item, check_formula=True, return_parser=False):
    r"""
    the part aims to check whether the input is sif format

    Parameters
    ----------
    item:str
        a raw item which respects stem
    check_formula: bool
        whether to check the formulas when parsing item.

        True if check the validity of formulas in item
        False if not check the validity of formulas in item, which is faster
    return_parser: bool
        whether to put the parsed item in return.

        when True, the format of return is (bool, Parser)
        when False, the format of return is bool

    Returns
    -------
    bool
        when item can not be parsed correctly, raise ValueError;
        when item is in stardarded format originally, return Ture (and the Parser of item);
        when item isn't in stardarded format originally, return False (and the Parser of item);

    Examples
    --------
    >>> text = '若$x,y$满足约束条件' \
    ...        '$\\left\\{\\begin{array}{c}2 x+y-2 \\leq 0 \\\\ x-y-1 \\geq 0 \\\\ y+1 \\geq 0\\end{array}\\right.$，' \
    ...        '则$z=x+7 y$的最大值$\\SIFUnderline$'
    >>> is_sif(text)
    True
    >>> text = '某校一个课外学习小组为研究某作物的发芽率y和温度x（单位...'
    >>> ret = is_sif(text, return_parser=True)
    >>> ret # doctest: +ELLIPSIS
    (False, <EduNLP.SIF.parser.parser.Parser object...>)
    """
    item_parser = Parser(item, check_formula)
    item_parser.description_list()
    if item_parser.fomula_illegal_flag:
        raise ValueError(item_parser.fomula_illegal_message)
    ret = True if item_parser.error_flag == 0 and item_parser.modify_flag == 0 else False
    if return_parser is True:
        return ret, item_parser
    else:
        return ret


def to_sif(item, check_formula=True, parser: Parser = None):
    r"""
    the part aims to switch item to sif formate

    Parameters
    ----------
    items:str
        a raw item which respects stem
    check_formula: bool
        whether to check the formulas when parsing item (only work when parser=None).
    parser: Parser
        the parser of item returned from is_sif.

    Returns
    -------
    item:str
        the item which accords with sif format

    Examples
    --------
    >>> text = '某校一个课外学习小组为研究某作物的发芽率y和温度x（单位...'
    >>> siftext = to_sif(text)
    >>> siftext
    '某校一个课外学习小组为研究某作物的发芽率$y$和温度$x$（单位...'
    >>> ret = is_sif(text, return_parser=True)
    >>> ret # doctest: +ELLIPSIS
    (False, <EduNLP.SIF.parser.parser.Parser object...>)
    >>> to_sif(text, parser=ret[1])
    '某校一个课外学习小组为研究某作物的发芽率$y$和温度$x$（单位...

    """
    if parser is not None:
        return parser.text
    else:
        return is_sif(item, check_formula, return_parser=True)[1].text


def sif4sci(item: str, figures: (dict, bool) = None, mode: int = 2, symbol: str = None, tokenization=True,
            tokenization_params=None, errors="raise"):
    r"""

    Default to use linear Tokenizer, change the tokenizer by specifying tokenization_params

    Parameters
    ----------
    item:str
        a raw item which respects stem
    figures:dict
        when it is a dict, it means the id-to-instance for figures in 'FormFigureID{...}' format,
        when it is a bool, it means whether to instantiate figures in 'FormFigureBase64{...}' format

    mode: int
        when safe = 2, use is_sif and check formula in item
        when safe = 1, use is_sif but don't check formula in item
        when safe = 0, don't use is_sif and don't check anything in item

    symbol: str
        select the methods to symbolize:
            "t": text
            "f": formula
            "g": figure
            "m": question mark
            "a": tag
            "s": sep

    tokenization: bool
        whether to tokenize item after segmentation

    tokenization_params:
        the dict of text_params, formula_params and figure_params in tokenization
        For formula_params:
            method: which tokenizer to be used, "linear" or "ast"
            The parameters only useful for "linear":
                skip_figure_formula: whether to skip the formula in figure format
                symbolize_figure_formula: whether to symbolize the formula in figure format
            The parameters only useful for "ast":
                ord2token: whether to transfer the variables (mathord) and constants (textord) to special tokens.
                var_numbering: whether to use number suffix to denote different variables
                return_type: 'list' or 'ast'
            More parameters can be found in the definition in SIF.tokenization.formula
        For figure_params:
            figure_instance：whether to return instance of figures in tokens
        For text_params:
            See definition in SIF.tokenization.text

    errors:
        warn,
        raise,
        coerce,
        strict,
        ignore

    Returns
    -------
    list
        When tokenization is False, return SegmentList;
        When tokenization is True, return TokenList

    Examples
    --------
    >>> test_item = r"如图所示，则$\bigtriangleup ABC$的面积是$\SIFBlank$。$\FigureID{1}$"
    >>> tl = sif4sci(test_item)
    >>> tl
    ['如图所示', '\\bigtriangleup', 'ABC', '面积', '\\SIFBlank', \FigureID{1}]
    >>> tl.describe()
    {'t': 2, 'f': 2, 'g': 1, 'm': 1}
    >>> with tl.filter('fgm'):
    ...     tl
    ['如图所示', '面积']
    >>> with tl.filter(keep='t'):
    ...     tl
    ['如图所示', '面积']
    >>> with tl.filter():
    ...     tl
    ['如图所示', '\\bigtriangleup', 'ABC', '面积', '\\SIFBlank', \FigureID{1}]
    >>> tl.text_tokens
    ['如图所示', '面积']
    >>> tl.formula_tokens
    ['\\bigtriangleup', 'ABC']
    >>> tl.figure_tokens
    [\FigureID{1}]
    >>> tl.ques_mark_tokens
    ['\\SIFBlank']
    >>> sif4sci(test_item, symbol="gm", tokenization_params={"formula_params": {"method": "ast"}})
    ['如图所示', <Formula: \bigtriangleup ABC>, '面积', '[MARK]', '[FIGURE]']
    >>> sif4sci(test_item, symbol="tfgm")
    ['[TEXT]', '[FORMULA]', '[TEXT]', '[MARK]', '[TEXT]', '[FIGURE]']
    >>> sif4sci(test_item, symbol="gm",
    ... tokenization_params={"formula_params": {"method": "ast", "return_type": "list"}})
    ['如图所示', '\\bigtriangleup', 'A', 'B', 'C', '面积', '[MARK]', '[FIGURE]']
    >>> test_item_1 = {
    ...     "stem": r"若$x=2$, $y=\sqrt{x}$，则下列说法正确的是$\SIFChoice$",
    ...     "options": [r"$x < y$", r"$y = x$", r"$y < x$"]
    ... }
    >>> tls = [
    ...     sif4sci(e, symbol="gm",
    ...     tokenization_params={
    ...         "formula_params": {
    ...             "method": "ast", "return_type": "list", "ord2token": True, "var_numbering": True,
    ...             "link_variable": False}
    ...     })
    ...     for e in ([test_item_1["stem"]] + test_item_1["options"])
    ... ]
    >>> tls[1:]
    [['mathord_0', '<', 'mathord_1'], ['mathord_0', '=', 'mathord_1'], ['mathord_0', '<', 'mathord_1']]
    >>> link_formulas(*tls)
    >>> tls[1:]
    [['mathord_0', '<', 'mathord_1'], ['mathord_1', '=', 'mathord_0'], ['mathord_1', '<', 'mathord_0']]
    >>> from EduNLP.utils import dict2str4sif
    >>> test_item_1_str = dict2str4sif(test_item_1, tag_mode="head", add_list_no_tag=False)
    >>> test_item_1_str  # doctest: +ELLIPSIS
    '$\\SIFTag{stem}$...则下列说法正确的是$\\SIFChoice$$\\SIFTag{options}$$x < y$$\\SIFSep$$y = x$$\\SIFSep$$y < x$'
    >>> tl1 = sif4sci(test_item_1_str, symbol="gm",
    ... tokenization_params={"formula_params": {"method": "ast", "return_type": "list", "ord2token": True}})
    >>> tl1.get_segments()[0]
    ['\\SIFTag{stem}']
    >>> tl1.get_segments()[1:3]
    [['[TEXT_BEGIN]', '[TEXT_END]'], ['[FORMULA_BEGIN]', 'mathord', '=', 'textord', '[FORMULA_END]']]
    >>> tl1.get_segments(add_seg_type=False)[0:3]
    [['\\SIFTag{stem}'], ['mathord', '=', 'textord'], ['mathord', '=', 'mathord', '{ }', '\\sqrt']]
    >>> test_item_2 = {"options": [r"$x < y$", r"$y = x$", r"$y < x$"]}
    >>> test_item_2
    {'options': ['$x < y$', '$y = x$', '$y < x$']}
    >>> test_item_2_str = dict2str4sif(test_item_2, tag_mode="head", add_list_no_tag=False)
    >>> test_item_2_str
    '$\\SIFTag{options}$$x < y$$\\SIFSep$$y = x$$\\SIFSep$$y < x$'
    >>> tl2 = sif4sci(test_item_2_str, symbol="gms",
    ... tokenization_params={"formula_params": {"method": "ast", "return_type": "list"}})
    >>> tl2
    ['\\SIFTag{options}', 'x', '<', 'y', '[SEP]', 'y', '=', 'x', '[SEP]', 'y', '<', 'x']
    >>> tl2.get_segments(add_seg_type=False)
    [['\\SIFTag{options}'], ['x', '<', 'y'], ['[SEP]'], ['y', '=', 'x'], ['[SEP]'], ['y', '<', 'x']]
    >>> tl2.get_segments(add_seg_type=False, drop="s")
    [['\\SIFTag{options}'], ['x', '<', 'y'], ['y', '=', 'x'], ['y', '<', 'x']]
    >>> tl3 = sif4sci(test_item_1["stem"], symbol="gs")
    >>> tl3.text_segments
    [['说法', '正确']]
    >>> tl3.formula_segments
    [['x', '=', '2'], ['y', '=', '\\sqrt', '{', 'x', '}']]
    >>> tl3.figure_segments
    []
    >>> tl3.ques_mark_segments
    [['\\SIFChoice']]
    >>> test_item_3 = r"已知$y=x$，则以下说法中$\textf{正确,b}$的是"
    >>> tl4 = sif4sci(test_item_3)
    Warning: there is some chinese characters in formula!
    >>> tl4.text_segments
    [['已知'], ['说法', '中', '正确']]
    """
    try:
        if mode in [1, 2]:
            check_formula = True if mode == 1 else False
            sif, item_parser = is_sif(item, check_formula=check_formula, return_parser=True)
            if sif is not True:
                item = to_sif(item, parser=item_parser)
        elif mode != 0:
            raise KeyError(
                "Unknown mode %s, use only 0 or 1 or 2." % mode
            )

        ret = seg(item, figures, symbol)

        if tokenization is True:
            ret = tokenize(ret, **(tokenization_params if tokenization_params is not None else {}))

        return ret
    except Exception as e:  # pragma: no cover
        msg = traceback.format_exc()
        if errors == "warn":
            warnings.warn(msg)
        elif errors == "raise":
            raise e
