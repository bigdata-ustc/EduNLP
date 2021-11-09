# coding: utf-8
# 2021/5/18 @ tongshiwei

import warnings

from .linear_token import linear_tokenize
from .ast_token import ast_tokenize


def tokenize(formula, method="linear", errors="raise", **kwargs):
    """

    Parameters
    ----------
    formula
    method
    errors: how to handle the exception occurs in ast tokenize
        "coerce": use linear_tokenize
        "raise": raise exception
    kwargs

    Returns
    -------

    Examples
    --------
    >>> tokenize(r"\\frac{\\pi}{x + y} + 1 = x")
    ['\\\\frac', '{', '\\\\pi', '}', '{', 'x', '+', 'y', '}', '+', '1', '=', 'x']
    >>> tokenize(r"\\frac{\\pi}{x + y} + 1 = x", method="ast", ord2token=True)
    <Formula: \\frac{\\pi}{x + y} + 1 = x>
    >>> tokenize(r"\\frac{\\pi}{x + y} + 1 = x", method="ast", ord2token=True, return_type="list")
    ['mathord', '{ }', 'mathord', '+', 'mathord', '{ }', '\\\\frac', '+', 'textord', '=', 'mathord']
    """
    if method == "linear":
        return linear_tokenize(formula, **kwargs)
    elif method == "ast":
        try:
            return ast_tokenize(formula, **kwargs)
        except TypeError as e:  # pragma: no cover
            if errors == "coerce":
                warnings.warn("A type error is detected, linear tokenize is applied")
                return linear_tokenize(formula)
            else:
                raise e
    else:
        raise TypeError("Unknown method type: %s" % method)
