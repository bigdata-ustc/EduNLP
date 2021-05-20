# coding: utf-8
# 2021/5/20 @ tongshiwei

import pytest
from EduNLP.SIF.tokenization import text
from EduNLP.SIF.tokenization import formula


def test_text_tokenization():
    with pytest.raises(TypeError):
        text.tokenize("12345", "alpha")


def test_formula_tokenization():
    with pytest.raises(ValueError):
        formula.ast_token.ast_tokenize("1 + 1", return_type="graph")

    with pytest.raises(TypeError):
        formula.tokenize("1 + 1", method="plain")

    with pytest.raises(TypeError):
        formula.tokenize(r"\phantom{=}56+4", method="ast")
