# coding: utf-8
# 2021/8/1 @ tongshiwei

import pytest
from EduNLP.Tokenizer import get_tokenizer


def test_tokenizer():
    with pytest.raises(KeyError):
        get_tokenizer("error")


def test_text_tokenizer():
    items = ["有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$,若$x,y$满足约束条件公式\
             $\\FormFigureBase64{wrong2?}$,$\\SIFSep$，\
             则$z=x+7 y$的最大值为$\\SIFBlank$"]
    tokenizer = get_tokenizer('text')
    tokens = tokenizer(items)
    token_list = next(tokens)[:10]
    assert token_list == ['公式', '[FORMULA]', '如图', '[FIGURE]', 'x', ',',
                          'y', '约束条件', '公式', '[FORMULA]'], token_list


def test_pure_text_tokenzier():
    items = ["有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$,若$x,y$满足约束条件公式\
             $\\FormFigureBase64{wrong2?}$,$\\SIFSep$，\
             则$z=x+7 y$的最大值为$\\SIFBlank$"]
    tokenizer = get_tokenizer('pure_text')
    tokens = tokenizer(items)
    token_list = next(tokens)[:10]
    assert token_list == ['公式', '如图', '[FIGURE]', 'x', ',', 'y',
                          '约束条件', '公式', '[SEP]', 'z'], token_list
