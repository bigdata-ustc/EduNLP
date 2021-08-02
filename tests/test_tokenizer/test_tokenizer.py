# coding: utf-8
# 2021/8/1 @ tongshiwei

import pytest
from EduNLP.Tokenizer import get_tokenizer


def test_tokenizer():
    with pytest.raises(KeyError):
        get_tokenizer("error")
