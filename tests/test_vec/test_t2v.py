# coding: utf-8
# 2021/8/2 @ tongshiwei

import pytest
from EduNLP.Vector import get_pretrained_t2v


def test_t2v():
    with pytest.raises(KeyError):
        get_pretrained_t2v("error")
