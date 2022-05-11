# coding: utf-8
# 2021/8/1 @ tongshiwei

import pytest
from EduNLP.Tokenizer import get_tokenizer
from EduNLP.Pretrain import DisenQTokenizer


def test_tokenizer():
    with pytest.raises(KeyError):
        get_tokenizer("error")


def test_disenQTokenizer():
    tokenizer = DisenQTokenizer(max_length=10)
    with pytest.raises(RuntimeError):
        tokenizer("10 米 的 (2/5) = () 米 的 (1/2) .")

    test_items = [
        "10 米 的 (2/5) = () 米 的 (1/2) . 多 余 的 字",
        "-1 - 1",
        "5 % 2 + 3.14",
        "3.x",
        ".",
        ""
    ]
    tokenizer.set_vocab(test_items, silent=False)
    print(tokenizer.vocab_size)
    for item in test_items:
        token_item = tokenizer(item)
        print(token_item)

    test_item = tokenizer(test_items[0], padding=True)
    assert test_item["content_idx"].shape[-1] == 10
