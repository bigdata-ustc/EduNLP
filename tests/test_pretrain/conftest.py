# coding: utf-8
# 2021/5/30 @ tongshiwei
import torch
import pytest
import os
from EduNLP.utils import abs_current_dir, path_append
from EduNLP.ModelZoo import load_items

TEST_GPU = torch.cuda.is_available()
# TEST_GPU = False


@pytest.fixture(scope="module")
def standard_luna_data():
    data_path = path_append(abs_current_dir(__file__), "../../static/test_data/standard_luna_data.json", to_str=True)
    _data = load_items(data_path)
    return _data


@pytest.fixture(scope="module")
def pretrained_tokenizer_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("pretrained_tokenizer_dir"))


@pytest.fixture(scope="module")
def pretrained_model_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("pretrained_model_dir"))


@pytest.fixture(scope="module")
def pretrained_pp_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("pretrained_pp_dir"))
