# coding: utf-8
# 2021/5/30 @ tongshiwei

import codecs
import json
import pytest
from EduNLP.utils import abs_current_dir, path_append


@pytest.fixture(scope="module")
def data():
    _data = []
    with codecs.open(path_append(abs_current_dir(__file__), "test.json", to_str=True), encoding="utf-8") as f:
        for line in f.readlines():
            _data.append(json.loads(line))
    return _data


@pytest.fixture(scope="module")
def quesnet_data():
    _data = []
    data_path = path_append(abs_current_dir(__file__), "test_data/quesnet_data.json", to_str=True)
    with codecs.open(data_path, encoding="utf-8") as f:
        for line in f.readlines():
            _data.append(json.loads(line))
    return _data
