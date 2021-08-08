# coding: utf-8
# 2021/5/18 @ tongshiwei

import os
from EduNLP.utils import abs_current_dir, path_append

DEFAULT_FILEPATH = os.path.abspath(
    path_append(abs_current_dir(__file__), "..", "..", "..", "meta_data", "sif_stopwords.txt")
)


def get_stopwords(filepath=DEFAULT_FILEPATH):
    _stopwords = set()
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            _stopwords.add(line.strip())

    return _stopwords


DEFAULT_STOPWORDS = get_stopwords()
