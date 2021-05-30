# coding: utf-8
# 2021/5/30 @ tongshiwei

import pytest
from EduNLP.Pretrain import train_vector
from EduNLP.Vector import W2V
from EduNLP.SIF import sif4sci


@pytest.fixture(scope="module")
def stem_data(data):
    _data = []
    for e in data[:10]:
        d = sif4sci(
            e["stem"],
            symbol="gm",
            tokenization_params={
                "formula_params": {
                    "method": "ast",
                    "return_type": "list",
                    "ord2token": True
                }
            },
            errors="ignore"
        )
        if d is not None:
            _data.append(d.tokens)
    assert _data
    return _data


@pytest.mark.parametrize("method", ["sg", "cbow"])
def test_w2v(stem_data, tmpdir, method):
    filepath_prefix = tmpdir.mkdir(method).join("stem_tf_")
    filepath = train_vector(
        stem_data,
        filepath_prefix,
        10,
        method=method
    )
    w2v = W2V(filepath)
    w2v(stem_data[0])
    assert len(w2v[["FIGURE"]]) == 10


def test_fasttext(stem_data, tmpdir):
    method = "fasttext"
    filepath_prefix = tmpdir.mkdir(method).join("stem_tf_")
    filepath = train_vector(
        stem_data,
        filepath_prefix,
        10,
        method=method
    )
    w2v = W2V(filepath, fasttext=True)
    w2v(stem_data[0])
    assert len(w2v[["FIGURE"]]) == 10
