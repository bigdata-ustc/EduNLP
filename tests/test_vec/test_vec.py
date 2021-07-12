# coding: utf-8
# 2021/5/30 @ tongshiwei

import numpy as np
import pytest
from EduNLP.Pretrain import train_vector, GensimWordTokenizer
from EduNLP.Vector import W2V, D2V


@pytest.fixture(scope="module")
def stem_data(data):
    _data = []
    tokenizer = GensimWordTokenizer()
    for e in data[:10]:
        d = tokenizer(e["stem"])
        if d is not None:
            _data.append(d.tokens)
    assert _data
    return _data


@pytest.mark.parametrize("method", ["sg", "cbow", "fasttext"])
@pytest.mark.parametrize("binary", [True, False, None])
def test_w2v(stem_data, tmpdir, method, binary):
    filepath_prefix = str(tmpdir.mkdir(method).join("stem_tf_"))
    filepath = train_vector(
        stem_data,
        filepath_prefix,
        10,
        method=method,
        binary=binary,
        train_params=dict(min_count=0)
    )
    w2v = W2V(filepath, method=method, binary=binary)
    assert w2v.vector_size == 10
    w2v(*stem_data[0])
    w2v.key_to_index(stem_data[0][0])
    assert len(w2v["[FIGURE]"]) == 10
    assert len(list(w2v("[FIGURE]"))) == 1
    assert np.array_equal(w2v["[UNK]"], np.zeros((10,)))
    assert np.array_equal(w2v["[PAD]"], np.zeros((10,)))
    assert w2v.vectors.shape == (len(w2v.wv.vectors) + len(w2v.constants), w2v.vector_size)
    assert w2v.key_to_index("[UNK]") == 0
    assert w2v.key_to_index("OOV") == 0


def test_d2v(stem_data, tmpdir):
    method = "d2v"
    filepath_prefix = str(tmpdir.mkdir(method).join("stem_tf_"))
    filepath = train_vector(
        stem_data,
        filepath_prefix,
        10,
        method=method,
        train_params=dict(min_count=0)
    )
    d2v = D2V(filepath)
    assert len(d2v(stem_data[0])) == 10
    assert d2v.vector_size == 10


@pytest.mark.parametrize("method", ["bow", "tfidf"])
def test_d2v_bow_tfidf(stem_data, tmpdir, method):
    filepath_prefix = str(tmpdir.mkdir(method).join("stem_tf_"))
    filepath = train_vector(
        stem_data,
        filepath_prefix,
        method=method
    )
    d2v = D2V(filepath, method=method)
    d2v(stem_data[0])


def test_exception(stem_data, tmpdir):
    filepath_prefix = str(tmpdir.mkdir("error").join("stem_tf_"))
    with pytest.raises(ValueError):
        train_vector(
            stem_data,
            filepath_prefix,
            10,
            method="error",
            train_params=dict(min_count=0)
        )
    with pytest.raises(ValueError):
        D2V("error_path", method="error")
