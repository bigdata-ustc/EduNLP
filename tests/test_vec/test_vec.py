# coding: utf-8
# 2021/5/30 @ tongshiwei

import numpy as np
import pytest
from EduNLP.Pretrain import train_vector, GensimWordTokenizer
from EduNLP.Vector import W2V, D2V, RNNModel


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


@pytest.fixture(scope="module")
def stem_data_general(data):
    test_items = [
        {'stem': '有公式$\\FormFigureID{wrong1?}$和公式$\\FormFigureBase64{wrong2?}$，\
            如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$,若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'},
        {'stem': '如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$, \
            若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}
    ]
    data = test_items + data
    _data = []
    tokenizer = GensimWordTokenizer(symbol="gmas", general=True)
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
    assert len(w2v.infer_vector(stem_data[0])) == w2v.vector_size
    w2v.key_to_index(stem_data[0][0])
    assert len(w2v) > 0
    assert len(w2v["[FIGURE]"]) == 10
    assert len(list(w2v("[FIGURE]"))) == 1
    assert np.array_equal(w2v["[UNK]"], np.zeros((10,)))
    assert np.array_equal(w2v["[PAD]"], np.zeros((10,)))
    assert w2v.vectors.shape == (len(w2v.wv.vectors) + len(w2v.constants), w2v.vector_size)
    assert w2v.key_to_index("[UNK]") == 0
    assert w2v.key_to_index("OOV") == 0


def test_rnn(stem_data, tmpdir):
    method = "sg"
    filepath_prefix = str(tmpdir.mkdir(method).join("stem_tf_"))
    filepath = train_vector(
        stem_data,
        filepath_prefix,
        10,
        method=method,
        train_params=dict(min_count=0)
    )
    w2v = W2V(filepath, method=method)

    with pytest.raises(TypeError):
        RNNModel("Error", w2v, 20)

    for rnn_type in ["ElMo", "Rnn", "lstm", "GRU"]:
        rnn = RNNModel(rnn_type, w2v, 20)

        tokens = rnn.infer_tokens(stem_data[:1])
        item = rnn.infer_vector(stem_data[:1])
        assert tokens.shape == (1, len(stem_data[0]), 20 * (2 if rnn.rnn.bidirectional else 1))
        assert item.shape == ((2 if rnn.rnn.bidirectional else 1), 1, 20)


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
def test_d2v_bow_tfidf(stem_data_general, tmpdir, method):
    filepath_prefix = str(tmpdir.mkdir(method).join("stem_tf_"))
    filepath = train_vector(
        stem_data_general,
        filepath_prefix,
        method=method
    )
    d2v = D2V(filepath, method=method)
    d2v(stem_data_general[0])
    assert d2v.vector_size > 0


def test_exception(stem_data, tmpdir):
    filepath_prefix = str(tmpdir.mkdir("error").join("stem_tf_"))
    with pytest.raises(ValueError):
        train_vector(
            stem_data,
            filepath_prefix,
            embedding_dim=10,
            method="error",
            train_params=dict(min_count=0)
        )
    with pytest.raises(ValueError):
        D2V("error_path", method="error")
