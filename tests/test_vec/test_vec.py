# coding: utf-8
# 2021/5/30 @ tongshiwei

import torch
import numpy as np
import pytest
from EduNLP.Pretrain import train_vector, GensimWordTokenizer
from EduNLP.Vector import W2V, D2V, RNNModel, T2V, Embedding
from EduNLP.I2V import D2V as I_D2V


@pytest.fixture(scope="module")
def stem_data(data):
    _data = []
    for e in data[:10]:
        d = e["stem"]
        _data.append(d)
    assert _data
    return _data


@pytest.fixture(scope="module")
def stem_tokens(stem_data):
    _data = []
    tokenizer = GensimWordTokenizer()
    for e in stem_data:
        d = tokenizer(e)
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
def test_w2v(stem_tokens, tmpdir, method, binary):
    filepath_prefix = str(tmpdir.mkdir(method).join("stem_tf_"))
    filepath = train_vector(
        stem_tokens,
        filepath_prefix,
        10,
        method=method,
        binary=binary,
        train_params=dict(min_count=0)
    )
    w2v = W2V(filepath, method=method, binary=binary)
    assert w2v.vector_size == 10
    w2v(*stem_tokens[0])
    assert len(w2v.infer_vector([stem_tokens[0]])[0]) == w2v.vector_size
    w2v.key_to_index(stem_tokens[0][0])
    assert len(w2v) > 0
    assert len(w2v["[FIGURE]"]) == 10
    assert len(list(w2v("[FIGURE]"))) == 1
    assert np.array_equal(w2v["[UNK]"], np.zeros((10,)))
    assert np.array_equal(w2v["[PAD]"], np.zeros((10,)))
    assert w2v.vectors.shape == (len(w2v.wv.vectors) + len(w2v.constants), w2v.vector_size)
    assert w2v.key_to_index("[UNK]") == 0
    assert w2v.key_to_index("OOV") == 0

    t2v = T2V("w2v", filepath=filepath, method=method, binary=binary)
    assert len(t2v(stem_tokens[:1])[0]) == t2v.vector_size

    for _w2v in [[filepath, method, binary], dict(filepath=filepath, method=method, binary=binary)]:
        embedding = Embedding(_w2v, device="cpu")
        items, item_len = embedding(stem_tokens[:5])
        assert items.shape == (5, max(item_len), embedding.embedding_dim)


def test_embedding():
    with pytest.raises(TypeError):
        Embedding("error")


def test_rnn(stem_tokens, tmpdir):
    method = "sg"
    filepath_prefix = str(tmpdir.mkdir(method).join("stem_tf_"))
    filepath = train_vector(
        stem_tokens,
        filepath_prefix,
        10,
        method=method,
        train_params=dict(min_count=0)
    )
    w2v = W2V(filepath, method=method)

    with pytest.raises(TypeError):
        RNNModel("Error", w2v, 20)

    for rnn_type in ["ElMo", "Rnn", "lstm", "GRU"]:
        rnn = RNNModel(rnn_type, w2v, 20, device="cpu")

        tokens = rnn.infer_tokens(stem_tokens[:1])
        item = rnn.infer_vector(stem_tokens[:1])
        assert tokens.shape == (1, len(stem_tokens[0]), 20 * (2 if rnn.bidirectional else 1))
        assert item.shape == (1, rnn.vector_size)
        item_vec = rnn.infer_vector(stem_tokens[:1])
        assert torch.equal(item, item_vec)

        t2v = T2V(rnn_type, w2v, 20)
        assert len(t2v(stem_tokens[:1])[0]) == t2v.vector_size

        saved_params = rnn.save(str((tmpdir / method).join("stem_tf_rnn.params")), save_embedding=True)

        rnn = RNNModel(rnn_type, w2v, 20, device="cpu", model_params=saved_params)
        rnn.train()
        assert rnn.is_frozen is False
        rnn.freeze()
        assert rnn.is_frozen is True
        item_vec1 = rnn.infer_vector(stem_tokens[:1])
        assert torch.equal(item, item_vec1)


def test_d2v(stem_tokens, tmpdir, stem_data):
    method = "d2v"
    filepath_prefix = str(tmpdir.mkdir(method).join("stem_tf_"))
    filepath = train_vector(
        stem_tokens,
        filepath_prefix,
        10,
        method=method,
        train_params=dict(min_count=0)
    )
    d2v = D2V(filepath)
    assert len(d2v(stem_tokens[0])) == 10
    assert d2v.vector_size == 10

    t2v = T2V("d2v", filepath)
    assert len(t2v(stem_tokens[:1])[0]) == t2v.vector_size

    i2v = I_D2V("text", "d2v", filepath)
    i_vec, t_vec = i2v(stem_data[:1])
    assert len(i_vec[0]) == i2v.vector_size
    assert t_vec is None

    cfg_path = str(tmpdir / method / "i2v_config.json")
    i2v.save(config_path=cfg_path)
    i2v = I_D2V.load(cfg_path)

    i_vec = i2v.infer_item_vector(stem_data[:1])
    assert len(i_vec[0]) == i2v.vector_size

    t_vec = i2v.infer_token_vector(stem_data[:1])
    assert t_vec is None


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


def test_exception(stem_tokens, tmpdir):
    filepath_prefix = str(tmpdir.mkdir("error").join("stem_tf_"))
    with pytest.raises(ValueError):
        train_vector(
            stem_tokens,
            filepath_prefix,
            embedding_dim=10,
            method="error",
            train_params=dict(min_count=0)
        )
    with pytest.raises(ValueError):
        D2V("error_path", method="error")
