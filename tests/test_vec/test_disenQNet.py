import torch
import numpy as np
import pytest
import os
from EduNLP.Pretrain import DisenQTokenizer, train_disenQNet
from EduNLP.Vector import DisenQModel, T2V
from EduNLP.I2V import DisenQ, get_pretrained_i2v


@pytest.fixture(scope="module")
def disen_data(disen_raw_data):
    _data = []
    for e in disen_raw_data[:10]:
        _data.append(e)
    assert _data
    return _data


def test_disen_train(disen_data, tmpdir):
    output_dir = str(tmpdir.mkdir('disenQ'))
    train_params = {
        'epoch': 10,
        'batch': 16,
    }
    train_disenQNet(
        disen_data,
        output_dir,
        train_params=train_params
    )

    test_items = [
        "10 米 的 (2/5) = 多少 米 的 (1/2),有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$",
        "10 米 的 (2/5) = 多少 米 的 (1/2),有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$,若$x,y$满足约束条件公式"
    ]
    
    vocab_path =  os.path.join(output_dir, "vocab.list")
    config_path = os.path.join(output_dir, "model_config.json")
    tokenizer_kwargs = {
        "vocab_path": vocab_path, 
        "config_path": config_path,
    }
    i2v = DisenQ('disenQ', 'disenQ', output_dir, tokenizer_kwargs=tokenizer_kwargs)

    i_vec, t_vec = i2v(test_items)
    assert len(i_vec[0]) == i2v.vector_size
    assert len(t_vec[0][0]) == i2v.vector_size

    i_vec = i2v.infer_item_vector(test_items)
    assert len(i_vec[0]) == i2v.vector_size

    t_vec = i2v.infer_token_vector(test_items)
    assert len(t_vec[0][0]) == i2v.vector_size
