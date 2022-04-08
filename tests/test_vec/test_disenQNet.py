import torch
import numpy as np
import pytest
import os
from EduNLP.Pretrain import DisenQTokenizer, train_disenQNet
from EduNLP.Vector import DisenQModel, T2V
from EduNLP.I2V import DisenQ, get_pretrained_i2v
import warnings
from copy import deepcopy


def test_dataset(disen_train_data, disen_test_data, tmpdir):
    output_dir = str(tmpdir.mkdir('disenq_dataset'))
    train_params = {
        'epoch': 2,
        'batch': 1,
        'trim_min': 1,
    }
    train_disenQNet(
        deepcopy(disen_train_data[:100]),
        output_dir,
        output_dir,
        train_params=train_params,
        test_items=disen_test_data[:100],
    )
    # for test Datasets
    os.remove(os.path.join(output_dir, "vocab.list"))
    with pytest.warns(UserWarning):
        train_disenQNet(
            deepcopy(disen_train_data[:100]),
            output_dir,
            output_dir,
            train_params=train_params,
            test_items=None,
        )


def test_disen_train(disen_train_data, disen_test_data, tmpdir):
    output_dir = str(tmpdir.mkdir('disenq'))
    train_params = {
        'epoch': 2,
        'batch': 16,
        'trim_min': 5,
    }

    train_disenQNet(
        disen_train_data[-100:],
        output_dir,
        output_dir,
        train_params=train_params,
        test_items=disen_test_data[-100:],
    )

    pretrained_dir = output_dir
    tokenizer_kwargs = {
        "tokenizer_config_dir": pretrained_dir,
    }
    i2v = DisenQ('disenq', 'disenq', pretrained_dir, tokenizer_kwargs=tokenizer_kwargs, device="cpu")

    test_items = [
        {"content": "10 米 的 (2/5) = 多少 米 的 (1/2),有 公 式"},
        {"content": "10 米 的 (2/5) = 多少 米 的 (1/2),有 公 式 , 如 图 , 若 $x,y$ 满 足 约 束 条 件 公 式"},
    ]

    t_vec = i2v.infer_token_vector(test_items[0], key=lambda x: x["content"])
    i_vec_k = i2v.infer_item_vector(test_items[0], key=lambda x: x["content"], vector_type="k")
    i_vec_i = i2v.infer_item_vector(test_items[0], key=lambda x: x["content"], vector_type="i")
    assert i_vec_k.shape == torch.Size([1, 128])
    assert i_vec_i.shape == torch.Size([1, 128])
    assert t_vec.shape == torch.Size([1, 11, 128])
    assert i2v.vector_size == i_vec_k.shape[1]

    i_vec, t_vec = i2v.infer_vector(test_items[0], key=lambda x: x["content"], vector_type=None)
    assert len(i_vec) == 2
    assert i_vec[0].shape == torch.Size([1, 128])
    assert i_vec[1].shape == torch.Size([1, 128])
    assert t_vec.shape == torch.Size([1, 11, 128])

    with pytest.raises(KeyError):
        i_vec = i2v.infer_item_vector(test_items[0], key=lambda x: x["content"], vector_type="x")


def test_disen_i2v(tmpdir):
    test_items = [
        {"content": "10 米 的 (2/5) = 多少 米 的 (1/2),有 公 式"},
        {"content": "10 米 的 (2/5) = 多少 米 的 (1/2),有 公 式 , 如 图 , 若 $x,y$ 满 足 约 束 条 件 公 式"},
    ]

    pretrained_dir = str(tmpdir.mkdir('pretrained'))
    i2v = get_pretrained_i2v("disenq_pub_128", model_dir=pretrained_dir)
    i_vec, t_vec = i2v(test_items[0], key=lambda x: x["content"])
    assert len(i_vec) == 2
    assert t_vec.shape[2] == i2v.vector_size

    t_vec = i2v.infer_token_vector(test_items[0], key=lambda x: x["content"])
    i_vec_k = i2v.infer_item_vector(test_items[0], key=lambda x: x["content"], vector_type="k")
    i_vec_i = i2v.infer_item_vector(test_items[0], key=lambda x: x["content"], vector_type="i")
    assert i_vec_k.shape == torch.Size([1, 128])
    assert i_vec_i.shape == torch.Size([1, 128])
    assert t_vec.shape == torch.Size([1, 11, 128])
    assert i2v.vector_size == i_vec_k.shape[1]

    i_vec, t_vec = i2v.infer_vector(test_items[0], key=lambda x: x["content"], vector_type=None)
    assert len(i_vec) == 2
    assert i_vec[0].shape == torch.Size([1, 128])
    assert i_vec[1].shape == torch.Size([1, 128])
    assert t_vec.shape == torch.Size([1, 11, 128])

    i_vec, t_vec = i2v(test_items, key=lambda x: x["content"])
    assert t_vec.shape == torch.Size([2, 23, 128])
