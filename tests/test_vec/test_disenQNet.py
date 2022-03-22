import torch
import numpy as np
import pytest
import os
from EduNLP.Pretrain import DisenQTokenizer, train_disenQNet
from EduNLP.Vector import DisenQModel, T2V
from EduNLP.I2V import DisenQ, get_pretrained_i2v


@pytest.fixture(scope="module")
def disen_data_train(disen_train_data):
    return disen_train_data[:100]

@pytest.fixture(scope="module")
def disen_data_test(disen_test_data):
    return disen_test_data[:100]

def test_disen_train(disen_data_train, disen_data_test, tmpdir):
    output_dir = str(tmpdir.mkdir('disenq'))
    train_params = {
        'epoch': 5,
        'batch': 16,
        'trim_min': 5,
    }
    train_disenQNet(
        disen_data_train,
        output_dir,
        train_params=train_params,
        test_items=disen_data_test,
    )

    test_items = [
        "10 米 的 (2/5) = 多少 米 的 (1/2),有 公 式",
        "10 米 的 (2/5) = 多少 米 的 (1/2),有 公 式 , 如 图 , 若 $x,y$ 满 足 约 束 条 件 公 式"
    ]

    vocab_path =  os.path.join(output_dir, "vocab.list")
    tokenizer_kwargs = {
        "vocab_path": vocab_path,
        "max_length": 150,
        "text_tokenzier": "space",
    }
    i2v = DisenQ('disenQ', 'disenq', output_dir, tokenizer_kwargs=tokenizer_kwargs, device="cuda")
    
    test_items = [
        {"content": "10 米 的 (2/5) = 多少 米 的 (1/2),有 公 式"},
        {"content": "10 米 的 (2/5) = 多少 米 的 (1/2),有 公 式 , 如 图 , 若 $x,y$ 满 足 约 束 条 件 公 式"},
    ]

    t_vec = i2v.infer_token_vector(test_items[1], key=lambda x:x["content"])
    i_vec = i2v.infer_item_vector(test_items[1], key=lambda x:x["content"], vector_type="k")
    assert i_vec.shape == torch.Size([1, 128])
    assert t_vec.shape == torch.Size([1, 150, 128])

    i_vec, t_vec = i2v.infer_vector(test_items[1], key=lambda x:x["content"], vector_type=None)
    assert len(i_vec) == 2
    assert i_vec[0].shape == torch.Size([1, 128])
    assert i_vec[1].shape == torch.Size([1, 128])
    assert t_vec.shape == torch.Size([1, 150, 128])
