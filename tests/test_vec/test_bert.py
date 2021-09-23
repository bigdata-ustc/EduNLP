import torch
import numpy as np
import pytest
from EduNLP.Pretrain import BertTokenizer, finetune_bert


@pytest.fixture(scope="module")
def stem_data_bert(data):
    test_items = [
        {'stem': '有公式$\\FormFigureID{wrong1?}$和公式$\\FormFigureBase64{wrong2?}$，\
            如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$,若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'},
        {'stem': '如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$, \
            若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}
    ]
    data = test_items + data
    _data = []
    tokenizer = BertTokenizer()
    for e in data[:10]:
        d = tokenizer(e["stem"])
        if d is not None:
            _data.append(d)
    assert _data
    return _data


def test_bert_without_param(stem_data_bert, tmpdir):
    output_dir = str(tmpdir.mkdir('finetuneBert'))
    finetune_bert(
        stem_data_bert,
        output_dir
    )


def test_bert(stem_data_bert, tmpdir):
    output_dir = str(tmpdir.mkdir('finetuneBert'))
    train_params = {
        'epochs': 1,
        'save_steps': 100,
        'batch_size': 8,
        'logging_steps': 3
    }
    finetune_bert(
        stem_data_bert,
        output_dir,
        train_params=train_params
    )
