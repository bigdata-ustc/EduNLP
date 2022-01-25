import torch
import numpy as np
import pytest
import os
from EduNLP.Pretrain import ElmoVocab, ElmoDataset
from EduNLP.Vector import ElmoModel, train_elmo, T2V
from EduNLP.I2V import Elmo, get_pretrained_i2v
from EduNLP.Tokenizer import PureTextTokenizer


@pytest.fixture(scope="module")
def stem_data_elmo(data):
    test_items = [
        {'stem': '有公式$\\FormFigureID{wrong1?}$和公式$\\FormFigureBase64{wrong2?}$，\
                如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$,\
                若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'},
        {'stem': '如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$, \
                若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}
    ]
    data = test_items + data
    _data = []
    for e in data[:10]:
        _data.append(e['stem'])
    assert _data
    return _data


def test_elmo_without_param(stem_data_elmo, tmpdir):
    output_dir = str(tmpdir.mkdir('elmo_test'))
    vocab = ElmoVocab()
    train_elmo(stem_data_elmo, output_dir)
    model = ElmoModel(output_dir + '/' + 'elmo_test')
    item = {'stem': '如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$, \
                若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}
    inputs = vocab.tokenize(item['stem'])
    output = model(inputs)
    assert model.vector_size > 0
    assert output.shape[-1] == model.vector_size
    t2v = T2V('elmo', output_dir, '/elmo_test')
    assert t2v(inputs).shape[-1] == t2v.vector_size
    assert t2v.infer_vector(inputs).shape[-1] == t2v.vector_size
