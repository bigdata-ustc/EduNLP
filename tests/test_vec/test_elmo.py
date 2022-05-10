import torch
import numpy as np
import pytest
import os
from EduNLP.Pretrain import ElmoTokenizer, ElmoDataset, train_elmo
from EduNLP.Vector import ElmoModel, T2V
from EduNLP.I2V import Elmo, get_pretrained_i2v


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
    tokenizer = ElmoTokenizer()
    train_text = [tokenizer.tokenize(item=data, freeze_vocab=False) for data in stem_data_elmo]
    train_elmo(train_text, output_dir)
    model = ElmoModel(output_dir)
    item = {'stem': '如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$, \
                若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}
    inputs, length = tokenizer(item=item['stem'], freeze_vocab=True, pad_to_max_length=False)
    output = model(inputs)
    assert model.vector_size > 0
    assert output.shape[-1] == model.vector_size
    t2v = T2V('elmo', output_dir)
    assert t2v(inputs).shape[-1] == t2v.vector_size
    assert t2v.infer_vector(inputs).shape[-1] == t2v.vector_size


def test_elmo_i2v(stem_data_elmo, tmpdir):
    output_dir = str(tmpdir.mkdir('elmo_test'))
    tokenizer = ElmoTokenizer()
    train_text = [tokenizer.tokenize(item=data, freeze_vocab=False) for data in stem_data_elmo]
    train_elmo(train_text, output_dir)
    tokenizer_kwargs = {"path": os.path.join(output_dir, "vocab.json")}
    i2v = Elmo('elmo', 'elmo', output_dir, tokenizer_kwargs=tokenizer_kwargs, pretrained_t2v=False)
    item = {'stem': '如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$, \
                若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}
    i_vec, t_vec = i2v(item['stem'])
    assert len(i_vec) == i2v.vector_size
    assert len(t_vec[0]) == i2v.vector_size
    i_vec = i2v.infer_item_vector(item['stem'])
    assert len(i_vec) == i2v.vector_size
    t_vec = i2v.infer_token_vector(item['stem'])
    assert len(t_vec[0]) == i2v.vector_size

    i_vec, t_vec = i2v([item['stem'], item['stem'], item['stem']])
    assert len(i_vec[0]) == i2v.vector_size
    assert len(t_vec[0][0]) == i2v.vector_size


def test_pretrained_elmo_i2v(stem_data_elmo, tmpdir):
    output_dir = str(tmpdir.mkdir('elmo_test'))
    i2v = get_pretrained_i2v("elmo_test", output_dir)
    item = {'stem': '如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$, \
                    若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}
    i_vec, t_vec = i2v(item['stem'])
    assert len(i_vec) == i2v.vector_size
    assert len(t_vec[0]) == i2v.vector_size
    i_vec = i2v.infer_item_vector(item['stem'])
    assert len(i_vec) == i2v.vector_size
    t_vec = i2v.infer_token_vector(item['stem'])
    assert len(t_vec[0]) == i2v.vector_size
