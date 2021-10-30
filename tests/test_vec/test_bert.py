import torch
import numpy as np
import pytest
from EduNLP.Pretrain import BertTokenizer, finetune_bert
from EduNLP.Vector import BertModel, T2V
from EduNLP.I2V import Bert, get_pretrained_i2v


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
    tokenizer = BertTokenizer(output_dir)
    model = BertModel(output_dir)
    item = {'stem': '如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$, \
            若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}
    inputs = tokenizer(item['stem'], return_tensors='pt')
    output = model(inputs)
    assert model.vector_size > 0
    assert output.shape[-1] == model.vector_size
    t2v = T2V('bert', output_dir)
    assert t2v(inputs).shape[-1] == t2v.vector_size
    assert t2v.infer_vector(inputs).shape[-1] == t2v.vector_size
    assert t2v.infer_tokens(inputs).shape[-1] == t2v.vector_size


def test_bert_i2v(stem_data_bert, tmpdir):
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

    item = {'stem': '如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$, \
            若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}
    tokenizer_kwargs = {"pretrain_model": output_dir}
    i2v = Bert('bert', 'bert', output_dir, tokenizer_kwargs=tokenizer_kwargs)
    i_vec, t_vec = i2v([item['stem'], item['stem']])
    assert len(i_vec[0]) == i2v.vector_size
    assert len(t_vec[0][0]) == i2v.vector_size

    i_vec = i2v.infer_item_vector([item['stem'], item['stem']])
    assert len(i_vec[0]) == i2v.vector_size

    t_vec = i2v.infer_token_vector([item['stem'], item['stem']])
    assert len(t_vec[0][0]) == i2v.vector_size
