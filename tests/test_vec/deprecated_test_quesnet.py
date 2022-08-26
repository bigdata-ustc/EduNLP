import pytest
import torch
from EduNLP.Pretrain import QuesNetTokenizer, pretrain_quesnet
from EduNLP.I2V import QuesNet
from EduNLP.utils import abs_current_dir, path_append
from EduNLP.ModelZoo.quesnet import QuesNet as _QuesNet, ImageAE, MetaAE
import numpy as np
import os


def test_quesnet_tokenizer(quesnet_data, tmpdir):
    tokenizer = QuesNetTokenizer(meta=['know_name'], max_length=10,
                                 img_dir=path_append(abs_current_dir(__file__),
                                                     "test_data/quesnet_img", to_str=True))
    with pytest.raises(Exception):
        tokenizer.tokenize(quesnet_data[0], key=lambda x: x['ques_content'])

    # set_vocab
    tokenizer.set_vocab([{"ques_content": "已知", "know_name": ['立体几何', '空间几何体']}],
                        key=lambda x: x['ques_content'], trim_min_count=1)
    with pytest.raises(Exception):
        tokenizer.set_vocab([{"ques_content": "已知", "know_name": 1}],
                            key=lambda x: x['ques_content'], trim_min_count=1)

    tokenizer.set_vocab(quesnet_data[:100], key=lambda x: x['ques_content'],
                        trim_min_count=2, silent=False)

    # vocab_size, set_img_dir
    assert isinstance(tokenizer.vocab_size, int)
    tokenizer.set_img_dir(None)

    # tokenize
    tokenizer.tokenize({"ques_content": "已知"}, key=lambda x: x['ques_content'])
    tokenizer.tokenize({"ques_content": ""}, key=lambda x: x['ques_content'])
    tokenizer.tokenize(quesnet_data[:2], key=lambda x: x['ques_content'])

    # __call__
    ret = tokenizer({"ques_content": "已知", "know_name": ['立体几何', '空间几何体']},
                    key=lambda x: x['ques_content'], return_text=True, padding=True)
    assert 'content' in ret
    assert 'meta' in ret
    with pytest.raises(Exception):
        tokenizer({"ques_content": "已知", "know_name": 3}, key=lambda x: x['ques_content'])


def test_quesnet_pretrain(quesnet_data, tmpdir):
    output_dir = str(tmpdir.mkdir('quesnet'))

    tokenizer = QuesNetTokenizer(meta=['know_name'],
                                 img_dir=path_append(abs_current_dir(__file__),
                                                     "../../static/test_data/quesnet_img",
                                                     to_str=True))

    tokenizer.set_vocab(quesnet_data, key=lambda x: x['ques_content'],
                        trim_min_count=2, silent=False)
    tokenizer.save_pretrained(output_dir)

    tokenizer = QuesNetTokenizer.from_pretrained(output_dir,
                                                 img_dir=path_append(abs_current_dir(__file__),
                                                                     "../../static/test_data/quesnet_img",
                                                                     to_str=True))
    train_params = {
        'max_steps': 2,
        'feat_size': 256,
        'save_every': 1,
        'emb_size': 256
    }
    pretrain_quesnet(path_append(abs_current_dir(__file__),
                                 "../../static/test_data/quesnet_data.json", to_str=True),
                     output_dir, tokenizer, True, train_params)

    tokenizer_kwargs = {
        'tokenizer_config_dir': output_dir,
    }
    i2v = QuesNet('quesnet', 'quesnet', output_dir,
                  tokenizer_kwargs=tokenizer_kwargs, device="cpu")
    t_vec = i2v.infer_token_vector(quesnet_data[0], key=lambda x: x["ques_content"])
    i_vec = i2v.infer_item_vector(quesnet_data[0], key=lambda x: x["ques_content"])
    assert t_vec.shape[-1] == 256
    assert i_vec.shape == torch.Size([1, 256])
    t_vec = i2v.infer_token_vector(quesnet_data[:2], key=lambda x: x["ques_content"])
    i_vec = i2v.infer_item_vector(quesnet_data[:2], key=lambda x: x["ques_content"])
    assert t_vec.shape[0] == 2, t_vec.shape
    assert i_vec.shape[0] == 2, i_vec.shape

    assert i2v.vector_size == 256

    # test pretrained embeddings
    ie = ImageAE(train_params['emb_size'])
    meta_size = len(tokenizer.stoi['know_name'])
    me = MetaAE(meta_size, train_params['emb_size'])
    ie.load_state_dict(torch.load(os.path.join(output_dir, 'trained_ie.pt')))
    me.load_state_dict(torch.load(os.path.join(output_dir, 'trained_me.pt')))
    _QuesNet(_stoi=tokenizer.stoi, feat_size=train_params['feat_size'],
             emb_size=train_params['emb_size'],
             rnn='GRU',
             embs=np.load(os.path.join(output_dir, 'w2v_embs.npy')),
             pretrained_meta=me, pretrained_image=ie)
    with pytest.raises(ValueError):
        _QuesNet(_stoi=tokenizer.stoi, feat_size=train_params['feat_size'],
                 emb_size=train_params['emb_size'],
                 rnn='wrong_rnn',
                 embs=np.load(os.path.join(output_dir, 'w2v_embs.npy')),
                 pretrained_meta=me, pretrained_image=ie)
