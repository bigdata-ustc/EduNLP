import pytest
import torch
from EduNLP.Pretrain import QuesNetTokenizer, pretrain_QuesNet
from EduNLP.I2V import QuesNet
from EduNLP.utils import abs_current_dir, path_append


def test_quesnet_tokenizer(quesnet_data, tmpdir):
    tokenizer = QuesNetTokenizer(meta=['know_name'],
                                 img_dir=path_append(abs_current_dir(__file__),
                                                     "test_data/quesnet_img", to_str=True))
    with pytest.raises(Exception):
        tokenizer.tokenize(quesnet_data[0], key=lambda x: x['ques_content'])

    tokenizer.set_vocab(quesnet_data[:100], key=lambda x: x['ques_content'],
                        trim_min_count=2, silent=False)

    tokenizer.tokenize({"ques_content": "已知"}, key=lambda x: x['ques_content'])


def test_quesnet_pretrain(quesnet_data, tmpdir):
    output_dir = str(tmpdir.mkdir('quesnet'))

    tokenizer = QuesNetTokenizer(meta=['know_name'],
                                 img_dir=path_append(abs_current_dir(__file__),
                                                     "test_data/quesnet_img", to_str=True))

    tokenizer.set_vocab(quesnet_data, key=lambda x: x['ques_content'],
                        trim_min_count=2, silent=False)
    tokenizer.save_pretrained(output_dir)

    tokenizer = QuesNetTokenizer.from_pretrained(output_dir,
                                                 img_dir=path_append(abs_current_dir(__file__),
                                                                     "test_data/quesnet_img", to_str=True))
    train_params = {
        'max_steps': 2,
        'feat_size': 256
    }
    pretrain_QuesNet(path_append(abs_current_dir(__file__),
                                 "test_data/quesnet_data.json", to_str=True),
                     output_dir, tokenizer, train_params)

    tokenizer_kwargs = {
        'tokenizer_config_dir': output_dir,
    }
    i2v = QuesNet('quesnet', 'quesnet', output_dir,
                  tokenizer_kwargs=tokenizer_kwargs, device="cpu")
    t_vec = i2v.infer_token_vector(quesnet_data[0], key=lambda x: x["ques_content"])
    i_vec = i2v.infer_item_vector(quesnet_data[0], key=lambda x: x["ques_content"])
    assert t_vec.shape[-1] == 256
    assert i_vec.shape == torch.Size([256])

    assert i2v.vector_size == 256
