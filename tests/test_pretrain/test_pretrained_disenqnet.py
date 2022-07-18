import os
import pytest
import torch
from EduNLP.ModelZoo import load_items
from EduNLP.ModelZoo.rnn import ElmoLM
from EduNLP.Pretrain import DisenQTokenizer, train_disenqnet
from EduNLP.utils import abs_current_dir, path_append

TEST_GPU = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


# BASE_DIR = "/home/qlh/EduNLP/data/test/test_pretrain/bert"
# pretrained_tokenizer_dir = f"{BASE_DIR}/pretrained_tokenizer_dir"
# pretrained_model_dir = f"{BASE_DIR}/output_model_dir"
# pretrained_pp_dir = f"{BASE_DIR}/pretrained_pp_dir"
# if not os.path.exists(pretrained_tokenizer_dir):
#     os.makedirs(pretrained_tokenizer_dir, exist_ok=True)
# if not os.path.exists(pretrained_model_dir):
#     os.makedirs(pretrained_model_dir, exist_ok=True)
# if not os.path.exists(pretrained_pp_dir):
#     os.makedirs(pretrained_pp_dir, exist_ok=True)
# def get_standard_luna_data():
#     data_path = path_append(abs_current_dir(__file__), "../../static/test_data/standard_luna_data.json", to_str=True)
#     _data = load_items(data_path)
#     return _data
# standard_luna_data = get_standard_luna_data()


class PretrainDisenQNetTest:
    def test_tokenizer(standard_luna_data, pretrained_tokenizer_dir):
        pretrained_dir = pretrained_tokenizer_dir
        test_items = [
            {'ques_content': '有公式$\\FormFigureID{wrong1?}$和公式$\\FormFigureBase64{wrong2?}$，\
                    如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$,\
                    若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'},
            {'ques_content': '如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$, \
                    若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}
        ]
        tokenizer = DisenQTokenizer()
        tokenizer_size1 = len(tokenizer)
        tokenizer.set_vocab(standard_luna_data, key=lambda x: x["ques_content"])
        tokenizer_size2 = len(tokenizer)
        assert tokenizer_size1 < tokenizer_size2
        tokenizer.save_pretrained(pretrained_dir)
        tokenizer = DisenQTokenizer.from_pretrained(pretrained_dir)
        tokenizer_size3 = len(tokenizer)
        assert tokenizer_size2 == tokenizer_size3

        tokens = tokenizer.tokenize(test_items, key=lambda x: x["ques_content"])
        assert isinstance(tokens[0], list)
        tokens = tokenizer.tokenize(test_items[0], key=lambda x: x["ques_content"])
        assert isinstance(tokens[0], str)

        res = tokenizer(test_items, key=lambda x: x["ques_content"])
        assert len(res["seq_idx"].shape) == 2
        res = tokenizer(test_items[0], key=lambda x: x["ques_content"])
        assert len(res["seq_idx"].shape) == 1
        res = tokenizer(test_items, key=lambda x: x["ques_content"], return_tensors=False, return_text=True)
        assert isinstance(res["seq_idx"], list)
