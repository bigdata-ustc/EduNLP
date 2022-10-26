import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import torch
from EduNLP.ModelZoo.disenqnet import DisenQNet
from EduNLP.Pretrain import DisenQTokenizer, train_disenqnet

# TEST_GPU = torch.cuda.is_available()
from conftest import TEST_GPU


# TODO
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

    def test_train_disenq(self, standard_luna_data, pretrained_model_dir):
        print("[debug] test_train_disenq")
        test_items = [
            {'ques_content': '有公式$\\FormFigureID{wrong1?}$和公式$\\FormFigureBase64{wrong2?}$，\
                    如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$,\
                    若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'},
            {'ques_content': '如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$, \
                    若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}
        ]
        train_disenqnet(
            standard_luna_data,
            pretrained_model_dir,
            data_params={
                "stem_key": "ques_content"
            },
            train_params={
                "num_train_epochs": 3,
                "per_device_train_batch_size": 2,
                "per_device_eval_batch_size": 2,
                "no_cuda": not TEST_GPU,
            }
        )
        model = DisenQNet.from_pretrained(pretrained_model_dir)
        tokenizer = DisenQTokenizer.from_pretrained(pretrained_model_dir)

        # TODO: need to handle inference for T2V for batch or single
        # encodes = tokenizer(test_items[0], lambda x: x['ques_content'])
        # model(**encodes)
        encodes = tokenizer(test_items, lambda x: x['ques_content'])
        model(**encodes)
