import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import torch
from EduNLP.ModelZoo.disenqnet import DisenQNet
from EduNLP.Pretrain import DisenQTokenizer, train_disenqnet
from EduNLP.Vector import T2V, DisenQModel
from EduNLP.I2V import DisenQ, get_pretrained_i2v
from conftest import TEST_GPU


# TODO
class TestPretrainDisenQNet:
    def test_tokenizer(self, standard_luna_data, pretrained_tokenizer_dir):
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
                "stem_key": "ques_content",
                "data_formation": {
                    "knowledge": "know_name"
                }
            },
            train_params={
                "num_train_epochs": 3,
                "per_device_train_batch_size": 2,
                "per_device_eval_batch_size": 2,
                "no_cuda": not TEST_GPU,
            }
        )
        # train with a pretrained model
        train_disenqnet(
            standard_luna_data,
            pretrained_model_dir,
            pretrained_dir=pretrained_model_dir,
            eval_items=standard_luna_data,
            data_params={
                "stem_key": "ques_content",
                "data_formation": {
                    "knowledge": "know_name"
                }
            },
            train_params={
                "num_train_epochs": 1,
                "per_device_train_batch_size": 1,
                "per_device_eval_batch_size": 1,
                "no_cuda": not TEST_GPU,
                "gradient_accumulation_steps": 2
            },
            model_params={
                "hidden_size": 300
            },
            w2v_params={
                "min_count": 5
            }
        )
        model = DisenQNet.from_pretrained(pretrained_model_dir)
        tokenizer = DisenQTokenizer.from_pretrained(pretrained_model_dir)

        # TODO: need to handle inference for T2V for batch or single
        # encodes = tokenizer(test_items[0], lambda x: x['ques_content'])
        # model(**encodes)
        encodes = tokenizer(test_items, lambda x: x['ques_content'])
        model(**encodes)

    def test_disenq_t2v(self, pretrained_model_dir):
        items = [
            {'stem': '如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$, \
                若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}
        ]
        tokenizer = DisenQTokenizer.from_pretrained(pretrained_model_dir)

        t2v = DisenQModel(pretrained_model_dir)
        encodes = tokenizer(items, key=lambda x: x['stem'])
        outputs = t2v(encodes)
        assert len(outputs) == 3

        t2v = T2V('disenq', pretrained_model_dir)
        encodes = tokenizer(items, key=lambda x: x['stem'])
        t_vec = t2v.infer_tokens(encodes, key=lambda x: x["stem"])
        i_vec_k = t2v.infer_vector(encodes, key=lambda x: x["stem"], vector_type="k")
        i_vec_i = t2v.infer_vector(encodes, key=lambda x: x["stem"], vector_type="i")
        assert i_vec_k.shape[1] == t2v.vector_size
        assert i_vec_i.shape[1] == t2v.vector_size
        assert t_vec.shape[2] == t2v.vector_size

    def test_disenq_i2v(self, pretrained_model_dir):
        test_items = [
            {"content": "10 米 的 (2/5) = 多少 米 的 (1/2),有 公 式"},
            {"content": "10 米 的 (2/5) = 多少 米 的 (1/2),有 公 式 , 如 图 , 若 $x,y$ 满 足 约 束 条 件 公 式"},
        ]

        tokenizer_kwargs = {"tokenizer_config_dir": pretrained_model_dir}
        i2v = DisenQ('disenq', 'disenq', pretrained_model_dir, tokenizer_kwargs=tokenizer_kwargs, pretrained_t2v=False)

        t_vec = i2v.infer_token_vector(test_items, key=lambda x: x["content"])
        i_vec_k = i2v.infer_item_vector(test_items, key=lambda x: x["content"], vector_type="k")
        i_vec_i = i2v.infer_item_vector(test_items, key=lambda x: x["content"], vector_type="i")
        assert i_vec_k.shape[1] == i2v.vector_size
        assert i_vec_i.shape[1] == i2v.vector_size
        assert t_vec.shape[2] == i2v.vector_size

        i_vec, t_vec = i2v.infer_vector(test_items, key=lambda x: x["content"], vector_type=None)
        assert len(i_vec) == 2
        assert i_vec_k.shape[1] == i2v.vector_size
        assert i_vec_i.shape[1] == i2v.vector_size
        assert t_vec.shape[2] == i2v.vector_size

        with pytest.raises(KeyError):
            i_vec = i2v.infer_item_vector(test_items, key=lambda x: x["content"], vector_type="x")
