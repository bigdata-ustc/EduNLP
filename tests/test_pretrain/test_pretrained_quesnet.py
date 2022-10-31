from lib2to3.pgen2 import token
import os

from bson import encode
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import torch
from EduNLP.ModelZoo.quesnet import QuesNet
from EduNLP.Pretrain import QuesNetTokenizer, Question, pretrain_quesnet
# from EduNLP.Pretrain import train_quesnet_for_property_prediction, train_quesnet_for_knowledge_prediction
from EduNLP.Vector import T2V
from EduNLP.Vector.quesnet import QuesNetModel
from EduNLP.I2V import QuesNet as QuesNetI2V, get_pretrained_i2v
from EduNLP.utils import abs_current_dir, path_append

from conftest import TEST_GPU


class TestPretrainQuesNet:
    def test_tokenizer(self, standard_luna_data, pretrained_tokenizer_dir):
        pretrained_dir = pretrained_tokenizer_dir
        test_items = [
            {'ques_content': '有公式$\\FormFigureID{wrong1?}$和公式$\\FormFigureBase64{wrong2?}$，\
                    如图$\\FigureID{000004d6-0479-11ec-829b-797d5eb43535}$,\
                    若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'},
            {'ques_content': '如图$\\FigureID{000004d6-0479-11ec-829b-797d5eb43535}$, \
                    若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}
        ]
        img_dir = path_append(abs_current_dir(__file__), "test_data/quesnet_img", to_str=True)
        tokenizer = QuesNetTokenizer(max_length=10, img_dir=img_dir, add_specials=[])

        # set_vocab
        tokenizer_size1 = len(tokenizer)
        tokenizer.set_vocab(standard_luna_data, key=lambda x: x["ques_content"])
        tokenizer.set_meta_vocab(standard_luna_data)
        with pytest.raises(Exception):
            tokenizer.set_meta_vocab([{"ques_content": "已知", "know_name": 1}])
        tokenizer.set_img_dir(None)

        # save and load
        tokenizer_size2 = len(tokenizer)
        assert tokenizer_size1 < tokenizer_size2
        tokenizer.save_pretrained(pretrained_dir)
        tokenizer = QuesNetTokenizer.from_pretrained(pretrained_dir)
        tokenizer.load_meta_vocab('wrong_path')
        tokenizer.load_meta_vocab(tokenizer.meta_vocab_dir)
        tokenizer_size3 = len(tokenizer)
        assert tokenizer_size2 == tokenizer_size3

        # tokenize
        tokens = tokenizer.tokenize(test_items, key=lambda x: x["ques_content"])
        assert isinstance(tokens[0], list)
        tokens = tokenizer.tokenize(test_items[0], key=lambda x: x["ques_content"])
        assert isinstance(tokens[0], str)
        tokenizer.tokenize({"ques_content": "已知"}, key=lambda x: x['ques_content'])
        tokenizer.tokenize({"ques_content": ""}, key=lambda x: x['ques_content'])
        tokenizer.tokenize(test_items[:2], key=lambda x: x['ques_content'])

        # call
        res = tokenizer(test_items, key=lambda x: x["ques_content"])
        assert len(res["seq_idx"]) == 2
        res = tokenizer(test_items[0], key=lambda x: x["ques_content"])
        assert len(res["seq_idx"]) == 10
        ret = tokenizer({"ques_content": "已知", "know_name": ['立体几何', '空间几何体']},
                        key=lambda x: x['ques_content'], return_text=True, padding=True)
        assert 'seq_token' in ret
        assert 'meta' in ret
        with pytest.raises(Exception):
            tokenizer({"ques_content": "已知", "know_name": 3}, key=lambda x: x['ques_content'])

    def test_train_quesnet(self, standard_luna_data, pretrained_model_dir):
        test_items = [
            {'ques_content': '有公式$\\FormFigureID{wrong1?}$和公式$\\FormFigureBase64{wrong2?}$，\
                    如图$\\FigureID{000004d6-0479-11ec-829b-797d5eb43535}$,\
                    若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'},
            {'ques_content': '如图$\\FigureID{000004d6-0479-11ec-829b-797d5eb43535}$, \
                    若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}
        ]

        ques_file = path_append(abs_current_dir(__file__),
                                "../../static/test_data/quesnet_data.json", to_str=True)
        img_dir = path_append(abs_current_dir(__file__),
                              "../../static/test_data/quesnet_img", to_str=True)
        pretrain_quesnet(
            ques_file,
            pretrained_model_dir,
            img_dir=img_dir,
            save_embs=True,
            # data_params={
            #     "stem_key": "ques_content"
            # },
            train_params={
                # "num_train_epochs": 3,
                # "per_device_train_batch_size": 2,
                # "per_device_eval_batch_size": 2,
                # "no_cuda": not TEST_GPU,
                'max_steps': 2,
                'feat_size': 256,
                'save_every': 1,
                'emb_size': 256,
                'device': "cpu"
            }
        )

        tokenizer = QuesNetTokenizer.from_pretrained(pretrained_model_dir, img_dir=img_dir)
        model = QuesNet.from_pretrained(pretrained_model_dir)
        # TODO: need to handle inference for T2V for batch or single
        # encodes = tokenizer(test_items[0], lambda x: x['ques_content'])
        # model(**encodes)
        encodes = tokenizer(test_items, lambda x: x['ques_content'])
        qs = [Question("", encodes['seq_idx'][i],
                       [0], [[0], [0], [0]], encodes['meta_idx'][i]) for i in range(len(encodes))]
        # content = encodes['seq_idx']
        # meta_idx = encodes['meta_idx']
        # qs = [Question("", content[i], [0], [[0], [0], [0]], meta_idx[i]) for i in range(len(encodes))]
        batch = model.make_batch(qs, device="cpu")
        out = model(batch)
        print(out.embeded.shape)

    def test_quesnet_t2v(self, pretrained_model_dir):
        items = [
            {'ques_content': '如图$\\FigureID{000004d6-0479-11ec-829b-797d5eb43535}$, \
                若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}
        ]
        img_dir = path_append(abs_current_dir(__file__),
                              "../../static/test_data/quesnet_img", to_str=True)
        tokenizer = QuesNetTokenizer.from_pretrained(pretrained_model_dir, img_dir=img_dir)
        t2v = QuesNetModel(pretrained_model_dir)

        encodes = tokenizer(items, key=lambda x: x['ques_content'])
        output = t2v(encodes)
        assert len(output) == 2

        t2v = T2V('quesnet', pretrained_model_dir)
        encodes = tokenizer(items, key=lambda x: x['ques_content'])
        output_i = t2v(encodes)
        assert output_i.shape[1] == t2v.vector_size
        assert t2v.infer_vector(encodes).shape[1] == t2v.vector_size
        assert t2v.infer_tokens(encodes).shape[2] == t2v.vector_size

    def test_quesnet_i2v(self, pretrained_model_dir):
        items = [
            {'ques_content': '如图$\\FigureID{000004d6-0479-11ec-829b-797d5eb43535}$, \
                若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}
        ]
        img_dir = path_append(abs_current_dir(__file__),
                              "../../static/test_data/quesnet_img", to_str=True)
        tokenizer_kwargs = {
            "tokenizer_config_dir": pretrained_model_dir,
            "img_dir": img_dir
        }
        i2v = QuesNetI2V('quesnet', 'quesnet', pretrained_model_dir, tokenizer_kwargs=tokenizer_kwargs,
                         pretrained_t2v=False)

        i_vec, t_vec = i2v(items, key=lambda x: x["ques_content"])
        assert len(i_vec[0]) == i2v.vector_size
        assert len(t_vec[0][0]) == i2v.vector_size
        i_vec = i2v.infer_item_vector(items, key=lambda x: x['ques_content'])
        assert len(i_vec[0]) == i2v.vector_size
        t_vec = i2v.infer_token_vector(items, key=lambda x: x['ques_content'])
        assert len(t_vec[0][0]) == i2v.vector_size
