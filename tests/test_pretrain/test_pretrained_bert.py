import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from EduNLP.ModelZoo.bert import BertForPropertyPrediction, BertForKnowledgePrediction
from transformers import BertModel as HFBertModel
from EduNLP.Pretrain import BertTokenizer, finetune_bert
from EduNLP.Pretrain import finetune_bert_for_property_prediction, finetune_bert_for_knowledge_prediction
from EduNLP.Vector import T2V, BertModel
from EduNLP.I2V import Bert, get_pretrained_i2v

from conftest import TEST_GPU


class TestPretrainBert:
    def test_tokenizer(self, standard_luna_data, pretrained_tokenizer_dir):
        test_items = [
            {'ques_content': '有公式$\\FormFigureID{wrong1?}$和公式$\\FormFigureBase64{wrong2?}$，\
                    如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$,\
                    若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'},
            {'ques_content': '如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$, \
                    若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}
        ]
        text_params = {
            "granularity": "char",
            # "stopwords": None,
        }
        tokenizer = BertTokenizer(pretrained_model="bert-base-chinese", add_specials=True,
                                  tokenize_method="ast_formula", text_params=text_params)

        tokenizer_size1 = len(tokenizer)
        tokenizer.set_vocab(standard_luna_data, key=lambda x: x["ques_content"])
        tokenizer_size2 = len(tokenizer)
        assert tokenizer_size1 < tokenizer_size2
        tokenizer.save_pretrained(pretrained_tokenizer_dir)
        tokenizer = BertTokenizer.from_pretrained(pretrained_tokenizer_dir)
        tokenizer_size3 = len(tokenizer)
        assert tokenizer_size2 == tokenizer_size3
        tokens = tokenizer.tokenize(test_items, key=lambda x: x["ques_content"])
        assert isinstance(tokens[0], list)
        tokens = tokenizer.tokenize(test_items[0], key=lambda x: x["ques_content"])
        assert isinstance(tokens[0], str)
        res = tokenizer(test_items, key=lambda x: x["ques_content"])
        assert len(res["input_ids"].shape) == 2
        res = tokenizer(test_items[0], key=lambda x: x["ques_content"])
        assert len(res["input_ids"].shape) == 2
        res = tokenizer(test_items, key=lambda x: x["ques_content"], return_tensors=False)
        assert isinstance(res["input_ids"], list)

    def test_train_model(self, standard_luna_data, pretrained_model_dir, pretrained_tokenizer_dir):
        tokenizer = BertTokenizer.from_pretrained(pretrained_tokenizer_dir)
        items = [
            {'ques_content': '有公式$\\FormFigureID{wrong1?}$和公式$\\FormFigureBase64{wrong2?}$，\
                    如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$,\
                    若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'},
            {'ques_content': '如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$, \
                    若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}
        ]
        finetune_bert(
            standard_luna_data,
            pretrained_model_dir,
            data_params={
                "stem_key": "ques_content",
            },
            train_params={
                "num_train_epochs": 1,
                "per_device_train_batch_size": 2,
                "save_steps": 500,
                "no_cuda": not TEST_GPU,
            }
        )
        model = HFBertModel.from_pretrained(pretrained_model_dir)
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_dir)

        encodes = tokenizer(items[0], lambda x: x['ques_content'])
        model(**encodes)
        encodes = tokenizer(items, lambda x: x['ques_content'])
        model(**encodes)

    def test_train_pp(self, standard_luna_data, pretrained_pp_dir, pretrained_model_dir):
        data_params = {
            "stem_key": "ques_content",
            "label_key": "difficulty"
        }
        train_params = {
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "no_cuda": not TEST_GPU,
        }
        train_items = standard_luna_data
        # train without eval_items
        finetune_bert_for_property_prediction(
            train_items,
            pretrained_pp_dir,
            pretrained_model=pretrained_model_dir,
            train_params=train_params,
            data_params=data_params
        )
        # train with eval_items
        finetune_bert_for_property_prediction(
            train_items,
            pretrained_pp_dir,
            pretrained_model=pretrained_model_dir,
            eval_items=train_items,
            train_params=train_params,
            data_params=data_params
        )
        model = BertForPropertyPrediction.from_pretrained(pretrained_pp_dir)
        tokenizer = BertTokenizer.from_pretrained(pretrained_pp_dir)

        encodes = tokenizer(train_items[:8], lambda x: x['ques_content'])
        # TODO: need to handle inference for T2V for batch or single
        model(**encodes)

    def test_train_kp(self, standard_luna_data, pretrained_model_dir, pretrained_kp_dir):
        data_params = {
            "stem_key": "ques_content",
            "label_key": "know_list"
        }
        train_params = {
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "no_cuda": not TEST_GPU,
        }
        model_params = {
            "num_classes_list": [10, 27, 963],
            "num_total_classes": 1000,
        }
        train_items = standard_luna_data
        # train without eval_items
        finetune_bert_for_knowledge_prediction(
            train_items,
            pretrained_kp_dir,
            pretrained_model=pretrained_model_dir,
            train_params=train_params,
            data_params=data_params,
            model_params=model_params
        )
        # train with eval_items
        finetune_bert_for_knowledge_prediction(
            train_items,
            pretrained_kp_dir,
            pretrained_model=pretrained_model_dir,
            eval_items=train_items,
            train_params=train_params,
            data_params=data_params,
            model_params=model_params
        )
        model = BertForKnowledgePrediction.from_pretrained(pretrained_kp_dir)
        tokenizer = BertTokenizer.from_pretrained(pretrained_kp_dir)

        encodes = tokenizer(train_items[:8], lambda x: x['ques_content'])
        # TODO: need to handle inference for T2V for batch or single
        model(**encodes)

    def test_t2v(self, pretrained_model_dir):
        items = [
            {'stem': '如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$, \
                若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}
        ]
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_dir)
        encodes = tokenizer(items, key=lambda x: x['stem'])

        t2v = BertModel(pretrained_model_dir)
        output = t2v(encodes)
        assert output.shape[2] == t2v.vector_size

        t2v = T2V('bert', pretrained_model_dir)
        output = t2v(encodes)
        assert output.shape[-1] == t2v.vector_size
        assert t2v.infer_vector(encodes).shape[1] == t2v.vector_size
        assert t2v.infer_tokens(encodes).shape[2] == t2v.vector_size
        t2v.infer_vector(encodes, pooling_strategy='CLS')
        t2v.infer_vector(encodes, pooling_strategy='average')

    def test_i2v(self, pretrained_model_dir):
        items = [
            {'stem': '如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$, \
                若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}
        ]
        tokenizer_kwargs = {
            "tokenizer_config_dir": pretrained_model_dir
        }
        i2v = Bert('bert', 'bert', pretrained_model_dir, tokenizer_kwargs=tokenizer_kwargs)

        i_vec, t_vec = i2v(items, key=lambda x: x['stem'])
        assert len(i_vec[0]) == i2v.vector_size
        assert len(t_vec[0][0]) == i2v.vector_size

        i_vec = i2v.infer_item_vector(items, key=lambda x: x['stem'])
        assert len(i_vec[0]) == i2v.vector_size
        t_vec = i2v.infer_token_vector(items, key=lambda x: x['stem'])
        assert len(t_vec[0][0]) == i2v.vector_size

    #     output_dir = pretrained_model_dir
    #     i2v = get_pretrained_i2v("luna_pub_bert_math_base", output_dir)
    #     i_vec, t_vec = i2v(items, key=lambda x: x['stem'])
    #     assert len(i_vec[0]) == i2v.vector_size
    #     assert len(t_vec[0][0]) == i2v.vector_size

    #     i_vec = i2v.infer_item_vector(items, key=lambda x: x['stem'])
    #     assert len(i_vec[0]) == i2v.vector_size

    #     t_vec = i2v.infer_token_vector(items, key=lambda x: x['stem'])
    #     assert len(t_vec[0][0]) == i2v.vector_size
