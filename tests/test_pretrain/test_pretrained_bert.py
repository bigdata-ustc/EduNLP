import os
import pytest
import torch
import json
from EduNLP.utils import abs_current_dir, path_append
from EduNLP.ModelZoo import load_items
from EduNLP.ModelZoo.bert import BertForPropertyPrediction
from transformers import BertModel
from EduNLP.Pretrain import BertTokenizer, BertDataset, finetune_bert, finetune_bert_for_property_prediction

TEST_GPU = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


class PretrainBertTest:
    def test_tokenizer(standard_luna_data, pretrained_tokenizer_dir):
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

    def test_train_model(standard_luna_data, pretrained_model_dir, pretrained_tokenizer_dir):
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
                "num_train_epochs": 3,
                "per_device_train_batch_size": 2,
                "per_device_eval_batch_size": 2,
                "no_cuda": not TEST_GPU,
            }
        )
        model = BertModel.from_pretrained(pretrained_model_dir)
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_dir)

        encodes = tokenizer(items[0], lambda x: x['ques_content'])
        model(**encodes)
        encodes = tokenizer(items, lambda x: x['ques_content'])
        model(**encodes)

    def test_train_pp(standard_luna_data, pretrained_pp_dir, pretrained_model_dir):
        data_params = {
            "stem_key": "ques_content",
            "labal_key": "difficulty"
        }
        train_params = {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "no_cuda": not TEST_GPU,
        }
        train_items = standard_luna_data
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


# Old test
# def test_bert_tokenizer(stem_data_bert, tmpdir):
    # output_dir = str(tmpdir.mkdir('test_bert_tokenizer'))
    # tokenizer = BertTokenizer(add_special_tokens=True, tokenize_method='pure_text')
    # tokenizer.save_pretrained(output_dir)
    # tokenizer = BertTokenizer.from_pretrained(output_dir)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


# def test_train_bert(standard_luna_data, tmpdir_factory):
#     output_dir = tmpdir_factory.mktemp("output_dir")
#     assert os.path.exists(output_dir)
#     print("debbug test_train_bert: output_dir=", output_dir)
    # pretrained_tokenizer_dir = os.path.join(os.path.dirname(output_dir), "pretrained_tokenizer_dir")
    # assert os.path.exists(pretrained_tokenizer_dir)
    # print("debbug test_train_bert: pretrained_tokenizer_dir=", pretrained_tokenizer_dir)


# TODO: I2V test
# def test_bert_i2v(stem_data_bert, tmpdir):
    # ...
    # item = {'stem': '如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$, \
    #         若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}
    # tokenizer_kwargs = {"tokenizer_config_dir": output_dir}
    # i2v = Bert('bert', 'bert', output_dir, tokenizer_kwargs=tokenizer_kwargs)
    # i_vec, t_vec = i2v([item['stem'], item['stem']])
    # assert len(i_vec[0]) == i2v.vector_size
    # assert len(t_vec[0][0]) == i2v.vector_size

    # i_vec = i2v.infer_item_vector([item['stem'], item['stem']])
    # assert len(i_vec[0]) == i2v.vector_size

    # t_vec = i2v.infer_token_vector([item['stem'], item['stem']])
    # assert len(t_vec[0][0]) == i2v.vector_size

# TODO: pretrained_i2v_test
# def test_luna_pub_bert(stem_data_bert, tmpdir):
#     output_dir = str(tmpdir.mkdir('bert_test'))
#     i2v = get_pretrained_i2v("luna_pub_bert_math_base", output_dir)
#     item = {'stem': '如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$, \
#                         若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}
#     i_vec, t_vec = i2v([item['stem'], item['stem']])
#     assert len(i_vec[0]) == i2v.vector_size
#     assert len(t_vec[0][0]) == i2v.vector_size

#     i_vec = i2v.infer_item_vector([item['stem'], item['stem']])
#     assert len(i_vec[0]) == i2v.vector_size

#     t_vec = i2v.infer_token_vector([item['stem'], item['stem']])
#     assert len(t_vec[0][0]) == i2v.vector_size

# test_tokenizer(standard_luna_data, pretrained_tokenizer_dir)
# test_train_bert(standard_luna_data, output_dir)
# test_train_bert_pp(standard_luna_data, pretrained_pp_dir)
