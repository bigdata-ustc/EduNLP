import os
import pytest
import torch
from EduNLP.ModelZoo import load_items
from EduNLP.ModelZoo.rnn import ElmoLM
from EduNLP.Pretrain import ElmoTokenizer, train_elmo, train_elmo_for_property_prediction
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


class PretrainEmloTest:
    def test_tokenizer(standard_luna_data, pretrained_tokenizer_dir):
        pretrained_dir = pretrained_tokenizer_dir
        test_items = [
            {'ques_content': '有公式$\\FormFigureID{wrong1?}$和公式$\\FormFigureBase64{wrong2?}$，\
                    如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$,\
                    若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'},
            {'ques_content': '如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$, \
                    若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}
        ]
        tokenizer = ElmoTokenizer()
        tokenizer_size1 = len(tokenizer)
        tokenizer.set_vocab(standard_luna_data, key=lambda x: x["ques_content"])
        tokenizer_size2 = len(tokenizer)
        assert tokenizer_size1 < tokenizer_size2
        tokenizer.save_pretrained(pretrained_dir)
        tokenizer = ElmoTokenizer.from_pretrained(pretrained_dir)
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

    def test_train_elmo(standard_luna_data, pretrained_model_dir):
        test_items = [
            {'ques_content': '有公式$\\FormFigureID{wrong1?}$和公式$\\FormFigureBase64{wrong2?}$，\
                    如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$,\
                    若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'},
            {'ques_content': '如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$, \
                    若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}
        ]
        train_elmo(
            standard_luna_data,
            pretrained_model_dir,
            data_params={
                "feature_key": "ques_content"
            },
            train_params={
                "num_train_epochs": 3,
                "per_device_train_batch_size": 2,
                "per_device_eval_batch_size": 2,
                "no_cuda": not TEST_GPU,
            }
        )
        model = ElmoLM.from_pretrained(pretrained_model_dir)
        tokenizer = ElmoTokenizer.from_pretrained(pretrained_model_dir)

        # TODO: need to handle inference for T2V for batch or single
        encodes = tokenizer(test_items[0], lambda x: x['ques_content'])
        model(**encodes)
        encodes = tokenizer(test_items, lambda x: x['ques_content'])
        model(**encodes)

    def test_train_pp(standard_luna_data, pretrained_model_dir, pretrained_pp_dir):
        test_items = [
            {'ques_content': '有公式$\\FormFigureID{wrong1?}$和公式$\\FormFigureBase64{wrong2?}$，\
                    如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$,\
                    若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'},
            {'ques_content': '如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$, \
                    若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}
        ]
        data_params = {
            "feature_key": "ques_content",
            "labal_key": "difficulty"
        }
        train_params = {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "no_cuda": not TEST_GPU,
        }
        train_elmo_for_property_prediction(
            standard_luna_data,
            pretrained_pp_dir,
            pretrained_dir=pretrained_model_dir,

            eval_items=standard_luna_data,
            train_params=train_params,
            data_params=data_params
        )
        model = ElmoLM.from_pretrained(pretrained_pp_dir)
        tokenizer = ElmoTokenizer.from_pretrained(pretrained_pp_dir)

        # TODO: need to handle inference for T2V for batch or single
        encodes = tokenizer(test_items[0], lambda x: x['ques_content'])
        model(**encodes)
        encodes = tokenizer(test_items, lambda x: x['ques_content'])
        model(**encodes)


# TODO: T2V test
# model = ElmoModel(pretrained_model_dir)
# assert model.vector_size > 0
# assert output.shape[-1] == model.vector_size
# t2v = T2V('elmo', pretrained_model_dir)
# assert t2v(inputs).shape[-1] == t2v.vector_size
# assert t2v.infer_vector(inputs).shape[-1] == t2v.vector_size


def test_elmo_i2v(standard_luna_data):
    pass


# def test_elmo_i2v(stem_data_elmo, tmpdir):
#     output_dir = str(tmpdir.mkdir('elmo_test'))
#     tokenizer = ElmoTokenizer()
#     train_text = [tokenizer.tokenize(item=data, freeze_vocab=False) for data in stem_data_elmo]
#     train_elmo(train_text, output_dir)
#     tokenizer_kwargs = {"path": os.path.join(output_dir, "vocab.json")}
#     i2v = Elmo('elmo', 'elmo', output_dir, tokenizer_kwargs=tokenizer_kwargs, pretrained_t2v=False)
#     item = {'stem': '如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$, \
#                 若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}
#     i_vec, t_vec = i2v(item['stem'])
#     assert len(i_vec) == i2v.vector_size
#     assert len(t_vec[0]) == i2v.vector_size
#     i_vec = i2v.infer_item_vector(item['stem'])
#     assert len(i_vec) == i2v.vector_size
#     t_vec = i2v.infer_token_vector(item['stem'])
#     assert len(t_vec[0]) == i2v.vector_size

#     i_vec, t_vec = i2v([item['stem'], item['stem'], item['stem']])
#     assert len(i_vec[0]) == i2v.vector_size
#     assert len(t_vec[0][0]) == i2v.vector_size


# TODO: pretrained_i2v_test
def test_pretrained_elmo_i2v(standard_luna_data,):
    pass
# def test_pretrained_elmo_i2v(stem_data_elmo, tmpdir):
#     output_dir = str(tmpdir.mkdir('elmo_test'))
#     i2v = get_pretrained_i2v("elmo_test", output_dir)
#     item = {'stem': '如图$\\FigureID{088f15ea-8b7c-11eb-897e-b46bfc50aa29}$, \
#                     若$x,y$满足约束条件$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$'}
#     i_vec, t_vec = i2v(item['stem'])
#     assert len(i_vec) == i2v.vector_size
#     assert len(t_vec[0]) == i2v.vector_size
#     i_vec = i2v.infer_item_vector(item['stem'])
#     assert len(i_vec) == i2v.vector_size
#     t_vec = i2v.infer_token_vector(item['stem'])
#     assert len(t_vec[0]) == i2v.vector_size
