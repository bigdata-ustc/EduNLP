# coding: utf-8
# 2021/8/2 @ tongshiwei
import pytest
from EduNLP import get_pretrained_i2v
# from EduNLP.I2V.i2v import MODELS
from EduNLP.I2V import D2V, W2V
from EduNLP.Vector import get_pretrained_model_info, get_all_pretrained_models


def test_pretrained_i2v(tmp_path):

    d = tmp_path / "model"
    d.mkdir()

    url, t2v_name = get_pretrained_model_info("d2v_test_256")
    assert url != ""
    assert t2v_name == "d2v"
    model_names = get_all_pretrained_models()
    assert "d2v_test_256" in model_names

    get_pretrained_i2v("d2v_test_256", d)

    with pytest.raises(KeyError):
        get_pretrained_i2v("error")

    get_pretrained_i2v("w2v_test_256", d)

#     get_pretrained_i2v("quesnet_test_256", d)

#     get_pretrained_i2v("elmo_test", d)

#     # get_pretrained_i2v("tal_edu_bert", d)
