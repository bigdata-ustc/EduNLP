# coding: utf-8
# 2021/8/2 @ tongshiwei
import pytest
from EduNLP import get_pretrained_i2v
# from EduNLP.I2V.i2v import MODELS
from EduNLP.I2V import D2V, W2V


def test_pretrained_i2v(tmp_path):
    # PRETRAINED_MODELS["test_d2v"] = ["http://base.ustc.edu.cn/data/model_zoo/EduNLP/d2v/test_256.zip", "d2v"]
    # MODELS["d2v_test_256"] = [D2V, "d2v_test_256"]

    d = tmp_path / "model"
    d.mkdir()

    get_pretrained_i2v("d2v_test_256", d)

    with pytest.raises(KeyError):
        get_pretrained_i2v("error")

    # PRETRAINED_MODELS["test_w2v"] = ["http://base.ustc.edu.cn/data/model_zoo/EduNLP/w2v/test_w2v_256.zip", "w2v"]
    # MODELS["w2v_test_256"] = [W2V, "w2v_test_256"]

    get_pretrained_i2v("w2v_test_256", d)

    # get_pretrained_i2v("luna_bert", d)

    get_pretrained_i2v("tal_edu_bert", d)

    # get_pretrained_i2v("luna_pub_bert_math_base", d)
    with pytest.raises(KeyError):
        get_pretrained_i2v("error")
