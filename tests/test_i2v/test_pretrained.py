# coding: utf-8
# 2021/8/2 @ tongshiwei
import pytest
from EduNLP import get_pretrained_i2v
from EduNLP.Vector.t2v import PRETRAINED_MODELS
from EduNLP.I2V.i2v import MODELS
from EduNLP.I2V import D2V, W2V


def test_pretrained_i2v(tmp_path):
    PRETRAINED_MODELS["test_d2v"] = ["http://base.ustc.edu.cn/data/model_zoo/EduNLP/d2v/test_256.zip", "d2v"]
    MODELS["test_d2v"] = [D2V, "test_d2v"]

    d = tmp_path / "model"
    d.mkdir()

    get_pretrained_i2v("test_d2v", d)

    with pytest.raises(KeyError):
        get_pretrained_i2v("error")

    PRETRAINED_MODELS["test_w2v"] = ["http://base.ustc.edu.cn/data/model_zoo/EduNLP/w2v/test_w2v_256.zip", "w2v"]
    MODELS["test_w2v"] = [W2V, "test_w2v"]

    get_pretrained_i2v("test_w2v", d)

    with pytest.raises(KeyError):
        get_pretrained_i2v("error")
