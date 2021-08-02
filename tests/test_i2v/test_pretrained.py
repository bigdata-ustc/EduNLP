# coding: utf-8
# 2021/8/2 @ tongshiwei
import pytest
from EduNLP import get_pretrained_i2v
from EduNLP.Vector.t2v import PRETRAINED_MODELS
from EduNLP.I2V.i2v import MODELS
from EduNLP.I2V import D2V


def test_pretrained_i2v(tmp_path):
    PRETRAINED_MODELS["test"] = ["http://base.ustc.edu.cn/data/model_zoo/EduNLP/d2v/test_256.zip", "d2v"]
    MODELS["test"] = [D2V, "test"]

    d = tmp_path / "model"
    d.mkdir()

    get_pretrained_i2v("test", d)

    with pytest.raises(KeyError):
        get_pretrained_i2v("error")

    get_pretrained_i2v("test", d)
