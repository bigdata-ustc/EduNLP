# coding: utf-8
# 2021/8/2 @ tongshiwei
import pytest
from EduNLP import get_pretrained_i2v
# from EduNLP.I2V.i2v import MODELS
from EduNLP.I2V import D2V, W2V


def test_pretrained_i2v(tmp_path):

    d = tmp_path / "model"
    d.mkdir()

    get_pretrained_i2v("d2v_test_256", d)

    with pytest.raises(KeyError):
        get_pretrained_i2v("error")

    get_pretrained_i2v("w2v_test_256", d)

    get_pretrained_i2v("quesnet_test_256", d)

    get_pretrained_i2v("elmo_test", d)

    # get_pretrained_i2v("tal_edu_bert", d)

    with pytest.raises(KeyError):
        get_pretrained_i2v("error")
