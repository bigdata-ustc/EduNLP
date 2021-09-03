# coding: utf-8
# 2021/8/2 @ tongshiwei
import pytest
from EduNLP import get_pretrained_i2v
from EduNLP.Vector.t2v import PRETRAINED_MODELS
from EduNLP.I2V.i2v import MODELS
from EduNLP.I2V import D2V, W2V
import os


def test_pretrained_i2v(tmp_path):
    PRETRAINED_MODELS["test_d2v"] = ["http://base.ustc.edu.cn/data/model_zoo/EduNLP/d2v/test_256.zip", "d2v"]
    MODELS["test_d2v"] = [D2V, "test_d2v"]

    d = tmp_path / "model"
    d.mkdir()

    get_pretrained_i2v(name="test_d2v", model_dir=d)
    with pytest.raises(KeyError):
        get_pretrained_i2v(name="error")

    model_path = os.path.join(d, "test_256/test_256.bin")
    get_pretrained_i2v(local_type="d2v", local_path=model_path, source="local")

    with pytest.raises(KeyError):
        get_pretrained_i2v(local_type="error", local_path=model_path, source="local")

    with pytest.raises(ValueError):
        get_pretrained_i2v(source="local")

    PRETRAINED_MODELS["test_w2v"] = ["http://base.ustc.edu.cn/data/model_zoo/EduNLP/w2v/test_w2v_256.zip", "w2v"]
    MODELS["test_w2v"] = [W2V, "test_w2v"]

    get_pretrained_i2v(name="test_w2v", model_dir=d)
    with pytest.raises(KeyError):
        get_pretrained_i2v(name="error")

    model_path = os.path.join(d, "test_w2v_256/test_w2v_256.kv")
    get_pretrained_i2v(local_type="w2v", local_path=model_path, source="local")

    with pytest.raises(KeyError):
        get_pretrained_i2v(local_type="error", local_path=model_path, source="local")

    with pytest.raises(ValueError):
        get_pretrained_i2v(source="local")
