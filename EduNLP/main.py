# coding: utf-8
# 2021/8/2 @ tongshiwei

import fire

from EduNLP.I2V.i2v import MODELS


def list_i2v():
    print("\n".join(MODELS.keys()))


def cli():  # pragma: no cover
    fire.Fire({"i2v": list_i2v})
