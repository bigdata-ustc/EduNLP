# coding: utf-8
# 2021/8/2 @ tongshiwei

import fire


from EduNLP.Vector.t2v import get_all_pretrained_models


def list_i2v():
    print("\n".join(get_all_pretrained_models()))


def cli():  # pragma: no cover
    fire.Fire({"i2v": list_i2v})
