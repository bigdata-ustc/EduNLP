# coding: utf-8
# 2021/5/20 @ tongshiwei

import pytest

import os
from PIL import Image
from EduNLP.SIF.segment import seg
from EduNLP.utils import abs_current_dir, path_append, image2base64


def test_segment():
    img_dir = os.path.abspath(path_append(abs_current_dir(__file__), "..", "..", "asset", "_static"))
    figure0 = Image.open(path_append(img_dir, "item_formula.png", to_str=True))
    figure1 = Image.open(path_append(img_dir, "item_figure.png", to_str=True))
    seg(
        r"如图所示，则$\FormFigureID{0}$的面积是$\SIFBlank$。$\FigureID{1}$",
        figures={
            "0": figure0,
            "1": figure1
        }
    )

    figure0_base64 = image2base64(figure0)
    figure1_base64 = image2base64(figure1)
    s = seg(
        r"如图所示，则$\FormFigureBase64{%s}$的面积是$\SIFBlank$。$\FigureBase64{%s}$" % (figure0_base64, figure1_base64),
        figures=True
    )

    with pytest.raises(TypeError):
        s.append("123")
