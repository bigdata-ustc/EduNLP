# coding: utf-8
# 2021/5/20 @ tongshiwei

import pytest

from EduNLP.SIF.segment import seg
from EduNLP.utils import image2base64


def test_segment(figure0, figure1, figure0_base64, figure1_base64):
    seg(
        r"如图所示，则$\FormFigureID{0}$的面积是$\SIFBlank$。$\FigureID{1}$",
        figures={
            "0": figure0,
            "1": figure1
        }
    )
    s = seg(
        r"如图所示，则$\FormFigureBase64{%s}$的面积是$\SIFBlank$。$\FigureBase64{%s}$" % (figure0_base64, figure1_base64),
        figures=True
    )

    with pytest.raises(TypeError):
        s.append("123")
