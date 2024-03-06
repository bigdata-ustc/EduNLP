# 2024/3/5 @ yuheng

import pytest

from EduNLP.SIF.segment import seg


def test_ocr(figure0, figure1, figure0_base64, figure1_base64):
    seg(
        r"如图所示，则$\FormFigureID{0}$的面积是$\SIFBlank$。$\FigureID{1}$",
        figures={
            "0": figure0,
            "1": figure1
        },
        convert_image_to_latex=True
    )
    s = seg(
        r"如图所示，则$\FormFigureBase64{%s}$的面积是$\SIFBlank$。$\FigureBase64{%s}$" % (figure0_base64, figure1_base64),
        figures=True,
        convert_image_to_latex=True
    )
    with pytest.raises(TypeError):
        s.append("123")
    seg_test_text = seg(
        r"如图所示，有三组$\textf{机器人,bu}$在踢$\textf{足球,b}$",
        figures=True
    )
    assert seg_test_text.text_segments == ['如图所示，有三组机器人在踢足球']
