# coding: utf-8
# 2021/5/20 @ tongshiwei


from EduNLP.SIF import sif4sci


def test_sci4sif(figure0, figure1, figure0_base64, figure1_base64):
    repr(sif4sci(
        r"如图所示，则$\bigtriangleup ABC$的面积是$\SIFBlank$。$\FigureID{1}$",
        tokenization_params={
            "formula_params": {
                "method": "ast",
                "return_type": "ast"
            }
        }
    ))
    repr(sif4sci(
        r"如图所示，则$\FormFigureID{0}$的面积是$\SIFBlank$。$\FigureID{1}$",
        figures={
            "0": figure0,
            "1": figure1
        },
    ))
    repr(sif4sci(
        item=r"如图所示，则$\FormFigureBase64{%s}$的面积是$\SIFBlank$。$\FigureBase64{%s}$" % (
            figure0_base64, figure1_base64
        ),
        tokenization_params={
            "figure_params": {"figure_instance": True}
        }
    ))
