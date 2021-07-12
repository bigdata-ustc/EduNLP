# coding: utf-8
# 2021/5/20 @ tongshiwei

from EduNLP.SIF import is_sif
from EduNLP.SIF import to_sif
from EduNLP.SIF import sif4sci
import pytest


def test_is_sif():
    text = '若$x,y$满足约束条件' \
           '$\\left\\{\\begin{array}{c}2 x+y-2 \\leq 0 \\\\ x-y-1 \\geq 0 \\\\ y+1 \\geq 0\\end{array}\\right.$，' \
           '则$z=x+7 y$的最大值$\\SIFUnderline$'
    assert is_sif(text) == 1

    text = '公式需要满足完整性，完整的公式如' \
           '$\\begin{matrix} a & b \\\\ c & d \\end{matrix}$' \
           '，不完整的公式如$\\begin{matrix} a & b \\\\ c & d$'
    with pytest.raises(ValueError):
        is_sif(text)

    text = '公式需要满足符合katex的支持性，可支持的公式如' \
           '$\\begin{matrix} a & b \\\\ c & d \\end{matrix}$' \
           '，不可支持的公式如$\\frac{ \\dddot y }{ x }$'
    with pytest.raises(ValueError):
        is_sif(text)


def test_to_sif():
    text = '某校一个课外学习小组为研究某作物的发芽率y和温度x（单位...'
    siftext = to_sif(text)
    print(siftext)


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
