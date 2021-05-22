# coding: utf-8
# 2021/5/20 @ tongshiwei
import pytest
from EduNLP.Formula import Formula, FormulaGroup


def test_formula():
    formula = r"x + x"
    f = Formula(formula)
    f.variable_standardization(inplace=False)
    f.variable_standardization(inplace=True)
    assert len(f.ast_graph.nodes) == len(f.ast)
    f.to_str()

    formula = r"\frac{\pi}{2}"
    f = Formula(formula, variable_standardization=True)
    assert repr(f) == r"<Formula: \frac{\pi}{2}>"

    f = Formula(f.ast)
    assert f.resetable is False
    with pytest.raises(TypeError):
        f.reset_ast()

    fg = FormulaGroup([r"x + x", r"x + \frac{\pi}{2}"], variable_standardization=True)
    for f in fg:
        assert f in fg
    assert len(fg[0].ast) == 3
    fg.to_str()

    fg = FormulaGroup(["x", "y", "x"])
    assert len(fg.ast) == 3

    with pytest.raises(TypeError):
        FormulaGroup([{}])
