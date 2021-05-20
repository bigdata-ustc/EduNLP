# coding: utf-8
# 2021/5/20 @ tongshiwei

from EduNLP.Formula import Formula, FormulaGroup


def test_formula():
    formula = r"x + x"
    f = Formula(formula)
    f.variable_standardization(inplace=False)
    f.variable_standardization(inplace=True)
    assert len(f.ast.nodes) == len(f.element)
    f.to_str()

    formula = r"\frac{\pi}{2}"
    f = Formula(formula, variable_standardization=True)
    assert repr(f) == r"<Formula: \frac{\pi}{2}>"

    fg = FormulaGroup([r"x + x", r"x + \frac{\pi}{2}"], variable_standardization=True)
    for f in fg:
        assert f in fg
    assert len(fg[0].element) == 3
    fg.to_str()
