# coding: utf-8
# 2021/3/8 @ tongshiwei

import matplotlib.pyplot as plt
from EduNLP.Formula import Formula
from EduNLP.Formula.viz import ForestPlotter

f = Formula(r"\frac{\sqrt{x^2}}{\pi} + 1 = y", variable_standardization=True)

ForestPlotter().export(
    f.ast, root_list=[node["val"]["id"] for node in f.element if node["structure"]["father"] is None],
)
plt.show()