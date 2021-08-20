# coding: utf-8
# 2021/3/8 @ tongshiwei
#
from EduNLP.Formula import Formula, FormulaGroup, link_formulas
#
# f1 = Formula(r"x + y", variable_standardization=True)
# f2 = Formula(r"y + x", variable_standardization=True)
# f3 = Formula(r"z + y", variable_standardization=True)
#
# print(f1.element)
# print(f2.element)
# print(f3.element)
#
# print("-----------------------")
#
# link_formulas(f1, f2, f3)
#
# print("------------------------")
#
# print(f1.element)
# print(f2.element)
# print(f3.element)
#
# print("---------------------")
#
# fg = FormulaGroup(
#     [r"x + y", r"y + x", r"y + z"]
# )
# for f in fg:
#     print(f.element)

# fg = FormulaGroup(["x", "y", "x"])
# print(fg.elements)

fg = FormulaGroup(["x", Formula("y"), "x"])
print(fg.elements)