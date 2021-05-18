# coding: utf-8
# 2021/5/18 @ tongshiwei

from EduNLP.SIF import sif4sci

item = r"集合$A={x \in R | |x - 2| \leq 5}$中最小整数位$\SIFBlank$"
# print(sif4sci(item, symbol="t", tokenization=False))
print(sif4sci(item, symbol="t"))
# print(sif4sci(item, symbol="fgm", tokenization=False))
print(sif4sci(item, symbol="fgm"))
print(sif4sci(item, symbol="gm", tokenization_params={"formula_params": {"method": "ast"}}))
print(sif4sci(item, tokenization_params={"formula_params": {"method": "ast"}}))
