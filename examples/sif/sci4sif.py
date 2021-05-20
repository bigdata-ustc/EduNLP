# coding: utf-8
# 2021/5/18 @ tongshiwei

from EduNLP.SIF import sif4sci

item = r"若集合$A=\{x \in R | |x - 2| \leq 5\}$，则$A$中最小整数位是$\SIFChoice$"
print(item)
# print(sif4sci(item, symbol="fgm", tokenization=False))
print(sif4sci(item, symbol="fgm", tokenization=True))
# print(sif4sci(item, symbol="t"))
# print(sif4sci(item, symbol="fgm", tokenization=False))
# print(sif4sci(item, symbol="fgm"))
# print(sif4sci(item, symbol="gm", tokenization_params={"formula_params": {"method": "ast"}}))
# print(sif4sci(item, symbol="gm", tokenization_params={"formula_params": {"method": "linear"}}))
# print(sif4sci(item, tokenization_params={"formula_params": {"method": "ast", "ord2token": True}}))
# print(
#     sif4sci(item, tokenization_params={"formula_params": {"method": "ast", "ord2token": True, "var_numbering": True}}))
# print(sif4sci(item, tokenization_params={"formula_params": {"method": "ast", "return_type": "list"}}))
# print(
#     sif4sci(item, tokenization_params={"formula_params": {"method": "ast", "ord2token": True, "return_type": "list"}}).formula_tokens
# )
# print(
#     sif4sci(item, tokenization_params={
#         "formula_params": {"method": "ast", "ord2token": True, "var_numbering": True, "return_type": "list"}})
# )
# print(sif4sci(item, tokenization_params={"formula_params": {"method": "ast", "return_type": "ast"}}))
# print(sif4sci(item, tokenization_params={"formula_params": {"method": "ast", "return_type": "formula"}}))
