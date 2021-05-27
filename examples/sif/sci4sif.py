# coding: utf-8
# 2021/5/18 @ tongshiwei

from EduNLP.SIF import sif4sci, link_formulas

# item = r"若集合$A=\{x \in R | |x - 2| \leq 5\}$，则$A$中最小整数位是$\SIFChoice$"
# print(item)
# print(sif4sci(item, symbol="fgm", tokenization=False))
# print(sif4sci(item, symbol="fgm", tokenization=True))
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

e = r"$x$ 是 $y$ 那么 $y$ 和 $z$ 是什么 $x,y,z$"
# print(sif4sci(e, symbol="gm",
#               tokenization_params={
#                   "formula_params": {
#                       "method": "ast", "return_type": "list", "ord2token": True, "var_numbering": True,
#                   }
#               }))
#
# test_item_1 = [r"$x < y$", r"$y = x$", r"$y < x$"]
# tls = [
#     sif4sci(e, symbol="gm",
#             tokenization_params={
#                 "formula_params": {
#                     "method": "ast", "return_type": "list", "ord2token": True, "var_numbering": True,
#                 }
#             })
#     for e in test_item_1
# ]
# link_formulas(*tls)
# print(tls)
seg = sif4sci(e, tokenization=False)
with seg.filter(keep="t"):
    print(seg)
