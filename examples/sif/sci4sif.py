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

# e = r"$x$ 是 $y$ 那么 $y$ 和 $z$ 是什么 $x,y,z$"
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
# seg = sif4sci(e, tokenization=False)
# with seg.filter(keep="t"):
#     print(seg)
# e = r'某校一个课外学习小组为研究某作物的发芽率y和温度x（单位：$^{\circ} \mathrm{C}$）的关系，在20个不同温度条件下进行种子发芽实验，由实验数据$\left(x_{i}, y_{i}\right)(i=1,2, \cdots, 20)$得到下面的散点图：由此散点图，在10$^{\circ} \mathrm{C}$至40$^{\circ} \mathrm{C}$之间，下面四个回归方程类型中最适宜作为发芽率y和温度x的回归方程类型的是$\FigureID{3bf20b91-8af1-11eb-86ff-b46bfc50aa29}$$\FigureID{59b851d3-8af1-11eb-bd45-b46bfc50aa29}$$\FigureID{6310d375-8b75-11eb-bf70-b46bfc50aa29}$$\FigureID{6a006175-8b76-11eb-aa57-b46bfc50aa29}$$\FigureID{088f15e7-8b7c-11eb-a8aa-b46bfc50aa29}$'
# # e = r"$x$ 是 $y$ 那么 $y$ 和 $z$ 是什么 $x,y,z$"

# e = r'已知集合$A=\left\{x \mid x^{2}-3 x-4<0\right\}, \quad B=\{-4,1,3,5\}, \quad$ 则 $A \cap B=$'

from EduNLP.utils import dict2str4sif

test_item_1 = {
    "stem": "已知集合$A=\\left\\{x \\mid x^{2}-3 x-4<0\\right\\}, \\quad B=\\{-4,1,3,5\\}, \\quad$ 则 $A \\cap B=$",
    "options": ['\\{-4,1\\}', '\\{1,5\\}', '\\{3,5\\}', '\\{1,3\\}'],
}
e = dict2str4sif(test_item_1, tag_mode="head", add_list_no_tag=False)
seg = sif4sci(
    e,
    symbol="gm",
    tokenization_params={
        "formula_params": {
            "method": "ast", "return_type": "list", "ord2token": True
        }
    },
    errors="raise"
)
print(seg.get_segments())
#
# import json
# from tqdm import tqdm
#
#
# def load_items():
#     with open("../../data/OpenLUNA.json", encoding="utf-8") as f:
#         for line in f:
#             yield json.loads(line)
#
#
# from EduNLP.SIF import sif4sci
#
# sif_items = []
# for i, item in tqdm(enumerate(load_items()), "sifing"):
#     if i > 100:
#         break
#     sif_item = sif4sci(
#         item["stem"],
#         symbol="gm",
#         tokenization_params={"formula_params": {
#             "method": "ast",
#             "return_type": "list",
#             "ord2token": True,
#         }}
#     )
#     if sif_item:
#         sif_items.append(sif_item.tokens)
