# coding: utf-8
# 2021/5/18 @ tongshiwei

from .cut import linear_tokenize
import networkx as nx
from EduNLP.Formula import Formula


# def inorder_traversal(ast: nx.DiGraph):
#     visit = set()
#     nodes = []
#
#     def _inorder_traversal(_node):
#         if _node in visit:
#             return
#         successors = list(ast.successors(_node))
#         if successors:
#             if len(successors) <= 2:
#                 _inorder_traversal(successors[0])
#                 nodes.append(_node)
#                 visit.add(_node)
#                 if len(successors) == 2:
#                     _inorder_traversal(successors[1])
#             else:
#                 nodes.append(_node)
#                 for successor in successors:
#                     if successor in visit:
#                         continue
#                     _inorder_traversal(successor)
#         else:
#             nodes.append(_node)
#
#     for node in ast.nodes:
#         if node in visit or list(ast.predecessors(node)):
#             continue
#         _inorder_traversal(node)
#     return nodes


def ast_tokenize(formula, ord2token=False, var_numbering=False):
    tokens = []
    ast = Formula(formula).variable_standardization(inplace=True).ast
    for i in nx.dfs_postorder_nodes(ast):
        node = ast.nodes[i]
        if ord2token is True and node["type"] in ["mathord", "textord"]:
            if node["type"] == "mathord" and var_numbering is True:
                tokens.append("%s_%s" % (node["type"], node.get("var", "con")))
            else:
                tokens.append(node["type"])
        else:
            tokens.append(node["text"])
    return tokens


def tokenize(formula, method="ast", errors="raise", **kwargs):
    """

    Parameters
    ----------
    formula
    method
    errors: how to handle the exception occurs in ast tokenize
        "coerce": use linear_tokenize
        "raise": raise exception
    kwargs

    Returns
    -------

    """
    if method == "linear":
        return linear_tokenize(formula, **kwargs)
    elif method == "ast":
        try:
            return ast_tokenize(formula, **kwargs)
        except TypeError as e:
            if errors == "coerce":
                linear_tokenize(formula)
            else:
                raise e
    else:
        raise TypeError("Unknown method type: %s" % method)

# print(Formula(r"x^\frac{1}{2}"))
# print(Formula(r"x + y"))
# print(Formula(r"{x + y}"))
# ast_nodes = Formula(r"{x + y}^\frac{1}{2} + 1 = 0").variable_standardization(inplace=True).ast.nodes
# for i in ast_nodes:
#     print(ast_nodes[i])
# print(list(nx.dfs_tree(Formula(r"x^\frac{1}{2}").ast)))
# print(ast_tokenize(r"{x + y}^\frac{\pi}{2} + 1 = 0"))
# print(ast_tokenize(r"{x + y}^\frac{\pi}{2} + 1 = 0", ord2token=True))
# print(ast_tokenize(r"{x + y}^\frac{\pi}{2} + 1 = 0", ord2token=True, var_numbering=True))
