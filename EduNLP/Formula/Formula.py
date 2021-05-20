# coding: utf-8
# 2021/3/8 @ tongshiwei

from pprint import pformat
from typing import List
import networkx as nx
from copy import deepcopy

from .ast import str2ast, get_edges, ast, link_variable

CONST_MATHORD = {r"\pi"}

__all__ = ["Formula", "FormulaGroup", "CONST_MATHORD"]


class Formula(object):
    def __init__(self, formula, is_str=True, variable_standardization=False, const_mathord=None):
        self._formula = formula
        self._ast = str2ast(formula) if is_str else formula
        if variable_standardization:
            const_mathord = CONST_MATHORD if const_mathord is None else const_mathord
            self.variable_standardization(inplace=True, const_mathord=const_mathord)

    def variable_standardization(self, inplace=False, const_mathord=None):
        const_mathord = const_mathord if const_mathord is not None else CONST_MATHORD
        ast_tree = self._ast if inplace else deepcopy(self._ast)
        variables = {}
        index = 0
        for node in ast_tree:
            if node["val"]["type"] == "mathord":
                var = node["val"]["text"]
                if var in const_mathord:
                    continue
                else:
                    if var not in variables:
                        variables[var], index = index, index + 1
                    node["val"]["var"] = variables[var]
        if inplace:
            return self
        else:
            return Formula(ast_tree, is_str=False)

    @property
    def element(self):
        return self._ast

    @property
    def ast(self) -> (nx.Graph, nx.DiGraph):
        edges = [(edge[0], edge[1]) for edge in get_edges(self._ast) if edge[2] == 3]
        tree = nx.DiGraph()
        for node in self._ast:
            tree.add_node(
                node["val"]["id"],
                **node["val"]
            )
        tree.add_edges_from(edges)
        return tree

    def to_str(self):
        return pformat(self._ast)

    def __repr__(self):
        return "<Formula: %s>" % self._formula


class FormulaGroup(object):
    def __init__(self, formula_list: List[str], variable_standardization=False, const_mathord=None):
        """

        Parameters
        ----------
        formula_list: List[str]
        """
        forest_begin = 0
        forest = []
        formula_sep_index = []
        for index in range(0, len(formula_list)):
            formula_sep_index.append(forest_begin)
            tree = ast(
                formula_list[index],
                forest_begin=forest_begin,
                is_str=True
            )
            forest_begin += len(tree)
            forest += tree
        else:
            formula_sep_index.append(len(forest))
        forest = link_variable(forest)
        self._forest = forest
        self._formulas = []
        for i, sep in enumerate(formula_sep_index[:-1]):
            self._formulas.append(Formula(forest[sep: formula_sep_index[i + 1]], is_str=False))
        if variable_standardization:
            self.variable_standardization(inplace=True, const_mathord=const_mathord)

    def __iter__(self):
        return iter(self._formulas)

    def __getitem__(self, item) -> Formula:
        return self._formulas[item]

    def __contains__(self, item) -> bool:
        return item in self._formulas

    def variable_standardization(self, inplace=False, const_mathord=None):
        ret = []
        for formula in self._formulas:
            ret.append(formula.variable_standardization(inplace=inplace, const_mathord=const_mathord))
        return ret

    def to_str(self):
        return pformat(self._formulas)

    def __repr__(self):
        return "<FormulaGroup: %s>" % ";".join([repr(_formula) for _formula in self._formulas])
