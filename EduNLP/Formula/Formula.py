# coding: utf-8
# 2021/3/8 @ tongshiwei

from pprint import pformat
from typing import List, Dict
import networkx as nx
from copy import deepcopy

from .ast import str2ast, get_edges, link_variable

CONST_MATHORD = {r"\pi"}

__all__ = ["Formula", "FormulaGroup", "CONST_MATHORD", "link_formulas"]


class Formula(object):
    def __init__(self, formula: (str, List[Dict]), variable_standardization=False, const_mathord=None,
                 *args, **kwargs):
        self._formula = formula
        self._ast = None
        self.reset_ast(
            formula_ensure_str=False,
            variable_standardization=variable_standardization,
            const_mathord=const_mathord, *args, **kwargs
        )

    def variable_standardization(self, inplace=False, const_mathord=None, variable_connect_dict=None):
        const_mathord = const_mathord if const_mathord is not None else CONST_MATHORD
        ast_tree = self._ast if inplace else deepcopy(self._ast)
        var_code = variable_connect_dict["var_code"] if variable_connect_dict is not None else {}
        for node in ast_tree:
            if node["val"]["type"] == "mathord":
                var = node["val"]["text"]
                if var in const_mathord:
                    continue
                else:
                    if var not in var_code:
                        var_code[var] = len(var_code)
                    node["val"]["var"] = var_code[var]
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
        if isinstance(self._formula, str):
            return "<Formula: %s>" % self._formula
        else:
            return super(Formula, self).__repr__()

    def reset_ast(self, formula_ensure_str=True, variable_standardization=False, const_mathord=None, *args, **kwargs):
        if formula_ensure_str is True and self.resetable is True:
            raise TypeError("formula must be str, now is %s" % type(self._formula))
        self._ast = str2ast(self._formula, *args, **kwargs) if isinstance(self._formula, str) else self._formula
        if variable_standardization:
            const_mathord = CONST_MATHORD if const_mathord is None else const_mathord
            self.variable_standardization(inplace=True, const_mathord=const_mathord)
        return self._ast

    @property
    def resetable(self):
        return isinstance(self._formula, str)


class FormulaGroup(object):
    def __init__(self,
                 formula_list: (List[(str, Formula, dict)]),
                 variable_standardization=False,
                 const_mathord=None,
                 detach=True
                 ):
        """

        Parameters
        ----------
        formula_list: List[str]
        """
        forest = []
        self._formulas = []
        for index in range(0, len(formula_list)):
            formula = formula_list[index]
            if isinstance(formula, str):
                tree = str2ast(
                    formula,
                    forest_begin=len(forest),
                )
                self._formulas.append(Formula(tree))
            elif isinstance(formula, Formula):
                if detach:
                    formula = deepcopy(formula)
                tree = formula.reset_ast(
                    formula_ensure_str=True,
                    variable_standardization=False,
                )
                self._formulas.append(formula)
            else:
                raise TypeError(
                    "the element in formula_list should be either str or Formula, now is %s" % type(Formula)
                )
            forest += tree
        variable_connect_dict = link_variable(forest)
        self._forest = forest
        if variable_standardization:
            self.variable_standardization(
                inplace=True,
                const_mathord=const_mathord,
                variable_connect_dict=variable_connect_dict
            )

    def __iter__(self):
        return iter(self._formulas)

    def __getitem__(self, item) -> Formula:
        return self._formulas[item]

    def __contains__(self, item) -> bool:
        return item in self._formulas

    def variable_standardization(self, inplace=False, const_mathord=None, variable_connect_dict=None):
        ret = []
        for formula in self._formulas:
            ret.append(formula.variable_standardization(inplace=inplace, const_mathord=const_mathord,
                                                        variable_connect_dict=variable_connect_dict))
        return ret

    def to_str(self):
        return pformat(self._forest)

    def __repr__(self):
        return "<FormulaGroup: %s>" % ";".join([repr(_formula) for _formula in self._formulas])

    @property
    def element(self):
        return self._forest

    @property
    def ast(self) -> (nx.Graph, nx.DiGraph):
        edges = [(edge[0], edge[1]) for edge in get_edges(self._forest) if edge[2] == 3]
        tree = nx.DiGraph()
        for node in self._forest:
            tree.add_node(
                node["val"]["id"],
                **node["val"]
            )
        tree.add_edges_from(edges)
        return tree


def link_formulas(*formula: Formula, **kwargs):
    forest = []
    for form in formula:
        forest += form.reset_ast(
            forest_begin=len(forest),
            **kwargs
        )
    variable_connect_dict = link_variable(forest)
    for form in formula:
        form.variable_standardization(inplace=True, variable_connect_dict=variable_connect_dict, **kwargs)
