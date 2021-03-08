# coding: utf-8
# 2021/3/8 @ tongshiwei

from pprint import pformat
from typing import List, Dict
import networkx as nx
from copy import deepcopy

from .watex import watex

CONST_MATHORD = {r"\pi"}


def str2ast(formula: str):
    return ast(formula, is_str=True)


def ast(formula: (str, List[Dict]), index=0, forest_begin=0, father_tree=None, is_str=False):
    """
    The origin code author is https://github.com/hxwujinze

    Parameters
    ----------
    formula: str or List[Dict]

    index: int
        本子树在树上的位置
    forest_begin: int
        本树在森林中的起始位置
    father_tree: List[Dict]
        父亲树
    is_str: bool


    Returns
    ----------
    tree: List[Dict]
        重新解析形成的特征树

    """
    tree = []
    index += forest_begin
    json_ast: List[Dict] = watex.katex.__parse(formula).to_list() if is_str else formula
    last_node = None

    for item in json_ast:
        private_index = len(tree)
        role = None
        if 'role' in item:
            role = item['role']
        tree_node = {
            'val': {'id': private_index + index, 'type': None, 'text': None, 'role': role},
            'structure': {'bro': [None, None], 'child': None, 'father': None, 'forest': None}
        }

        tree_node['val']['type'] = item['type']

        if index > forest_begin:
            tree_node['structure']['father'] = index - 1
        if tree_node['val']['type'] == "mathord" or tree_node['val']['type'] == "textord":
            tree_node['val']['text'] = item['text'].replace('\\prime', '’')
            tree.append(tree_node)
        elif tree_node['val']['type'] == "atom":
            tree_node['val']['text'] = item['text'].replace('\\cdotp', '·')
            tree_node['val']['type'] = item['family']
            tree.append(tree_node)

        elif tree_node['val']['type'] == "op":
            tree_node['val']['text'] = item['name']
            tree.append(tree_node)

        elif tree_node['val']['type'] == "genfrac":
            item['numer']['role'] = 'numer'
            item['denom']['role'] = 'denom'
            tree_node['val']['text'] = '\\frac'
            tree_node['structure']['child'] = [1 + private_index + index]
            tree.append(tree_node)
            tree += ast([item['numer'], item['denom']], index=len(tree) + index, father_tree=tree)

        elif tree_node['val']['type'] == 'sqrt':
            tree_node['structure']['child'] = [1 + private_index + index]
            tree_node['val']['text'] = '\\sqrt'
            tree.append(tree_node)
            item['body']['role'] = 'body'
            if item['index']:
                item['index']['role'] = 'index'
                tree += ast([item['body'], item['index']], index=len(tree) + index, father_tree=tree)
            else:
                tree += ast([item['body']], index=len(tree) + index, father_tree=tree)

        elif tree_node['val']['type'] == 'array':
            tree_node['structure']['child'] = [1 + private_index + index]
            tree_node['val']['arraydims'] = item['arraystretch']
            tree_node['val']['text'] = '\\begin {matrix} \\end {matrix}'
            tree.append(tree_node)
            bodys = []
            for litem in item['body']:
                for citem in litem:
                    citem['role'] = 'body'
                    bodys.append(citem)
            tree += ast(bodys, index=len(tree) + index, father_tree=tree)


        elif tree_node['val']['type'] == 'styling':
            tree_node['structure']['child'] = [1 + private_index + index]
            tree.append(tree_node)
            for citem in item['body']:
                citem['role'] = 'body'
            tree += ast(item['body'], index=len(tree) + index, father_tree=tree)

        elif tree_node['val']['type'] == 'xArrow':
            tree_node['structure']['child'] = [1 + private_index + index]
            tree_node['val']['text'] = item['label']
            tree.append(tree_node)
            item['body']['role'] = 'body'
            item['below']['role'] = 'below'
            tree += ast([item['body'], item['below']], index=len(tree) + index, father_tree=tree)

        elif tree_node['val']['type'] == 'overline':
            tree_node['structure']['child'] = [1 + private_index + index]
            tree_node['val']['text'] = '\\' + tree_node['val']['type']
            tree.append(tree_node)
            item['body']['role'] = 'body'
            tree += ast([item['body']], index=len(tree) + index, father_tree=tree)

        elif tree_node['val']['type'] == "accent":

            tree_node['val']['text'] = item['label']
            tree_node['structure']['child'] = [1 + private_index + index]
            tree.append(tree_node)
            item['base']['role'] = 'base'
            tree += ast([item['base']], index=len(tree) + index, father_tree=tree)

        elif tree_node['val']['type'] == "supsub":
            item['base']['role'] = 'base'

            if 'sup' in item and item['sup']:
                bp = 'sup'
                bptext = '^'
            else:
                bp = 'sub'
                bptext = '_'

            if 'text' in item:
                bptext = item['text']

            tree_node['val']['text'] = bptext
            item[bp]['role'] = bp
            tree_node['structure']['child'] = [1 + private_index + index]
            tree.append(tree_node)
            tree += ast([item['base'], item[bp]], index=len(tree) + index, father_tree=tree)

        elif tree_node['val']['type'] == "ordgroup":
            tree_node['structure']['child'] = [1 + private_index + index]
            tree_node['val']['text'] = '{ }'
            tree.append(tree_node)
            for citem in item['body']:
                citem['role'] = 'body'
            tree += ast(item['body'], index=len(tree) + index, father_tree=tree)

        elif tree_node['val']['type'] == "mclass":
            for citem in item['body']:
                citem['role'] = 'body'
            tree += ast(item['body'], index=len(tree) + index, father_tree=tree)

        elif tree_node['val']['type'] == 'leftright':
            tree_node['structure']['child'] = [1 + private_index + index]
            tree_node['val']['text'] = item['left']
            tree_node['val']['right'] = item['right']
            tree.append(tree_node)
            for citem in item['body']:
                citem['role'] = 'body'
            tree += ast(item['body'], index=len(tree) + index, father_tree=tree)

        else:
            tree_node['structure']['child'] = [1 + private_index + index]

            if "text" in item:
                tree_node['val']['text'] = item["text"]
            tree_node['val']['type'] = "other"
            tree.append(tree_node)
            Role = ['body', 'base', 'sup', 'sub', 'numer', 'denom', 'index', 'blew', 'other']
            childrole = []

            for role_item in Role:
                if role_item in item:
                    item[role_item]['role'] = role_item
                    childrole.append(item[role_item])
            tree += ast(childrole, index=len(tree) + index, father_tree=tree)
        if item:
            if item != json_ast[0]:
                tree[private_index]['structure']['bro'][0] = last_node + index
                last_node = private_index
            else:
                last_node = private_index
            if item != json_ast[-1]:
                tree[private_index]['structure']['bro'][1] = len(tree) + index
                if father_tree:
                    father_tree[tree[private_index]['structure']['father'] - index]['structure']['child'].append(
                        len(tree) + index
                    )

    return tree


def link_variable(forest):
    """
    建森林

    Parameters
    ----------
    forest: List[Dict]

    Returns
    -------
    trees: List[Dict]

    """
    forest_connect_dict = {}
    for node in forest:
        if node['val']['type'] == 'mathord':
            if node['val']['text'] not in forest_connect_dict:
                forest_connect_dict[node['val']['text']] = []
            forest_connect_dict[node['val']['text']].append(forest.index(node))

    for k, v in forest_connect_dict.items():
        if len(v) > 1:
            for i in range(0, len(v)):
                l_v = [] + v
                index = l_v.pop(i)
                forest[index]['structure']['forest'] = l_v

    return forest


def get_edges(forest):
    """
    构造边集合

    Parameters
    ----------
    forest: List[Dict]
        森林

    Returns
    ----------
    edges: list of tuple(src,dst,type)
        边集合
    """
    edges = []
    for node in forest:
        index = forest.index(node)
        edges.append((index, index, 1))
        if node['structure']['bro'][1] is not None:
            edges.append((index, node['structure']['bro'][1], 2))
        if node['structure']['bro'][0] is not None:
            edges.append((index, node['structure']['bro'][0], 2))
        if node['structure']['child'] is not None:
            for item in node['structure']['child']:
                edges.append((index, item, 3))
        if node['structure']['father'] is not None:
            edges.append((index, node['structure']['father'], 4))
        if node['structure']['forest'] is not None:
            for item in node['structure']['forest']:
                edges.append((index, item, 5))
    return edges


class Formula(object):
    def __init__(self, formula, is_str=True, variable_standardization=False, const_mathord=None):
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

    def __repr__(self):
        return pformat(self._ast)


class FormulaGroup(object):
    def __init__(self, formula_list, variable_standardization=False, const_mathord=None):
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

    def __next__(self) -> Formula:
        return next(self._formulas)

    def __getitem__(self, item) -> Formula:
        return self._formulas[item]

    def __contains__(self, item) -> bool:
        return item in self._formulas

    def variable_standardization(self, inplace=False, const_mathord=None):
        ret = []
        for formula in self._formulas:
            ret.append(formula.variable_standardization(inplace=inplace, const_mathord=const_mathord))
        return ret

    def __repr__(self):
        return pformat(self._formulas)
