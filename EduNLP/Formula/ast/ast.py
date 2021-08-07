# coding: utf-8
# 2021/5/20 @ tongshiwei
from typing import List, Dict
from .katex import katex


__all__ = ["str2ast", "get_edges", "ast", "link_variable", "katex_parse"]


def katex_parse(formula):
    return katex.katex.__parse(formula,{'displayMode':True,'trust': True}).to_list()


def str2ast(formula: str, *args, **kwargs):
    return ast(formula, is_str=True, *args, **kwargs)


def ast(formula: (str, List[Dict]), index=0, forest_begin=0, father_tree=None, is_str=False):
    """
    The origin code author is https://github.com/hxwujinze

    Parameters
    ----------
    formula: str or List[Dict]
        公式字符串或通过katex解析得到的结构体
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

    todo: finish all types

    Notes
    ----------
    Some functions are not supportd in ``katex``
    e.g.,

    1. tag
        - ``\\begin{equation} \\tag{tagName} F=ma \\end{equation}``
        - ``\\begin{align} \\tag{1} y=x+z \\end{align}``
        - ``\\tag*{hi} x+y^{2x}``
    2. dddot
        - ``\\frac{ \\dddot y }{ x }``

    For more information, refer to
    `katex support table <https://github.com/KaTeX/KaTeX/blob/master/docs/support_table.md>`_
    """
    tree = []
    index += forest_begin
    json_ast: List[Dict] = katex.katex.__parse(formula,{'displayMode':True,'trust': True}).to_list() if is_str else formula
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
            tree_node['val']['text'] = "\\op" if 'name' not in item else item['name']
            if item['symbol'] and 'body' in item:
                tree_node['structure']['child'] = [1 + private_index + index]
                tree.append(tree_node)
                tree += ast(item['body'], index=len(tree) + index, father_tree=tree)
            else:
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
            _tree = []
            if 'base' in item and item['base'] is not None:
                item['base']['role'] = 'base'
                _tree.append(item['base'])
            if 'sub' in item and item['sub']:
                item['sub']['role'] = 'sub'
                _tree.append(item['sub'])
            if 'sup' in item and item['sup']:
                item['sup']['role'] = 'sup'
                _tree.append(item['sup'])

            tree_node['val']['text'] = "\\supsub"
            if _tree != []:
                tree_node['structure']['child'] = [1 + private_index + index]
                tree.append(tree_node)
                tree += ast(_tree, index=len(tree) + index, father_tree=tree)
            else:
                tree.append(tree_node)

        elif tree_node['val']['type'] == "ordgroup":
            tree_node['structure']['child'] = [1 + private_index + index]
            tree_node['val']['text'] = '{ }'
            tree.append(tree_node)
            for citem in item['body']:
                citem['role'] = 'body'
            tree += ast(item['body'], index=len(tree) + index, father_tree=tree)

        elif tree_node['val']['type'] == "mclass":
            tree_node['val']['text'] = item['mclass']
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

        elif tree_node['val']['type'] in {"kern"}:
            # \quad
            tree_node['val']['text'] = tree_node['val']['type']
            tree_node['val']['type'] = "ignore"
            tree.append(tree_node)

        elif tree_node['val']['type'] == "text":
            # \text{}
            tree_node['val']['text'] = "".join([e['text'] for e in item["body"]])
            tree.append(tree_node)
        # --------------------- new node --------------------- # 
        elif tree_node['val']['type'] == "size":
            # nknown usage : different from "sizing"
            continue
        elif tree_node['val']['type'] == "internal": 
            # unknown usage
            continue
        elif tree_node['val']['type'] == "cr":
            # new line
            continue
        elif tree_node['val']['type'] == "infix":
            continue
        elif tree_node['val']['type'] == "rule":
            # ignore layout setting
            continue
        elif tree_node['val']['type'] == "cdlabel":
            tree_node['val']['text'] = item['side']
            tree_node['structure']['child'] = [1 + private_index + index]
            tree.append(tree_node)      
            item['label']['role'] = 'label'
            tree += ast([item['label']], index=len(tree) + index, father_tree=tree)
        elif tree_node['val']['type'] == "cdlabelparent":
            tree_node['val']['text'] = "\\cdlabelparent"
            tree_node['structure']['child'] = [1 + private_index + index]
            tree.append(tree_node)
            item['fragment']['role'] = 'fragment'
            tree += ast([item['fragment']], index=len(tree) + index, father_tree=tree)
        elif tree_node['val']['type'] == "color":
            tree_node['val']['text'] = "\\color"
            tree_node['structure']['child'] = [1 + private_index + index]
            tree.append(tree_node)
            tree += ast(item['body'], index=len(tree) + index, father_tree=tree)
        elif tree_node['val']['type'] == "color-token":
            tree_node['val']['text'] = "\\color-token"
            tree_node['structure']['child'] = [1 + private_index + index]
            tree.append(tree_node)
            tree += ast([item['body']], index=len(tree) + index, father_tree=tree)
        elif tree_node['val']['type'] == "raw":
            tree_node['val']['text'] = item['string']
            tree.append(tree_node)
        elif tree_node['val']['type'] == "styling":
            # to be confirmed
            tree_node['val']['text'] = "\\styling" 
            # tree_node['val']['text'] = item["style"] ！= None ? item["style"]: "\\styling"
            tree_node['structure']['child'] = [1 + private_index + index]
            tree.append(tree_node)
            tree += ast(item['body'], index=len(tree) + index, father_tree=tree)
        elif tree_node['val']['type'] == "tag":
            continue
            # not supported in Katex yet
            # tree_node['structure']['child'] = [1 + private_index + index]
            # tree_node['val']['text'] = '\\tag' # equations with order number
            # tree.append(tree_node)
            # body_item = {'type':'nodelist','role': 'body','body': item['body']}
            # tag_item = {'type':'nodelist','role': 'tag','body': item['tag']}
            # tree += ast([body_item, tag_item], index=len(tree) + index, father_tree=tree)
        elif tree_node['val']['type'] == "verb":
            tree_node['val']['text'] = item['body'] # "original copy", source code
            tree.append(tree_node)
        elif tree_node['val']['type'] in ["spacing","accent-token","op-token"]:
            tree_node['val']['text'] = item['text']
            tree.append(tree_node) 
        elif tree_node['val']['type'] in ["accent","accentUnder"]: 
            tree_node['structure']['child'] = [1 + private_index + index]
            tree_node['val']['text'] = item["label"]
            tree.append(tree_node)
            item['base']['role'] = 'base'
            tree += ast([item['base']], index=len(tree) + index, father_tree=tree)
        elif tree_node['val']['type'] == "delimsizing":
            # contains symbols for size settings, including "(",")", etc
            tree_node['val']['text'] = item['delim']
            tree.append(tree_node)
        elif tree_node['val']['type'] == "enclose":
            # setting deleting line effect
            tree_node['val']['text'] = item['label']
            tree_node['structure']['child'] = [1 + private_index + index]
            tree.append(tree_node)
            tree += ast([item['body']], index=len(tree) + index, father_tree=tree)
        elif tree_node['val']['type'] == "environment":
            tree_node['val']['text'] = item['name']
            tree_node['structure']['child'] = [1 + private_index + index]
            tree.append(tree_node)
            item['nameGroup']['role'] = 'nameGroup'
            tree += ast([item['nameGroup']], index=len(tree) + index, father_tree=tree)
        elif tree_node['val']['type'] == "url":
            # continue
            tree_node['val']['text'] = item['url']
            tree.append(tree_node)
        elif tree_node['val']['type'] == "href":
            # continue
            tree_node['val']['text'] = item['href']
            tree_node['structure']['child'] = [1 + private_index + index]
            tree.append(tree_node)
            tree += ast(item['body'], index=len(tree) + index, father_tree=tree)
        elif tree_node['val']['type'] == "html":
            # continue
            tree_node['structure']['child'] = [1 + private_index + index]
            tree_node['val']['text'] = "\\html"  
            tree.append(tree_node)
            tree += ast(item['body'], index=len(tree) + index, father_tree=tree)
        elif tree_node['val']['type'] == "htmlmathml":
            # continue
            tree_node['structure']['child'] = [1 + private_index + index]
            tree_node['val']['text'] = "\\htmlmathml"
            tree.append(tree_node)
            html_item = {'type':'nodelist','role': 'html','body': item['html']} # ?
            mathml_item = {'type':'nodelist','role': 'mathml','body': item['mathml']} # ?
            tree += ast([html_item,mathml_item], index=len(tree) + index, father_tree=tree)
        elif tree_node['val']['type'] == "includegraphics":
            # continue
            tree_node['val']['text'] = item['src']
            tree.append(tree_node)
        elif tree_node['val']['type'] == "font":
            tree_node['structure']['child'] = [1 + private_index + index]
            tree_node['val']['text'] = item["font"]  # font name
            tree.append(tree_node)
            item['body']['role'] = 'body'
            tree += ast([item['body']], index=len(tree) + index, father_tree=tree)
        elif tree_node['val']['type'] == "hbox":
            tree_node['structure']['child'] = [1 + private_index + index]
            tree_node['val']['text'] = '\\hbox' # box layout
            tree.append(tree_node)
            tree += ast(item['body'], index=len(tree) + index, father_tree=tree)
        elif tree_node['val']['type'] == "vcenter":
            tree_node['structure']['child'] = [1 + private_index + index]
            tree_node['val']['text'] = '\\vcenter' # box layout
            tree.append(tree_node)
            tree += ast([item['body']], index=len(tree) + index, father_tree=tree)
        elif tree_node['val']['type'] == "horizBrace":
            tree_node['structure']['child'] = [1 + private_index + index]
            tree_node['val']['text'] = item['label']
            tree.append(tree_node)
            item['base']['role'] = 'base'
            tree += ast([item['base']], index=len(tree) + index, father_tree=tree)
        elif tree_node['val']['type'] == "lap":
            # layout setting (overlap) 
            tree_node['structure']['child'] = [1 + private_index + index]
            tree_node['val']['text'] = item["alignment"] # methods of overlap (llap | rlap)
            tree.append(tree_node)
            tree += ast([item['body']], index=len(tree) + index, father_tree=tree)
        elif tree_node['val']['type'] == "sizing":
            # consider ignoring size
            tree_node['structure']['child'] = [1 + private_index + index]
            tree_node['val']['text'] = "\\sizing"
            tree.append(tree_node)
            tree += ast(item['body'], index=len(tree) + index, father_tree=tree)
        elif tree_node['val']['type'] == "smash":
            # layout setting : smash (height | width)
            tree_node['structure']['child'] = [1 + private_index + index]
            tree_node['val']['text'] = "\\smash"
            tree.append(tree_node)
            tree += ast([item['body']], index=len(tree) + index, father_tree=tree)
        elif tree_node['val']['type'] == "mathchoice":
            # provides content that is dependent on the current style (display, text, script, or scriptscript).
            # eg: \mathchoice {#1}{#2}{#3}{#4}
            tree_node['structure']['child'] = [1 + private_index + index]
            tree_node['val']['text'] = "\\mathchoice"
            tree.append(tree_node)
            mathchoiceList = []
            for choice in ["display","text","script","scriptscript"]:
                citem = {'type':'nodelist','role':choice, 'body':item[choice]}
                mathchoiceList.append(citem)
            tree += ast(mathchoiceList, index=len(tree) + index, father_tree=tree)
        elif tree_node['val']['type'] == "operatorname":
            # unknown usage
            tree_node['structure']['child'] = [1 + private_index + index]
            tree_node['val']['text'] = "\\operatorname"
            tree.append(tree_node)
            tree += ast(item['body'], index=len(tree) + index, father_tree=tree)
        elif tree_node['val']['type'] in ["overline","underline"]:
            # consider ignoring line
            tree_node['structure']['child'] = [1 + private_index + index]
            tree_node['val']['text'] = "\\" + tree_node['val']['type']
            tree.append(tree_node)
            tree += ast([item['body']], index=len(tree) + index, father_tree=tree)
        elif tree_node['val']['type'] == "raisebox":
            # raise or lower the height of the text
            tree_node['structure']['child'] = [1 + private_index + index]
            tree_node['val']['text'] = "\\raisebox"
            tree.append(tree_node)
            tree += ast([item['body']], index=len(tree) + index, father_tree=tree)
        elif tree_node['val']['type'] == "leftright-right":
            # paired with leftright
            tree_node['val']['text'] = item["delim"]
            tree.append(tree_node)
        elif tree_node['val']['type'] == "middle":
            # symbols with height setting, such as "|"
            tree_node['val']['text'] = item["delim"]
            tree.append(tree_node)
        elif tree_node['val']['type'] in ["phantom","hphantom","vphantom"]:
            # set space distance by the length of content
            # continue 
            tree_node['structure']['child'] = [1 + private_index + index]
            tree_node['val']['text'] = "\\" + tree_node['val']['type']
            tree.append(tree_node)
            tree += ast([item['body']], index=len(tree) + index, father_tree=tree)
        elif tree_node['val']['type'] == 'nodelist':
            # process node list specially
            tree_node['structure']['child'] = [1 + private_index + index]
            tree_node['val']['text'] = "\\" + item["role"]
            tree.append(tree_node)
            tree += ast(item['body'], index=len(tree) + index, father_tree=tree)
        
        else:
            tree_node['structure']['child'] = [1 + private_index + index]

            if "text" in item:
                tree_node['val']['text'] = item["text"]
            else:
                tree_node['val']['text'] = item["type"]
            tree_node['val']['type'] = "other"
            tree.append(tree_node)
            Role = ['body', 'base', 'sup', 'sub', 'numer', 'denom', 'index', 'below','nameGroup','fragment','label', 'other']
            childrole = []
            for role_item in Role:
                if role_item in item:
                    if role_item == "body" and isinstance(item[role_item], dict) is False:
                        # \text{}
                        childrole.extend(item[role_item])
                    else:
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
    variable_connect_dict = {
        "var2id": forest_connect_dict,
        "var_code": {}
    }
    return variable_connect_dict


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
        index = node["val"]["id"]
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
