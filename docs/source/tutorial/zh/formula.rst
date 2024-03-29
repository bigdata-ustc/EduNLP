公式语法结构解析
===========================


公式语法结构解析
--------------------

公式解析（formula-tokenization）：理科类文本中常常含有公式。将一个符合 latex 语法的公式解析为标记字符列表或抽象语法树的过程称为“公式解析”。



本小节主要介绍如何获取不同格式的公式解析结果。公式解析的底层实现请参考：`EduNLP.Formula` 部分。


本功能主要由EduNLP.Formula模块实现，具有检查传入的公式是否合法，并将合法的公式转换为art树的形式。从实际使用的角度，本模块常作为中间处理过程，调用相应的模型即可自动选择本模块的相关参数，故一般不需要特别关注。

主要内容介绍
+++++++++++++++

1.Formula:对传入的单个公式进行判断，判断传入的公式是否为str形式，如果是则使用ast的方法进行处理，否则进行报错。此外，提供了variable_standardization参数，当此参数为True时，使用变量标准化方法，即同一变量拥有相同的变量编号。

2.FormulaGroup:如果需要传入公式集则可调用此接口，最终将形成ast森林，森林中树的结构同Formula。

Formula
>>>>>>>>>>>>

Formula 首先在分词功能中对原始文本的公式做切分处理，另外提供 ``公式解析树`` 功能，可以将数学公式的抽象语法分析树用文本或图片的形式表示出来。  

本模块另提供公式变量标准化的功能，如判断几个子公式内的‘x’为同一变量。

调用库
+++++++++

::

   import matplotlib.pyplot as plt
   from EduNLP.Formula import Formula
   from EduNLP.Formula.viz import ForestPlotter

初始化
+++++++++

传入参数：item 

item为str 或 List[Dict]类型，具体内容为latex 公式 或 公式经解析后产生的抽象语法分析树。

::

   >>> f=Formula("x^2 + x+1 = y")
   >>> f
   <Formula: x^2 + x+1 = y>

查看公式切分后的具体内容
++++++++++++++++++++++++++++

- 查看公式切分后的结点元素

::

   >>> f.elements
   [{'id': 0, 'type': 'supsub', 'text': '\\supsub', 'role': None},
   {'id': 1, 'type': 'mathord', 'text': 'x', 'role': 'base'},
   {'id': 2, 'type': 'textord', 'text': '2', 'role': 'sup'},
   {'id': 3, 'type': 'bin', 'text': '+', 'role': None},
   {'id': 4, 'type': 'mathord', 'text': 'x', 'role': None},
   {'id': 5, 'type': 'bin', 'text': '+', 'role': None},
   {'id': 6, 'type': 'textord', 'text': '1', 'role': None},
   {'id': 7, 'type': 'rel', 'text': '=', 'role': None},
   {'id': 8, 'type': 'mathord', 'text': 'y', 'role': None}]

- 查看公式的抽象语法分析树

::

   >>> f.ast
   [{'val': {'id': 0, 'type': 'supsub', 'text': '\\supsub', 'role': None},
   'structure': {'bro': [None, 3],'child': [1, 2],'father': None,'forest': None}},
   {'val': {'id': 1, 'type': 'mathord', 'text': 'x', 'role': 'base'},
   'structure': {'bro': [None, 2], 'child': None, 'father': 0, 'forest': None}},
   {'val': {'id': 2, 'type': 'textord', 'text': '2', 'role': 'sup'},
   'structure': {'bro': [1, None], 'child': None, 'father': 0, 'forest': None}},
   {'val': {'id': 3, 'type': 'bin', 'text': '+', 'role': None},
   'structure': {'bro': [0, 4], 'child': None, 'father': None, 'forest': None}},
   {'val': {'id': 4, 'type': 'mathord', 'text': 'x', 'role': None},
   'structure': {'bro': [3, 5], 'child': None, 'father': None, 'forest': None}},
   {'val': {'id': 5, 'type': 'bin', 'text': '+', 'role': None},
   'structure': {'bro': [4, 6], 'child': None, 'father': None, 'forest': None}},
   {'val': {'id': 6, 'type': 'textord', 'text': '1', 'role': None},
   'structure': {'bro': [5, 7], 'child': None, 'father': None, 'forest': None}},
   {'val': {'id': 7, 'type': 'rel', 'text': '=', 'role': None},
   'structure': {'bro': [6, 8], 'child': None, 'father': None, 'forest': None}},
   {'val': {'id': 8, 'type': 'mathord', 'text': 'y', 'role': None},
   'structure': {'bro': [7, None],'child': None,'father': None,'forest': None}}]

   >>> print('nodes: ',f.ast_graph.nodes)
   nodes:  [0, 1, 2, 3, 4, 5, 6, 7, 8]
   >>> print('edges: ' ,f.ast_graph.edges)
   edges:  [(0, 1), (0, 2)]

- 将抽象语法分析树用图片表示

::

   >>> ForestPlotter().export(f.ast_graph, root_list=[node["val"]["id"] for node in f.ast if node["structure"]["father"] is None],)
   >>> plt.show()


.. figure:: ../../_static/formula.png


变量标准化
+++++++++++

此参数使得同一变量拥有相同的变量编号。

如：``x`` 变量的编号为 ``0``， ``y`` 变量的编号为 ``1``。

::

   >>> f.variable_standardization().elements
   [{'id': 0, 'type': 'supsub', 'text': '\\supsub', 'role': None},
   {'id': 1, 'type': 'mathord', 'text': 'x', 'role': 'base', 'var': 0},
   {'id': 2, 'type': 'textord', 'text': '2', 'role': 'sup'},
   {'id': 3, 'type': 'bin', 'text': '+', 'role': None},
   {'id': 4, 'type': 'mathord', 'text': 'x', 'role': None, 'var': 0},
   {'id': 5, 'type': 'bin', 'text': '+', 'role': None},
   {'id': 6, 'type': 'textord', 'text': '1', 'role': None},
   {'id': 7, 'type': 'rel', 'text': '=', 'role': None},
   {'id': 8, 'type': 'mathord', 'text': 'y', 'role': None, 'var': 1}]

FormulaGroup
>>>>>>>>>>>>>>>

调用 ``FormulaGroup`` 类解析公式方程组，相关的属性和函数方法同上。

::

   import matplotlib.pyplot as plt
   from EduNLP.Formula import Formula
   from EduNLP.Formula import FormulaGroup
   from EduNLP.Formula.viz import ForestPlotter
   >>> fs = FormulaGroup(["x^2 = y", "x^3 = y^2", "x + y = \pi"])
   >>> fs
   <FormulaGroup: <Formula: x^2 = y>;<Formula: x^3 = y^2>;<Formula: x + y = \pi>>
   >>> fs.elements
   [{'id': 0, 'type': 'supsub', 'text': '\\supsub', 'role': None},
   {'id': 1, 'type': 'mathord', 'text': 'x', 'role': 'base'},
   {'id': 2, 'type': 'textord', 'text': '2', 'role': 'sup'},
   {'id': 3, 'type': 'rel', 'text': '=', 'role': None},
   {'id': 4, 'type': 'mathord', 'text': 'y', 'role': None},
   {'id': 5, 'type': 'supsub', 'text': '\\supsub', 'role': None},
   {'id': 6, 'type': 'mathord', 'text': 'x', 'role': 'base'},
   {'id': 7, 'type': 'textord', 'text': '3', 'role': 'sup'},
   {'id': 8, 'type': 'rel', 'text': '=', 'role': None},
   {'id': 9, 'type': 'supsub', 'text': '\\supsub', 'role': None},
   {'id': 10, 'type': 'mathord', 'text': 'y', 'role': 'base'},
   {'id': 11, 'type': 'textord', 'text': '2', 'role': 'sup'},
   {'id': 12, 'type': 'mathord', 'text': 'x', 'role': None},
   {'id': 13, 'type': 'bin', 'text': '+', 'role': None},
   {'id': 14, 'type': 'mathord', 'text': 'y', 'role': None},
   {'id': 15, 'type': 'rel', 'text': '=', 'role': None},
   {'id': 16, 'type': 'mathord', 'text': '\\pi', 'role': None}]
   >>> fs.ast
   [{'val': {'id': 0, 'type': 'supsub', 'text': '\\supsub', 'role': None},
   'structure': {'bro': [None, 3],
      'child': [1, 2],
      'father': None,
      'forest': None}},
   {'val': {'id': 1, 'type': 'mathord', 'text': 'x', 'role': 'base'},
   'structure': {'bro': [None, 2],
      'child': None,
      'father': 0,
      'forest': [6, 12]}},
   {'val': {'id': 2, 'type': 'textord', 'text': '2', 'role': 'sup'},
   'structure': {'bro': [1, None], 'child': None, 'father': 0, 'forest': None}},
   {'val': {'id': 3, 'type': 'rel', 'text': '=', 'role': None},
   'structure': {'bro': [0, 4], 'child': None, 'father': None, 'forest': None}},
   {'val': {'id': 4, 'type': 'mathord', 'text': 'y', 'role': None},
   'structure': {'bro': [3, None],
      'child': None,
      'father': None,
      'forest': [10, 14]}},
   {'val': {'id': 5, 'type': 'supsub', 'text': '\\supsub', 'role': None},
   'structure': {'bro': [None, 8],
      'child': [6, 7],
      'father': None,
      'forest': None}},
   {'val': {'id': 6, 'type': 'mathord', 'text': 'x', 'role': 'base'},
   show more (open the raw output data in a text editor) ...
   >>> fs.variable_standardization()[0]
   [{'id': 0, 'type': 'supsub', 'text': '\\supsub', 'role': None}, {'id': 1, 'type': 'mathord', 'text': 'x', 'role': 'base', 'var': 0}, {'id': 2, 'type': 'textord', 'text': '2', 'role': 'sup'}, {'id': 3, 'type': 'rel', 'text': '=', 'role': None}, {'id': 4, 'type': 'mathord', 'text': 'y', 'role': None, 'var': 1}]
   >>> ForestPlotter().export(fs.ast_graph, root_list=[node["val"]["id"] for node in fs.ast if node["structure"]["father"] is None],)

.. figure:: ../../_static/formulagroup.png

