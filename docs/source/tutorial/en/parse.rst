Syntax Parsing
=================

In educational resources, texts and formulas have internal implicit or explicit syntax structures. It is of great benefit for further processing to extract these structures.

* Text syntax structure parsing

* Formula syntax structure parsing

The purpose is as follows:


1. Represent underlines of blanks and brackets of choices with special identifiers. And the alphabets and formulas should be wrapped with $$, so that items of different types can be cut accurately through the symbol $.
2. Determine whether the current item is legal and report the error type.

Specific processing content
--------------------------------

1.Its function is to match alphabets and numbers other than formulas. Only the alphabets and numbers between two Chinese characters will be corrected, and the rest of the cases are regarded as formulas that do not conform to latex syntax.

2.Match brackets like "( )" (both English format and Chinese format), that is, brackets with no content or spaces, which should be replaced with ``$\\SIFChoice$``

3.Match continuous underscores or underscores with spaces and replace them with ``$\\SIFBlank$``.

4.Match latex formulas，check the completeness and analyzability of latex formulas, and report an error for illegal formula.

Formula syntax structure parsing
-------------------------------------

This section is mainly realized by EduNLP. Formula modules, which can determine if the text has syntax errors and convert the syntax formula into the form of ast tree. In practice, this module is often used as part of an intermediate process, and the relevant parameters of this module can be automatically chosen by calling the corresponding model, so it generally does not need special attention.

Introduction of Main Introduction
+++++++++++++++++++++++++++++++++++++++

1.Formula: determine whether the single formula passed in is in str form. If so, use the ast method for processing, otherwise an error will be reported. In addition, parameter variable_standardization is given. If this parameter is true, the variable standardization method will be used to make sure the same variable has the same variable number.

2.FormulaGroup: If you need to pass in a formula set, you can call this interface to get an ast forest. The tree structure in the forest is the same as that of Formula.

Formula
>>>>>>>>>>>>

Formula: firstly, in the word segmentation function, the formula of the original text is segmented. In addition, ``Formula parse tree`` function is provided, which can represent the abstract syntax analysis tree of mathematical formula in the form of text or picture.

This module also provides the function of formula variable standardization, such as determining whether 'x' in several sub formulas is the same variable.

Call the library
+++++++++++++++++++++

::

   import matplotlib.pyplot as plt
   from EduNLP.Formula import Formula
   from EduNLP.Formula.viz import ForestPlotter

Initialization
+++++++++++++++

Incoming parameters: item

Item is the latex formula or the abstract syntax parse tree generated after the formula is parsed and its type is str or List[Dict].

::

   >>> f=Formula("x^2 + x+1 = y")
   >>> f
   <Formula: x^2 + x+1 = y>

View the specific content after formula segmentation
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

- View node elements after formula segmentation

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

- View the abstract parse tree of formulas

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

- show the abstract parse tree by a picture

::

   >>> ForestPlotter().export(f.ast_graph, root_list=[node["val"]["id"] for node in f.ast if node["structure"]["father"] is None],)
   >>> plt.show()


.. figure:: ../../_static/formula.png


Variable standardization
+++++++++++++++++++++++++++++

This parameter makes the same variable have the same variable number.

For example: the number of variable ``x`` is ``0`` and the number of variable ``y`` is ``1``.

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

Call ``FormulaGroup`` class to parse the equations. The related attributes and functions are the same as those above.

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


Text syntax structure parsing
------------------------------------

This section is mainly realized by EduNLP.SIF.Parse module. Its main function is to extract letters and numbers in the text and convert them into standard format.

This module is mainly used as an *middle module* to parse the input text. In general, users do not call this module directly.

Introduction of main content
+++++++++++++++++++++++++++++++++++

1. Judge the type of the incoming text in the following order

* is_chinese: its function is to match Chinese characters[\u4e00-\u9fa5].
 
* is_alphabet: its function is to match alphabets other than formulas. Only the alphabets between two Chinese characters will be corrected (wrapped with $$), and the rest of the cases are regarded as formulas that do not conform to latex syntax.
 
* is_number: its function is to match numbers other than formulas. Only the numbers between two Chinese characters will be corrected, and the rest of the cases are regarded as formulas that do not conform to latex syntax.
 
2. Match latex formula

* If Chinese characters appear in latex, print warning only once.
 
* Use _is_formula_legal function, check the completeness and analyzability of latex formula, and report an error for formulas that do not conform to latex syntax.

Call the library
>>>>>>>>>>>>>>>>>>>

::

   from EduNLP.SIF.Parser import Parser

Input
>>>>>>>

Types: str

Content: question text

::

   >>> text1 = '生产某种零件的A工厂25名工人的日加工零件数_   _'
   >>> text2 = 'X的分布列为(   )'
   >>> text3 = '① AB是⊙O的直径，AC是⊙O的切线，BC交⊙O于点E．AC的中点为D'
   >>> text4 = '支持公式如$\\frac{y}{x}$，$\\SIFBlank$，$\\FigureID{1}$，不支持公式如$\\frac{ \\dddot y}{x}$'

Parsing
>>>>>>>>>>>>>>>>>>>>

::

   >>> text_parser1 = Parser(text1)
   >>> text_parser2 = Parser(text2)
   >>> text_parser3 = Parser(text3)
   >>> text_parser4 = Parser(text4)

Related parameters description
>>>>>>>>>>>>

- Try to convert text to standard format

::

   >>> text_parser1.description_list()
   >>> print('text_parser1.text:',text_parser1.text)
   text_parser1.text: 生产某种零件的$A$工厂$25$名工人的日加工零件数$\SIFBlank$
   >>> text_parser2.description_list()
   >>> print('text_parser2.text:',text_parser2.text)
   text_parser2.text: $X$的分布列为$\SIFChoice$

- Determine if the text has syntax errors

::

   >>> text_parser3.description_list()
   >>> print('text_parser3.error_flag: ',text_parser3.error_flag)
   text_parser3.error_flag:  1
   >>> text_parser4.description_list()
   >>> print('text_parser4.fomula_illegal_flag: ',text_parser4.fomula_illegal_flag)
   text_parser4.fomula_illegal_flag:  1

