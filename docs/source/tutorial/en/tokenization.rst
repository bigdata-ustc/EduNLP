Tokenization
============================

This module provides tokenization to questions, converting sentence with formula or picture into `tokens`.
It actually does a step-forward processing on different output elements from `Segmentation <tokenize.rst>`_, and get the final multimodal `tokens` sequence.

.. note::
   In natural language processing, tokenization often refers to segmentation (converting a sentences to word sequence). For multimodal education resource, we define `tokenization` as: converting education resource with different elements to **token sequence with different types of tokens**.

Tokenization functions
----------------------------


Low-level interface
^^^^^^^^^^^^^^^^^^^^^^

The corresponding instance is `EduNLP.SIF.tokenize`.
.. note::
   This part is only for declaring specific operations(segmentation, tokenizing) in tokenization. If there is no need to modify low-level interfaces, we suggest you read **Standard interface**, or more convenient **Tokenization container**.

::
   >>> from EduNLP.SIF.tokenization import tokenize
   >>> items = "如图所示，则三角形$ABC$的面积是$\\SIFBlank$。$\\FigureID{1}$"
   >>> tokenize(seg(items))
   ['如图所示', '三角形', 'ABC', '面积', '\\\\SIFBlank', \\FigureID{1}]
   >>> tokenize(seg(items), formula_params={"method": "ast"})
   ['如图所示', '三角形', <Formula: ABC>, '面积', '\\\\SIFBlank', \\FigureID{1}]


Standard interface
^^^^^^^^^^^^^^^^^^^^^^

Standard interface is an abstraction of SIF, Segmentation and Tokenizing, with sufficient parameters for more needs.
Examples:


.. nbgallery::
    :caption: This is a thumbnail gallery:
    :name: tokenization_gallery_en1
    :glob:

    Standard Tokenization  <../../build/blitz/sif/sif4sci.ipynb>


Tokenization container
----------------------------

We provide various encapsulated tokenization containers for simple using, please check `EduNLP.Tokenizer` and `EduNLP.Pretrain` for more instances. Here is a tokenization container list:
Generalized base tokenization containers:

- CharTokenizer
- SpaceTokenizer
- CustomTokenizer
- PureTextTokenizer
- AstFormulaTokenizer
- GensimSegTokenizer
- GensimWordTokenizer

Adapted tokenization containers for specific models:

- ElmoTokenizer
- BertTokenizer
- DisenQTokenizer
- QuesNetTokenizer

.. note::

   "Adapted tokenization containers for specific models" are used with corresponding T2V container, more discussion is in vectorization and pretrain parts.


CharTokenizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Character token parser. This token parser extracts each character of the text individually to form a token sequence.

In addition, we provide a key parameter to select the pending content in the incoming item. You can also specify your own stop words to filter some text messages.

::

   >>> items = [{
   ...     "stem": "已知集合$A=\\left\\{x \\mid x^{2}-3 x-4<0\\right\\}, \\quad B=\\{-4,1,3,5\\}, \\quad$ 则 $A \\cap B=$",
   ...     "options": ["1", "2"]
   ... }]

   >>> tokenizer = get_tokenizer("char")
   >>> tokens = tokenizer(items, key=lambda x: x["stem"])
   >>> print(next(tokens))
   ['文', '具', '店', '有', '$', '600', '$', '本', '练', '习', '本', '卖', '出', '一', '些', '后', 
   '还', '剩', '$', '4', '$', '包', '每', '包', '$', '25', '$', '本', '卖', '出', '多', '少', '本']


SpaceTokenizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Space token parser. This token parser will tokenize the text based on the space to get the token sequence.

In addition, we provide a key parameter to select the pending content in the incoming item. You can also specify your own stop words to filter some text messages.

::

   >>> items = [{
   ...  "stem": "已知集合$A=\\left\\{x \\mid x^{2}-3 x-4<0\\right\\}, \\quad B=\\{-4,1,3,5\\}, \\quad$ 则 $A \\cap B=$",
   ...  "options": ["1", "2"]
   ...  }]

   >>> tokenizer = get_tokenizer("space", stop_words = [])
   >>> tokens = tokenizer(items, key=lambda x: x["stem"])
   >>> print(next(tokens))
   ['已知集合$A=\\left\\{x', '\\mid', 'x^{2}-3', 'x-4<0\\right\\},', '\\quad', 
    'B=\\{-4,1,3,5\\},', '\\quad$', '则', '$A', '\\cap', 'B=$']


CustomTokenizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Custom token parser. This token parser uses a linear parsing method to obtain a sequence of tokens. You can specify whether text, formulas, images, labels, separators, question gaps, etc. are converted into special characters and symbolized.

In addition, we provide a key parameter to select the pending content in the incoming item. You can also specify your own stop words to filter some text messages.

::

   >>> items = [{
   ...  "stem": "已知集合$A=\\left\\{x \\mid x^{2}-3 x-4<0\\right\\}, \\quad B=\\{-4,1,3,5\\}, \\quad$ 则 $A \\cap B=$",
   ...  "options": ["1", "2"]
   ...  }]

   >>> tokenizer_t = get_tokenizer("custom", symbol='t')
   >>> tokens = tokenizer_t(items, key=lambda x: x["stem"])
   >>> print(next(tokens))
   ['[TEXT]', 'A', '=', '\\left', '\\{', 'x', '\\mid', 'x', '^', '{', '2', '}', '-', '3', 'x', 
    '-', '4', '<', '0', '\\right', '\\}', ',', '\\quad', 'B', '=', '\\{', '-', '4', ',', '1', 
    ',', '3', ',', '5', '\\}', ',', '\\quad', '[TEXT]', 'A', '\\cap', 'B', '=']

   >>> tokenizer_f = get_tokenizer("custom", symbol='f')
   >>> tokens = tokenizer_f(items, key=lambda x: x["stem"])
   >>> print(next(tokens))
   ['已知', '集合', '[FORMULA]', '[FORMULA]']




PureTextTokenizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Plain text token parser. This token parser filters out specially processed formulas (e.g. '$FormFigureID{...} $` ， `$FormFigureBase64{...} $`) and preserves only text formatting formula.

In addition, we provide a key parameter to select the pending content in the incoming item. You can also specify your own stop words to filter some text messages.

::

   >>> tokenizer = PureTextTokenizer()

   >>> items = ["有公式$\\FormFigureID{1}$，如图$\\FigureID{088f15ea-xxx}$,若$x,y$满足约束条件公式$\\FormFigureBase64{2}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$"]

   >>> tokenizer = get_tokenizer("pure_text") # tokenizer = PureTextTokenizer()
   >>> tokens = tokenizer(items)
   >>> print(next(tokens))
   ['公式', '如图', '[FIGURE]', 'x', ',', 'y', '约束条件', '公式', '[SEP]', 'z', '=', 'x', '+', '7', 'y', '最大值', '[MARK]']



AstFormulaTokenizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Abstract formula parser. This token parser abstracts mathematical formulas from text and performs a series of processing. For example, variables that appear are recorded and marked in turn, while objects such as expressions and pictures are converted to special characters and symbolized.

In addition, we provide a key parameter to select the pending content in the incoming item. You can also specify your own stop words to filter some text messages.

::
   
   >>> items = ["有公式$\\FormFigureID{1}$，如图$\\FigureID{088f15ea-xxx}$,若$x,y$满足约束条件公式$\\FormFigureBase64{2}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$"]

   >>> tokenizer = get_tokenizer("ast_formula") 
   >>> tokens = tokenizer(items)
   >>> print(next(tokens))
   ['公式', '[FORMULA]', '如图', '[FIGURE]', 'mathord_0', ',', 'mathord_1', '约束条件', '公式', 
    '[FORMULA]', '[SEP]', 'mathord_2', '=', 'mathord_0', '+', 'textord', 'mathord_1', '最大值', '[MARK]']



GensimWordTokenizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, the pictures, blanks in the question text and other parts of the incoming item are converted into special characters for data security and the tokenization of text, formulas, labels and separators. Also, the tokenizer uses linear analysis method for text and abstract syntax tree method for formulas respectively. You can choose each of them by general parameter:

- true, it means that the incoming item conforms to SIF and the linear analysis method should be used.
- false, it means that the incoming item doesn't conform to SIF and the abstract syntax tree method should be used.


::

   >>> item = "已知有公式$\\FormFigureID{1}$，如图$\\FigureID{088f15ea-xxx}$, 若$x,y$满足约束条件公式$\\FormFigureBase64{2}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$"

   >>> tokenizer = GensimWordTokenizer(symbol="gmas")
   >>> token_item = tokenizer(item)
   >>> print(token_item.tokens)
   ['已知', '公式', \FormFigureID{1}, '如图', '[FIGURE]', 'mathord', ',', 'mathord', '约束条件', '公式', [FORMULA], '[SEP]', 'mathord', '=', 'mathord', '+', 'textord', 'mathord', '最大值', '[MARK]']

   >>> tokenizer = GensimWordTokenizer(symbol="gmas", general=True)
   >>> token_item = tokenizer(item)
   >>> print(token_item.tokens)
   ['已知', '公式', '[FORMULA]', '如图', '[FIGURE]', 'x', ',', 'y', '约束条件', '公式', '[FORMULA]', '[SEP]', 'z', '=', 'x', '+', '7', 'y', '最大值', '[MARK]']




GensimSegTokenizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, the pictures, separators, blanks in the question text and other parts of the incoming item are converted into special characters for data security and tokenization of text, formulas and labels. Also, the tokenizer uses linear analysis method for text and abstract analysis method of syntax tree for formulas.

Compared to GensimWordTokenizer, the main differences are:

* It provides the depth option for segmentation position, such as SIFSep and SIFTag.
* By default, labels are inserted in the header of item components (such as text and formulas).

Select segmentation level:

- depth=None: segmentation by components, which return text, formula, figure token list.
- depth=0: segmentation by `SEP` tag
- depth=1: segmentation by `TAG` tag
- depth=2: segmentation by `SEP` tag and `TAG` tag

::

   item = "已知有公式$\\FormFigureID{1}$，如图$\\FigureID{088f15ea-xxx}$, 若$x,y$满足约束条件公式$\\FormFigureBase64{2}$，$\\SIFSep$则$z=x+7 y$的最大值为$\\SIFBlank$"

   tokenizer = GensimSegTokenizer(symbol="gmas")
   token_item = tokenizer(item)
   print(len(token_item), token_item)
   # 10 [['已知', '公式'], [\FormFigureID{1}], ['如图'], ['[FIGURE]'], ['mathord', ',', 'mathord'], ['约束条件', '公式'], [[FORMULA]], ['mathord', '=', 'mathord', '+', 'textord', 'mathord'], ['最大值'], ['[MARK]']]

   # segment at Tag and Sep
   tokenizer = GensimSegTokenizer(symbol="gmas", depth=2)
   token_item = tokenizer(item)
   print(len(token_item), token_item)
   # 2 [['[TEXT_BEGIN]', '已知', '公式', '[FORMULA_BEGIN]', \FormFigureID{1}, '[TEXT_BEGIN]', '如图', '[FIGURE]', '[FORMULA_BEGIN]', 'mathord', ',', 'mathord', '[TEXT_BEGIN]', '约束条件', '公式', '[FORMULA_BEGIN]', [FORMULA], '[SEP]'], ['[FORMULA_BEGIN]', 'mathord', '=', 'mathord', '+', 'textord', 'mathord', '[TEXT_BEGIN]', '最大值', '[MARK]']]


More examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. nbgallery::
    :caption: This is a thumbnail gallery:
    :name: tokenization_gallery2
    :glob:

    Tokenization container  <../../build/blitz/tokenizer/tokenizer.ipynb>
