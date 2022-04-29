令牌化
==============


本模块提供题目的令牌化解析，将带公式、图片的句子切分为若干标记的过程。每个标记为一个“令牌”（token），方便后续向量化表征试题。

此模块实际上是对 `语法解析 <tokenize.rst>`_ 的结果做进一步处理。


.. note::
   在自然语言处理中，令牌化通常指分词，即将一个句子转化为词序列。针对多模态的教育资源，我们定义令牌化为：将含有不同成分的教育资源（如题目）转化为 **含有不同类型令牌的令牌序列**。

令牌化函数
----------------------------


底层接口
^^^^^^^^^^^^^^^^^^^^^^

此功能对应的实现函数为 `EduNLP.SIF.tokenize` ，将已经经过结构成分分解后的item传入其中即可得到所需结果。

::

   >>> items = "如图所示，则三角形$ABC$的面积是$\\SIFBlank$。$\\FigureID{1}$"
   >>> tokenize(seg(items))
   ['如图所示', '三角形', 'ABC', '面积', '\\\\SIFBlank', \\FigureID{1}]
   >>> tokenize(seg(items), formula_params={"method": "ast"})
   ['如图所示', '三角形', <Formula: ABC>, '面积', '\\\\SIFBlank', \\FigureID{1}]



标准接口
^^^^^^^^^^^^^^^^^^^^^^

标准接口将标准化检验、成分分解、语法解析封装为一个整体，含有三个步骤的所有功能。

具体使用方法见示例：

.. nbgallery::
    :caption: This is a thumbnail gallery:
    :name: t2v_gallery1
    :glob:
    
    令牌化  <../../build/blitz/tokenizer/sif4sci.ipynb>



令牌化容器
----------------------------

我们提供了多种已经封装好的令牌化容器供用户便捷调用，通过查看 `EduNLP.Tokenizer` 及 `EduNLP.Pretrain` 可以查看更多令牌化器。下面是一个完整的令牌化器列表:

通用的令牌化容器

- TextTokenizer
- PureTextTokenizer

特定的令牌化容器

- GensimSegTokenizer
- GensimWordTokenizer
- ElmoTokenizer
- BertTokenizer
- DisenQTokenizer
- QuesNetTokenizer


TextTokenizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

即文本令牌解析器，在默认情况下对传入的item中的图片、标签、分隔符、题目空缺符等部分则转换成特殊字符进行保护，从而对文本、公式进行令牌化操作。此外，此令牌解析器对文本、公式均采用线性的分析方法，并提供的key参数用于对传入的item进行预处理，待未来根据需求进行开发。

::

   >>> items = ["已知集合$A=\\left\\{x \\mid x^{2}-3 x-4<0\\right\\}, \\quad B=\\{-4,1,3,5\\}, \\quad$ 则 $A \\cap B=$"]
   >>> tokenizer = TextTokenizer()
   >>> tokens = tokenizer(items)
   >>> next(tokens)  # doctest: +NORMALIZE_WHITESPACE
   ['已知', '集合', 'A', '=', '\\left', '\\{', 'x', '\\mid', 'x', '^', '{', '2', '}', '-', '3', 'x', '-', '4', '<',
   '0', '\\right', '\\}', ',', '\\quad', 'B', '=', '\\{', '-', '4', ',', '1', ',', '3', ',', '5', '\\}', ',',
   '\\quad', 'A', '\\cap', 'B', '=']
   >>> items = [{
   ... "stem": "已知集合$A=\\left\\{x \\mid x^{2}-3 x-4<0\\right\\}, \\quad B=\\{-4,1,3,5\\}, \\quad$ 则 $A \\cap B=$",
   ... "options": ["1", "2"]
   ... }]
   >>> tokens = tokenizer(items, key=lambda x: x["stem"])
   >>> next(tokens)  # doctest: +NORMALIZE_WHITESPACE
   ['已知', '集合', 'A', '=', '\\left', '\\{', 'x', '\\mid', 'x', '^', '{', '2', '}', '-', '3', 'x', '-', '4', '<',
   '0', '\\right', '\\}', ',', '\\quad', 'B', '=', '\\{', '-', '4', ',', '1', ',', '3', ',', '5', '\\}', ',',
   '\\quad', 'A', '\\cap', 'B', '=']


PureTextTokenizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

即纯净型文本令牌解析器，在默认情况下对传入的item中的图片、标签、分隔符、题目空缺符等部分则转换成特殊字符进行保护，并对特殊公式(例如：$\\FormFigureID{...}$， $\\FormFigureBase64{...}$)进行筛除，从而对文本、纯文本公式进行令牌化操作。此外，此令牌解析器对文本、公式均采用线性的分析方法，并提供的key参数用于对传入的item进行预处理，待未来根据需求进行开发。


::

   >>> tokenizer = PureTextTokenizer()
   >>> items = ["有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$,\
   ... 若$x,y$满足约束条件公式$\\FormFigureBase64{wrong2?}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$"]
   >>> tokens = tokenizer(items)
   >>> next(tokens)[:10]
   ['公式', '如图', '[FIGURE]', 'x', ',', 'y', '约束条件', '公式', '[SEP]', 'z']
   >>> items = ["已知集合$A=\\left\\{x \\mid x^{2}-3 x-4<0\\right\\}, \\quad B=\\{-4,1,3,5\\}, \\quad$ 则 $A \\cap B=$"]
   >>> tokens = tokenizer(items)
   >>> next(tokens)  # doctest: +NORMALIZE_WHITESPACE
   ['已知', '集合', 'A', '=', '\\left', '\\{', 'x', '\\mid', 'x', '^', '{', '2', '}', '-', '3', 'x', '-', '4', '<',
   '0', '\\right', '\\}', ',', '\\quad', 'B', '=', '\\{', '-', '4', ',', '1', ',', '3', ',', '5', '\\}', ',',
   '\\quad', 'A', '\\cap', 'B', '=']
   >>> items = [{
   ... "stem": "已知集合$A=\\left\\{x \\mid x^{2}-3 x-4<0\\right\\}, \\quad B=\\{-4,1,3,5\\}, \\quad$ 则 $A \\cap B=$",
   ... "options": ["1", "2"]
   ... }]
   >>> tokens = tokenizer(items, key=lambda x: x["stem"])
   >>> next(tokens)  # doctest: +NORMALIZE_WHITESPACE
   ['已知', '集合', 'A', '=', '\\left', '\\{', 'x', '\\mid', 'x', '^', '{', '2', '}', '-', '3', 'x', '-', '4', '<',
   '0', '\\right', '\\}', ',', '\\quad', 'B', '=', '\\{', '-', '4', ',', '1', ',', '3', ',', '5', '\\}', ',',
   '\\quad', 'A', '\\cap', 'B', '=']

GensimWordTokenizer
+++++++++++++++++++++++

此令牌解析器在默认情况下对传入的item中的图片、题目空缺符等部分转换成特殊字符进行保护，从而对文本、公式、标签、分隔符进行令牌化操作。此外，从令牌化方法而言，此令牌解析器对文本均采用线性的分析方法，而对公式采用抽象语法树的分析方法，提供了general参数可供使用者选择：当general为true的时候则代表着传入的item并非标准格式，此时对公式也使用线性的分析方法；当general为false时则代表使用抽象语法树的方法对公式进行解析。

GensimSegTokenizer
++++++++++++++++++++

此令牌解析器在默认情况下对传入的item中的图片、分隔符、题目空缺符等部分则转换成特殊字符进行保护，从而对文本、公式、标签进行令牌化操作。此外，从令牌化方法而言，此令牌解析器对文本均采用线性的分析方法，而对公式采用抽象语法树的分析方法。

与GensimWordTokenizer相比，GensimSegTokenizer解析器主要区别是：

* 提供了切分深度的选项，即可以在sep标签或者tag标签处进行切割
* 默认在item组分（如text、formula）的头部插入开始标签

Examples
----------

::

   >>> tokenizer = GensimWordTokenizer(symbol="gmas", general=True)
   >>> token_item = tokenizer("有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$,\
   ... 若$x,y$满足约束条件公式$\\FormFigureBase64{wrong2?}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$")
   >>> print(token_item.tokens[:10])
   ['公式', '[FORMULA]', '如图', '[FIGURE]', 'x', ',', 'y', '约束条件', '公式', '[FORMULA]']
   >>> tokenizer = GensimWordTokenizer(symbol="fgmas", general=False)
   >>> token_item = tokenizer("有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$,\
   ... 若$x,y$满足约束条件公式$\\FormFigureBase64{wrong2?}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$")
   >>> print(token_item.tokens[:10])
   ['公式', '[FORMULA]', '如图', '[FIGURE]', '[FORMULA]', '约束条件', '公式', '[FORMULA]', '[SEP]', '[FORMULA]']
