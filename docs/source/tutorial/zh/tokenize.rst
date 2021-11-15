令牌化
=======

令牌化是自然语言处理中一项基本但是非常重要的步骤，它更令人为所熟知的名字是分句和分词。
在EduNLP中我们将令牌化分为不同的粒度，为避免歧义，我们定义如下：

* 词/字级别：分词

* 句级别：分句

* 资源级别：令牌化

本模块提供题目文本的令牌化解析（Tokenization），将题目转换成令牌序列，方便后续向量化表征试题。

在进入此模块前需要先后将item经过 `语法解析 <parse.rst>`_ 和 `成分分解 <seg.rst>`_ 处理，之后对切片后的item中的各个元素进行分词，提供深度选项，可以按照需求选择所有地方切分或者在部分标签处切分（比如\SIFSep、\SIFTag处）；对标签添加的位置也可以进行选择，可以在头尾处添加或仅在头或尾处添加。

具有两种模式，一种是linear模式，用于对文本进行处理（使用jieba库进行分词）；一种是ast模式，用于对公式进行解析。

分词
-------

词解析（text-tokenization）：一个句子（不含公式）是由若干“词”按顺序构成的，将一个句子切分为若干词的过程称为“词解析”。根据词的粒度大小，又可细分为“词组解析”和"单字解析"。

::

   - 词组解析 (word-tokenization)：每一个词组为一个“令牌”（token）。
   
   - 单字解析 (char-tokenization)：单个字符即为一个“令牌”（token）。
    

词解析分为两个主要步骤：

1. 分词：  

   - 词组解析：使用分词工具切分并提取题目文本中的词。本项目目前支持的分词工具有：`jieba`

   - 单字解析：按字符划分。

2. 筛选：过滤指定的停用词。   

   本项目默认使用的停用词表：`[stopwords] <https://github.com/bigdata-ustc/EduNLP/blob/master/EduNLP/meta_data/sif_stopwords.txt>`_  
   你也可以使用自己的停用词表，具体使用方法见下面的示例。

Examples：

::

   from EduNLP.SIF.tokenization.text import tokenize 
   >>> text = "三角函数是基本初等函数之一"
   >>> tokenize(text, granularity="word")
   ['三角函数', '初等', '函数']
   
   >>> tokenize(text, granularity="char")
   ['三', '角', '函', '数', '基', '初', '函', '数']
    
分句
-------

将较长的文档切分成若干句子的过程称为“分句”。每个句子为一个“令牌”（token）（待实现）。

令牌化
-------
即综合解析，将带公式的句子切分为若干标记的过程。每个标记为一个“令牌”（token）。

此功能对应的实现函数为tokenize，将已经经过结构成分分解后的item传入其中即可得到所需结果。

::

   from EduNLP.Tokenizer import get_tokenizer
   >>> items = "如图所示，则三角形$ABC$的面积是$\\SIFBlank$。$\\FigureID{1}$"
   >>> tokenize(SegmentList(items))
   ['如图所示', '三角形', 'ABC', '面积', '\\\\SIFBlank', \\FigureID{1}]
   >>> tokenize(SegmentList(items),formula_params={"method": "ast"})
   ['如图所示', '三角形', <Formula: ABC>, '面积', '\\\\SIFBlank', \\FigureID{1}]



我们提供了多种已经封装好的令牌化器供用户便捷调用，通过查看 ``./EduNLP/Tokenizer/tokenizer.py`` 及 ``./EduNLP/Pretrain/gensim_vec.py`` 可以查看更多令牌化器，下面是一个完整的令牌化器列表:

- TextTokenizer

- PureTextTokenizer

- GensimSegTokenizer

- GensimWordTokenizer


TextTokenizer
+++++++++++++++++++++

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
+++++++++++++++++++++

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
