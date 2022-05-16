语法解析
=========

在教育资源中，文本、公式都具有内在的隐式或显式的语法结构，提取这种结构对表征学习是大有裨益的：

* 文本语法结构解析

* 公式语法结构解析

文本语法结构解析
--------------------

根据题目文本切分粒度的大小，文本解析又分为 **“句解析”** 和 **“词解析”**。


句解析（sentence-tokenization）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

将较长的文档切分成若干句子的过程称为“分句”。每个句子为一个“令牌”（token）。（待实现）    
  

词解析（text-tokenization）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

一个句子（不含公式）是由若干“词”按顺序构成的，将一个句子切分为若干词的过程称为“词解析”。根据词的粒度大小，又可细分为“词组解析”和"单字解析"。

主要步骤
"""""""""""""""""""""""""

（1）分词

- 词组解析：使用分词工具切分并提取题目文本中的词。  
    本项目目前支持的分词工具有：`jieba`   
- 单字解析：按字符划分。

（2） 过滤停用词

- 本项目默认使用的停用词表：`stopwords <https://github.com/bigdata-ustc/EduNLP/blob/master/EduNLP/meta_data/sif_stopwords.txt>`_
- 你也可以使用自己的停用词表，具体使用方法见下面的示例。


示例
"""""""""""""""""""""""""

导入模块

::

  from EduNLP.SIF.tokenization.text import tokenize 


输入

::

  text = "三角函数是基本初等函数之一"


词组解析

::

  # 输出：默认使用 EduNLP 项目提供的停用词表
  >>> tokenize(text, granularity="word")
  ['三角函数', '初等', '函数']


单字解析

::

  # 输出：默认使用 EduNLP 项目提供的停用词表
  >>> tokenize(text, granularity="char")
  ['三', '角', '函', '数', '基', '初', '函', '数']


使用自己的停用词表

::

  >>> spath = "test_stopwords.txt"
  >>> from EduNLP.SIF.tokenization.text.stopwords import get_stopwords
  >>> stopwords = get_stopwords(spath)
  >>> stopwords
  {'一旦', '一时', '一来', '一样', '一次', '一片', '一番', '一直', '一致'}
  >>> tokenize(text, granularity="word", stopwords=stopwords)
  ['三角函数', '是', '基本', '初等', '函数', '之一']


公式语法结构解析
--------------------

公式解析（formula-tokenization）：理科类文本中常常含有公式。将一个符合 latex 语法的公式解析为标记字符列表或抽象语法树的过程称为“公式解析”。

包括两种方案

- 公式线性解析
- 公式AST解析

.. note::

  本小节主要介绍如何获取不同格式的公式解析结果。公式解析的底层实现请参考：`EduNLP.Formula` 部分。


（1）公式线性解析
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果您想按 latex 语法标记拆分公式的各个部分，并得到顺序序列结果，输出方法可以选择：`linear`

::
  >>> tokenize(formula, method="linear")
  ['\\frac', '{', '\\pi', '}', '{', 'x', '+', 'y', '}', '+', '1', '=', 'x']


（2） 公式AST解析
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果您想得到公式解析出的语法分析树序列，输出方法可以选择：`ast`

> 抽象语法分析树，简称语法树（Syntax tree），是源代码语法结构的一种抽象表示。它以树状的形式表现编程语言的语法结构，树上的每个节点都表示源代码中的一种结构。  
> 因此，ast 可以看做是公式的语法结构表征。

::
  >>> tokenize(formula, method="ast", return_type="list", ord2token=False)
  ['\\pi', '{ }', 'x', '+', 'y', '{ }', '\\frac', '+', '1', '=', 'x']


（3）公式AST解析+变量符号化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果您只是关心公式的结构和类型，并不关心变量具体是什么，比如二元二次方程 `x^2 + y = 1` ，它从公式结构和类型上来说，和 `w^2 + z = 1` 没有区别。  
此时，您可以设置如下参数：`ord2token = True`，将公式变量名转换成 token

::
  >>> tokenize(formula, method="ast", return_type="list", ord2token=True)
  ['mathord', '{ }', 'mathord', '+', 'mathord', '{ }', '\\frac', '+', 'textord', '=', 'mathord']


（4） 公式AST解析+变量标准化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果您除了 （3） 中提供的功能之外，还需要区分不同的变量。此时可以另外设置参数：`var_numbering=True`

::
  >>> tokenize(formula, method="ast", return_type="list", ord2token=True, var_numbering=True)
  ['mathord_con', '{ }', 'mathord_0', '+', 'mathord_1', '{ }', '\\frac', '+', 'textord', '=', 'mathord_0']

