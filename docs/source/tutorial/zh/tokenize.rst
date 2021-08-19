令牌化
=======

令牌化是自然语言处理中一项基本但是非常重要的步骤，它更令人为所熟知的名字是分句和分词。
在EduNLP中我们将令牌化分为不同的粒度，为避免歧义，我们定义如下：

* 词/字级别：分词

* 句级别：分句

* 资源级别：令牌化

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

对切片后的item中的各个元素进行分词，提供深度选项，可以按照需求选择所有地方切分或者在部分标签处切分（比如\SIFSep、\SIFTag处）；对标签添加的位置也可以进行选择，可以在头尾处添加或仅在头或尾处添加。

具有两种模式，一种是linear模式，用于对文本进行处理（使用jieba库进行分词）；一种是ast模式，用于对公式进行解析。

Examples

::
    
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
我们提供了多种已经封装好的令牌化器供用户便捷调用，下面是一个示例:


    Examples
    ------------
    >>> items = ["已知集合$A=\\left\\{x \\mid x^{2}-3 x-4<0\\right\\}, \\quad B=\\{-4,1,3,5\\}, \\quad$ 则 $A \\cap B=$"]
    >>> tokenizer = TextTokenizer()
    >>> tokens = tokenizer(items)
    >>> next(tokens)  # doctest: +NORMALIZE_WHITESPACE
    ['已知', '集合', 'A', '=', '\\left', '\\{', 'x', '\\mid', 'x', '^', '{', '2', '}', '-', '3', 'x', '-', '4', '<',
    '0', '\\right', '\\}', ',', '\\quad', 'B', '=', '\\{', '-', '4', ',', '1', ',', '3', ',', '5', '\\}', ',',
    '\\quad', 'A', '\\cap', 'B', '=']





通过查看"./EduNLP/Tokenizer/tokenizer.py"及"./EduNLP/Pretrain/gensim_vec.py"可以查看更多令牌化器，下面是一个完整的令牌化器列表

.. toctree::
  :maxdepth: 1
  :titlesonly:

  TextTokenizer
  GensimSegTokenizer
  GensimWordTokenizer

