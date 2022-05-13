Tokenization
==============

Tokenization, known as word segmentation and sentence segmentation, is a basic but very important step in the field of NLP.
In EduNLP, we divided Tokenization into different levels according to different granularity. To avoid ambiguity, we define as follows:

* Word/char level: word segmentation

* Sentence level: sentence segmentation

* Resource level: tokenization

This module provides tokenization function of question text, converting questions into token sequences to facilitate the vectorization of questions. After that, each element in the sliced item needs word segmentation. In this step, there is a depth option. You can select all positions or some labels for segmentation according to your needs, such as \SIFSep and \SIFTag. You can also select where to add labels, either at the head and tail or only at the head or tail.

There are two modes: one is linear mode, which is used for text processing (word segmentation using jieba library). The other one is ast mode, which is used to parse the formula.

Word Segmentation
---------------------

Text-tokenization: A sentence (without formulas) consists of several "words" in order. The process of dividing a sentence into several words is called "Text-tokenization". According to the granularity of "words", it can be subdivided into "Word-tokenization" and "Char-tokenization".

::

   - Word-tokenization: each phrase is a token.
   
   - Char-tokenization: each character is a token.
    

Text-tokenization is divided into two main steps:

1. Text-tokenization:

   - Word-tokenization: use the word segmentation tool to segment and extract words from the question text. Our project supports `jieba`.

   - Char-tokenization: process text by character.

2. Filter: filter the specified stopwords.

   The default stopwords used in this project:`[stopwords] <https://github.com/bigdata-ustc/EduNLP/blob/master/EduNLP/meta_data/sif_stopwords.txt>`_
   You can also use your own stopwords. The following example demonstrates how to use.

Examples:

::

   from EduNLP.SIF.tokenization.text import tokenize 
   >>> text = "三角函数是基本初等函数之一"
   >>> tokenize(text, granularity="word")
   ['三角函数', '初等', '函数']
   
   >>> tokenize(text, granularity="char")
   ['三', '角', '函', '数', '基', '初', '函', '数']
    
Sentence Segmentation
----------------------------

During the process of sentence segmentation, a long document is divided into several sentences. Each sentence is a "token" (to be realized).

Tokenization
--------------

Tokenization is comprehensive analysis. In this process, sentences with formulas are segmented into several markers. Each marker is a "token".

The implementation of this function is tokenize function. The required results can be obtained by passing in items after Structural Component Segmentation.

::

   from EduNLP.Tokenizer import get_tokenizer
   >>> items = "如图所示，则三角形$ABC$的面积是$\\SIFBlank$。$\\FigureID{1}$"
   >>> tokenize(SegmentList(items))
   ['如图所示', '三角形', 'ABC', '面积', '\\\\SIFBlank', \\FigureID{1}]
   >>> tokenize(SegmentList(items),formula_params={"method": "ast"})
   ['如图所示', '三角形', <Formula: ABC>, '面积', '\\\\SIFBlank', \\FigureID{1}]



You can view ``./EduNLP/Tokenizer/tokenizer.py`` and ``./EduNLP/Pretrain/gensim_vec.py`` for more tokenizers. We provide some encapsulated tokenizers for users to call them conveniently. Following is a complete list of tokenizers:

- TextTokenizer

- PureTextTokenizer

- GensimSegTokenizer

- GensimWordTokenizer


TextTokenizer
+++++++++++++++++++++

By default, the pictures, labels, separators, blanks in the question text and other parts of the incoming item are converted into special characters for data security and tokenization of text and formulas. Also, the tokenizer uses linear analysis method for text and formulas, and the ``key`` parameter provided is used to preprocess the incoming item, which will be improved based on users' requirements in the future.

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

By default, the pictures, labels, separators, blanks in the question text and other parts of the incoming item are converted into special characters for data security. At the same time, special formulas such as $\\FormFigureID{...}$ and $\\FormFigureBase64{...}$ are screened out to facilitate the tokenization of text and plain text formulas. Also, the tokenizer uses linear analysis method for text and formulas, and the ``key`` parameter provided is used to preprocess the incoming item, which will be improved based on users' requirements in the future.

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

By default, the pictures, blanks in the question text and other parts of the incoming item are converted into special characters for data security and the tokenization of text, formulas, labels and separators. Also, the tokenizer uses linear analysis method for text and abstract syntax tree method for formulas respectively. You can choose each of them by ``general`` parameter:

-true, it means that the incoming item conforms to SIF and the linear analysis method should be used.
-false, it means that the incoming item doesn't conform to SIF and the abstract syntax tree method should be used.

GensimSegTokenizer
++++++++++++++++++++

By default, the pictures, separators, blanks in the question text and other parts of the incoming item are converted into special characters for data security and tokenization of text, formulas and labels. Also, the tokenizer uses linear analysis method for text and abstract analysis method of syntax tree for formulas.

Compared to GensimWordTokenizer, the main differences are:

* It provides the depth option for segmentation position, such as \SIFSep and \SIFTag.
* By default, labels are inserted in the header of item components (such as text and formulas).

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
