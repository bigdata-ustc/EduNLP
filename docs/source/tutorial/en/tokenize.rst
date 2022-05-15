Syntax Parsing
=================

In educational resources, texts and formulas have internal implicit or explicit syntax structures. It is of great benefit for further processing to extract these structures.

* Text syntax structure parsing

* Formula syntax structure parsing


Text syntax structure parsing
--------------------------------
According to the granularity, text syntax parsing can be **sentence-tokenization** and **text-tokenization**

Sentence-tokenization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Sentence-tokenization` divides a long document into sentences, each sentence is a `token`. (TODO)

Text-tokenization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
`Text-tokenization` divides a sentence into "several words", each word is a `token`. According to different word granularity, we have `phrase-parsing` and `word-parsing`.


Overall steps
"""""""""""""""""""""""""

1. Segmentation

- Phrase parsing: using existing tools to do segmentation and extraction
    Currently supported tools: `jieba`
- Word paring: segmentation with word by word

2. Filter

- Default stop-word table: `stopwords <https://github.com/bigdata-ustc/EduNLP/blob/master/EduNLP/meta_data/sif_stopwords.txt>`_
- You are allowed to use custom stop-word table, please follow the example.


Example
"""""""""""""""""""""""""

Import modules

::

  from EduNLP.SIF.tokenization.text import tokenize


Input

::

  text = "三角函数是基本初等函数之一"


Phrase parsing

::

  # EduNLP default stop-word table
  >>> tokenize(text, granularity="word")
  ['三角函数', '初等', '函数']


Word parsing

::

  # EduNLP default stop-word table
  >>> tokenize(text, granularity="char")
  ['三', '角', '函', '数', '基', '初', '函', '数']


Custom stop words

::

  >>> spath = "test_stopwords.txt"
  >>> from EduNLP.SIF.tokenization.text.stopwords import get_stopwords
  >>> stopwords = get_stopwords(spath)
  >>> stopwords
  {'一旦', '一时', '一来', '一样', '一次', '一片', '一番', '一直', '一致'}
  >>> tokenize(text, granularity="word", stopwords=stopwords)
  ['三角函数', '是', '基本', '初等', '函数', '之一']


Formula syntax parsing
--------------------

Formula tokenization: science text usually contains formula, "formula tokenization" converts formula to character list or abstract syntax tree.

Including two ways:
- Linear formula parsing
- AST formula parsing

.. note::

  This part is mainly about how to get different format of formula parsing result. Please refer to `EduNLP.Formula` for low-level implementation.


1. Linear formula parsing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set method to `linear`.

::
  >>> tokenize(formula, method="linear")
  ['\\frac', '{', '\\pi', '}', '{', 'x', '+', 'y', '}', '+', '1', '=', 'x']


2. AST formula parsing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set method to `ast`


::
  >>> tokenize(formula, method="ast", return_type="list", ord2token=False)
  ['\\pi', '{ }', 'x', '+', 'y', '{ }', '\\frac', '+', '1', '=', 'x']


3. AST formula parsing and variable symbolization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you care only about the structure and type of formula, and variables themself is ignorable.
For example, `x^2 + y = 1` is structurally identical with `w^2 + z = 1`.
You can convert variable names into tokens, by set `ord2token=True`.

::
  >>> tokenize(formula, method="ast", return_type="list", ord2token=True)
  ['mathord', '{ }', 'mathord', '+', 'mathord', '{ }', '\\frac', '+', 'textord', '=', 'mathord']


4. AST formula parsing and variable normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want variables distinguishable in **3**, just set `var_numbering=True`.

::
  >>> tokenize(formula, method="ast", return_type="list", ord2token=True, var_numbering=True)
  ['mathord_con', '{ }', 'mathord_0', '+', 'mathord_1', '{ }', '\\frac', '+', 'textord', '=', 'mathord_0']

