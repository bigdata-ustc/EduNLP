Tokenization
--------------
Tokenization is comprehensive analysis. In this process, sentences with formulas are segmented into several markers. Each marker is a "token".
We provide some encapsulated tokenizers for users to call them conveniently. The following is a complete list of tokenizers.

Examples
    
::

    >>> items = ["已知集合$A=\\left\\{x \\mid x^{2}-3 x-4<0\\right\\}, \\quad B=\\{-4,1,3,5\\}, \\quad$ 则 $A \\cap B=$"]
    >>> tokenizer = TextTokenizer()
    >>> tokens = tokenizer(items)
    >>> next(tokens)  # doctest: +NORMALIZE_WHITESPACE
    ['已知', '集合', 'A', '=', '\\left', '\\{', 'x', '\\mid', 'x', '^', '{', '2', '}', '-', '3', 'x', '-', '4', '<',
    '0', '\\right', '\\}', ',', '\\quad', 'B', '=', '\\{', '-', '4', ',', '1', ',', '3', ',', '5', '\\}', ',',
    '\\quad', 'A', '\\cap', 'B', '=']



You can view ``./EduNLP/Tokenizer/tokenizer.py`` and ``./EduNLP/Pretrain/gensim_vec.py`` for more tokenizers. Following is a complete list of tokenizers:

.. toctree::
  :maxdepth: 1
  :titlesonly:

  ../tokenization/TextTokenizer
  ../tokenization/PureTextTokenizer
  ../tokenization/GensimSegTokenizer
  ../tokenization/GensimWordTokenizer
