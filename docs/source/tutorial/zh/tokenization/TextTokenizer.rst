

Examples
--------

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
