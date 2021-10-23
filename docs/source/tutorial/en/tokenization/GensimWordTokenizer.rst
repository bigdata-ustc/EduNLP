GensimWordTokenizer
=====================

By default, the pictures, blanks in the question text and other parts of the incoming item are converted into special characters for data security and the tokenization of text, formulas, labels and separators. Also, the tokenizer uses linear analysis method for text and abstract syntax tree method for formulas respectively. You can choose each of them by ``general`` parameter:

-true, it means that the incoming item conforms to SIF and the linear analysis method should be used.
-false, it means that the incoming item doesn't conform to SIF and the abstract syntax tree method should be used.

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
