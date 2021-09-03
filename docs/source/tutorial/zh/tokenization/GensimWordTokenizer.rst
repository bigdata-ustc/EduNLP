GensimWordTokenizer
=====================

此令牌解析器在默认情况下对传入的item中的图片、题目空缺符等部分转换成特殊字符进行保护，从而对文本、公式、标签、分隔符进行令牌化操作。此外，从令牌化方法而言，此令牌解析器对文本均采用线性的分析方法，而对公式采用抽象语法树的分析方法，提供了general参数可供使用者选择：当general为true的时候则代表着传入的item并非标准格式，此时对公式也使用线性的分析方法；当general为false时则代表使用抽象语法树的方法对公式进行解析。

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
