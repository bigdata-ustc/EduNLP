训练模型
---------

基本步骤
##################

1.确定模型的类型，选择适合的Tokenizer（GensimWordTokenizer、 GensimSegTokenizer），使之令牌化；

2.调用train_vector函数，即可得到所需的预训练模型。

Examples：

::

        >>> tokenizer = GensimWordTokenizer(symbol="gmas", general=True)
        >>> token_item = tokenizer("有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$,\
        ... 若$x,y$满足约束条件公式$\\FormFigureBase64{wrong2?}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$")
        >>> print(token_item.tokens[:10])
        ['公式', '[FORMULA]', '如图', '[FIGURE]', 'x', ',', 'y', '约束条件', '公式', '[FORMULA]']
        
        # 10 dimension with fasstext method
        train_vector(sif_items, "../../../data/w2v/gensim_luna_stem_tf_", 10, method="d2v")
