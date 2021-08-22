训练模型
------------

如需训练模型则可直接train_vector函数接口，来使使训练模型更加方便。模块调用gensim库中的相关训练模型，目前提供了"sg"、 "cbow"、 "fastext"、 "d2v"、 "bow"、 "tfidf"的训练方法，并提供了embedding_dim参数，使之可以按照需求确定向量的维度。

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
