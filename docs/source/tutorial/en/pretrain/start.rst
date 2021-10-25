Train the model
------------------

Call train_Vector function interface directly to make the training model easier. This section calls the relevant training models in the gensim library. At present, the training methods of "sg"、 "cbow"、 "fastext"、 "d2v"、 "bow"、 "tfidf" are provided. Parameter embedding_dim is also provided for users to determine vector dimension according to their needs.

Basic Steps
##################

1.Determine the type of model and select the appropriate tokenizer (GensimWordTokenizer、 GensimSegTokenizer) to finish tokenization.

2.Call train_vector function to get the required pre-trained model。

Examples：

::

        >>> tokenizer = GensimWordTokenizer(symbol="gmas", general=True)
        >>> token_item = tokenizer("有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$,\
        ... 若$x,y$满足约束条件公式$\\FormFigureBase64{wrong2?}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$")
        >>> print(token_item.tokens[:10])
        ['公式', '[FORMULA]', '如图', '[FIGURE]', 'x', ',', 'y', '约束条件', '公式', '[FORMULA]']
        
        # 10 dimension with fasstext method
        train_vector(sif_items, "../../../data/w2v/gensim_luna_stem_tf_", 10, method="d2v")
