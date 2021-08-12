预训练
=======

在自然语言处理领域中，预训练语言模型（Pre-trained Language Models）已成为非常重要的基础技术。
我们将在本章节介绍EduNLP中预训练工具：

* 如何从零开始用一份语料训练得到一个预训练模型
* 如何加载预训练模型
* 公开的预训练模型


训练模型
---------

基本步骤：
1.确定模型的类型，选择适合的Tokenizer（GensimWordTokenizer、 GensimSegTokenizer），使之令牌化；

2.调用train_vector函数，即可得到所需的预训练模型。

Examples
        >>> tokenizer = GensimWordTokenizer(symbol="gmas", general=True)
        >>> token_item = tokenizer("有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$,\
        ... 若$x,y$满足约束条件公式$\\FormFigureBase64{wrong2?}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$")
        >>> print(token_item.tokens[:10])
        ['公式', '[FORMULA]', '如图', '[FIGURE]', 'x', ',', 'y', '约束条件', '公式', '[FORMULA]']
        
        # 10 dimension with fasstext method
        train_vector(sif_items, "../../../data/w2v/gensim_luna_stem_tf_", 10, method="d2v")

装载模型
--------
将所得到的模型传入I2V模块即可装载模型
 
Examples
        
        >>> model_path = "../test_model/test_gensim_luna_stem_tf_d2v_256.bin"
        >>> i2v = D2V("text","d2v",filepath=model_path, pretrained_t2v = False)


公开模型一览
------------

模型训练数据说明：

* 当前【词向量w2v】【句向量d2v】模型所用的数据均为 【高中学段】 的题目
* 测试数据：`[OpenLUNA.json]<http://base.ustc.edu.cn/data/OpenLUNA/OpenLUNA.json>`_

当前提供以下模型，更多分学科、分题型模型正在训练中，敬请期待
    "d2v_all_256"(全科)，"d2v_sci_256"(理科)，"d2v_eng_256"（文科），"d2v_lit_256"(英语)
