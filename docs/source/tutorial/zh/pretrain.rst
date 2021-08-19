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
##################

1.确定模型的类型，选择适合的Tokenizer（GensimWordTokenizer、 GensimSegTokenizer），使之令牌化；

2.调用train_vector函数，即可得到所需的预训练模型。

Examples

::

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

::

        >>> model_path = "../test_model/test_gensim_luna_stem_tf_d2v_256.bin"
        >>> i2v = D2V("text","d2v",filepath=model_path, pretrained_t2v = False)


公开模型一览
------------

版本说明
##################

一级版本

* 全量版本（luna_private ）: 高考 + 考试中心 + 题库中国 + 学库宝
* 公开版本1（luna_pub1）：高考
* 公开版本2（luna_pub2）：高考 + 题库中国
* 当前luna标准库（luna_v0）: 当前为高考 + 题库中国 + 考试中心

二级版本：

* 小科（Chinese,Math,English,History,Geography,Politics,Biology,Physics,Chemistry）
* 大科（理科science、文科literal、全科all）

三级版本：【待完成】

* 不使用第三方初始化词表
* 使用第三方初始化词表 



模型命名规则：一级版本 + 二级版本 + gensim_luna_stem +  分词规则 + 模型方法 + 维度

Examples

::

     全量版本-全学科的D2V模型路径：
     `/share/qlh/d2v_model/luna_private/luna_private_all_gensim_luna_stem_general_d2v_256.bin`
     （备注：一个D2V模型含4个bin后缀的文件）

模型训练数据说明
##################

* 当前【词向量w2v】【句向量d2v】模型所用的数据均为 【高中学段】 的题目
* 测试数据：`[OpenLUNA.json] <http://base.ustc.edu.cn/data/OpenLUNA/OpenLUNA.json>`_

当前提供以下模型，更多分学科、分题型模型正在训练中，敬请期待
    "d2v_all_256"(全科)，"d2v_sci_256"(理科)，"d2v_eng_256"（文科），"d2v_lit_256"(英语)

模型训练案例
------------

获得数据集
####################

.. toctree::
   :maxdepth: 1
   :titlesonly:

   prepare_dataset  <../../build/blitz/pretrain/prepare_dataset.ipynb>

gensim模型d2v例子
####################

.. toctree::
   :maxdepth: 1
   :titlesonly:

   d2v_bow_tfidf  <../../build/blitz/pretrain/gensim/d2v_bow_tfidf.ipynb>
   d2v_general  <../../build/blitz/pretrain/gensim/d2v_general.ipynb>
   d2v_stem_tf  <../../build/blitz/pretrain/gensim/d2v_stem_tf.ipynb>

gensim模型w2v例子
####################

.. toctree::
   :maxdepth: 1
   :titlesonly:

   w2v_stem_text  <../../build/blitz/pretrain/gensim/w2v_stem_text.ipynb>
   w2v_stem_tf  <../../build/blitz/pretrain/gensim/w2v_stem_tf.ipynb>

seg_token例子
####################

.. toctree::
   :maxdepth: 1
   :titlesonly:

   d2v.ipynb  <../../build/blitz/pretrain/seg_token/d2v.ipynb>
   d2v_d1  <../../build/blitz/pretrain/seg_token/d2v_d1.ipynb>
   d2v_d2  <../../build/blitz/pretrain/seg_token/d2v_d2.ipynb>
