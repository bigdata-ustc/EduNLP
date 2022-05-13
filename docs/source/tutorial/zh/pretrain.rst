=======
预训练
=======

在自然语言处理领域中，预训练语言模型（Pre-trained Language Models）已成为非常重要的基础技术。
我们将在本章节介绍EduNLP中预训练工具：

* 如何从零开始用一份语料训练得到一个预训练模型
* 如何加载预训练模型
* 公开的预训练模型


训练模型
-----------------------

模型模块的接口定义在 `EduNLP.Pretrain` 中，包含令牌化容器、数据处理、模型定义、模型训练等功能。


基本步骤：
#######################################

以训练word2vec为例说明：

- 确定模型的类型，选择适合的Tokenizer（如GensimWordTokenizer、PureTextTokenizer等），使之令牌化；

- 调用train_vector函数，即可得到所需的预训练模型。


Examples：

::
   
   from EduNLP.Tokenizer import PureTextTokenizer
   from EduNLP.Pretrain import train_vector

   items = [
      r"题目一：如图几何图形．此图由三个半圆构成，三个半圆的直径分别为直角三角形$ABC$的斜边$BC$, 直角边$AB$, $AC$.$\bigtriangleup ABC$的三边所围成的区域记为$I$,黑色部分记为$II$, 其余部分记为$III$.在整个图形中随机取一点，此点取自$I,II,III$的概率分别记为$p_1,p_2,p_3$,则$\SIFChoice$$\FigureID{1}$",
      r"题目二: 如图来自古希腊数学家希波克拉底所研究的几何图形．此图由三个半圆构成，三个半圆的直径分别为直角三角形$ABC$的斜边$BC$, 直角边$AB$, $AC$.$\bigtriangleup ABC$的三边所围成的区域记为$I$,黑色部分记为$II$, 其余部分记为$III$.在整个图形中随机取一点，此点取自$I,II,III$的概率分别记为$p_1,p_2,p_3$,则$\SIFChoice$$\FigureID{1}$"
   ]

   tokenizer = PureTextTokenizer()
   token_items = [t for t in tokenizer(raw_items)]
   
   print(token_items[0[:10])
   # ['公式', '[FORMULA]', '如图', '[FIGURE]', 'x', ',', 'y', '约束条件', '公式', '[FORMULA]']
   
   # 10 dimension with fasstext method
   train_vector(sif_items, "../../../data/w2v/gensim_luna_stem_tf_", 10, method="d2v")


装载模型
-----------------------

将所得到的模型传入I2V模块即可装载模型，通过向量化获取题目表征。
 
Examples：

::

   model_path = "../test_model/d2v/test_gensim_luna_stem_tf_d2v_256.bin"
   i2v = D2V("text","d2v",filepath=model_path, pretrained_t2v = False)



更多模型训练案例
-----------------------

获得数据集
########################################

.. toctree::
   :maxdepth: 1
   :titlesonly:

   prepare_dataset  <../../build/blitz/pretrain/prepare_dataset.ipynb>

gensim模型d2v例子
########################################

.. toctree::
   :maxdepth: 1
   :titlesonly:

   d2v_bow_tfidf  <../../build/blitz/pretrain/gensim/d2v_bow_tfidf.ipynb>
   d2v_general  <../../build/blitz/pretrain/gensim/d2v_general.ipynb>
   d2v_stem_tf  <../../build/blitz/pretrain/gensim/d2v_stem_tf.ipynb>

gensim模型w2v例子
########################################

.. toctree::
   :maxdepth: 1
   :titlesonly:

   w2v_stem_text  <../../build/blitz/pretrain/gensim/w2v_stem_text.ipynb>
   w2v_stem_tf  <../../build/blitz/pretrain/gensim/w2v_stem_tf.ipynb>


进阶表征模型示例
########################################

.. nbgallery::
    :caption: This is a thumbnail gallery:
    :name: pretrain_gallery1
    :glob:

    Emlo预训练  <../../build/blitz/pretrain/elmo.ipynb>

    Bert预训练  <../../build/blitz/pretrain/bert.ipynb>


.. nbgallery::
    :caption: This is a thumbnail gallery:
    :name: pretrain_gallery2
    :glob:

    DisenQNet预训练  <../../build/blitz/pretrain/disenq.ipynb>
    
    QuesNet预训练  <../../build/blitz/pretrain/quesnet.ipynb>
