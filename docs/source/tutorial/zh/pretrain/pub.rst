公开模型一览
------------

版本说明
##################

一级版本

* 公开版本1（luna_pub）：高考
* 公开版本2（ luna_pub_large）：高考 + 地区试题

二级版本：

* 小科（Chinese,Math,English,History,Geography,Politics,Biology,Physics,Chemistry）
* 大科（理科science、文科literal、全科all）

三级版本：【待完成】

* 不使用第三方初始化词表
* 使用第三方初始化词表 

模型训练数据说明
##################

* 当前【词向量w2v】【句向量d2v】模型所用的数据均为 【高中学段】 的题目
* 测试数据：`[OpenLUNA.json] <http://base.ustc.edu.cn/data/OpenLUNA/OpenLUNA.json>`_

当前提供以下模型，更多分学科、分题型模型正在训练中，敬请期待
    "d2v_all_256"(全科)，"d2v_sci_256"(理科)，"d2v_eng_256"（英语），"d2v_lit_256"(文科)

模型训练案例
------------

获得数据集
####################

.. toctree::
   :maxdepth: 1
   :titlesonly:

   prepare_dataset  <../../../build/blitz/pretrain/prepare_dataset.ipynb>

gensim模型d2v例子
####################

.. toctree::
   :maxdepth: 1
   :titlesonly:

   d2v_bow_tfidf  <../../../build/blitz/pretrain/gensim/d2v_bow_tfidf.ipynb>
   d2v_general  <../../../build/blitz/pretrain/gensim/d2v_general.ipynb>
   d2v_stem_tf  <../../../build/blitz/pretrain/gensim/d2v_stem_tf.ipynb>

gensim模型w2v例子
####################

.. toctree::
   :maxdepth: 1
   :titlesonly:

   w2v_stem_text  <../../../build/blitz/pretrain/gensim/w2v_stem_text.ipynb>
   w2v_stem_tf  <../../../build/blitz/pretrain/gensim/w2v_stem_tf.ipynb>

seg_token例子
####################

.. toctree::
   :maxdepth: 1
   :titlesonly:

   d2v.ipynb  <../../../build/blitz/pretrain/seg_token/d2v.ipynb>
   d2v_d1  <../../../build/blitz/pretrain/seg_token/d2v_d1.ipynb>
   d2v_d2  <../../../build/blitz/pretrain/seg_token/d2v_d2.ipynb>
