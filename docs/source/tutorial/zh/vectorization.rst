=========
向量化
=========

向量化过程是将原始题目（item）转成向量（vector）的过程，它包括两个步骤：

- 使用 `Tokenizer` 令牌化容器 将原始题目（item）转化为令牌化序列（tokens）;
- 使用 `T2V` 向量化容器 将令牌化序列（tokens）转成向量（vector）。


I2V 向量化容器
==================
为了使用户能直接使用本地的（或公开的）预训练模型，我们提供了`I2V向量化容器`, 将令牌化、向量化操作同时封装起来。

`I2V` 模块提供两种向量化方法：

- 使用开源预训练模型
- 使用本地预训练模型

输入数据
---------------------------------------------------

输入原始的题目列表，题目内容以文本或字典的形式给出

::

   items = [
      r"题目一：如图几何图形．此图由三个半圆构成，三个半圆的直径分别为直角三角形$ABC$的斜边$BC$, 直角边$AB$, $AC$.$\bigtriangleup ABC$的三边所围成的区域记为$I$,黑色部分记为$II$, 其余部分记为$III$.在整个图形中随机取一点，此点取自$I,II,III$的概率分别记为$p_1,p_2,p_3$,则$\SIFChoice$$\FigureID{1}$",
      r"题目二: 如图来自古希腊数学家希波克拉底所研究的几何图形．此图由三个半圆构成，三个半圆的直径分别为直角三角形$ABC$的斜边$BC$, 直角边$AB$, $AC$.$\bigtriangleup ABC$的三边所围成的区域记为$I$,黑色部分记为$II$, 其余部分记为$III$.在整个图形中随机取一点，此点取自$I,II,III$的概率分别记为$p_1,p_2,p_3$,则$\SIFChoice$$\FigureID{1}$"
   ]


使用开源预训练模型
---------------------------------------------------
调用 `get_pretrained_i2v` 加载开源模型，并获取对应的 `I2V` 向量化容器将给定的题目转成向量。

- 优点：用户不需要研究令牌化和模型加载的细节。令牌化和向量化的参数已由预训练模型的参数文件定义好。
- 缺点：不适合修改预训练的模型参数或令牌化容器参数

模型选择与使用
^^^^^^^^^^^^^^^^^^^^^^

根据题目所属学科选择预训练模型：(以下为部分开源模型示例)

+--------------------+------------------------+
|    预训练模型名称  |          学科          |
+====================+========================+
|    d2v_math_300    |         数学           |
+--------------------+------------------------+
|    w2v_math_300    |         数学           |
+--------------------+------------------------+
|    elmo_math_2048  |         数学           |
+--------------------+------------------------+
|    bert_math_768   |         数学           |
+--------------------+------------------------+
|    bert_taledu_768 |         数学           |
+--------------------+------------------------+
|    disenq_math_256 |         数学           |
+--------------------+------------------------+
|    quesnet_math_512|         数学           |
+--------------------+------------------------+


具体用法
^^^^^^^^^^^^^^^^^^^^^^

::

   from EduNLP import get_pretrained_i2v

   i2v = get_pretrained_i2v("w2v_eng_300")
   item_vector, token_vector = i2v(items)



使用本地预训练模型
------------------------------------

使用自己提供的任一预训练模型（给出模型存放路径即可）将给定的题目文本转成向量。

* 优点：可以使用自己的模型，另可调整训练参数，灵活性强。


提供的I2V容器
^^^^^^^^^^^^^^^^^^^^^^

+--------+---------+
| 名称   | I2V容器 |
+========+=========+
| w2v    | W2V     |
+--------+---------+
| d2v    | D2V     |
+--------+---------+
| elmo   | Elmo    |
+--------+---------+
| bert   | Bert    |
+--------+---------+
| disenq | DisenQ  |
+--------+---------+
| quesnet| QuesNet |
+--------+---------+

具体用法
^^^^^^^^^^^^^^^^^^^^^^

::

   from EduNLP.I2V import W2V
   
   # 加载向量化容器
   pretrained_path = "path/to/model"
   i2v = W2V("pure_text", "w2v", pretrained_path)
   
   # 向量化
   item_vector, token_vector = i2v(items)
   # or
   item_vector, token_vector = i2v.infer_vector(items)
   # or
   item_vector = i2v.infer_item_vector(items)
   token_vector = i2v.infer_token_vector(items)


.. note::

   不同模型的I2V容器在使用时略有差别，建议使用时查看对应的API文档或用法样例。


更多I2V容器使用示例
------------------------------------

.. nbgallery::
    :caption: This is a thumbnail gallery:
    :name: i2v_gallery1
    :glob:
    
    W2V向量化  <../../build/blitz/i2v/i2v_w2v.ipynb>
    
    D2V向量化  <../../build/blitz/i2v/i2v_d2v.ipynb>
    
    Elmo向量化  <../../build/blitz/i2v/i2v_elmo.ipynb>


.. nbgallery::
    :caption: This is a thumbnail gallery:
    :name: i2v_gallery2
    :glob:
    
    Bert向量化  <../../build/blitz/i2v/i2v_bert.ipynb>
    
    DisenQNet向量化  <../../build/blitz/i2v/i2v_disenq.ipynb>
    
    QuesNet向量化  <../../build/blitz/i2v/i2v_quesnet.ipynb>




T2V 向量化容器
==================

`T2V` 向量化容器能将题目的令牌序列（tokens）转成向量（vector）。

- 优点：此容器与令牌化容器相互分离，用户可以自主调整令牌化容器和向量化容器的参数，可用于个性化的需求。

`T2V` 模块提供两种向量化方法：

- 使用开源预训练模型
- 使用本地预训练模型

输入数据
------------------------------------

`T2V` 向量化容器的输入为题目的令牌化序列。因此，在调用 `T2V` 向量化容器之前，必须先使用 `Tokenizer` 令牌化容器获取 令牌序列列（token）。


::
   
   from EduNLP.Tokenizer import PureTextTokenize

   raw_items = [
      r"题目一：如图几何图形．此图由三个半圆构成，三个半圆的直径分别为直角三角形$ABC$的斜边$BC$, 直角边$AB$, $AC$.$\bigtriangleup ABC$的三边所围成的区域记为$I$,黑色部分记为$II$, 其余部分记为$III$.在整个图形中随机取一点，此点取自$I,II,III$的概率分别记为$p_1,p_2,p_3$,则$\SIFChoice$$\FigureID{1}$",
      r"题目二: 如图来自古希腊数学家希波克拉底所研究的几何图形．此图由三个半圆构成，三个半圆的直径分别为直角三角形$ABC$的斜边$BC$, 直角边$AB$, $AC$.$\bigtriangleup ABC$的三边所围成的区域记为$I$,黑色部分记为$II$, 其余部分记为$III$.在整个图形中随机取一点，此点取自$I,II,III$的概率分别记为$p_1,p_2,p_3$,则$\SIFChoice$$\FigureID{1}$"
   ]

   tokenizer = PureTextTokenizer()
   token_items = [t for t in tokenizer(raw_items)]


使用开源预训练模型
--------------------------------------------

.. note::

   开源模型列表同I2V部分


加载源预训练模型到W2V容器中：

::

   from EduNLP.Vector import get_pretrained_t2v

   model_dir = "path/to/save/model"
   t2v = get_pretrained_t2v("test_w2v", model_dir=model_dir)

   item_vector = t2v.infer_vector(token_items)
   # [array(), ..., array()]
   token_vector = t2v.infer_tokens(token_items)
   # [[array(), ..., array()], [...], [...]]


使用本地预训练模型
------------------------------------

提供的T2V容器：

+---------+------------+
| 名称    | T2V容器    |
+=========+============+
| w2v     | W2V        |
+---------+------------+
| d2v     | D2V        |
+---------+------------+
| elmo    | ElmoModel  |
+---------+------------+
| bert    | BertModel  |
+---------+------------+
| dienq   |DisenQModel |
+---------+------------+
|quesnet  |QuesNetModel|
+---------+------------+

加载本地模型到W2V容器中：

::

   from EduNLP.Vector import T2V, W2V

   path = "path_to_model"
   t2v = T2V('w2v', filepath=path)
   # 或
   # t2v = W2V(path)

   tem_vector = t2v.infer_vector(token_items)
   # [array(), ..., array()]
   token_vector = t2v.infer_tokens(token_items)
   # [[array(), ..., array()], [...], [...]]


.. note::

   不同模型的T2V容器在使用时略有差别，建议使用时查看对应的API文档或用法样例。


更多T2V容器使用示例
------------------------------------

.. nbgallery::
    :caption: This is a thumbnail gallery:
    :name: t2v_gallery1
    :glob:
    
    W2V向量化  <../../build/blitz/t2v/t2v_w2v.ipynb>

    D2V向量化  <../../build/blitz/t2v/t2v_d2v.ipynb>

    Elmo向量化  <../../build/blitz/t2v/t2v_elmo.ipynb>


.. nbgallery::
    :caption: This is a thumbnail gallery:
    :name: t2v_gallery2
    :glob:
    
    Bert向量化  <../../build/blitz/t2v/t2v_bert.ipynb>
    
    DisenQNet向量化  <../../build/blitz/t2v/t2v_disenq.ipynb>
    
    QuesNet向量化  <../../build/blitz/t2v/t2v_quesnet.ipynb>
