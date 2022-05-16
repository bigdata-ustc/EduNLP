==================
Vectorization
==================

The vectorization process is divided into two steps:
- Use `Tokenizer` to convert original questions (`item`) to tokenization sequence (`tokens`)
- Use `T2V` to convert tokenization sequence (`tokens`) to vectors


I2V container
=====================
`I2V container` makes it easy to use models(from local or open-source) for vectorization, it contains operations from tokenization to vectorization together in one pipeline.

`I2V` provides two ways of vectorization:

- Use a pretrained model from local
- Use an open-source pretrained model


Items to input
---------------------------------------------------
A list of questions or one question, and the text of a question is given as a `dictionary` or `string`

::

   items = [
      r"题目一：如图几何图形．此图由三个半圆构成，三个半圆的直径分别为直角三角形$ABC$的斜边$BC$, 直角边$AB$, $AC$.$\bigtriangleup ABC$的三边所围成的区域记为$I$,黑色部分记为$II$, 其余部分记为$III$.在整个图形中随机取一点，此点取自$I,II,III$的概率分别记为$p_1,p_2,p_3$,则$\SIFChoice$$\FigureID{1}$",
      r"题目二: 如图来自古希腊数学家希波克拉底所研究的几何图形．此图由三个半圆构成，三个半圆的直径分别为直角三角形$ABC$的斜边$BC$, 直角边$AB$, $AC$.$\bigtriangleup ABC$的三边所围成的区域记为$I$,黑色部分记为$II$, 其余部分记为$III$.在整个图形中随机取一点，此点取自$I,II,III$的概率分别记为$p_1,p_2,p_3$,则$\SIFChoice$$\FigureID{1}$"
   ]


Use an open-source pretrained model
---------------------------------------------

Use `get_pretrained_i2v` function to obtain the open-source pretrained models provided by EduNLP.

- Advantages: simple and convenient, easy to use even if you don’t understand the code powering the models.

- Disadvantages: NOT designed for model parameters tuning


Models selection and usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Select the pre-training model according to the subject (here are some of the open-source models):

+----------------------------+--------------------------------+
|   Pre-training model name  | Subject of model training data |
+============================+================================+
|    d2v_math_300            |         math                   |
+----------------------------+--------------------------------+
|    w2v_math_300            |         math                   |
+----------------------------+--------------------------------+
|    elmo_math_2048          |         math                   |
+----------------------------+--------------------------------+
|    bert_math_768           |         math                   |
+----------------------------+--------------------------------+
|    bert_taledu_768         |         math                   |
+----------------------------+--------------------------------+
|    disenq_math_256         |         math                   |
+----------------------------+--------------------------------+
|    quesnet_math_512        |         math                   |
+----------------------------+--------------------------------+


Example
^^^^^^^^^^^^^^^^^^^^^^

::

   from EduNLP import get_pretrained_i2v

   i2v = get_pretrained_i2v("w2v_eng_300")
   item_vector, token_vector = i2v(items)


Use a pretrained model from local
--------------------------------------------------------------------------

You can use any pretrained model provided by yourself (with the storage path of the model) to convert the given question text into vectors.

* Advantages: flexible, free to tune your model parameters.

All I2V containers provided
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-------------+---------------+
| Name        | I2V container |
+=============+===============+
| w2v         |  W2V          |
+-------------+---------------+
| d2v         |  D2V          |
+-------------+---------------+
| elmo        |  Elmo         |
+-------------+---------------+
| bert        |  Bert         |
+-------------+---------------+
| disenq      |  DisenQ       |
+-------------+---------------+
| quesnet     |  QuesNet      |
+-------------+---------------+

.. note::

   The complete list of the public pretrained models provided by `EduNLP` can be found through `Vector.t2v.get_all_pretrained_models` or `Modelhub <https://modelhub.bdaa.pro/>`_


Example
^^^^^^^^^^^^^^^^^^^^^^

::

   from EduNLP.I2V import W2V

   # load vectorization container
   pretrained_path = "path/to/model"
   i2v = W2V("pure_text", "w2v", pretrained_path)

   # vectorization
   item_vector, token_vector = i2v(items)
   # or
   item_vector, token_vector = i2v.infer_vector(items)
   # or
   item_vector = i2v.infer_item_vector(items)
   token_vector = i2v.infer_token_vector(items)


.. note::

   I2V container of different models can be slightly different in use, please refer to the specific API guide.

Specific I2V examples
------------------------------------

.. nbgallery::
    :caption: This is a thumbnail gallery:
    :name: i2v_gallery_en1
    :glob:

    W2V  <../../build/blitz/i2v/i2v_w2v.ipynb>

    D2V  <../../build/blitz/i2v/i2v_d2v.ipynb>

    Elmo  <../../build/blitz/i2v/i2v_elmo.ipynb>


.. nbgallery::
    :caption: This is a thumbnail gallery:
    :name: i2v_gallery_en2
    :glob:

    Bert  <../../build/blitz/i2v/i2v_bert.ipynb>

    DisenQNet  <../../build/blitz/i2v/i2v_disenq.ipynb>

    QuesNet  <../../build/blitz/i2v/i2v_quesnet.ipynb>


T2V container
=======================

`T2V` is designed to convert tokenization sequence (tokens) to vectors.

- Advantages: separated from tokenization, users are free to configure tokenization and vectorization parameters.

`I2V` also provides two ways of vectorization:

- Use a pretrained model from local
- Use an open-source pretrained model

Items to input
---------------------------------------------------
`T2V` accepts only tokenization sequence (`tokens`) as input, please use `Tokenizer` to obtain `tokens` before this.

::

   from EduNLP.Tokenizer import PureTextTokenize

   raw_items = [
      r"题目一：如图几何图形．此图由三个半圆构成，三个半圆的直径分别为直角三角形$ABC$的斜边$BC$, 直角边$AB$, $AC$.$\bigtriangleup ABC$的三边所围成的区域记为$I$,黑色部分记为$II$, 其余部分记为$III$.在整个图形中随机取一点，此点取自$I,II,III$的概率分别记为$p_1,p_2,p_3$,则$\SIFChoice$$\FigureID{1}$",
      r"题目二: 如图来自古希腊数学家希波克拉底所研究的几何图形．此图由三个半圆构成，三个半圆的直径分别为直角三角形$ABC$的斜边$BC$, 直角边$AB$, $AC$.$\bigtriangleup ABC$的三边所围成的区域记为$I$,黑色部分记为$II$, 其余部分记为$III$.在整个图形中随机取一点，此点取自$I,II,III$的概率分别记为$p_1,p_2,p_3$,则$\SIFChoice$$\FigureID{1}$"
   ]

   tokenizer = PureTextTokenizer()
   token_items = [t for t in tokenizer(raw_items)]


Use an open-source pretrained model
---------------------------------------------

.. note::

   The open-source models are same as `I2V`


Example: load a pretrained model to W2V:

::

   from EduNLP.Vector import get_pretrained_t2v

   model_dir = "path/to/save/model"
   t2v = get_pretrained_t2v("test_w2v", model_dir=model_dir)

   item_vector = t2v.infer_vector(token_items)
   # [array(), ..., array()]
   token_vector = t2v.infer_tokens(token_items)
   # [[array(), ..., array()], [...], [...]]


Use a pretrained model from local
------------------------------------

All T2V containers provided:

+---------+--------------+
| Name    |T2V container |
+=========+==============+
| w2v     | W2V          |
+---------+--------------+
| d2v     | D2V          |
+---------+--------------+
| elmo    | ElmoModel    |
+---------+--------------+
| bert    | BertModel    |
+---------+--------------+
| dienq   |DisenQMode    |
+---------+--------------+
|quesnet  |QuesNetModel  |
+---------+--------------+

Example: load a local models to W2V container:

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

   I2V container of different models can be slightly different in use, please refer to the specific API guide.


Specific T2V examples
------------------------------------
.. nbgallery::
    :caption: This is a thumbnail gallery:
    :name: t2v_gallery_en1
    :glob:

    W2V  <../../build/blitz/t2v/t2v_w2v.ipynb>

    D2V  <../../build/blitz/t2v/t2v_d2v.ipynb>

    Elmo  <../../build/blitz/t2v/t2v_elmo.ipynb>


.. nbgallery::
    :caption: This is a thumbnail gallery:
    :name: t2v_gallery_en2
    :glob:

    Bert  <../../build/blitz/t2v/t2v_bert.ipynb>

    DisenQNet  <../../build/blitz/t2v/t2v_disenq.ipynb>

    QuesNet  <../../build/blitz/t2v/t2v_quesnet.ipynb>

