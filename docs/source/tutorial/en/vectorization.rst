Vectorization
==================

This section provides a simple interface to convert the incoming items into vectors directly. Currently, the option of whether to use the pre training model is provided. You can choose according to your needs. If you don't want to use the pre-trained model, you can call D2V directly, or call get_pretrained_i2v function if you want to use the pre-trained model.

- Don't use the pre-trained model

- Use the pre-trained model

Overview Flow
---------------------------

1.Perform `syntax parsing <parse.rst>`_ on incoming items to get items in SIF format；

2.Perform `component segmentation <seg.rst>`_ on sif_items;

3.Perform `tokenization <tokenize.rst>`_ on segmented items;

4.Use the existing or pre-trained model we provided to convert the tokenized items into vectors.


Don't use the pre-trained model: call existing models directly
--------------------------------------------------------------------------

You can use any pre-trained model provided by yourself (just give the storage path of the model) to convert the given question text into vectors.

* Advantages: it is flexible to use your own model and its parameters can be adjusted freely.

Import modules
+++++++++++++++++++++++

::

   from EduNLP.I2V import D2V,W2V,get_pretrained_i2v
   from EduNLP.Vector import T2V,get_pretrained_t2v

Models provided
++++++++++++++++++++

- W2V

- D2V

- T2V

W2V
<<<<<<<<<

This model directly uses the relevant model methods in the gensim library to convert words into vectors. Currently, there are four methods:

 - FastText

 - Word2Vec

 - KeyedVectors

::

   >>> i2v = get_pretrained_i2v("test_w2v", "examples/test_model/data/w2v") # doctest: +ELLIPSIS
   >>> item_vector, token_vector = i2v(["有学者认为：‘学习’，必须适应实际"])
   >>> item_vector # doctest: +ELLIPSIS
   array([[...]], dtype=float32)

D2V
<<<<<<<<<<<<

This model is a comprehensive processing method which can convert items into vectors. Currently, the following methods are provided:

- d2v: call doc2vec module in gensim library to convert items into vectors.

- BowLoader: call corpora module in gensim library to convert docs into bows.

- TfidfLoader: call TfidfModel module in gensim library to convert docs into bows.

::

   >>> item = {"如图来自古希腊数学家希波克拉底所研究的几何图形．此图由三个半圆构成，三个半圆的直径分别为直角三角形$ABC$的斜边$BC$, 直角边$AB$, $AC$.$\bigtriangleup ABC$的三边所围成的区域记为$I$,黑色部分记为$II$, 其余部分记为$III$.在整个图形中随机取一点，此点取自$I,II,III$的概率分别记为$p_1,p_2,p_3$,则$\SIFChoice$$\FigureID{1}$"}
   >>> model_path = "../test_model/test_gensim_luna_stem_tf_d2v_256.bin"
   >>> i2v = D2V("text","d2v",filepath=model_path, pretrained_t2v = False)
   >>> i2v(item)
   ([array([ 4.76559885e-02, -1.60574958e-01,  1.94614579e-03,  2.40295693e-01,
   2.24517003e-01, -3.24351490e-02,  4.35789041e-02, -1.65670961e-02,...

T2V
<<<<<<<<<<

You can use any pre-trained model provided by yourself to represent the segmentation sequences of a group of questions as vectors (just give the storage path of the model).

- Advantages: the model and its parameters can be adjusted independently and has strong flexibility.

Input
^^^^^^^^^^

Types: list
Contents: the combination of each question segmentation sequences in one question group.
>You can transfer question text (`str` type) to tokens using ``GensimWordTokenizer`` model

::

   >>> token_items=['公式','[FORMULA]','公式','[FORMULA]','如图','[FIGURE]','x',',','y','约束条件','[SEP]','z','=','x','+','7','y','最大值','[MARK]']
   >>> path = "../test_model/test_gensim_luna_stem_tf_d2v_256.bin"
   >>> t2v = T2V('d2v',filepath=path)
   >>> t2v(token_items)
   [array([ 0.0256574 ,  0.06061139, -0.00121044, -0.0167674 , -0.0111706 ,
   0.05325712, -0.02097339, -0.01613594,  0.02904145,  0.0185046 ,...

Specific process of processing
++++++++++++++++++++++++++++++++++++++++

1.Call get_tokenizer function to get the result after word segmentation;

2.Select the model provided for vectorization depending on the model used.


Use the pre-training model: call get_pretrained_i2v directly
---------------------------------------------

Use the pre-training model provided by EduNLP to convert the given question text into vectors.

* Advantages: Simple and convenient.

* Disadvantages: Only the model given in the project can be used, which has great limitations.

* Call this function to obtain the corresponding pre-training model. At present, the following pre training models are provided: d2v_all_256, d2v_sci_256, d2v_eng_256 and d2v_lit_256.

Selection and Use of Models
####################################

Select the pre-training model according to the subject:

+--------------------+------------------------+
|   Pre-training model name  | Subject of model training data |
+====================+========================+
|    d2v_all_256     |        all subject          |
+--------------------+------------------------+
|    d2v_sci_256     |         Science           |
+--------------------+------------------------+
|    d2v_lit_256     |         Arts           |
+--------------------+------------------------+
|    d2v_eng_256     |         English           |
+--------------------+------------------------+


The concrete process of processing
####################################

1.Download the corresponding preprocessing model

2.Transfer the obtained model to D2V and process it with D2V
  Convert the obtained model into D2V and process it through D2V

Examples:

::

  >>> i2v = get_pretrained_i2v("d2v_sci_256")
  >>> i2v(item)
