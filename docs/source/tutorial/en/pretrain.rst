Pre-training
==============

In the field of NLP, Pre-trained Language Models has become a very important basic technology.
In this chapter, we will introduce the pre training tools in EduNLP:

* How to train with a corpus to get a pre-trained model
* How to load the pre-trained model
* Public pre-trained models

Import modules
---------------

::

   from EduNLP.I2V import get_pretrained_i2v
   from EduNLP.Vector import get_pretrained_t2v

Train the Model
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


Load models
----------------

Transfer the obtained model to the I2V module to load the model.
 
Examples：

::

   >>> model_path = "../test_model/test_gensim_luna_stem_tf_d2v_256.bin"
   >>> i2v = D2V("text","d2v",filepath=model_path, pretrained_t2v = False)

The overview of our public model
------------------------------------

Version description
#######################

First level version:

* Public version 1 (luna_pub): college entrance examination
* Public version 2 (luna_pub_large): college entrance examination + regional examination

Second level version:

* Minor subjects(Chinese,Math,English,History,Geography,Politics,Biology,Physics,Chemistry)
* Major subjects(science, arts and all subject)

Third level version【to be finished】:

* Don't use third-party initializers
* Use third-party initializers

Description of train data in models
##############################################

* Currently, the data used in w2v and d2v models are the subjects of senior high school.
* test data:`[OpenLUNA.json] <http://base.ustc.edu.cn/data/OpenLUNA/OpenLUNA.json>`_

At present, the following models are provided. More models of different subjects and question types are being trained. Please look forward to it.
    "d2v_all_256" (all subject), "d2v_sci_256" (Science), "d2v_eng_256" (English)，"d2v_lit_256" (Arts)


Examples of Model Training
------------------------------------

Get the dataset
####################

.. toctree::
   :maxdepth: 1
   :titlesonly:

   prepare_dataset  <../../build/blitz/pretrain/prepare_dataset.ipynb>

An example of d2v in gensim model
##################################

.. toctree::
   :maxdepth: 1
   :titlesonly:

   d2v_bow_tfidf  <../../build/blitz/pretrain/gensim/d2v_bow_tfidf.ipynb>
   d2v_general  <../../build/blitz/pretrain/gensim/d2v_general.ipynb>
   d2v_stem_tf  <../../build/blitz/pretrain/gensim/d2v_stem_tf.ipynb>

An example of w2v in gensim model
##################################

.. toctree::
   :maxdepth: 1
   :titlesonly:

   w2v_stem_text  <../../build/blitz/pretrain/gensim/w2v_stem_text.ipynb>
   w2v_stem_tf  <../../build/blitz/pretrain/gensim/w2v_stem_tf.ipynb>

An example of seg_token
#############################

.. toctree::
   :maxdepth: 1
   :titlesonly:

   d2v.ipynb  <../../build/blitz/pretrain/seg_token/d2v.ipynb>
   d2v_d1  <../../build/blitz/pretrain/seg_token/d2v_d1.ipynb>
   d2v_d2  <../../build/blitz/pretrain/seg_token/d2v_d2.ipynb>