The overview of our public model
------------------------------------


Version Description
#########################

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
#######################################

* Currently, the data used in w2v and d2v models are the subjects of senior high school.
* test data:`[OpenLUNA.json] <http://base.ustc.edu.cn/data/OpenLUNA/OpenLUNA.json>`_

At present, the following models are provided. More models of different subjects and question types are being trained. Please look forward to it.
    "d2v_all_256" (all subject), "d2v_sci_256" (Science), "d2v_eng_256" (English)，"d2v_lit_256" (Arts)

Examples of model training
----------------------------

Get the dataset
####################

.. toctree::
   :maxdepth: 1
   :titlesonly:

   prepare_dataset  <../../../build/blitz/pretrain/prepare_dataset.ipynb>

An example of d2v in gensim model
####################################

.. toctree::
   :maxdepth: 1
   :titlesonly:

   d2v_bow_tfidf  <../../../build/blitz/pretrain/gensim/d2v_bow_tfidf.ipynb>
   d2v_general  <../../../build/blitz/pretrain/gensim/d2v_general.ipynb>
   d2v_stem_tf  <../../../build/blitz/pretrain/gensim/d2v_stem_tf.ipynb>

An example of w2v in gensim model
####################################

.. toctree::
   :maxdepth: 1
   :titlesonly:

   w2v_stem_text  <../../../build/blitz/pretrain/gensim/w2v_stem_text.ipynb>
   w2v_stem_tf  <../../../build/blitz/pretrain/gensim/w2v_stem_tf.ipynb>

An example of seg_token
############################

.. toctree::
   :maxdepth: 1
   :titlesonly:

   d2v.ipynb  <../../../build/blitz/pretrain/seg_token/d2v.ipynb>
   d2v_d1  <../../../build/blitz/pretrain/seg_token/d2v_d1.ipynb>
   d2v_d2  <../../../build/blitz/pretrain/seg_token/d2v_d2.ipynb>
