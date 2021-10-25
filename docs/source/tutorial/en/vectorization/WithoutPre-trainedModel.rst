Don't use the pre-trained model: call existing models directly
----------------------------------------------------------------

You can use any pre-trained model provided by yourself (just give the storage path of the model) to convert the given question text into vectors.

* Advantages: it is flexible to use your own model and its parameters can be adjusted freely.

Specific process of processing
+++++++++++++++++++++++++++++++++++

1.Call get_tokenizer function to get the result after word segmentation;

2.Select the model provided for vectorization depending on the model used.

Examples：

::

  >>> model_path = "../test_model/test_gensim_luna_stem_tf_d2v_256.bin"
  >>> i2v = D2V("text","d2v",filepath=model_path, pretrained_t2v = False)
  >>> i2v(item)
