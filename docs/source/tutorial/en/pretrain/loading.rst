Load models
----------------

Transfer the obtained model to the I2V module to load the model.
 
Examples：

::

        >>> model_path = "../test_model/test_gensim_luna_stem_tf_d2v_256.bin"
        >>> i2v = D2V("text","d2v",filepath=model_path, pretrained_t2v = False)
