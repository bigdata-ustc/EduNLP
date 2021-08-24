装载模型
--------

将所得到的模型传入I2V模块即可装载模型
 
Examples：

::

        >>> model_path = "../test_model/test_gensim_luna_stem_tf_d2v_256.bin"
        >>> i2v = D2V("text","d2v",filepath=model_path, pretrained_t2v = False)
