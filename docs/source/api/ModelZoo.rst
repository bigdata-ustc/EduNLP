EduNLP.ModelZoo
==================

base_model
-----------

.. automodule:: EduNLP.ModelZoo.base_model
   :members:

::
    相关方法中的参数说明：

    save_pretrained(output_dir)：
          output_dir: str
          The path you want to save your model

    classmethodfrom_pretrained(pretrained_model_path, *args, **kwargs):
          pretrained_model_path: str
          The path where you load your checkpoint from

    save_config(config_dir):
          config_dir: str
          The path you want to save the config file
    
    @classmethod 
    from_config(config_path, *args, **kwargs):
          config_path: str
          The path where you load the config file
   


rnn
-----------

.. automodule:: EduNLP.ModelZoo.rnn
   :members:
   :imported-members:

::
    参数补充说明：
    @classmethod from_config(config_path, **kwargs)：
          config_path: str
          The path where you load the config file



disenqnet
-----------

.. automodule:: EduNLP.ModelZoo.disenqnet
   :members:
   :imported-members:

::
    参数补充说明：
    @classmethod from_config(config_path, **kwargs)：
          config_path: str
          The path where you load the config file

quesnet
-----------

.. automodule:: EduNLP.ModelZoo.quesnet
   :members:
   :imported-members:

utils
-----------

.. automodule:: EduNLP.ModelZoo.utils
   :members:
   :imported-members:
