=======
预训练
=======

在自然语言处理领域中，预训练语言模型（Pre-trained Language Models）已成为非常重要的基础技术。
我们将在本章节介绍EduNLP中预训练工具：

* 如何从零开始用一份语料训练得到一个预训练模型
* 如何加载预训练模型


训练模型
-----------------------

训练模块的接口定义在 `EduNLP.Pretrain` 中，包含令牌化容器、数据处理、模型训练等功能。


预训练工具
#######################################
为了方便研究人员构造自定义的训练过程，我们提供了机模型训练的常用功能，包括：

* 语料库词典 (Vocab)
* 预训练令牌化容器 (Tokenizer)
* 数据容器 (Datasets)

语料库词典
>>>>>>>>>>>>>>>>>>>>>>>>

语料库词典是预训练中为了便于用户进行后续处理而引入的工具。它支持用户导入并自定义词典内容，并对语料库的信息进行一定的处理。例如：

::

   >>> from EduNLP.Pretrain.pretrian_utils import EduVocab
   >>> vocab = EduVocab()
   >>> print(vocab.tokens)
   ['[PAD]', '[UNK]', '[BOS]', '[EOS]']

   >>> token_list = ['An', 'apple', 'a', 'day', 'keeps', 'doctors', 'away']
   >>> vocab.add_tokens(token_list)
   >>> test_token_list = ['An', 'banana', 'is', 'a', 'kind', 'of', 'fruit']
   >>> res = vocab.convert_sequence_to_token(vocab.convert_sequence_to_idx(test_token_list))
   >>> print(res)
   ['An', '[UNK]', '[UNK]', 'a', '[UNK]', '[UNK]', '[UNK]']


预训练令牌化容器
>>>>>>>>>>>>>>>>>>>>>>>>

在训练模型前, 需要将题目文本进行分词, 并将其转化为语料库字典中的词ID, 同时进行必要的预处理, 如OOV、Padding等常见操作。
我们将 `EduNLP` 的令牌化操作和通用的预处理操作封装在了自定义的基类 `PretrainedEduTokenizer` 中。

此外, 为了兼容Huggingface的预训练库 `Transformers`, 我们提供了基类 `TokenizerForHuggingface`。例如 `EduNLP.Pretrin.BertTokenizer` 兼容 `transformers.BertTokenizer`
具体用法参考如下。

.. nbgallery::
    :caption: This is a thumbnail gallery:
    :name: pretrained_tokenizer_gallery
    :glob:

    自定义分词器  <../../build/blitz/pretrain/pretrained_tokenizer.ipynb>
    
    Huggingface分词器  <../../build/blitz/pretrain/hugginface_tokenizer.ipynb>


预训练Dataset
>>>>>>>>>>>>>>>>>>>>>>>>
我们提供EduDataset基类, 封装了对教育数据的预处理，此外，考虑到教育数据规模巨大的特点, 我们提供并行处理操作和本地保存、加载等操作，加速数据预处理过程。
详细使用参考API文档 `Pretrain.pretrian_utils` 部分。



基本步骤
#######################################

训练模型
-----------------------

以训练word2vec为例说明：

- 确定模型的类型，选择适合的Tokenizer（如GensimWordTokenizer、PureTextTokenizer等），使之令牌化；

- 调用train_vector函数，即可得到所需的预训练模型。


Examples：

::
   
   from EduNLP.Tokenizer import PureTextTokenizer
   from EduNLP.Pretrain import train_vector

   items = [
      r"题目一：如图几何图形．此图由三个半圆构成，三个半圆的直径分别为直角三角形$ABC$的斜边$BC$, 直角边$AB$, $AC$.$\bigtriangleup ABC$的三边所围成的区域记为$I$,黑色部分记为$II$, 其余部分记为$III$.在整个图形中随机取一点，此点取自$I,II,III$的概率分别记为$p_1,p_2,p_3$,则$\SIFChoice$$\FigureID{1}$",
      r"题目二: 如图来自古希腊数学家希波克拉底所研究的几何图形．此图由三个半圆构成，三个半圆的直径分别为直角三角形$ABC$的斜边$BC$, 直角边$AB$, $AC$.$\bigtriangleup ABC$的三边所围成的区域记为$I$,黑色部分记为$II$, 其余部分记为$III$.在整个图形中随机取一点，此点取自$I,II,III$的概率分别记为$p_1,p_2,p_3$,则$\SIFChoice$$\FigureID{1}$"
   ]

   tokenizer = PureTextTokenizer()
   token_items = [t for t in tokenizer(raw_items)]
   
   print(token_items[0[:10])
   # ['公式', '[FORMULA]', '如图', '[FIGURE]', 'x', ',', 'y', '约束条件', '公式', '[FORMULA]']
   
   # 10 dimension with fasstext method
   train_vector(sif_items, "../../../data/w2v/gensim_luna_stem_tf_", 10, method="d2v")


加载预训练模型
-----------------------

我们提供三种加载方式

* 原生加载接口: `MODEL.from_pretrained()`. 适用于研究者，提供模型原生加载接口，方便应用到自定义的下游模型。
* 向量化容器: `I2V` 或 `T2V`. 适用于表征级应用，直接提供模型的向量化操作。
* 流水线容器：`Pipeline`. 适用于任务级应用，针对对具体的任务，封装了完整的令牌化、向量化、结果预测的等流水线操作。

原生加载接口可参考 `Modelzoo` 各个具体的模型, `I2V` 和 `Pipeline` 的用法可分别参考向量化部分和流水线部分的教程和及API。
**这里以向量化容器I2V举例, 通过向量化获取题目表征。**

Examples：

::

   model_path = "../test_model/d2v/test_gensim_luna_stem_tf_d2v_256.bin"
   i2v = D2V("text","d2v",filepath=model_path, pretrained_t2v = False)


更多模型训练案例
-----------------------

获得数据集
########################################

.. toctree::
   :maxdepth: 1
   :titlesonly:

   prepare_dataset  <../../build/blitz/pretrain/prepare_dataset.ipynb>

gensim模型d2v例子
########################################

.. toctree::
   :maxdepth: 1
   :titlesonly:

   d2v_bow_tfidf  <../../build/blitz/pretrain/gensim/d2v_bow_tfidf.ipynb>
   d2v_general  <../../build/blitz/pretrain/gensim/d2v_general.ipynb>
   d2v_stem_tf  <../../build/blitz/pretrain/gensim/d2v_stem_tf.ipynb>

gensim模型w2v例子
########################################

.. toctree::
   :maxdepth: 1
   :titlesonly:

   w2v_stem_text  <../../build/blitz/pretrain/gensim/w2v_stem_text.ipynb>
   w2v_stem_tf  <../../build/blitz/pretrain/gensim/w2v_stem_tf.ipynb>


进阶表征模型示例
########################################

.. nbgallery::
   :caption: This is a thumbnail gallery:
   :name: pretrain_gallery1
   :glob:

   Elmo预训练  <../../build/blitz/pretrain/elmo.ipynb>

   Bert预训练  <../../build/blitz/pretrain/bert.ipynb>


.. nbgallery::
   :caption: This is a thumbnail gallery:
   :name: pretrain_gallery2
   :glob:

   DisenQNet预训练  <../../build/blitz/pretrain/disenq.ipynb>
   
   QuesNet预训练  <../../build/blitz/pretrain/quesnet.ipynb>
