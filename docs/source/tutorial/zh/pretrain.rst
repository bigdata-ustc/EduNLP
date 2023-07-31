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

::
  
   EduVocab相关参数说明和函数定义：
   class EduVocab(object):
    """The vocabulary container for a corpus.

    Parameters
    ----------
    vocab_path : str, optional
        vocabulary path to initialize this container, by default None
    corpus_items : List[str], optional
        corpus items to update this vocabulary, by default None
    bos_token : str, optional
        token representing for the start of a sentence, by default "[BOS]"
    eos_token : str, optional
        token representing for the end of a sentence, by default "[EOS]"
    pad_token : str, optional
        token representing for padding, by default "[PAD]"
    unk_token : str, optional
        token representing for unknown word, by default "[UNK]"
    specials : List[str], optional
        spacials tokens in vocabulary, by default None
    lower : bool, optional
        wheather to lower the corpus items, by default False
    trim_min_count : int, optional
        the lower bound number for adding a word into vocabulary, by default 1
    """
    def __init__(self, vocab_path: str = None, corpus_items: List[str] = None, bos_token: str = "[BOS]",
                 eos_token: str = "[EOS]", pad_token: str = "[PAD]", unk_token: str = "[UNK]",
                 specials: List[str] = None, lower: bool = False, trim_min_count: int = 1, **kwargs):
        super(EduVocab, self).__init__()

        self._tokens = []
        self.idx_to_token = dict()
        self.token_to_idx = dict()
        self.frequencies = dict()
        # 定义特殊词
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self._special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]

        if specials:
            self._special_tokens += specials
        for st in self._special_tokens:
            self._add(st)
        # 加载词典
        if vocab_path is not None:
            self.load_vocab(vocab_path)
        elif corpus_items is not None:
            self.set_vocab(corpus_items, lower, trim_min_count)

        self.bos_idx = self.token_to_idx[self.bos_token]
        self.eos_idx = self.token_to_idx[self.eos_token]
        self.pad_idx = self.token_to_idx[self.pad_token]
        self.unk_idx = self.token_to_idx[self.unk_token]

    def __len__(self):
        return len(self._tokens)

    @property
    def vocab_size(self):
        return len(self._tokens)

    @property
    def special_tokens(self):
        return self._special_tokens

    @property
    def tokens(self):
        return self._tokens

    def to_idx(self, token):
        """convert token to index"""
        return self.token_to_idx.get(token, self.unk_idx)

    def to_token(self, idx):
        """convert index to index"""
        return self.idx_to_token.get(idx, self.unk_token)

    def convert_sequence_to_idx(self, tokens, bos=False, eos=False):
        """convert sentence of tokens to sentence of indexs"""
        res = [self.to_idx(t) for t in tokens]
        if bos is True:
            res = [self.bos_idx] + res
        if eos is True:
            res = res + [self.eos_idx]
        return res

    def convert_sequence_to_token(self, idxs, **kwargs):
        """convert sentence of indexs to sentence of tokens"""
        return [self.to_token(i) for i in idxs]

    def set_vocab(self, corpus_items: List[str], lower: bool = False, trim_min_count: int = 1, silent=True):
        """Update the vocabulary with the tokens in corpus items

        Parameters
        ----------
        corpus_items : List[str], optional
            corpus items to update this vocabulary, by default None
        lower : bool, optional
            wheather to lower the corpus items, by default False
        trim_min_count : int, optional
            the lower bound number for adding a word into vocabulary, by default 1
        """
        word2cnt = dict()
        for item in corpus_items:
            for word in item:
                word = word.lower() if lower else word
                word2cnt[word] = word2cnt.get(word, 0) + 1
        words = [w for w, c in word2cnt.items() if c >= trim_min_count and w not in self._special_tokens]
        for token in words:
            self._add(token)
        if not silent:
            keep_word_cnts = sum(word2cnt[w] for w in words)
            all_word_cnts = sum(word2cnt.values())
            print(f"save words(trim_min_count={trim_min_count}): {len(words)}/{len(word2cnt)} = {len(words) / len(word2cnt):.4f}\
                  with frequency {keep_word_cnts}/{all_word_cnts}={keep_word_cnts / all_word_cnts:.4f}")

    def load_vocab(self, vocab_path: str):
        """Load the vocabulary from vocab_file

        Parameters
        ----------
        vocab_path : str
            path to save vocabulary file
        """
        with open(vocab_path, "r", encoding="utf-8") as file:
            self._tokens = file.read().strip().split('\n')
            self.token_to_idx = {token: idx for idx, token in enumerate(self._tokens)}
            self.idx_to_token = {idx: token for idx, token in enumerate(self._tokens)}

    def save_vocab(self, vocab_path: str):
        """Save the vocabulary into vocab_file

        Parameters
        ----------
        vocab_path : str
            path to save vocabulary file
        """
        with open(vocab_path, 'w', encoding='utf-8') as file:
            for i in range(self.vocab_size):
                token = self._tokens[i]
                file.write(f"{token}\n")

    def _add(self, token: str):
        if token not in self._tokens:
            idx = len(self._tokens)
            self._tokens.append(token)
            self.idx_to_token[idx] = token
            self.token_to_idx[token] = idx

    def add_specials(self, tokens: List[str]):
        """Add special tokens into vocabulary"""
        for token in tokens:
            if token not in self._special_tokens:
                self._special_tokens += [token]
                self._add(token)

    def add_tokens(self, tokens: List[str]):
        """Add tokens into vocabulary"""
        for token in tokens:
            self._add(token)


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
>>>>>>>>>>>>>>>>>>>>>>>>

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

::
   
   PureTextTokenizer在“令牌化”部分已有详细说明
   train_vector定义如下：
   def train_vector(items, w2v_prefix, embedding_dim=None, method="sg", binary=None, train_params=None):
    """

    Parameters
    ----------
    items：str
        the text of question
    w2v_prefix
    embedding_dim:int
        vector_size
    method:str
        the method of training,
        e.g.: sg, cbow, fasttext, d2v, bow, tfidf
    binary: model format
        True:bin;
        False:kv
    train_params: dict
        the training parameters passed to model

    Returns
    ----------
    tokenizer: Tokenizer

    """
    monitor = MonitorCallback(["word", "I", "less"])
    _train_params = dict(
        min_count=0,
        vector_size=embedding_dim,
        workers=multiprocessing.cpu_count(),
        callbacks=[monitor]
    )
    if method in {"sg", "cbow"}:
        sg = 1 if method == "sg" else 0
        _train_params["sg"] = sg
        if train_params is not None:
            _train_params.update(train_params)
        model = gensim.models.Word2Vec(
            items, **_train_params
        )
        binary = binary if binary is not None else False
    elif method == "fasttext":
        if train_params is not None:
            _train_params.update(train_params)
        model = gensim.models.FastText(
            sentences=items,
            **_train_params
        )
        binary = binary if binary is not None else True
    elif method == "d2v":
        if train_params is not None:
            _train_params.update(train_params)
        docs = [TaggedDocument(doc, [i]) for i, doc in enumerate(items)]
        model = gensim.models.Doc2Vec(
            docs, **_train_params
        )
        binary = binary if binary is not None else True
    elif method == "bow":
        model = gensim.corpora.Dictionary(items)
        binary = binary if binary is not None else True
    elif method == "tfidf":
        dictionary_path = train_vector(items, w2v_prefix, method="bow")
        dictionary = BowLoader(dictionary_path)
        corpus = [dictionary.infer_vector(item) for item in items]
        model = gensim.models.TfidfModel(corpus)
        binary = binary if binary is not None else True
    else:
        raise ValueError("Unknown method: %s" % method)

    filepath = w2v_prefix + method
    if embedding_dim is not None:
        filepath = filepath + "_" + str(embedding_dim)

    if binary is True:
        filepath += ".bin"
        logger.info("model is saved to %s" % filepath)
        model.save(filepath)
    else:
        if method in {"fasttext", "d2v"}:  # pragma: no cover
            logger.warning("binary should be True for %s, otherwise all vectors for ngrams will be lost." % method)
        filepath += ".kv"
        logger.info("model is saved to %s" % filepath)
        model.wv.save(filepath)
    return filepath
   

加载预训练模型
>>>>>>>>>>>>>>>>>>>>>>>>

我们提供三种加载方式

* 原生加载接口: `MODEL.from_pretrained()`. 适用于研究者，提供模型原生加载接口，方便应用到自定义的下游模型。
* 向量化容器: `I2V` 或 `T2V`. 适用于表征级应用，直接提供模型的向量化操作。
* 流水线容器：`Pipeline`. 适用于任务级应用，针对对具体的任务，封装了完整的令牌化、向量化、结果预测的等流水线操作。

原生加载接口可参考 `Modelzoo` 各个具体的模型, `I2V` 和 `Pipeline` 的用法可分别参考向量化部分和流水线部分的教程和及API。
**这里以向量化容器I2V举例, 通过向量化获取题目表征。**

Examples：

::

   from EduNLP.I2V import D2V
   
   model_path = "../test_model/d2v/test_gensim_luna_stem_tf_d2v_256.bin"
   i2v = D2V("text", "d2v", filepath=model_path, pretrained_t2v=False)


:: 
   
   以D2V为例，具体定义如下：（其余接口可参考EduNLP/I2V下的各个定义）
   class D2V(I2V):
    """
    The model aims to transfer item to vector directly.

    Bases
    -------
    I2V

    Parameters
    -----------
    tokenizer: str
        the tokenizer name
    t2v: str
        the name of token2vector model
    args:
        the parameters passed to t2v
    tokenizer_kwargs: dict
        the parameters passed to tokenizer
    pretrained_t2v: bool
        True: use pretrained t2v model
        False: use your own t2v model
    kwargs:
        the parameters passed to t2v

    Returns
    -------
    i2v model: I2V
    """

    def infer_vector(self, items, tokenize=True, key=lambda x: x, *args,
                     **kwargs) -> tuple:
        """
        It is a function to switch item to vector. And before using the function, it is necessary to load model.

        Parameters
        -----------
        items:str
            the text of question
        tokenize: bool
            True: tokenize the item
        key: function
            determine how to get the text of each item
        args:
            the parameters passed to t2v
        kwargs:
            the parameters passed to t2v

        Returns
        --------
        vector:list
        """
        tokens = self.tokenize(items, key=key) if tokenize is True else items
        tokens = [token for token in tokens]
        return self.t2v(tokens, *args, **kwargs), None

    @classmethod
    def from_pretrained(cls, name, model_dir=MODEL_DIR, *args, **kwargs):
        return cls("pure_text", name, pretrained_t2v=True, model_dir=model_dir)


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
