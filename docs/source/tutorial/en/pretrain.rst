Pretraining
==============

In the field of NLP, Pretrained Language Models has become a very important basic technology.
In this chapter, we will introduce the pre training tools in EduNLP:

* How to train with a corpus to get a pretrained model
* How to load the pretrained model
* Public pretrained models

Import modules
---------------

::

   from EduNLP.I2V import get_pretrained_i2v
   from EduNLP.Vector import get_pretrained_t2v

Train a model
------------------

The module interface definition is in `EduNLP.Pretrain`, including tokenization, data processing, model definition, model training.


Pretrain tools
#######################################


Corpus dictionary
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

The corpus dictionary is a tool introduced in pre-training to facilitate user post-processing. It allows users to import and customize dictionary content and process the information of the corpus. For example:

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
   EduVocab() related parameter descriptions and function definitions:
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




Basic Steps
##################

1.Determine the type of model and select the appropriate tokenizer (GensimWordTokenizer、 GensimSegTokenizer) to finish tokenization.

2.Call `train_vector` function to get the required pretrained model。

Examples：

::

   >>> tokenizer = GensimWordTokenizer(symbol="gmas", general=True)
   >>> token_item = tokenizer("有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$,\
   ... 若$x,y$满足约束条件公式$\\FormFigureBase64{wrong2?}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$")
   >>> print(token_item.tokens[:10])
   ['公式', '[FORMULA]', '如图', '[FIGURE]', 'x', ',', 'y', '约束条件', '公式', '[FORMULA]']
   
   # 10 dimension with fasstext method
   train_vector(sif_items, "../../../data/w2v/gensim_luna_stem_tf_", 10, method="d2v")

::
   Definition of train_vector():
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
   


Load models
----------------

Transfer the obtained model to the I2V module to load the model.
 
Examples：

::

   >>> model_path = "../test_model/d2v/test_gensim_luna_stem_tf_d2v_256.bin"
   >>> i2v = D2V("text","d2v",filepath=model_path, pretrained_t2v = False)

::

   Taking D2V as an example, the specific definitions are as follows: (For other interfaces, please refer to the definitions under EduNLP/I2V)

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

Examples of Model Training
------------------------------------

Get the dataset
####################

.. toctree::
   :maxdepth: 1
   :titlesonly:

   prepare_dataset  <../../build/blitz/pretrain/prepare_dataset.ipynb>

Examples of d2v in gensim model
##################################

.. toctree::
   :maxdepth: 1
   :titlesonly:

   d2v_bow_tfidf  <../../build/blitz/pretrain/gensim/d2v_bow_tfidf.ipynb>
   d2v_general  <../../build/blitz/pretrain/gensim/d2v_general.ipynb>
   d2v_stem_tf  <../../build/blitz/pretrain/gensim/d2v_stem_tf.ipynb>

Examples of w2v in gensim model
##################################

.. toctree::
   :maxdepth: 1
   :titlesonly:

   w2v_stem_text  <../../build/blitz/pretrain/gensim/w2v_stem_text.ipynb>
   w2v_stem_tf  <../../build/blitz/pretrain/gensim/w2v_stem_tf.ipynb>

Examples of seg_token
#############################

.. toctree::
   :maxdepth: 1
   :titlesonly:

   d2v.ipynb  <../../build/blitz/pretrain/seg_token/d2v.ipynb>
   d2v_d1  <../../build/blitz/pretrain/seg_token/d2v_d1.ipynb>
   d2v_d2  <../../build/blitz/pretrain/seg_token/d2v_d2.ipynb>

Examples of advanced models
#############################

.. nbgallery::
    :caption: This is a thumbnail gallery:
    :name: pretrain_gallery_en1
    :glob:

    ELMo pretrain  <../../build/blitz/pretrain/elmo.ipynb>

    BERT pretrain <../../build/blitz/pretrain/bert.ipynb>


.. nbgallery::
    :caption: This is a thumbnail gallery:
    :name: pretrain_gallery_en2
    :glob:

    DisenQNet pretrain  <../../build/blitz/pretrain/disenq.ipynb>

    QuesNet pretrain <../../build/blitz/pretrain/quesnet.ipynb>