# coding: utf-8
# 2021/5/29 @ tongshiwei
from EduNLP import logger
import multiprocessing
import gensim
from gensim.models import word2vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec
from EduNLP.SIF.sif import sif4sci
from EduNLP.Vector import D2V, BowLoader
from copy import deepcopy
import itertools as it

__all__ = ["GensimWordTokenizer", "train_vector", "GensimSegTokenizer"]


class GensimWordTokenizer(object):
    """

    Parameters
    ----------
    symbol: str
        select the methods to symbolize:
            "t": text,
            "f": formula,
            "g": figure,
            "m": question mark,
            "a": tag,
            "s": sep,
        e.g.: gm, fgm, gmas, fgmas
    general: bool

        True: when item isn't in standard format, and want to tokenize formulas(except formulas in figure) linearly.

        False: when use 'ast' mothed to tokenize formulas instead of 'linear'.

    Returns
    ----------
    tokenizer: Tokenizer

    Examples
    ----------
    >>> tokenizer = GensimWordTokenizer(symbol="gmas", general=True)
    >>> token_item = tokenizer("有公式$\\FormFigureID{1}$，如图$\\FigureID{088f15ea-xxx}$,\
    ... 若$x,y$满足约束条件公式$\\FormFigureBase64{2}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$")
    >>> print(token_item.tokens[:10])
    ['公式', '[FORMULA]', '如图', '[FIGURE]', 'x', ',', 'y', '约束条件', '公式', '[FORMULA]']
    >>> tokenizer = GensimWordTokenizer(symbol="fgmas", general=False)
    >>> token_item = tokenizer("有公式$\\FormFigureID{1}$，如图$\\FigureID{088f15ea-xxx}$,\
    ... 若$x,y$满足约束条件公式$\\FormFigureBase64{2}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$")
    >>> print(token_item.tokens[:10])
    ['公式', '[FORMULA]', '如图', '[FIGURE]', '[FORMULA]', '约束条件', '公式', '[FORMULA]', '[SEP]', '[FORMULA]']
    """
    def __init__(self, symbol="gm", general=False):
        self.symbol = symbol
        if general is True:
            self.tokenization_params = {
                "formula_params": {
                    "method": "linear",
                    "symbolize_figure_formula": True
                }
            }
        else:
            self.tokenization_params = {
                "formula_params": {
                    "method": "ast",
                    "return_type": "list",
                    "ord2token": True
                }
            }

    def batch_process(self, *items):
        pass

    def __call__(self, item):
        return sif4sci(
            item, symbol=self.symbol, tokenization_params=self.tokenization_params, errors="ignore"
        )


class GensimSegTokenizer(object):  # pragma: no cover
    """

    Parameters
    ----------
    symbol:str
        select the methods to symbolize:
            "t": text,
            "f": formula,
            "g": figure,
            "m": question mark,
            "a": tag,
            "s": sep,
        e.g. gms, fgm

    depth: int or None

        0: only separate at \\SIFSep ;
        1: only separate at \\SIFTag ;
        2: separate at \\SIFTag and \\SIFSep ;
        otherwise, separate all segments ;

    Returns
    ----------
    tokenizer: Tokenizer

    Examples
    ----------
    >>> tokenizer = GensimSegTokenizer(symbol="gms", depth=None)
    >>> token_item = tokenizer("有公式$\\FormFigureID{1}$，如图$\\FigureID{088f15ea-xxx}$,\
    ... 若$x,y$满足约束条件公式$\\FormFigureBase64{2}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$")
    >>> print(token_item[:10])
    [['公式'], [\\FormFigureID{1}], ['如图'], ['[FIGURE]'],...['最大值'], ['[MARK]']]
    >>> tokenizer = GensimSegTokenizer(symbol="fgm", depth=None)
    >>> token_item = tokenizer("有公式$\\FormFigureID{1}$，如图$\\FigureID{088f15ea-xxx}$,\
    ... 若$x,y$满足约束条件公式$\\FormFigureBase64{2}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$")
    >>> print(token_item[:10])
    [['公式'], ['[FORMULA]'], ['如图'], ['[FIGURE]'], ['[FORMULA]'],...['[FORMULA]'], ['最大值'], ['[MARK]']]
    """
    def __init__(self, symbol="gms", depth=None, flatten=False, **kwargs):
        self.symbol = symbol
        self.tokenization_params = {
            "formula_params": {
                "method": "ast",
                "return_type": "list",
                "ord2token": True
            }
        }
        self.kwargs = dict(
            add_seg_type=True if depth in {0, 1, 2} else False,
            add_seg_mode="head",
            depth=depth,
            drop="s" if depth not in {0, 1, 2} else ""
        )
        self.kwargs.update(kwargs)
        self.flatten = flatten

    def __call__(self, item, flatten=None, **kwargs):
        flatten = self.flatten if flatten is None else flatten
        tl = sif4sci(
            item, symbol=self.symbol, tokenization_params=self.tokenization_params, errors="ignore"
        )
        if kwargs:
            _kwargs = deepcopy(self.kwargs)
            _kwargs.update(kwargs)
        else:
            _kwargs = self.kwargs
        if tl:
            ret = tl.get_segments(**_kwargs)
            if flatten is True:
                return it.chain(*ret)
            return ret
        return tl


class MonitorCallback(CallbackAny2Vec):
    """record the loss in each epoch"""
    def __init__(self, test_words):
        self.epoch = 0
        self._test_words = test_words

    def on_epoch_end(self, model):
        logger.info("Epoch #{}: loss-{:.4f} ".format(self.epoch, model.get_latest_training_loss()))
        self.epoch += 1


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

    Examples
    ----------
    >>> tokenizer = GensimSegTokenizer(symbol="gms", depth=None)
    >>> token_item = tokenizer("有公式$\\FormFigureID{1}$，如图$\\FigureID{088f15ea-xxx}$,\
    ... 若$x,y$满足约束条件公式$\\FormFigureBase64{2}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$")
    >>> print(token_item[:10])
    [['公式'], [\\FormFigureID{1}], ['如图'], ['[FIGURE]'],...['最大值'], ['[MARK]']]
    >>> train_vector(token_item[:10], "examples/test_model/w2v/gensim_luna_stem_t_", 100) #doctest: +ELLIPSIS
    'examples/test_model/w2v/gensim_luna_stem_t_sg_100.kv'
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
