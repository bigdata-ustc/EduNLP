# coding: utf-8
# 2021/5/29 @ tongshiwei
from EduNLP import logger
import multiprocessing
import gensim
from gensim.models import word2vec
from gensim.models.callbacks import CallbackAny2Vec
from EduNLP.SIF.sif import sif4sci

__all__ = ["GensimWordTokenizer", "train_vector"]


class GensimWordTokenizer(object):
    def __init__(self, symbol="gm"):
        self.symbol = symbol
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
            item, symbol=self.symbol, tokenization_params=self.tokenization_params
        )


class MonitorCallback(CallbackAny2Vec):
    def __init__(self, test_words):
        self.epoch = 0
        self._test_words = test_words

    def on_epoch_end(self, model):
        logger.info("Epoch #{}: loss-{:.4f} ".format(self.epoch, model.get_latest_training_loss()))
        self.epoch += 1


def train_vector(items, w2v_prefix, embedding_dim, method="sg", binary=None, train_params=None):
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
    else:
        raise TypeError()

    filepath = w2v_prefix + "%s_%s" % (method, embedding_dim)
    if binary is True:
        filepath += ".bin"
        logger.info("model is saved to %s" % filepath)
        model.save(filepath)
    else:
        if method == "fasttext":
            logger.warning("binary should be True for fasttext, otherwise all vectors for ngrams will be lost.")
        filepath += ".kv"
        logger.info("model is saved to %s" % filepath)
        model.wv.save(filepath)
    return filepath
