# coding: utf-8
# 2021/5/29 @ tongshiwei

import numpy as np
from pathlib import PurePath
from gensim.models import KeyedVectors, Word2Vec, FastText, Doc2Vec, TfidfModel
from gensim import corpora
import re
from .const import UNK, PAD
from .meta import Vector


class W2V(Vector):
    def __init__(self, filepath, method=None, binary=None):
        """

        Parameters
        ----------
        filepath:
            path to the pretrained model file
        method
        binary
        """
        fp = PurePath(filepath)
        self.binary = binary if binary is not None else (True if fp.suffix == ".bin" else False)
        if self.binary is True:
            if method == "fasttext":
                self.wv = FastText.load(filepath).wv
            else:
                self.wv = Word2Vec.load(filepath).wv
        else:
            self.wv = KeyedVectors.load(filepath, mmap="r")

        self.method = method
        self.constants = {UNK: 0, PAD: 1}

    def __len__(self):
        return len(self.constants) + len(self.wv.key_to_index)

    def key_to_index(self, word):
        if word in self.constants:
            return self.constants[word]
        else:
            if word in self.wv.key_to_index:
                return self.wv.key_to_index[word] + len(self.constants)
            else:
                return self.constants[UNK]

    @property
    def vectors(self):
        return np.concatenate([np.zeros((len(self.constants), self.vector_size)), self.wv.vectors], axis=0)

    @property
    def vector_size(self):
        return self.wv.vector_size

    def __call__(self, *words):
        for word in words:
            yield self[word]

    def __getitem__(self, item):
        return self.wv[item] if item not in self.constants else np.zeros((self.vector_size,))

    def infer_vector(self, items, agg="mean", *args, **kwargs) -> np.ndarray:
        tokens = self.infer_tokens(items, *args, **kwargs)
        return eval("np.%s" % agg)(tokens, axis=1)

    def infer_tokens(self, items, *args, **kwargs) -> list:
        return [list(self(*item)) for item in items]


class BowLoader(object):
    def __init__(self, filepath):
        self.dictionary = corpora.Dictionary.load(filepath)

    def infer_vector(self, item, return_vec=False):
        item = self.dictionary.doc2bow(item)
        if not return_vec:
            return item  # return dic as default
        vec = [0 for i in range(len(self.dictionary.keys()))]
        for i, v in item:
            vec[i] = v
        return vec

    @property
    def vector_size(self):
        return len(self.dictionary.keys())


class TfidfLoader(object):
    def __init__(self, filepath):
        self.tfidf_model = TfidfModel.load(filepath)
        # 'tfidf' model shold be used based on 'bow' model
        dictionary_path = re.sub(r"(.*)tfidf", r"\1bow", filepath)
        self.dictionary = corpora.Dictionary.load(dictionary_path)

    def infer_vector(self, item, return_vec=False):
        dic_item = self.dictionary.doc2bow(item)
        tfidf_item = self.tfidf_model[dic_item]
        # return dic as default
        if not return_vec:
            return tfidf_item  # pragma: no cover
        vec = [0 for i in range(len(self.dictionary.keys()))]
        for i, v in tfidf_item:
            vec[i] = v
        return vec

    @property
    def vector_size(self):
        return len(self.dictionary.token2id)


class D2V(Vector):
    def __init__(self, filepath, method="d2v"):
        self._method = method
        self._filepath = filepath
        if self._method == "d2v":
            self.d2v = Doc2Vec.load(filepath)
        elif self._method == "bow":
            self.d2v = BowLoader(filepath)
        elif self._method == "tfidf":
            self.d2v = TfidfLoader(filepath)
        else:
            raise ValueError("Unknown method: %s" % method)

    def __call__(self, item):
        if self._method == "d2v":
            return self.d2v.infer_vector(item)
        else:
            return self.d2v.infer_vector(item, return_vec=True)

    @property
    def vector_size(self):
        if self._method == "d2v":
            return self.d2v.vector_size
        elif self._method == "bow":
            return self.d2v.vector_size
        elif self._method == "tfidf":
            return self.d2v.vector_size

    def infer_vector(self, items, *args, **kwargs) -> list:
        return [self(item) for item in items]

    def infer_tokens(self, item, *args, **kwargs) -> ...:
        raise NotImplementedError
