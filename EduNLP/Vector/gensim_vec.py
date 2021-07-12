# coding: utf-8
# 2021/5/29 @ tongshiwei

import numpy as np
from pathlib import PurePath
from gensim.models import KeyedVectors, Word2Vec, FastText, Doc2Vec, TfidfModel
from gensim import corpora
import re
from .const import UNK, PAD


class W2V(object):
    def __init__(self, filepath, method = "sg", binary=None):
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


class TfidfLoader(object):
    def __init__(self, filepath):
        self.tfidf_model = TfidfModel.load(filepath)
        # 'tfidf' model shold be used based on 'bow' model
        dictionary_path = re.sub(r"(.*)tfidf", r"\1bow", filepath)
        self.dictionary = corpora.Dictionary.load(dictionary_path)

    def infer_vector(self, item):
        item = self.dictionary.doc2bow(item)
        return self.tfidf_model[item]


class D2V(object):
    def __init__(self, filepath, method="d2v"):
        self._method = method
        self._filepath = filepath
        if self._method == "d2v":
            self.d2v = Doc2Vec.load(filepath)
        elif self._method == "bow":
            self.d2v = corpora.Dictionary.load(filepath)
        elif self._method == "tfidf":
            self.d2v = TfidfLoader(filepath)
        else:
            raise ValueError("Unknown method: %s" % method)

    def __call__(self, item):
        if self._method == "bow":
            return self.d2v.doc2bow(item)
        else:
            return self.d2v.infer_vector(item)

    @property
    def vector_size(self):
        if self._method == "d2v":
            return self.d2v.vector_size
        else:  # pragma: no cover
            raise NotImplementedError  # todo: enable this feature for bow and tfidf
