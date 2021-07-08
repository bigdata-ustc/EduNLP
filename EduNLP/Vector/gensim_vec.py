# coding: utf-8
# 2021/5/29 @ tongshiwei

from pathlib import PurePath
from gensim.models import KeyedVectors, Word2Vec, FastText, Doc2Vec


class W2V(object):
    def __init__(self, filepath, method, binary=None):
        fp = PurePath(filepath)
        self.binary = binary if binary is not None else (True if fp.suffix == ".bin" else False)
        if self.binary is True:
            if method == "fasttext":
                self.wv = FastText.load(filepath).wv
            else:
                self.wv = Word2Vec.load(filepath).wv  # pragma: no cover
        else:
            self.wv = KeyedVectors.load(filepath, mmap="r")

    def __call__(self, *words):
        for word in words:
            yield self.wv[word]

    def __getitem__(self, item):
        return self.wv[item]


class D2V(object):
    def __init__(self, filepath):
        self.d2v = Doc2Vec.load(filepath)

    def __call__(self, item):
        return self.d2v.infer_vector(item)
