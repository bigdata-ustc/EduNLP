# coding: utf-8
# 2021/5/29 @ tongshiwei

from pathlib import PurePath
from gensim.models import KeyedVectors, Word2Vec, FastText, Doc2Vec,TfidfModel
from gensim import corpora


class W2V(object):
    def __init__(self, filepath, method, binary=None):
        fp = PurePath(filepath)
        self.binary = binary if binary is not None else (True if fp.suffix == ".bin" else False)
        if self.binary is True:
            if method == "fasttext":
                self.wv = FastText.load(filepath).wv
            else:
                self.wv = Word2Vec.load(filepath).wv
        else:
            self.wv = KeyedVectors.load(filepath, mmap="r")

    def __call__(self, *words):
        for word in words:
            yield self.wv[word]

    def __getitem__(self, item):
        return self.wv[item]


class D2V(object):
    def __init__(self, filepath, method = "d2v"):
        self._method = method
        self._filepath = filepath
        if self._method == "d2v":
            self.d2v = Doc2Vec.load(filepath)
        elif self._method == "bow":
            self.d2v = corpora.Dictionary.load(filepath)
        elif self._method == "tfidf":
            self.d2v = TfidfModel.load(filepath)
        else:
            pass

    def __call__(self, item):
        if self._method == "d2v":
            return self.d2v.infer_vector(item)
        elif self._method == "bow":
            return self.d2v.doc2bow(item)
        elif self._method == "tfidf":
            # 'tfidf' model shold be used based on 'bow' model
            dictionary_path = self._filepath.replace("tfidf","bow")
            dictionary = D2V(dictionary_path, method = "bow")
            item = dictionary(item)
            return self.d2v[item]
        else:
            pass
        
