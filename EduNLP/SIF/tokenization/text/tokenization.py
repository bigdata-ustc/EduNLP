# coding: utf-8
# 2021/5/18 @ tongshiwei
import jieba


def tokenize(text, granularity="word", stopwords=None):
    stopwords = stopwords if stopwords is not None else {}
    if granularity == "word":
        return [token for token in jieba.cut(text) if token not in stopwords]
    elif granularity == "char":
        return [token for token in text if token not in stopwords]
    else:
        raise TypeError("Unknown granularity %s" % granularity)