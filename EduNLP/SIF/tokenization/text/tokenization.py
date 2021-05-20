# coding: utf-8
# 2021/5/18 @ tongshiwei
import logging
import jieba
from .stopwords import DEFAULT_STOPWORDS

jieba.setLogLevel(logging.INFO)


def tokenize(text, granularity="word", stopwords=DEFAULT_STOPWORDS):
    stopwords = stopwords if stopwords is not None else {}
    if granularity == "word":
        return [token for token in jieba.cut(text) if token not in stopwords and token.strip()]
    elif granularity == "char":
        return [token for token in text if token not in stopwords and token.strip()]
    else:
        raise TypeError("Unknown granularity %s" % granularity)
