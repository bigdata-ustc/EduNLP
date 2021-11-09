# coding: utf-8
# 2021/5/18 @ tongshiwei
import logging
import jieba
from .stopwords import DEFAULT_STOPWORDS

jieba.setLogLevel(logging.INFO)


def tokenize(text, granularity="word", stopwords="default"):
    """

    Parameters
    ----------
    text
    granularity
    stopwords: str, None or set

    Returns
    -------

    Examples
    --------
    >>> tokenize("三角函数是基本初等函数之一")
    ['三角函数', '初等', '函数']
    >>> tokenize("三角函数是基本初等函数之一", granularity="char")
    ['三', '角', '函', '数', '基', '初', '函', '数']
    """
    stopwords = DEFAULT_STOPWORDS if stopwords == "default" else stopwords
    stopwords = stopwords if stopwords is not None else {}
    if granularity == "word":
        return [token for token in jieba.cut(text) if token not in stopwords and token.strip()]
    elif granularity == "char":
        stopwords = stopwords if stopwords is not None else {}
        return [token for token in text if token not in stopwords and token.strip()]
    else:
        raise TypeError("Unknown granularity %s" % granularity)
