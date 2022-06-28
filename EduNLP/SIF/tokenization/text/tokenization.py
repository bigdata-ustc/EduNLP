# coding: utf-8
# 2021/5/18 @ tongshiwei
import logging
import jieba
from .stopwords import DEFAULT_STOPWORDS

jieba.setLogLevel(logging.INFO)


def is_chinese(word):
    """判断一个char或者string是否是汉字(串)"""
    for char in word:
        if char < u'\u4e00' or char > u'\u9fa5':
            return False
    return True

def tokenize(text, granularity="word", stopwords="default"):
    """
    Using jieba library to tokenize item by word or char.

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
        jieba_tokens = [token for token in jieba.cut(text) if token not in stopwords and token.strip()]
        print("[debug] jieba_tokens:", jieba_tokens)
        # Use jieba_tokens to hangle sentence with mixed chinese and english.
        split_tokens = []
        for token in jieba_tokens:
            if is_chinese(token):
                split_tokens.extend(list(token))
            else:
                split_tokens.append(token)
        return split_tokens
    else:
        raise TypeError("Unknown granularity %s" % granularity)
