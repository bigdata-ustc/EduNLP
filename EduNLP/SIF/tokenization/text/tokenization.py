# coding: utf-8
# 2021/5/18 @ tongshiwei
import logging
import jieba
from nltk.tokenize import word_tokenize
import nltk
import spacy
import tokenizers as huggingface_tokenizer
from tokenizers.trainers import BpeTrainer
from .stopwords import DEFAULT_STOPWORDS

jieba.setLogLevel(logging.INFO)


def is_chinese(word):
    """判断一个char或者string是否是汉字(串)"""
    for char in word:
        if char < u'\u4e00' or char > u'\u9fa5':
            return False
    return True


def tokenize(text,
             granularity="word",
             stopwords="default",
             tokenizer="jieba",
             tok_model="en_core_web_sm",
             bpe_json='bpe.tokenizer.json',
             bpe_trainfile=None):
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
    ['三', '角', '函', '数', '初', '等', '函', '数']
    """
    stopwords = DEFAULT_STOPWORDS if stopwords == "default" else stopwords
    stopwords = stopwords if stopwords is not None else {}

    if (tokenizer == 'jieba'):
        if granularity == "word":
            return [
                token for token in jieba.cut(text)
                if token not in stopwords and token.strip()
            ]
        elif granularity == "char":
            jieba_tokens = [
                token for token in jieba.cut(text)
                if token not in stopwords and token.strip()
            ]
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

    elif (tokenizer == 'nltk'):
        try:
            return [
                token for token in word_tokenize(text)
                if token not in stopwords and token.strip()
            ]
        except LookupError:
            nltk.download('punkt')
        return [
            token for token in word_tokenize(text)
            if token not in stopwords and token.strip()
        ]

    elif (tokenizer == 'spacy'):
        try:
            spacy_tokenizer = spacy.load(tok_model)
        except OSError:
            spacy.cli.download(tok_model)
            spacy_tokenizer = spacy.load(tok_model)

        return [
            token.text for token in spacy_tokenizer(text)
            if token.text not in stopwords
        ]

    elif (tokenizer == 'bpe'):
        tokenizer = huggingface_tokenizer.Tokenizer(
            huggingface_tokenizer.models.BPE())
        if (bpe_trainfile is None):
            raise LookupError("bpe train file not found, using %s." % bpe_trainfile)
        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        tokenizer.train(files=[bpe_trainfile], trainer=trainer)
        output = tokenizer.encode(text)
        return [
            token for token in output.tokens if token not in stopwords
        ]
    else:
        raise TypeError("Invalid Spliter: %s" % tokenizer)
