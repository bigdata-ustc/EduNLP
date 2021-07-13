# coding: utf-8
# 2021/7/12 @ tongshiwei

import torch
from ..gensim_vec import W2V
from ..embedding import Embedding
from EduNLP.ModelZoo import rnn, pad_sequence


class RNNModel(object):
    """
    Examples
    --------
    >>> model = RNNModel("ELMO", None, 2, vocab_size=4, embedding_dim=3)
    >>> seq_idx = [[1, 2, 3], [1, 2, 0], [3, 0, 0]]
    >>> output, hn = model(seq_idx, indexing=False, padding=False)
    >>> seq_idx = [[1, 2, 3], [1, 2], [3]]
    >>> output, hn = model(seq_idx, indexing=False, padding=True)
    >>> output.shape
    torch.Size([3, 3, 4])
    >>> hn.shape
    torch.Size([2, 3, 2])
    >>> tokens = model.infer_tokens(seq_idx, indexing=False)
    >>> tokens.shape
    torch.Size([3, 3, 4])
    >>> tokens = model.infer_tokens(seq_idx, agg="mean", indexing=False)
    >>> tokens.shape
    torch.Size([3, 4])
    >>> item = model.infer_vector(seq_idx, indexing=False)
    >>> item.shape
    torch.Size([2, 3, 2])
    >>> item = model.infer_vector(seq_idx, agg="mean", indexing=False)
    >>> item.shape
    torch.Size([3, 2])
    >>> item = model.infer_vector(seq_idx, agg=-1, indexing=False)
    >>> item.shape
    torch.Size([3, 4])
    """

    def __init__(self, rnn_type, w2v: (W2V, tuple, list, dict, None), hidden_size, freeze_pretrained=True, **kwargs):
        self.embedding = Embedding(w2v, freeze_pretrained, **kwargs)
        for key in ["vocab_size", "embedding_dim"]:
            if key in kwargs:
                kwargs.pop(key)
        self.rnn = rnn.LM(
            rnn_type,
            self.embedding.vocab_size,
            self.embedding.embedding_dim,
            hidden_size=hidden_size,
            embedding=self.embedding.embedding,
            **kwargs
        )
        self.freeze_pretrained = freeze_pretrained

    def __call__(self, seq, indexing=True, padding=True, **kwargs):
        seq_idx = [[self.embedding.key_to_index(w) for w in s] for s in seq] if indexing is True else seq
        seq_len = [len(_idx) for _idx in seq_idx]
        pad_seq_idx = pad_sequence(seq_idx, pad_val=self.embedding.pad_val) if padding is True else seq_idx

        tokens, item = self.rnn(torch.LongTensor(pad_seq_idx), torch.LongTensor(seq_len))

        return tokens, item

    def infer_vector(self, seq, agg=None, **kwargs):
        vector = self(seq, **kwargs)[1]
        if agg is not None:
            if agg == -1:
                return torch.reshape(vector, (vector.shape[1], -1))
            return eval("torch.%s" % agg)(vector, dim=0)
        return vector

    def infer_tokens(self, seq, agg=None, **kwargs):
        tokens = self(seq, **kwargs)[0]
        if agg is not None:
            return eval("torch.%s" % agg)(tokens, dim=1)
        return tokens
