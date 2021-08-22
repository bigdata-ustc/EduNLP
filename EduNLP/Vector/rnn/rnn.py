# coding: utf-8
# 2021/7/12 @ tongshiwei

import torch
from ..gensim_vec import W2V
from ..embedding import Embedding
from ..meta import Vector
from EduNLP.ModelZoo import rnn, set_device
from baize.torch import save_params


class RNNModel(Vector):
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
    torch.Size([3, 4])
    >>> item = model.infer_vector(seq_idx, agg="mean", indexing=False)
    >>> item.shape
    torch.Size([3, 2])
    >>> item = model.infer_vector(seq_idx, agg=None, indexing=False)
    >>> item.shape
    torch.Size([2, 3, 2])
    """

    def __init__(self, rnn_type, w2v: (W2V, tuple, list, dict, None), hidden_size,
                 freeze_pretrained=True, model_params=None, device=None,
                 **kwargs):
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
            model_params=model_params,
            **kwargs
        )
        self.bidirectional = self.rnn.rnn.bidirectional
        self.hidden_size = self.rnn.hidden_size
        self.freeze_pretrained = freeze_pretrained
        if device is not None:
            self.set_device(device)

    def __call__(self, items, indexing=True, padding=True, **kwargs):
        seq_idx, seq_len = self.embedding(items, indexing=indexing, padding=padding, vectorization=False)

        tokens, item = self.rnn(torch.LongTensor(seq_idx), torch.LongTensor(seq_len))

        return tokens, item

    def infer_vector(self, items, agg: (int, str, None) = -1, indexing=True, padding=True, *args,
                     **kwargs) -> torch.Tensor:
        vector = self(items, indexing=indexing, padding=padding, **kwargs)[1]
        if agg is not None:
            if agg == -1:
                return torch.reshape(vector, (vector.shape[1], -1))
            return eval("torch.%s" % agg)(vector, dim=0)
        return vector

    def infer_tokens(self, items, agg=None, *args, **kwargs) -> torch.Tensor:
        tokens = self(items, **kwargs)[0]
        if agg is not None:
            return eval("torch.%s" % agg)(tokens, dim=1)
        return tokens

    @property
    def vector_size(self) -> int:
        return self.hidden_size * (1 if self.bidirectional is False else 2)

    def set_device(self, device):
        self.rnn = set_device(self.rnn, device)

    def save(self, filepath, save_embedding=False):
        save_params(filepath, self.rnn, select=None if save_embedding is True else '^(?!.*embedding)')
        return filepath

    def freeze(self, *args, **kwargs):
        return self.eval()

    @property
    def is_frozen(self):
        return not self.rnn.training

    def eval(self):
        self.rnn.eval()
        return self

    def train(self, mode=True):
        self.rnn.train(mode)
        return self
