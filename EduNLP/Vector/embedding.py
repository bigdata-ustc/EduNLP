# coding: utf-8
# 2021/7/12 @ tongshiwei

import torch
from .gensim_vec import W2V
from .const import PAD


class Embedding(object):
    def __init__(self, w2v: (W2V, tuple, list, dict, None), freeze=True, **kwargs):
        if w2v is None:
            self.w2v = None
        elif isinstance(w2v, (tuple, list)):
            self.w2v = W2V(*w2v)
        elif isinstance(w2v, dict):
            self.w2v = W2V(**w2v)
        elif isinstance(w2v, W2V):
            self.w2v = w2v
        else:
            raise TypeError("w2v argument must be one of W2V, tuple, list, dict or None")

        if self.w2v is not None:
            self.vocab_size = len(self.w2v)
            self.embedding_dim = self.w2v.vector_size
        else:
            self.vocab_size = kwargs["vocab_size"]
            self.embedding_dim = kwargs["embedding_dim"]

        self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim)

        self.pad_val = 0
        if self.w2v is not None:
            self.embedding.from_pretrained(torch.Tensor(self.w2v.vectors), freeze)
            self.pad_val = self.w2v.constants[PAD]
        self.key_to_index = self.w2v.key_to_index if w2v is not None else lambda x: x

    def __call__(self, *args, **kwargs):  # todo
        raise NotImplementedError

    def infer_vector(self):  # todo
        raise NotImplementedError

    def infer_tokens(self):  # todo
        raise NotImplementedError
