# coding: utf-8
# 2021/7/12 @ tongshiwei

from typing import List
import torch
from .gensim_vec import W2V
from .const import PAD
from EduNLP.ModelZoo import pad_sequence, set_device


class Embedding(object):
    def __init__(self, w2v: (W2V, tuple, list, dict, None), freeze=True, device=None, **kwargs):
        if w2v is None:
            self.w2v = None
        elif isinstance(w2v, (tuple, list)):
            self.w2v = W2V(*w2v)
        elif isinstance(w2v, dict):
            self.w2v = W2V(**w2v)
        elif isinstance(w2v, W2V):
            self.w2v = w2v
        else:
            raise TypeError("w2v argument must be one of W2V, tuple, list, dict or None, now is %s" % type(w2v))

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

        if device is not None:
            self.set_device(device)

    def __call__(self, items: List[List[str]], indexing=True, padding=True, vectorization=True, *args,
                 **kwargs) -> tuple:

        items, item_len = self.indexing(items, padding=padding, indexing=indexing)
        items = self.infer_token_vector(items, indexing=False)[0] if vectorization else items
        return items, item_len

    def infer_token_vector(self, items: List[List[str]], indexing=True) -> tuple:
        items, item_len = self.indexing(items, padding=True, indexing=indexing)
        item_embedding = self.embedding(torch.LongTensor(items))
        return item_embedding, item_len

    def indexing(self, items: List[List[str]], padding=False, indexing=True) -> tuple:
        """

        Parameters
        ----------
        items: list of list of str(word/token)
        padding: bool
            whether padding the returned list with default pad_val to make all item in items have the same length
        indexing: bool

        Returns
        -------
        token_idx: list of list of int
            the list of the tokens of each item
        token_len: list of int
            the list of the length of tokens of each item
        """
        items_idx = [[self.key_to_index(word) for word in item] for item in items] if indexing else items
        item_len = [len(_idx) for _idx in items_idx]
        padded_items_idx = pad_sequence(items_idx, pad_val=self.pad_val) if padding is True else items_idx
        return padded_items_idx, item_len

    def set_device(self, device):
        self.embedding = set_device(self.embedding, device)
        return self
