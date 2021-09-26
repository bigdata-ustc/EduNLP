import numpy as np
from pathlib import PurePath
from transformers import AutoModel
from EduNLP.Pretrain import BertTokenizer
from .const import UNK, PAD
from .meta import Vector
import torch


class BertModel(Vector):
    """
    Examples
    --------
    >>> tokenizer = BertTokenizer("bert-base-chinese")
    >>> model = BertModel("bert-base-chinese", tokenizer=tokenizer)
    >>> item = "有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$，若$x,y$满足约束"
    >>> inputs = tokenizer(item, return_tensors='pt')
    >>> output = model(inputs)
    >>> output.shape
    torch.Size([1, 12, 768])
    >>> tokens = model.infer_tokens(inputs)
    >>> tokens.shape
    torch.Size([1, 10, 768])
    >>> tokens = model.infer_tokens(inputs, return_special_tokens=True)
    >>> tokens.shape
    torch.Size([1, 12, 768])
    >>> item = model.infer_vector(inputs)
    >>> item.shape
    torch.Size([1, 768])
    """
    def __init__(self, pretrained_model, tokenizer=None):
        self.model = AutoModel.from_pretrained(pretrained_model)
        if tokenizer:
            self.model.resize_token_embeddings(len(tokenizer.tokenizer))

    def __call__(self, items):
        # 1, sent_len, embedding_size
        tokens = self.model(**items).last_hidden_state
        return tokens

    def infer_vector(self, items) -> torch.Tensor:
        vector = self(items)
        return vector[:, 0, :]

    def infer_tokens(self, items, return_special_tokens=False) -> torch.Tensor:
        tokens = self(items)
        if return_special_tokens:
            return tokens
        else:
            return tokens[:, 1:-1, :]

    @property
    def vector_size(self):
        return self.model.config.hidden_size
