import numpy as np
from pathlib import PurePath
from transformers import AutoModel
from .const import UNK, PAD
from .meta import Vector
import torch


class BertModel(Vector):
    """
    Examples
    --------
    >>> from EduNLP.Pretrain import BertTokenizer
    >>> tokenizer = BertTokenizer("bert-base-chinese", add_special_tokens=False)
    >>> model = BertModel("bert-base-chinese")
    >>> item = ["有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$，若$x,y$满足约束",
    ... "有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$，若$x,y$满足约束"]
    >>> inputs = tokenizer(item, return_tensors='pt')
    >>> output = model(inputs)
    >>> output.shape
    torch.Size([2, 14, 768])
    >>> tokens = model.infer_tokens(inputs)
    >>> tokens.shape
    torch.Size([2, 12, 768])
    >>> tokens = model.infer_tokens(inputs, return_special_tokens=True)
    >>> tokens.shape
    torch.Size([2, 14, 768])
    >>> item = model.infer_vector(inputs)
    >>> item.shape
    torch.Size([2, 768])
    """

    def __init__(self, pretrained_model):
        self.model = AutoModel.from_pretrained(pretrained_model)

    def __call__(self, items: dict):
        # batch_size, sent_len, embedding_size
        tokens = self.model(**items).last_hidden_state
        return tokens

    def infer_vector(self, items: dict, pooling_strategy='CLS') -> torch.Tensor:
        vector = self(items)
        if pooling_strategy == 'CLS':
            return vector[:, 0, :]
        elif pooling_strategy == 'average':
            # the average of word embedding of the last layer
            # batch_size, sent_len, embedding_dim
            mask = items['attention_mask'].unsqueeze(-1).expand(vector.size())
            mul_mask = vector * mask
            # batch_size, embedding_dim
            return mul_mask.sum(1) / (mask.sum(1) + 1e-10)

    def infer_tokens(self, items: dict, return_special_tokens=False) -> torch.Tensor:
        tokens = self(items)
        if return_special_tokens:
            # include embedding of [CLS] and [SEP]
            return tokens
        else:
            # ignore embedding of [CLS] and [SEP]
            return tokens[:, 1:-1, :]

    @property
    def vector_size(self):
        return self.model.config.hidden_size
