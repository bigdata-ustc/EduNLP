import imp
from pickle import NONE
import numpy as np
from pathlib import PurePath
from transformers import AutoModel
# from .const import UNK, PAD
# from .meta import Vector
import torch
from  EduNLP.ModelZoo.DisenQNet.DisenQNet import DisenQNet


"""
注意修改DisenQNet中输入数据的格式，使用tokenizer处理后的令牌序列？
"""

class DisenQModel(object): # Vector
    """
    Examples
    --------

    """
    def __init__(self, pretrained_model, device="cpu"):
        """
        Parameters
        ----------
        pretrained_model: str
            the path to pretrained model
        device: str
            cpu or cuda, default is cpu
        """
        self.device = device
        self.model = DisenQNet.from_pretrained(pretrained_model)

    def __call__(self, item: dict):
        embed, k_hidden, i_hidden = self.model.inference(item, device=self.device)
        return embed, k_hidden, i_hidden

    def infer_vector(self, item: dict, vector_type=None) -> torch.Tensor:
        _, k_hidden, i_hidden = self(item)
        if vector_type == None:
            return k_hidden, i_hidden
        elif vector_type == "k":
            return k_hidden
        elif vector_type == "i":
            return i_hidden
        else:
            raise KeyError("vector_type must be one of (None, 'k', 'i') ")

    def infer_tokens(self, item: dict) -> torch.Tensor:
        embed, _, _ = self(item)
        return embed

    @property
    def vector_size(self):
        return self.model.hidden_dim
