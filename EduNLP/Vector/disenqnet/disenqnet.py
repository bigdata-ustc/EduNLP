import imp
from pickle import NONE
import numpy as np
from pathlib import PurePath
import torch
from EduNLP.ModelZoo.disenqnet.disenqnet import DisenQNet


class DisenQModel(object):
    def __init__(self, pretrained_dir, device="cpu"):
        """
        Parameters
        ----------
        pretrained_dir: str
            the dirname to pretrained model
        device: str
            cpu or cuda, default is cpu
        """
        self.device = device
        self.model = DisenQNet.from_pretrained(pretrained_dir)
        self.model.to(self.device)

    def __call__(self, items: dict, **kwargs):
        embed, k_hidden, i_hidden = self.model.inference(items)
        return embed, k_hidden, i_hidden

    def infer_vector(self, items: dict, vector_type=None, **kwargs) -> torch.Tensor:
        """
        Parameters
        ----------
        vector_type: str
            choose the type of items tensor to return.
            Default is None, which means return both (k_hidden, i_hidden)
            when vector_type="k", return k_hidden;
            when vector_type="i", return i_hidden;
        """
        _, k_hidden, i_hidden = self(items)
        if vector_type is None:
            return k_hidden, i_hidden
        elif vector_type == "k":
            return k_hidden
        elif vector_type == "i":
            return i_hidden
        else:
            raise KeyError("vector_type must be one of (None, 'k', 'i') ")

    def infer_tokens(self, items: dict, **kwargs) -> torch.Tensor:
        embed, _, _ = self(items)
        return embed

    @property
    def vector_size(self):
        return self.model.hidden_dim
