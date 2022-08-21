from pathlib import PurePath
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as tud
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from EduNLP.Pretrain import train_elmo, ElmoTokenizer
from EduNLP.ModelZoo.rnn import ElmoLM
from .meta import Vector
import json
from typing import Dict, List, Tuple

class ElmoModel(Vector):
    def __init__(self, pretrained_dir: str):
        """
        Parameters
        ----------
        pretrained_model_path: str
        """
        super(ElmoModel, self).__init__()
        self.model = ElmoLM.from_pretrained(pretrained_dir)
        self.model.eval()

    def __call__(self, *args, **kwargs):
        return self.infer_vector(*args, **kwargs)

    def infer_vector(self, items: Tuple[dict, List[dict]], *args, **kwargs) -> torch.Tensor:
        # TODO: handle batch and unbatch format for inputs and outputs
        is_batch = isinstance(items, list)
        # items = items if is_batch else [items]
        outputs = self.model(**items)
        item_embeds = torch.cat(
            (outputs.forward_output[torch.arange(len(items["seq_len"])), torch.tensor(items["seq_len"]) - 1],
             outputs.backward_output[torch.arange(len(items["seq_len"])), 0]),
            dim=-1)
        return item_embeds

    def infer_tokens(self, items, *args, **kwargs) -> torch.Tensor:
        # is_batch = isinstance(items, list)
        outputs = self.model(**items)
        forward_hiddens = outputs.forward_output
        backward_hiddens = outputs.backward_output
        return torch.cat((forward_hiddens, backward_hiddens), dim=-1)
        # if is_batch:
        #     ret = []
        #     for fh, bh, lg in zip(forward_hiddens, backward_hiddens, items.seq_len):
        #         _bh = torch.cat((torch.flip(bh[:lg], [0]), bh[lg:]), dim=0)
        #         ret.append(torch.cat((fh, _bh), dim=-1))
        #     return torch.stack(tuple(ret))
        # else:
        #     return torch.cat((forward_hiddens[0], torch.flip(backward_hiddens, [1])[0]), dim=-1)

    @property
    def vector_size(self):
        return 2 * self.model.hidden_size
