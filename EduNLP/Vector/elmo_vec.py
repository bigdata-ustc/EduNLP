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


class ElmoModel(Vector):
    def __init__(self, pretrained_model_path: str):
        """
        Parameters
        ----------
        pretrained_model_path: str
        """
        super(ElmoModel, self).__init__()
        with open(os.path.join(pretrained_model_path, 'config.json'), 'r') as f:
            config = json.load(f)
        self.model = ElmoLM(vocab_size=config['vocab_size'], embedding_dim=config['emb_dim'],
                            hidden_size=config['hid_dim'])
        self.model.load_state_dict(torch.load(os.path.join(pretrained_model_path, 'weight.pt')))
        self.model.eval()

    def __call__(self, *args, **kwargs):
        return self.infer_vector(*args, **kwargs)

    def infer_vector(self, items, *args, **kwargs) -> torch.Tensor:
        is_batch = isinstance(items[0], list)
        items = items if is_batch else [items]
        lengths = kwargs.get('length', [len(i) for i in items])
        pred_forward, pred_backward, forward_hiddens, backward_hiddens = self.model(torch.tensor(items),
                                                                                    torch.tensor(lengths,
                                                                                                 dtype=torch.int64))
        ret = torch.cat((forward_hiddens[:, -1, :], backward_hiddens[:, 0, :]), dim=-1) if is_batch else torch.cat(
            (forward_hiddens[0, -1, :], backward_hiddens[0, 0, :]), dim=-1)
        return ret

    def infer_tokens(self, items, *args, **kwargs) -> torch.Tensor:
        is_batch = isinstance(items[0], list)
        items = items if is_batch else [items]
        lengths = kwargs.get('length', [len(i) for i in items])
        pred_forward, pred_backward, forward_hiddens, backward_hiddens = self.model(torch.tensor(items),
                                                                                    torch.tensor(lengths,
                                                                                                 dtype=torch.int64))

        ret = torch.cat((forward_hiddens[:], torch.flip(backward_hiddens, [1])[:]), dim=-1) if is_batch else torch.cat(
            (forward_hiddens[0], torch.flip(backward_hiddens, [1])[0]), dim=-1)
        return ret

    @property
    def vector_size(self):
        return 2 * self.model.hidden_size
