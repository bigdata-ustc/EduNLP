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
    """

    Parameters
    --------
    pretrain_model: str, required
        The pretrained model name.
    torch.Size([1024])

    """

    def __init__(self, pretrained_model_path):
        super(ElmoModel, self).__init__()
        self.tokenizer = ElmoTokenizer()
        self.tokenizer.load_vocab(os.path.join(pretrained_model_path, 'vocab.json'))
        with open(os.path.join(pretrained_model_path, 'config.json'), 'r') as f:
            config = json.load(f)
        self.model = ElmoLM(vocab_size=len(self.tokenizer), embedding_dim=config['emb_dim'],
                            hidden_size=config['hid_dim'])
        self.model.load_state_dict(torch.load(os.path.join(pretrained_model_path, 'weight.pt')))
        self.model.eval()

    def __call__(self, *args, **kwargs):
        return self.infer_vector(*args, **kwargs)

    def get_contextual_emb(self, item_indices: list, token_idx: int, scale: int = 1):
        return self.model.get_contextual_emb(item_indices, token_idx, scale)

    def infer_vector(self, item, *args, **kwargs) -> torch.Tensor:
        item = [0 if token not in self.tokenizer.t2id else self.tokenizer.t2id[token] for token in item]
        pred_forward, pred_backward, forward_hiddens, backward_hiddens = self.model(torch.tensor([item]),
                                                                                    torch.tensor([len(item)]),
                                                                                    torch.device('cpu'))
        ret = torch.cat((forward_hiddens[0, -1, :], backward_hiddens[0, 0, :]), dim=-1)
        return ret

    def infer_tokens(self, item, *args, **kwargs) -> torch.Tensor:
        item = [0 if token not in self.tokenizer.t2id else self.tokenizer.t2id[token] for token in item]
        pred_forward, pred_backward, forward_hiddens, backward_hiddens = self.model(torch.tensor([item]),
                                                                                    torch.tensor([len(item)]),
                                                                                    torch.device('cpu'))

        ret = torch.cat((forward_hiddens[0], torch.flip(backward_hiddens, [1])[0]), dim=-1)
        return ret

    @property
    def vector_size(self):
        return 2 * self.model.hidden_size
