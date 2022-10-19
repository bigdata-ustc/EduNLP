# -*- coding: utf-8 -*-

import logging
import torch
from torch import nn
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR
import os
import json
from gensim.models import KeyedVectors

from .modules import TextEncoder, AttnModel, ConceptEstimator, MIEstimator, DisenEstimator
from .utils import get_mask
from ..utils import set_device
from ..base_model import BaseModel
from transformers.modeling_outputs import ModelOutput
from transformers import PretrainedConfig


class DisenQNetOutput(ModelOutput):
    """
    Output type of [`DisenQNet`]

    Parameters
    ----------
    embed: Tensor of (batch_size, seq_len, hidden_size), word embedding
    k_hidden: Tensor of (batch_size, hidden_size) or None, concept representation of question
    i_hidden: Tensor of (batch_size, hidden_size) or None, individual representation of question
    """
    embeded: torch.FloatTensor = None
    k_hidden: torch.FloatTensor = None
    i_hidden: torch.FloatTensor = None


class DisenQNet(BaseModel):
    base_model_prefix = 'disenq'
    """
    DisenQNet question representation model

    Parameters
    ----------
    vocab_size: int
        size of vocabulary
    hidden_size: int
        size of word and question embedding
    dropout_rate: float
        dropout rate
    wv: torch.Tensor
        Tensor of (vocab_size, hidden_size) or None, initial word embedding, default = None
    """

    def __init__(self, vocab_size: int, hidden_size: int, dropout_rate: float, wv=None, **argv):
        super(DisenQNet, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = TextEncoder(vocab_size, hidden_size, dropout_rate, wv=wv)
        self.k_model = AttnModel(hidden_size, dropout_rate)
        self.i_model = AttnModel(hidden_size, dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate)
        # config
        config = {k: v for k, v in locals().items() if k != "self" and k != "__class__" and k != "argv" and k!= 'wv'}
        config.update(argv)
        config['architecture'] = 'DisenQNet'
        self.config = PretrainedConfig.from_dict(config)

    def load_wv(self, wv):
        if isinstance(wv, torch.Tensor):
            # tensor
            self.encoder.load_wv(wv)
        else:
            # path
            print(f"load word2vec from {wv}")
            word2vec = torch.load(wv)
            self.encoder.load_wv(word2vec)

    def forward(self, seq_idx, seq_len, get_vk=True, get_vi=True) -> ModelOutput:
        """
        Parameters
        ----------
        input: Tensor of (batch_size, seq_len)
            word index
        length: Tensor of (batch_size)
            valid sequence length of each batch
        get_vk: bool
            whether to return vk
        get_vi: bool
            whether to return vi

        Returns
        -------
        DisenQNetOutput
            - embed: Tensor of (batch_size, seq_len, hidden_size), word embedding
            - k_hidden: Tensor of (batch_size, hidden_size) or None, concept representation of question
            - i_hidden: Tensor of (batch_size, hidden_size) or None, individual representation of question
        """
        # embed: batch_size * seq_len * hidden_size
        # q_hidden: batch_size * hidden_size
        embed, q_hidden = self.encoder(seq_idx)
        # batch_size * seq_len, 0 means valid, 1 means pad
        mask = get_mask(seq_idx.size(1), seq_len)
        embed.masked_fill_(mask.unsqueeze(-1), 0)
        k_hidden, i_hidden = None, None
        q_hidden_dp = self.dropout(q_hidden)
        embed_dp = self.dropout(embed)
        # batch_size * hidden_size
        if get_vk:
            k_hidden, _ = self.k_model(q_hidden_dp, embed_dp, embed_dp, mask)
        if get_vi:
            i_hidden, _ = self.i_model(q_hidden_dp, embed_dp, embed_dp, mask)
        return DisenQNetOutput(
            embeded=embed,
            k_hidden=k_hidden,
            i_hidden=i_hidden
        )

    @classmethod
    def from_config(cls, config_path, **argv):
        with open(config_path, "r", encoding="utf-8") as rf:
            model_config = json.load(rf)
            model_config.update(argv)
            return cls(
                vocab_size=model_config['vocab_size'],
                hidden_size=model_config['hidden_size'],
                dropout_rate=model_config['dropout_rate'],
            )


class DisenQNetForPreTrainingOutput(ModelOutput):
    """
    Output type of [`DisenQNetForPreTraining`]

    Parameters
    ----------
    loss
    embed: Tensor of (batch_size, seq_len, hidden_size), word embedding
    k_hidden: Tensor of (batch_size, hidden_size) or None, concept representation of question
    i_hidden: Tensor of (batch_size, hidden_size) or None, individual representation of question
    """
    loss: torch.FloatTensor = None
    embeded: torch.FloatTensor = None
    k_hidden: torch.FloatTensor = None
    i_hidden: torch.FloatTensor = None


class DisenQNetForPreTraining(BaseModel):
    base_model_prefix = 'disenq'

    def __init__(self, vocab_size, concept_size, hidden_size, dropout_rate, pos_weight,
                 w_cp, w_mi, w_dis, warmup, n_adversarial, wv=None, **argv):
        super(DisenQNetForPreTraining, self).__init__()
        self.disenq = DisenQNet(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            wv=wv,
            **argv)
        self.mi_estimator = MIEstimator(hidden_size, hidden_size * 2, dropout_rate)
        self.concept_estimator = ConceptEstimator(hidden_size, concept_size, pos_weight, dropout_rate)
        self.disen_estimator = DisenEstimator(hidden_size, dropout_rate)
        self.w_cp = w_cp
        self.w_mi = w_mi
        self.w_dis = w_dis
        self.hidden_size = hidden_size
        self.warming_up = False
        self.params = {
            "vocab_size": vocab_size,
            "concept_size": concept_size,
            "hidden_size": hidden_size,
            "dropout": dropout_rate,
            "pos_weight": pos_weight,
            "w_cp": w_cp,
            "w_mi": w_mi,
            "w_dis": w_dis,
            'warmup': warmup,
            'n_adversarial': n_adversarial,
        }
        self.modules = (self.disenq, self.mi_estimator, self.concept_estimator, self.disen_estimator)

        config = {k: v for k, v in locals().items() if k != "self" and k != "__class__" and k != "argv" and k!= 'wv'}
        config.update(argv)
        config['architecture'] = 'DisenQNetForPreTraining'
        self.config = PretrainedConfig.from_dict(config)

        model_params = list()
        for params in [list(self.disenq.parameters()),
                       list(self.mi_estimator.parameters()), list(self.concept_estimator.parameters())]:
            model_params.extend(params)
        self.model_params = model_params
        self.adv_params = list(self.disen_estimator.parameters())

    def forward(self, seq_idx=None, seq_len=None, concept=None) -> ModelOutput:
        # train enc
        # embed, k_hidden, i_hidden = self.disenq(seq_idx, seq_len)
        outputs = self.disenq(seq_idx, seq_len)
        embed = outputs.embeded
        k_hidden = outputs.k_hidden
        i_hidden = outputs.i_hidden
        print(k_hidden.shape, i_hidden.shape)
        hidden = torch.cat((k_hidden, i_hidden), dim=-1)
        # max mi
        mi_loss = - self.mi_estimator(embed, hidden, seq_len)
        # min concept_loss
        cp_loss = self.concept_estimator(k_hidden, concept)
        if self.warming_up:
            loss = self.w_mi * mi_loss + self.w_cp * cp_loss
        else:
            # min dis
            dis_loss = self.disen_estimator(k_hidden, i_hidden)
            loss = self.w_mi * mi_loss + self.w_cp * cp_loss + self.w_dis * dis_loss
        return DisenQNetForPreTrainingOutput(
            loss=loss,
            embeded=embed,
            k_hidden=k_hidden,
            i_hidden=i_hidden
        )

    @classmethod
    def from_config(cls, config_path, **argv):
        with open(config_path, "r", encoding="utf-8") as rf:
            model_config = json.load(rf)
            model_config.update(argv)
            return cls(
                vocab_size=model_config['vocab_size'],
                concept_size=model_config['concept_size'],
                hidden_size=model_config['hidden_size'],
                dropout_rate=model_config['dropout_rate'],
                pos_weight=model_config['pos_weight'],
                w_cp=model_config['w_cp'],
                w_mi=model_config['w_mi'],
                w_dis=model_config['w_dis'],
                warmup=model_config['warmup'],
                n_adversarial=model_config['n_adversarial'],
            )
