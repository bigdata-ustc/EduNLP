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


class QuestionEncoder(nn.Module):
    """
    DisenQNet question representation model

    Parameters
    ----------
    vocab_size: int
        size of vocabulary
    hidden_dim: int
        size of word and question embedding
    dropout: float
        dropout rate
    wv: torch.Tensor
        Tensor of (vocab_size, hidden_dim) or None, initial word embedding, default = None
    """

    def __init__(self, vocab_size, hidden_dim, dropout, wv=None):
        super(QuestionEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = TextEncoder(vocab_size, hidden_dim, dropout, wv=wv)
        self.k_model = AttnModel(hidden_dim, dropout)
        self.i_model = AttnModel(hidden_dim, dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input, length, get_vk=True, get_vi=True):
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
        (embed, k_hidden, i_hidden)
            - embed: Tensor of (batch_size, seq_len, hidden_dim), word embedding
            - k_hidden: Tensor of (batch_size, hidden_dim) or None, concept representation of question
            - i_hidden: Tensor of (batch_size, hidden_dim) or None, individual representation of question
        """
        # embed: batch_size * seq_len * hidden_dim
        # q_hidden: batch_size * hidden_dim
        embed, q_hidden = self.encoder(input)
        # batch_size * seq_len, 0 means valid, 1 means pad
        mask = get_mask(input.size(1), length)
        embed.masked_fill_(mask.unsqueeze(-1), 0)
        k_hidden, i_hidden = None, None
        q_hidden_dp = self.dropout(q_hidden)
        embed_dp = self.dropout(embed)
        # batch_size * hidden_dim
        if get_vk:
            k_hidden, _ = self.k_model(q_hidden_dp, embed_dp, embed_dp, mask)
        if get_vi:
            i_hidden, _ = self.i_model(q_hidden_dp, embed_dp, embed_dp, mask)
        return embed, k_hidden, i_hidden


class DisenQNet(object):
    """
    DisenQNet training and evaluation model

    Parameters
    ----------
    vocab_size: int
        size of vocabulary
    concept_size: int
        number of concept classes
    hidden_dim: int
        size of word and question embedding
    dropout: float
        dropout rate
    pos_weight: float
        positive sample weight in unbalanced multi-label concept classifier
    w_cp: float
        weight of concept loss
    w_mi: float
        weight of mutual information loss
    w_dis: float
        weight of disentangling loss
    wv: torch.Tensor
        Tensor of (vocab_size, hidden_dim) or None, initial word embedding, default = None
    device: str, defaults as 'cpu'
        Set device for model, examples 'cpu'、'cuda'、'cuda:0,2'
    """
    def __init__(self, vocab_size, concept_size, hidden_dim, dropout, pos_weight, w_cp, w_mi, w_dis,
                 wv=None, device="cpu"):
        super(DisenQNet, self).__init__()
        self.disen_q_net = QuestionEncoder(vocab_size, hidden_dim, dropout, wv)
        self.mi_estimator = MIEstimator(hidden_dim, hidden_dim * 2, dropout)
        self.concept_estimator = ConceptEstimator(hidden_dim, concept_size, pos_weight, dropout)
        self.disen_estimator = DisenEstimator(hidden_dim, dropout)
        self.w_cp = w_cp
        self.w_mi = w_mi
        self.w_dis = w_dis
        self.hidden_dim = hidden_dim
        self.params = {
            "vocab_size": vocab_size,
            "concept_size": concept_size,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "pos_weight": pos_weight,
            "w_cp": w_cp,
            "w_mi": w_mi,
            "w_dis": w_dis,
        }
        self.modules = (self.disen_q_net, self.mi_estimator, self.concept_estimator, self.disen_estimator)
        self.to(device)

    def train(self, train_data, test_data, epoch, lr, step_size, gamma, warm_up, n_adversarial, silent):
        """
        train DisenQNet

        Parameters
        ----------
        train_data:
            train dataloader, contains text, length, concept
            - text: Tensor of (batch_size, seq_len)
            - length: Tensor of (batch_size)
            - concept: Tensor of (batch_size, class_size)
        test_data:
            test dataloader
        epoch: int
            number of epoch
        lr: float
            initial learning rate
        step_size: int
            step_size for StepLR, period of learning rate decay
        gamma: float
            gamma for StepLR, multiplicative factor of learning rate decay
        warm_up: int
            number of epoch for warming up, without adversarial process for dis_loss
        n_adversarial: int
            ratio of disc/enc training for adversarial process
        silent: bool
            whether to log loss
        """
        if not silent:
            print("Start training the disenQNet...")
        # optimizer & scheduler
        model_params = list()
        for params in [list(self.disen_q_net.parameters()),
                       list(self.mi_estimator.parameters()), list(self.concept_estimator.parameters())]:
            model_params.extend(params)

        adv_params = list(self.disen_estimator.parameters())
        optimizer = Adam(model_params, lr=lr)
        adv_optimizer = Adam(adv_params, lr=lr)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        adv_scheduler = StepLR(adv_optimizer, step_size=step_size, gamma=gamma)
        # train
        epoch = warm_up + epoch
        for epoch_idx in range(epoch):
            epoch_idx += 1
            # warming_up: cp_loss & mi_loss only, ignore adversarial dis_loss
            warming_up = (epoch_idx <= warm_up)
            self.set_mode(True)
            for data in train_data:
                text, length, concept = data
                text, length, concept = text.to(self.device), length.to(self.device), concept.to(self.device)
                # WGAN-like adversarial training: min_enc max_disc dis_loss
                # train disc
                if not warming_up:
                    _, k_hidden, i_hidden = self.disen_q_net(text, length)
                    # stop gradient propagation to encoder
                    k_hidden, i_hidden = k_hidden.detach(), i_hidden.detach()
                    # max dis_loss
                    dis_loss = - self.disen_estimator(k_hidden, i_hidden)
                    dis_loss = n_adversarial * self.w_dis * dis_loss
                    adv_optimizer.zero_grad()
                    dis_loss.backward()
                    adv_optimizer.step()
                    # Lipschitz constrain for Disc of WGAN
                    self.disen_estimator.spectral_norm()
                # train enc
                embed, k_hidden, i_hidden = self.disen_q_net(text, length)
                hidden = torch.cat((k_hidden, i_hidden), dim=-1)
                # max mi
                mi_loss = - self.mi_estimator(embed, hidden, length)
                # min concept_loss
                cp_loss = self.concept_estimator(k_hidden, concept)
                if warming_up:
                    loss = self.w_mi * mi_loss + self.w_cp * cp_loss
                else:
                    # min dis
                    dis_loss = self.disen_estimator(k_hidden, i_hidden)
                    loss = self.w_mi * mi_loss + self.w_cp * cp_loss + self.w_dis * dis_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if not warming_up:
                scheduler.step()
                adv_scheduler.step()
            # test
            train_loss = self.eval(train_data)
            if test_data is not None and not warming_up:
                test_loss = self.eval(test_data)
                if not silent:
                    print(f"[Epoch {epoch_idx:2d}] train loss: {train_loss:.4f}, eval loss: {test_loss:.4f}")
            elif not silent:
                print(f"[Epoch {epoch_idx:2d}] train loss: {train_loss:.4f}")
        return

    def inference(self, items: dict):
        """
        DisenQNet for i2v inference. Now not support for batch !

        Parameters
        ----------
        items:  dict
            which contains content_idx and  content_len
            - content_idx: Tensor of (batch_size, seq_len)
            - content_len: Tensor of (batch_size)
        device: str
            cpu or cuda

        Returns
        ---------
        embed: torch.Tensor
            Tensor of (batch_size, seq_len, hidden_dim)
        k_hidden: torch.Tensor
            Tensor of (batch_size, hidden_dim)
        i_hidden: torch.Tensor
            Tensor of (batch_size, hidden_dim)
        """
        self.set_mode(False)
        text, length = items["content_idx"].to(self.device), items["content_len"].to(self.device)
        embed, k_hidden, i_hidden = self.disen_q_net(text, length)
        return embed, k_hidden, i_hidden

    def eval(self, test_data):
        """
        eval DisenQNet

        Parameters
        ----------
        test_data:
            iterable, train dataset, contains text, length, concept
            - text: Tensor of (batch_size, seq_len)
            - length: Tensor of (batch_size)
            - concept: Tensor of (batch_size, class_size)
        device: str
            cpu or cuda

        Returns
        ---------
        loss: float
            average loss for test dataset
        """
        total_size = 0
        total_loss = 0
        self.set_mode(False)
        with torch.no_grad():
            for data in test_data:
                text, length, concept = data
                text, length, concept = text.to(self.device), length.to(self.device), concept.to(self.device)
                embed, k_hidden, i_hidden = self.disen_q_net(text, length)
                hidden = torch.cat((k_hidden, i_hidden), dim=-1)
                mi_loss = - self.mi_estimator(embed, hidden, length)
                cp_loss = self.concept_estimator(k_hidden, concept)
                dis_loss = self.disen_estimator(k_hidden, i_hidden)
                loss = self.w_mi * mi_loss + self.w_cp * cp_loss + self.w_dis * dis_loss
                batch_size = text.size(0)
                total_size += batch_size
                total_loss += loss.item() * batch_size
        loss = total_loss / total_size
        return loss

    def save_pretrained(self, output_dir):
        filepath = os.path.join(output_dir, "disen_q_net.th")
        config_path = os.path.join(output_dir, "model_config.json")
        state_dicts = [module.state_dict() for module in self.modules]
        torch.save(state_dicts, filepath)
        self.save_config(config_path)
        return

    def load(self, filepath):
        state_dicts = torch.load(filepath, map_location='cpu')
        for module, state_dict in zip(self.modules, state_dicts):
            module.load_state_dict(state_dict)
        return

    def to(self, device):
        for module in self.modules:
            # module.to(device)
            set_device(module, device)
        self.device = "cpu" if device == "cpu" else "cuda"
        return

    def set_mode(self, train):
        for module in self.modules:
            if train:
                module.train()
            else:
                module.eval()
        return

    def save_config(self, config_path):
        with open(config_path, "w", encoding="utf-8") as wf:
            json.dump(self.params, wf, ensure_ascii=False, indent=2)

    @classmethod
    def from_config(cls, config_path):
        with open(config_path, "r", encoding="utf-8") as rf:
            model_config = json.load(rf)
            wv = torch.load(model_config["wv_path"],
                            map_location='cpu', mmap="r") if "wv_path" in model_config else None
            return cls(
                model_config["vocab_size"], model_config["concept_size"], model_config["hidden_dim"],
                model_config["dropout"], model_config["pos_weight"], model_config["w_cp"], model_config["w_mi"],
                model_config["w_dis"], wv=wv)

    @classmethod
    def from_pretrained(cls, model_dir):
        config_path = os.path.join(model_dir, "model_config.json")
        model_path = os.path.join(model_dir, "disen_q_net.th")
        model = cls.from_config(config_path)
        model.load(model_path)
        return model
