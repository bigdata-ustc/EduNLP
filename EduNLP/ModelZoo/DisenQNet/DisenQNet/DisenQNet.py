# -*- coding: utf-8 -*-

import logging
import torch
from torch import nn
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR

from .modules import TextEncoder, AttnModel, ConceptEstimator, MIEstimator, DisenEstimator
from .utils import MLP, get_mask, get_confuse_matrix, get_f1_score

class QuestionEncoder(nn.Module):
    """
        DisenQNet question representation model
        :param vocab_size: int, size of vocabulary
        :param hidden_dim: int, size of word and question embedding
        :param dropout: float, dropout rate
        :param wv: Tensor of (vocab_size, hidden_dim) or None, initial word embedding, default = None
    """

    def __init__(self, vocab_size, hidden_dim, dropout, wv=None):
        super(QuestionEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = TextEncoder(vocab_size, hidden_dim, dropout, wv=wv)
        self.k_model = AttnModel(hidden_dim, dropout)
        self.i_model = AttnModel(hidden_dim, dropout)
        self.dropout = nn.Dropout(p=dropout)
        return
    
    def forward(self, input, length, get_vk=True, get_vi=True):
        """
            :param input: Tensor of (batch_size, seq_len), word index
            :param length: Tensor of (batch_size), valid sequence length of each batch
            :param get_vk: bool, whether to return vk
            :param get_vi: bool, whether to return vi
            :returns: (embed, k_hidden, i_hidden)
                embed: Tensor of (batch_size, seq_len, hidden_dim), word embedding
                k_hidden: Tensor of (batch_size, hidden_dim) or None, concept representation of question
                i_hidden: Tensor of (batch_size, hidden_dim) or None, individual representation of question
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
        :param vocab_size: int, size of vocabulary
        :param concept_size: int, number of concept classes
        :param hidden_dim: int, size of word and question embedding
        :param dropout: float, dropout rate
        :param pos_weight: float, positive sample weight in unbalanced multi-label concept classifier
        :param w_cp: float, weight of concept loss
        :param w_mi: float, weight of mutual information loss
        :param w_dis: float, weight of disentangling loss
        :param wv: Tensor of (vocab_size, hidden_dim) or None, initial word embedding, default = None
    """

    def __init__(self, vocab_size, concept_size, hidden_dim, dropout, pos_weight, w_cp, w_mi, w_dis, wv=None):
        super(DisenQNet, self).__init__()
        self.disen_q_net = QuestionEncoder(vocab_size, hidden_dim, dropout, wv)
        self.mi_estimator = MIEstimator(hidden_dim, hidden_dim*2, dropout)
        self.concept_estimator = ConceptEstimator(hidden_dim, concept_size, pos_weight, dropout)
        self.disen_estimator = DisenEstimator(hidden_dim, dropout)
        self.w_cp = w_cp
        self.w_mi = w_mi
        self.w_dis = w_dis

        self.modules = (self.disen_q_net, self.mi_estimator, self.concept_estimator, self.disen_estimator)
        return
    
    def train(self, train_data, test_data, device, epoch, lr, step_size, gamma, warm_up, n_adversarial, silent):
        """
            DisenQNet train
            :param train_data: iterable, train dataset, contains text, length, concept
                text: Tensor of (batch_size, seq_len)
                length: Tensor of (batch_size)
                concept: Tensor of (batch_size, class_size)
            :param test_data: iterable, test dataset
            :param device: str, cpu or cuda
            :param epoch: int, number of epoch
            :param lr: float, initial learning rate
            :param step_size: int, step_size for StepLR, period of learning rate decay
            :param gamma: float, gamma for StepLR, multiplicative factor of learning rate decay
            :param warm_up: int, number of epoch for warming up, without adversarial process for dis_loss
            :param n_adversarial: int, ratio of disc/enc training for adversarial process
            :param silent: bool, whether to log loss
        """
        # optimizer & scheduler
        model_params = list(self.disen_q_net.parameters()) + list(self.mi_estimator.parameters()) + list(self.concept_estimator.parameters())
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
            
            self.to(device)
            self.set_mode(True)
            for data in train_data:
                text, length, concept = data
                text, length, concept = text.to(device), length.to(device), concept.to(device)
                # WGAN-like adversarial training: min_enc max_disc dis_loss
                # train disc
                if not warming_up:
                    _, k_hidden, i_hidden = self.disen_q_net(text, length)
                    # stop gradient propagation to encoder
                    k_hidden, i_hidden = k_hidden.detach(), i_hidden.detach()
                    # max dis_loss
                    dis_loss =  - self.disen_estimator(k_hidden, i_hidden)
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
            train_loss = self.eval(train_data, device)
            if test_data is not None and not warming_up:
                test_loss = self.eval(test_data, device)
                if not silent:
                    logging.info(f"[Epoch {epoch_idx:2d}] train loss: {train_loss:.4f}, eval loss: {test_loss:.4f}")
            elif not silent:
                logging.info(f"[Epoch {epoch_idx:2d}] train loss: {train_loss:.4f}")
        return
    
    def eval(self, test_data, device):
        """
            DisenQNet test
            :param test_data: iterable, train dataset, contains text, length, concept
                text: Tensor of (batch_size, seq_len)
                length: Tensor of (batch_size)
                concept: Tensor of (batch_size, class_size)
            :param device: str, cpu or cuda
            :returns: float, average loss for test dataset
        """
        total_size = 0
        total_loss = 0
        self.to(device)
        self.set_mode(False)
        with torch.no_grad():
            for data in test_data:
                text, length, concept = data
                text, length, concept = text.to(device), length.to(device), concept.to(device)
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
    
    def save(self, filepath):
        state_dicts = [module.state_dict() for module in self.modules]
        torch.save(state_dicts, filepath)
        return
    
    def load(self, filepath):
        state_dicts = torch.load(filepath)
        for module, state_dict in zip(self.modules, state_dicts):
            module.load_state_dict(state_dict)
        return
    
    def to(self, device):
        for module in self.modules:
            module.to(device)
        return
    
    def set_mode(self, train):
        for module in self.modules:
            if train:
                module.train()
            else:
                module.eval()
        return

class ConceptModel(object):
    """
        Concept prediction training and evaluation model, with pretrained and fixed DisenQNet
        :param concept_size: int, number of concept classes
        :param disen_q_net: QuestionEncoder, pretrained DisenQNet model
        :param dropout: float, dropout rate
        :param pos_weight: float, positive sample weight in unbalanced multi-label concept classifier
    """

    def __init__(self, concept_size, disen_q_net, dropout, pos_weight):
        super(ConceptModel, self).__init__()
        hidden_dim = disen_q_net.hidden_dim
        self.disen_q_net = disen_q_net
        self.classifier = MLP(hidden_dim, concept_size, hidden_dim, dropout, n_layers=2)
        pos_weight = torch.ones(concept_size) * pos_weight
        self.loss = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight)
        self.modules = (self.disen_q_net, self.classifier, self.loss)
        return
    
    def train(self, train_data, test_data, device, epoch, lr, step_size, gamma, silent, use_vi=False, top_k=2, reduction="micro"):
        """
            Concept model train
            :param train_data: iterable, train dataset, contains text, length, concept
                text: Tensor of (batch_size, seq_len)
                length: Tensor of (batch_size)
                concept: Tensor of (batch_size, class_size)
            :param test_data: iterable, test dataset
            :param device: str, cpu or cuda
            :param epoch: int, number of epoch
            :param lr: float, initial learning rate
            :param step_size: int, step_size for StepLR, period of learning rate decay
            :param gamma: float, gamma for StepLR, multiplicative factor of learning rate decay
            :param silent: bool, whether to log loss
            :param top_k: int, number of top k classes as positive label for multi-label classification
            :param reduction: str, macro or micro, reduction type for F1 score
        """
        # optimizer & scheduler
        optimizer = Adam(self.classifier.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        # train
        for epoch_idx in range(epoch):
            epoch_idx += 1

            self.to(device)
            self.set_mode(True)
            for data in train_data:
                text, length, concept = data
                text, length, concept = text.to(device), length.to(device), concept.to(device)
                if use_vi:
                    _, _, hidden = self.disen_q_net(text, length, get_vk=False)
                else:
                    _, hidden, _ = self.disen_q_net(text, length, get_vi=False)
                # stop gradient propagation to encoder
                hidden = hidden.detach()

                output = self.classifier(hidden)
                loss = self.loss(output, concept.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            # test
            train_loss, train_f1 = self.eval(train_data, device, use_vi, top_k, reduction)
            if test_data is not None:
                test_loss, test_f1 = self.eval(test_data, device, use_vi, top_k, reduction)
                if not silent:
                    logging.info(f"[Epoch {epoch_idx:2d}] train loss: {train_loss:.4f}, f1: {train_f1:.3f}, eval loss: {test_loss:.4f}, f1: {test_f1:.3f}")
            elif not silent:
                logging.info(f"[Epoch {epoch_idx:2d}] train loss: {train_loss:.4f}, f1: {train_f1:.3f}")
        return
    
    def eval(self, test_data, device, use_vi, top_k, reduction):
        """
            Concept model test
            :param test_data: iterable, test dataset, contains text, length, concept
                text: Tensor of (batch_size, seq_len)
                length: Tensor of (batch_size)
                concept: Tensor of (batch_size, class_size)
            :param device: str, cpu or cuda
            :param top_k: int, number of top k classes as positive label for multi-label classification
            :param reduction: str, macro or micro, reduction type for F1 score
            :returns: (loss, f1)
                loss: float, loss for test data
                f1: float, F1 score for multi-label classification for test data
        """
        self.to(device)
        self.set_mode(False)
        total_size = 0
        total_loss = 0
        confuse_matrix = 0
        with torch.no_grad():
            for data in test_data:
                text, length, concept = data
                text, length, concept = text.to(device), length.to(device), concept.to(device)
                if use_vi:
                    _, _, hidden = self.disen_q_net(text, length, get_vk=False)
                else:
                    _, hidden, _ = self.disen_q_net(text, length, get_vi=False)
                output = self.classifier(hidden)
                loss = self.loss(output, concept.float())
                cm = get_confuse_matrix(concept.long(), output.detach(), top_k)
                batch_size = text.size(0)
                total_size += batch_size
                total_loss += loss.item() * batch_size
                confuse_matrix += cm
        loss = total_loss / total_size
        f1 = get_f1_score(confuse_matrix, reduction)
        return loss, f1
    
    def save(self, filepath):
        state_dicts = [module.state_dict() for module in self.modules]
        torch.save(state_dicts, filepath)
        return
    
    def load(self, filepath):
        state_dicts = torch.load(filepath)
        for module, state_dict in zip(self.modules, state_dicts):
            module.load_state_dict(state_dict)
        return
    
    def to(self, device):
        for module in self.modules:
            module.to(device)
        return
    
    def set_mode(self, train):
        self.disen_q_net.eval()
        if train:
            self.classifier.train()
        else:
            self.classifier.eval()
        return
