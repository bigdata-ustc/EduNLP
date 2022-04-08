# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F


def get_mask(seq_len, lengths):
    device = lengths.device
    # batch_size
    batch_size = lengths.size(0)
    # seq_len
    pos_index = torch.arange(seq_len).to(device)
    # batch_size * seq_len
    mask = pos_index.unsqueeze(0).expand(batch_size, -1) >= lengths.unsqueeze(-1)
    return mask


def shuffle(real):
    # |0 1 2 3| => |1 2 3 0|
    device = real.device
    batch_size = real.size(0)
    shuffled_index = (torch.arange(batch_size) + 1) % batch_size
    shuffled_index = shuffled_index.to(device)
    shuffled = real.index_select(dim=0, index=shuffled_index)
    return shuffled


def spectral_norm(W, n_iteration=5):
    device = W.device
    # (o, i)
    # bias: (O) -> (o, 1)
    if W.dim() == 1:
        W = W.unsqueeze(-1)
    out_dim, in_dim = W.size()
    # (i, o)
    Wt = W.transpose(0, 1)
    # (1, i)
    u = torch.ones(1, in_dim).to(device)
    for _ in range(n_iteration):
        # (1, i) * (i, o) -> (1, o)
        v = torch.mm(u, Wt)
        v = v / v.norm(p=2)
        # (1, o) * (o, i) -> (1, i)
        u = torch.mm(v, W)
        u = u / u.norm(p=2)
    # (1, i) * (i, o) * (o, 1) -> (1, 1)
    sn = torch.mm(torch.mm(u, Wt), v.transpose(0, 1)).sum() ** 0.5
    return sn
