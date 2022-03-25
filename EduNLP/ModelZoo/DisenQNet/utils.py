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


class MLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden_dim, dropout, n_layers=2, act=F.leaky_relu):
        super(MLP, self).__init__()
        self.l_in = nn.Linear(in_dim, hidden_dim)
        self.l_hs = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2))
        self.l_out = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.act = act
        return

    def forward(self, input):
        hidden = self.act(self.l_in(self.dropout(input)))
        for l_h in self.l_hs:
            hidden = self.act(l_h(self.dropout(hidden)))
        output = self.l_out(self.dropout(hidden))
        return output


class TextCNN(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(TextCNN, self).__init__()
        kernel_sizes = [2, 3, 4, 5]
        channel_dim = hidden_dim // len(kernel_sizes)
        self.min_seq_len = max(kernel_sizes)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, channel_dim, k_size) for k_size in kernel_sizes])
        return

    def forward(self, embed):
        if embed.size(1) < self.min_seq_len:
            device = embed.device
            pad = torch.zeros(embed.size(0), self.min_seq_len - embed.size(1), embed.size(-1)).to(device)
            embed = torch.cat((embed, pad), dim=1)
        # (b, s, d) => (b, d, s) => (b, d', s') => (b, d', 1) => (b, d')
        # batch_size * dim * seq_len
        hidden = [F.leaky_relu(conv(embed.transpose(1, 2))) for conv in self.convs]
        # batch_size * dim
        hidden = [F.max_pool1d(h, kernel_size=h.size(2)).squeeze(-1) for h in hidden]
        hidden = torch.cat(hidden, dim=-1)
        return hidden


class Disc(nn.Module):
    def __init__(self, x_dim, y_dim, dropout):
        super(Disc, self).__init__()
        self.disc = MLP(x_dim + y_dim, 1, y_dim, dropout, n_layers=2)
        return

    def forward(self, x, y):
        input = torch.cat((x, y), dim=-1)
        # (b, 1) -> (b)
        score = self.disc(input).squeeze(-1)
        return score


class MI(nn.Module):
    def __init__(self, x_dim, y_dim, dropout):
        super(MI, self).__init__()
        self.disc = Disc(x_dim, y_dim, dropout)
        return

    def forward(self, x, y):
        # P(X,Y) = (x, y), P(X)P(Y) = (x, sy)
        sy = shuffle(y)
        mi = -F.softplus(-self.disc(x, y)).mean() - F.softplus(self.disc(x, sy)).mean()
        return mi
