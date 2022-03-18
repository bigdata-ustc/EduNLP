# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F

def get_mask(seq_len, lengths):
    """
        Get pad mask
        :param seq_len: int, padded length of data tensor
        :param lengths: Tensor of long (batch_size), valid length of each data tensor
        :returns: Tensor of bool (batch_size, seq_len), False means data, and True means pad
            [F F F T T T]
    """
    device = lengths.device
    # batch_size
    batch_size = lengths.size(0)
    # seq_len
    pos_index = torch.arange(seq_len).to(device)
    # batch_size * seq_len
    mask = pos_index.unsqueeze(0).expand(batch_size, -1) >= lengths.unsqueeze(-1)
    return mask

def shuffle(real):
    """
        shuffle data in a batch
        [1, 2, 3, 4, 5] -> [2, 3, 4, 5, 1]
        P(X,Y) -> P(X)P(Y) by shuffle Y in a batch
        P(X,Y) = [(1,1'),(2,2'),(3,3')] -> P(X)P(Y) = [(1,2'),(2,3'),(3,1')]
        :param real: Tensor of (batch_size, ...), data, batch_size > 1
        :returns: Tensor of (batch_size, ...), shuffled data
    """
    # |0 1 2 3| => |1 2 3 0|
    device = real.device
    batch_size = real.size(0)
    shuffled_index = (torch.arange(batch_size) + 1) % batch_size
    shuffled_index = shuffled_index.to(device)
    shuffled = real.index_select(dim=0, index=shuffled_index)
    return shuffled

def spectral_norm(W, n_iteration=5):
    """
        Spectral normalization for Lipschitz constrain in Disc of WGAN
        Following https://blog.csdn.net/qq_16568205/article/details/99586056
        |W|^2 = principal eigenvalue of W^TW through power iteration
        v = W^Tu/|W^Tu|
        u = Wv / |Wv|
        |W|^2 = u^TWv
        
        :param w: Tensor of (out_dim, in_dim) or (out_dim), weight matrix of NN
        :param n_iteration: int, number of iterations for iterative calculation of spectral normalization:
        :returns: Tensor of (), spectral normalization of weight matrix
    """
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

def get_topk_class(probability, top_k):
    """
        select top k class according probability for multi-label classification
        :param probability: Tensor of (batch_size, class_size), probability
        :param topk: int, number of selected classes
        :returns: Tensor of long (batch_size, class_size), whether class is selected, 1 means class selected, 0 means class unselected
    """
    # batch_size * class_size
    device = probability.device
    batch_size, class_size = probability.size()
    # batch_size * k
    label_index = probability.topk(top_k, dim=-1, sorted=False)[1]
    batch_index = torch.arange(batch_size).unsqueeze(-1).expand(-1, top_k)
    # batch_size * class_size
    label = torch.zeros(batch_size, class_size).long()
    batch_index = batch_index.to(device)
    label = label.to(device)
    label[batch_index, label_index] = 1
    return label

def get_confuse_matrix(label, probability, top_k):
    """
        Confuse matrix for multi-label classification by true label and prediction according top k probability
        :param label: Tensor of (batch_size, class_size), multi-label classification label, 1 means True, 0 means False
        :param probability: Tensor of (batch_size, class_size), predicted probability
        :param top_k: int, number of top k classes as positive label for multi-label classification
        :returns: Tensor of long (3, class_size), (TP, FP, FN) count for each class
    """
    prediction = get_topk_class(probability, top_k).bool()
    label = label.bool()
    # batch_size * class_size -> class_size
    tp = (label & prediction).sum(dim=0)
    fp = ((~label) & prediction).sum(dim=0)
    fn = (label & (~prediction)).sum(dim=0)
    # 3 * class_size
    cm = torch.stack((tp, fp, fn), dim=0)
    return cm

def get_f1_score(confuse_matrix, reduction="micro"):
    """
        F1 score for multi-label classification
        Follow https://www.cnblogs.com/fledlingbird/p/10675922.html
        :param confuse_matrix: Tensor of long (3, class_size), (TP, FP, FN) count for each class
        :param reduction: str, macro or micro, reduction type for F1 score
        :returns: float, f1 score for multi-label classification
    """
    # 3 * class_size -> class_size
    tp, fp, fn = confuse_matrix
    if reduction == "macro":
        precise = tp.float() / (tp + fp).float()
        recall = tp.float() / (tp + fn).float()
        f1 = (2 * precise * recall) / (precise + recall)
        f1[tp == 0] = 0
        f1 = f1.mean().item()
    elif reduction == "micro":
        tp, fp, fn = tp.sum(), fp.sum(), fn.sum()
        precise = tp.float() / (tp + fp).float()
        recall = tp.float() / (tp + fn).float()
        f1 = ((2 * precise * recall) / (precise + recall)).item()
        if tp.item() == 0:
            f1 = 0
    return f1

class MLP(nn.Module):
    """
        Multi-Layer Perceptron
        :param in_dim: int, size of input feature
        :param n_classes: int, number of output classes
        :param hidden_dim: int, size of hidden vector
        :param dropout: float, dropout rate
        :param n_layers: int, number of layers, at least 2, default = 2
        :param act: function, activation function, default = leaky_relu
    """

    def __init__(self, in_dim, n_classes, hidden_dim, dropout, n_layers=2, act=F.leaky_relu):
        super(MLP, self).__init__()
        self.l_in = nn.Linear(in_dim, hidden_dim)
        self.l_hs = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2))
        self.l_out = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.act = act
        return
    
    def forward(self, input):
        """
            :param input: Tensor of (batch_size, in_dim), input feature
            :returns: Tensor of (batch_size, n_classes), output class
        """
        hidden = self.act(self.l_in(self.dropout(input)))
        for l_h in self.l_hs:
            hidden = self.act(l_h(self.dropout(hidden)))
        output = self.l_out(self.dropout(hidden))
        return output

class TextCNN(nn.Module):
    """
        TextCNN
        :param embed_dim: int, size of word embedding
        :param hidden_dim: int, size of question embedding
    """

    def __init__(self, embed_dim, hidden_dim):
        super(TextCNN, self).__init__()
        kernel_sizes = [2, 3, 4, 5]
        channel_dim = hidden_dim // len(kernel_sizes)
        self.min_seq_len = max(kernel_sizes)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, channel_dim, k_size) for k_size in kernel_sizes])
        return
    
    def forward(self, embed):
        """
            :param embed: Tensor of (batch_size, seq_len, embed_dim), word embedding
            :returns: Tensor of (batch_size, hidden_dim), question embedding
        """
        if embed.size(1) < self.min_seq_len:
            device = embed.device
            pad = torch.zeros(embed.size(0), self.min_seq_len-embed.size(1), embed.size(-1)).to(device)
            embed = torch.cat((embed, pad), dim=1)
        # (b, s, d) => (b, d, s) => (b, d', s') => (b, d', 1) => (b, d')
        # batch_size * dim * seq_len
        hidden = [F.leaky_relu(conv(embed.transpose(1, 2))) for conv in self.convs]
        # batch_size * dim
        hidden = [F.max_pool1d(h, kernel_size=h.size(2)).squeeze(-1) for h in hidden]
        hidden = torch.cat(hidden, dim=-1)
        return hidden

class Disc(nn.Module):
    """
        2-layer discriminator for MI estimator
        :param x_dim: int, size of x vector
        :param y_dim: int, size of y vector
        :param dropout: float, dropout rate
    """
    def __init__(self, x_dim, y_dim, dropout):
        super(Disc, self).__init__()
        self.disc = MLP(x_dim+y_dim, 1, y_dim, dropout, n_layers=2)
        return
    
    def forward(self, x, y):
        """
            :param x: Tensor of (batch_size, hidden_dim), x
            :param y: Tensor of (batch_size, hidden_dim), y
            :returns: Tensor of (batch_size), score
        """
        input = torch.cat((x, y), dim=-1)
        # (b, 1) -> (b)
        score = self.disc(input).squeeze(-1)
        return score

class MI(nn.Module): 
    """
        MI JS estimator
        MI(X,Y) = E_pxy[-sp(-T(x,y))] - E_pxpy[sp(T(x,y))]
        
        :param x_dim: int, size of x vector
        :param y_dim: int, size of y vector
        :param dropout: float, dropout rate of discrimanator
    """

    def __init__(self, x_dim, y_dim, dropout):
        super(MI, self).__init__()
        self.disc = Disc(x_dim, y_dim, dropout)
        return
    
    def forward(self, x, y):
        """
            :param x: Tensor of (batch_size, hidden_dim), x
            :param y: Tensor of (batch_size, hidden_dim), y
            :returns: Tensor of (), MI
        """
        # P(X,Y) = (x, y), P(X)P(Y) = (x, sy)
        sy = shuffle(y)
        mi = -F.softplus(-self.disc(x, y)).mean() - F.softplus(self.disc(x, sy)).mean()
        return mi
