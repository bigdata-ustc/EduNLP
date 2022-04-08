import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden_dim, dropout, n_layers=2, act=F.leaky_relu):
        super(MLP, self).__init__()
        self.l_in = nn.Linear(in_dim, hidden_dim)
        self.l_hs = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2))  # doctest: +ELLIPSIS
        self.l_out = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.act = act

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
