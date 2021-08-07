# coding: utf-8
# 2021/7/12 @ tongshiwei

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from baize.torch import load_net


class LM(nn.Module):
    """
    Examples
    --------
    >>> import torch
    >>> seq_idx = torch.LongTensor([[1, 2, 3], [1, 2, 0], [3, 0, 0]])
    >>> seq_len = torch.LongTensor([3, 2, 1])
    >>> lm = LM("RNN", 4, 3, 2)
    >>> output, hn = lm(seq_idx, seq_len)
    >>> output.shape
    torch.Size([3, 3, 2])
    >>> hn.shape
    torch.Size([1, 3, 2])
    >>> lm = LM("RNN", 4, 3, 2, num_layers=2)
    >>> output, hn = lm(seq_idx, seq_len)
    >>> output.shape
    torch.Size([3, 3, 2])
    >>> hn.shape
    torch.Size([2, 3, 2])
    """

    def __init__(self, rnn_type: str, vocab_size: int, embedding_dim: int, hidden_size: int, num_layers=1,
                 bidirectional=False, embedding=None, model_params=None, **kwargs):
        super(LM, self).__init__()
        rnn_type = rnn_type.upper()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim) if embedding is None else embedding
        self.c = False
        if rnn_type == "RNN":
            self.rnn = torch.nn.RNN(
                embedding_dim, hidden_size, num_layers, bidirectional=bidirectional, **kwargs
            )
        elif rnn_type == "LSTM":
            self.rnn = torch.nn.LSTM(
                embedding_dim, hidden_size, num_layers, bidirectional=bidirectional, **kwargs
            )
            self.c = True
        elif rnn_type == "GRU":
            self.rnn = torch.nn.GRU(
                embedding_dim, hidden_size, num_layers, bidirectional=bidirectional, **kwargs
            )
        elif rnn_type == "ELMO":
            bidirectional = True
            self.rnn = torch.nn.LSTM(
                embedding_dim, hidden_size, num_layers, bidirectional=bidirectional, **kwargs
            )
            self.c = True
        else:
            raise TypeError("Unknown rnn_type %s" % rnn_type)

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        if bidirectional is True:
            self.num_layers *= 2
        self.hidden_size = hidden_size

        if model_params:
            load_net(model_params, self, allow_missing=True)

    def forward(self, seq_idx, seq_len):
        seq = self.embedding(seq_idx)
        pack = pack_padded_sequence(seq, seq_len, batch_first=True)
        h0 = torch.zeros(self.num_layers, seq.shape[0], self.hidden_size)
        if self.c is True:
            c0 = torch.zeros(self.num_layers, seq.shape[0], self.hidden_size)
            output, (hn, _) = self.rnn(pack, (h0, c0))
        else:
            output, hn = self.rnn(pack, h0)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return output, hn
