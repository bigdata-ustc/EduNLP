# coding: utf-8
# 2021/7/12 @ tongshiwei

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from baize.torch import load_net
import torch.nn.functional as F
import json


class LM(nn.Module):
    """

    Parameters
    ----------
    rnn_type：str
        Legal types including RNN, LSTM, GRU, BiLSTM
    vocab_size: int
    embedding_dim: int
    hidden_size: int
    num_layers
    bidirectional
    embedding
    model_params
    kwargs

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
        elif rnn_type == "BILSTM":
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
        """

        Parameters
        ----------
        seq_idx:Tensor
            a list of indices
        seq_len:Tensor
            length
        Returns
        --------
        sequence
            a PackedSequence object
        """
        seq = self.embedding(seq_idx)
        pack = pack_padded_sequence(seq, seq_len.cpu(), batch_first=True, enforce_sorted=False)
        h0 = torch.zeros(self.num_layers, seq.shape[0], self.hidden_size)
        if self.c is True:
            c0 = torch.zeros(self.num_layers, seq.shape[0], self.hidden_size).to(seq_idx.device)
            output, (hn, _) = self.rnn(pack, (h0, c0))
        else:
            output, hn = self.rnn(pack, h0)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return output, hn


class ElmoLM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, dropout_rate: float = 0.5,
                 batch_first=True):
        super(ElmoLM, self).__init__()
        self.LM_layer = LM("BiLSTM", vocab_size, embedding_dim, hidden_size, num_layers=2, batch_first=batch_first)
        self.pred_layer = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, seq_idx, seq_len):
        """
        Parameters
        ----------
        seq_idx:Tensor, of shape (batch_size, sequence_length)
            a list of indices
        seq_len:Tensor, of shape (batch_size)
            length

        Returns
        ----------
        pred_forward: of shape (batch_size, sequence_length)
        pred_backward: of shape (batch_size, sequence_length)
        forward_output: of shape (batch_size, sequence_length, hidden_size)
        backward_output: of shape (batch_size, sequence_length, hidden_size)
        """
        lm_output, _ = self.LM_layer(seq_idx, seq_len)
        forward_output = lm_output[:, :, :self.hidden_size]
        backward_output = lm_output[:, :, self.hidden_size:]
        forward_output = self.dropout(forward_output)
        backward_output = self.dropout(backward_output)
        pred_forward = F.softmax(input=self.pred_layer(forward_output), dim=-1)
        pred_backward = F.softmax(input=self.pred_layer(backward_output), dim=-1)

        return pred_forward, pred_backward, forward_output, backward_output
