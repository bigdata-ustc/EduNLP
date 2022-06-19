import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from baize.torch import load_net
import torch.nn.functional as F
import json
import os
from ..base_model import BaseModel
from transformers.modeling_outputs import ModelOutput
from transformers import PretrainedConfig


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


class ElmoLMOutput(ModelOutput):
    pred_forward: torch.FloatTensor = None
    pred_backward: torch.FloatTensor = None
    forward_output: torch.FloatTensor = None
    backward_output: torch.FloatTensor = None


class ElmoLM(BaseModel):
    base_model_prefix = 'elmo'
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, dropout_rate: float = 0.5,
                 batch_first=True):
        super(ElmoLM, self).__init__()
        self.LM_layer = LM("BiLSTM", vocab_size, embedding_dim, hidden_size, num_layers=2, batch_first=batch_first)
        self.pred_layer = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        config = {k: v for k, v in locals().items() if k != "self" and k != "__class__"}
        config['architecture'] = 'ElmoLM'
        self.config = PretrainedConfig.from_dict(config)

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

        return ElmoLMOutput(
            pred_forward=pred_forward,
            pred_backward=pred_backward,
            forward_output=forward_output,
            backward_output=backward_output
        )

    @classmethod
    def from_config(cls, config_path):
        with open(config_path, "r", encoding="utf-8") as rf:
            model_config = json.load(rf)
            return cls(
                vocab_size=model_config['vocab_size'],
                embedding_dim=model_config['embedding_dim'],
                hidden_size=model_config['hidden_size'],
                dropout_rate=model_config['dropout_rate'],
                batch_first=model_config['batch_first']
            )


class ElmoLMForPreTrainingOutput(ModelOutput):
    loss: torch.FloatTensor = None
    pred_forward: torch.FloatTensor = None
    pred_backward: torch.FloatTensor = None
    forward_output: torch.FloatTensor = None
    backward_output: torch.FloatTensor = None


class ElmoLMForPreTraining(BaseModel):
    base_model_prefix = 'elmo'
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, dropout_rate: float = 0.5,
                 batch_first=True):
        super(ElmoLMForPreTraining, self).__init__()
        self.elmo = ElmoLM(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            batch_first=batch_first
        )
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        config = {k: v for k, v in locals().items() if k != "self" and k != "__class__"}
        config['architecture'] = 'ElmoLMForPreTraining'
        self.config = PretrainedConfig.from_dict(config)

    def forward(self, seq_idx, seq_len, pred_mask, idx_mask):
        y = F.one_hot(seq_idx, self.elmo.vocab_size).to(seq_idx.device)
        outputs = self.elmo(seq_idx, seq_len)
        pred_forward, pred_backward = outputs.pred_forward, outputs.pred_backward

        loss_func = nn.BCELoss()
        pred_forward = pred_forward[pred_mask]
        pred_backward = pred_backward[torch.flip(pred_mask, [1])]
        y_rev = torch.flip(y, [1])[torch.flip(idx_mask, [1])]
        y = y[idx_mask]
        forward_loss = loss_func(pred_forward[:, :-1].double(), y[:, 1:].double())
        backward_loss = loss_func(pred_backward[:, :-1].double(), y_rev[:, 1:].double())
        loss = forward_loss + backward_loss

        return ElmoLMForPreTrainingOutput(
            loss=loss,
            pred_forward=pred_forward,
            pred_backward=pred_backward,
            forward_output=outputs.forward_output,
            backward_output=outputs.backward_output
        )

    @classmethod
    def from_config(cls, config_path):
        with open(config_path, "r", encoding="utf-8") as rf:
            model_config = json.load(rf)
            return cls(
                vocab_size=model_config['vocab_size'],
                embedding_dim=model_config['embedding_dim'],
                hidden_size=model_config['hidden_size'],
                dropout_rate=model_config['dropout_rate'],
                batch_first=model_config['batch_first']
            )
