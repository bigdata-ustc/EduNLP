import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from baize.torch import load_net
import torch.nn.functional as F
import json
import os
from typing import List
from transformers.modeling_outputs import ModelOutput
from transformers import PretrainedConfig
from typing import Optional
from ..base_model import BaseModel
from ..utils import torch_utils as mytorch
from .harnn import HAM

__all__ = ["LM", "ElmoLM", "ElmoLMForPreTraining", "ElmoLMForPropertyPrediction", "ElmoLMForKnowledgePrediction"]


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
                 bidirectional=False, embedding=None, model_params=None, use_pack_pad=True, **kwargs):
        super(LM, self).__init__()
        rnn_type = rnn_type.upper()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim) if embedding is None else embedding
        self.use_pack_pad = use_pack_pad
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
        if self.use_pack_pad:
            seq_or_pack = pack_padded_sequence(seq, seq_len.cpu(), batch_first=True, enforce_sorted=False)
        else:
            seq_or_pack = seq
        h0 = torch.zeros(self.num_layers, seq.shape[0], self.hidden_size).to(seq_idx.device)
        if self.c is True:
            c0 = torch.zeros(self.num_layers, seq.shape[0], self.hidden_size).to(seq_idx.device)
            output, (hn, _) = self.rnn(seq_or_pack, (h0, c0))
        else:
            output, hn = self.rnn(seq_or_pack, h0)
        if self.use_pack_pad:
            output, _ = pad_packed_sequence(output, batch_first=True)

        return output, hn


class ElmoLMOutput(ModelOutput):
    """
    Output type of [`ElmoLM`]

    Parameters
    ----------
    pred_forward: of shape (batch_size, sequence_length)
    pred_backward: of shape (batch_size, sequence_length)
    forward_output: of shape (batch_size, sequence_length, hidden_size)
    backward_output: of shape (batch_size, sequence_length, hidden_size)
    """
    pred_forward: torch.FloatTensor = None
    pred_backward: torch.FloatTensor = None
    forward_output: torch.FloatTensor = None
    backward_output: torch.FloatTensor = None


class ElmoLM(BaseModel):
    base_model_prefix = 'elmo'

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int = 2,
                 dropout_rate: float = 0.5, use_pack_pad=False, **argv):
        super(ElmoLM, self).__init__()
        self.LM_layer = LM("BiLSTM", vocab_size, embedding_dim, hidden_size, num_layers=num_layers,
                           use_pack_pad=use_pack_pad, **argv)
        self.pred_layer = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.config = {k: v for k, v in locals().items() if k != "self" and k != "__class__" and k != "argv"}
        self.config.update(argv)
        self.config['architecture'] = 'ElmoLM'
        self.config = PretrainedConfig.from_dict(self.config)

    def forward(self, seq_idx=None, seq_len=None) -> ModelOutput:
        """
        Parameters
        ----------
        seq_idx:Tensor, of shape (batch_size, sequence_length)
            a list of indices
        seq_len:Tensor, of shape (batch_size)
            length

        Returns
        ----------
        ElmoLMOutput
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
        pred_forward = self.pred_layer(forward_output)
        pred_backward = self.pred_layer(backward_output)

        return ElmoLMOutput(
            pred_forward=pred_forward,
            pred_backward=pred_backward,
            forward_output=forward_output,
            backward_output=backward_output
        )

    @classmethod
    def from_config(cls, config_path, **argv):
        with open(config_path, "r", encoding="utf-8") as rf:
            model_config = json.load(rf)
            model_config.update(argv)
            return cls(
                vocab_size=model_config['vocab_size'],
                embedding_dim=model_config['embedding_dim'],
                hidden_size=model_config['hidden_size'],
                dropout_rate=model_config['dropout_rate'],
                batch_first=model_config['batch_first']
            )


class ElmoLMForPreTrainingOutput(ModelOutput):
    """
    Output type of [`ElmoLMForPreTraining`].

    Parameters
    ----------
    loss:
    pred_forward: of shape (batch_size, sequence_length)
    pred_backward: of shape (batch_size, sequence_length)
    forward_output: of shape (batch_size, sequence_length, hidden_size)
    backward_output: of shape (batch_size, sequence_length, hidden_size)
    """
    loss: torch.FloatTensor = None
    pred_forward: torch.FloatTensor = None
    pred_backward: torch.FloatTensor = None
    forward_output: torch.FloatTensor = None
    backward_output: torch.FloatTensor = None


class ElmoLMForPreTraining(BaseModel):
    base_model_prefix = 'elmo'

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, dropout_rate: float = 0.5,
                 batch_first=True, use_pack_pad=False, **argv):
        super(ElmoLMForPreTraining, self).__init__()
        self.elmo = ElmoLM(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            batch_first=batch_first,
            use_pack_pad=use_pack_pad,
            **argv
        )
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.criterion = nn.CrossEntropyLoss()

        self.config = {k: v for k, v in locals().items() if k != "self" and k != "__class__" and k != "argv"}
        self.config.update(argv)
        self.config['architecture'] = 'ElmoLMForPreTraining'
        self.config = PretrainedConfig.from_dict(self.config)

    def forward(self, seq_idx=None, seq_len=None) -> ModelOutput:
        """

        Parameters
        ----------
        seq_idx:Tensor, of shape (batch_size, sequence_length)
            a list of indices
        seq_len:Tensor, of shape (batch_size)
            length
        pred_mask : Tensor, of shape(batch_size, sequence_length)
        idx_mask : Tensor, of shape (batch_size, sequence_length)

        Returns
        -------
        ElmoLMForPreTrainingOutput
            loss
            pred_forward: of shape (batch_size, sequence_length)
            pred_backward: of shape (batch_size, sequence_length)
            forward_output: of shape (batch_size, sequence_length, hidden_size)
            backward_output: of shape (batch_size, sequence_length, hidden_size)
        """
        batch_size, idx_len = seq_idx.shape
        max_len = seq_len.max().item() if self.config.use_pack_pad is True else idx_len
        # Note:
        # pred_mask matters when LM use pack_pad,
        # but it will break down for parallel GPU because of different seq_len between gpus
        pred_mask = torch.arange(max_len, device=seq_idx.device)[None, :] < seq_len[:, None]
        idx_mask = torch.arange(idx_len, device=seq_idx.device)[None, :] < seq_len[:, None]

        pred_forward_mask = pred_mask.clone()
        pred_forward_mask[torch.arange(batch_size).unsqueeze(1), seq_len.unsqueeze(1) - 1] = False
        pred_backward_mask = pred_mask.clone()
        pred_backward_mask[torch.arange(batch_size).unsqueeze(1), 0] = False

        idx_forward_mask = idx_mask.clone()
        idx_forward_mask[torch.arange(batch_size).unsqueeze(1), 0] = False
        idx_backward_mask = idx_mask.clone()
        idx_backward_mask[torch.arange(batch_size).unsqueeze(1), seq_len.unsqueeze(1) - 1] = False

        outputs = self.elmo(seq_idx, seq_len)
        pred_forward, pred_backward = outputs.pred_forward, outputs.pred_backward

        flat_pred_forward = pred_forward[pred_forward_mask]
        flat_pred_backward = pred_backward[pred_backward_mask]

        # _, flat_pred_idx_forward = torch.max(flat_pred_forward, dim=1)
        # _, flat_pred_idx_backward = torch.max(flat_pred_backward, dim=1)

        flat_y_backward = seq_idx[idx_backward_mask]
        flat_y_forward = seq_idx[idx_forward_mask]

        # diff_forward = torch.sum(flat_pred_idx_forward - flat_y_forward)
        # diff_backward = torch.sum(flat_pred_idx_backward - flat_y_backward)

        forward_loss = self.criterion(flat_pred_forward, flat_y_forward)
        backward_loss = self.criterion(flat_pred_backward, flat_y_backward)
        loss = forward_loss + backward_loss
        return ElmoLMForPreTrainingOutput(
            loss=loss,
            pred_forward=pred_forward,
            pred_backward=pred_backward,
            forward_output=outputs.forward_output,
            backward_output=outputs.backward_output
        )

    @classmethod
    def from_config(cls, config_path, **argv):
        with open(config_path, "r", encoding="utf-8") as rf:
            model_config = json.load(rf)
            model_config.update(argv)
            return cls(
                vocab_size=model_config['vocab_size'],
                embedding_dim=model_config['embedding_dim'],
                hidden_size=model_config['hidden_size'],
                dropout_rate=model_config['dropout_rate'],
                batch_first=model_config['batch_first']
            )


class PropertyPredictionOutput(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None


class ElmoLMForPropertyPrediction(BaseModel):
    base_model_prefix = 'elmo'

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, dropout_rate: float = 0.5,
                 batch_first=True, head_dropout=0.5, **argv):
        super(ElmoLMForPropertyPrediction, self).__init__()

        self.elmo = ElmoLM(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            batch_first=batch_first
        )
        self.head_dropout = head_dropout
        self.dropout = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(2 * hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.MSELoss()

        self.config = {k: v for k, v in locals().items() if k != "self" and k != "__class__" and k != "argv"}
        self.config.update(argv)
        self.config['architecture'] = 'ElmoLMForPreTraining'
        self.config = PretrainedConfig.from_dict(self.config)

    def forward(self, seq_idx=None, seq_len=None, labels=None) -> ModelOutput:
        outputs = self.elmo(seq_idx, seq_len)
        item_embeds = torch.cat(
            (outputs.forward_output[torch.arange(len(seq_len)), torch.tensor(seq_len) - 1],
             outputs.backward_output[torch.arange(len(seq_len)), 0]),
            dim=-1)
        item_embeds = self.dropout(item_embeds)

        logits = self.sigmoid(self.classifier(item_embeds))
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
        return PropertyPredictionOutput(
            loss=loss,
            logits=logits
        )

    @classmethod
    def from_config(cls, config_path, **argv):
        with open(config_path, "r", encoding="utf-8") as rf:
            model_config = json.load(rf)
            model_config.update(argv)
            return cls(
                vocab_size=model_config.get('vocab_size'),
                embedding_dim=model_config.get('embedding_dim'),
                hidden_size=model_config.get('hidden_size'),
                dropout_rate=model_config.get('dropout_rate'),
                batch_first=model_config.get('batch_first'),
                head_dropout=model_config.get('head_dropout', 0.5),
            )


class KnowledgePredictionOutput(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None


class ElmoLMForKnowledgePrediction(BaseModel):
    base_model_prefix = 'elmo'

    def __init__(self, vocab_size: int,
                 embedding_dim: int,
                 hidden_size: int,
                 num_classes_list: List[int],
                 num_total_classes: int,
                 dropout_rate: float = 0.5,
                 batch_first=True,
                 head_dropout: Optional[float] = 0.5,
                 flat_cls_weight: Optional[float] = 0.5,
                 attention_unit_size: Optional[int] = 256,
                 fc_hidden_size: Optional[int] = 512,
                 beta: Optional[float] = 0.5,
                 **argv):
        super(ElmoLMForKnowledgePrediction, self).__init__()

        self.elmo = ElmoLM(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            batch_first=batch_first
        )
        self.head_dropout = head_dropout
        self.dropout = nn.Dropout(head_dropout)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.MSELoss()
        self.flat_classifier = nn.Linear(in_features=2 * hidden_size, out_features=num_total_classes)
        self.ham_classifier = HAM(
            num_classes_list=num_classes_list,
            num_total_classes=num_total_classes,
            sequence_model_hidden_size=hidden_size * 2,
            attention_unit_size=attention_unit_size,
            fc_hidden_size=fc_hidden_size,
            beta=beta,
            dropout_rate=dropout_rate
        )
        self.flat_cls_weight = flat_cls_weight
        self.num_classes_list = num_classes_list
        self.num_total_classes = num_total_classes

        self.config = {k: v for k, v in locals().items() if k != "self" and k != "__class__" and k != "argv"}
        self.config.update(argv)
        self.config['architecture'] = 'ElmoLMForPreTraining'
        self.config = PretrainedConfig.from_dict(self.config)

    def forward(self, seq_idx=None, seq_len=None, labels=None) -> ModelOutput:
        outputs = self.elmo(seq_idx, seq_len)
        item_embeds = torch.cat(
            (outputs.forward_output[torch.arange(len(seq_len)), torch.tensor(seq_len) - 1],
             outputs.backward_output[torch.arange(len(seq_len)), 0]),
            dim=-1)
        tokens_embeds = torch.cat((outputs.forward_output, outputs.backward_output), dim=-1)
        item_embeds = self.dropout(item_embeds)
        tokens_embeds = self.dropout(tokens_embeds)
        flat_logits = self.sigmoid(self.flat_classifier(item_embeds))
        ham_outputs = self.ham_classifier(tokens_embeds)
        ham_logits = self.sigmoid(ham_outputs.scores)
        logits = self.flat_cls_weight * flat_logits + (1 - self.flat_cls_weight) * ham_logits
        loss = None
        if labels is not None:
            labels = torch.sum(torch.nn.functional.one_hot(labels, num_classes=self.num_total_classes), dim=1)
            labels = labels.float()
            loss = self.criterion(logits, labels)
        return KnowledgePredictionOutput(
            loss=loss,
            logits=logits
        )

    @classmethod
    def from_config(cls, config_path, **argv):
        with open(config_path, "r", encoding="utf-8") as rf:
            model_config = json.load(rf)
            model_config.update(argv)
            return cls(
                vocab_size=model_config.get('vocab_size'),
                embedding_dim=model_config.get('embedding_dim'),
                hidden_size=model_config.get('hidden_size'),
                num_total_classes=model_config.get('num_total_classes'),
                num_classes_list=model_config.get('num_classes_list'),
                dropout_rate=model_config.get('dropout_rate'),
                batch_first=model_config.get('batch_first'),
                head_dropout=model_config.get('head_dropout', 0.5),
                flat_cls_weight=model_config.get('flat_cls_weight', 0.5),
                attention_unit_size=model_config.get('attention_unit_size', 256),
                fc_hidden_size=model_config.get('fc_hidden_size', 512),
                beta=model_config.get('beta', 0.5),
            )
