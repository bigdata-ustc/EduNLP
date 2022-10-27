import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from baize.torch import load_net
import torch.nn.functional as F
import json
import os
from ..base_model import BaseModel
from transformers.modeling_outputs import ModelOutput
from transformers import BertModel
from typing import List, Optional
from ..rnn.harnn import HAM

__all__ = ["BertForPropertyPrediction", "BertForKnowledgePrediction"]


class BertForPPOutput(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None


class BertForPropertyPrediction(BaseModel):
    def __init__(self, pretrained_model_dir=None, head_dropout=0.5):
        super(BertForPropertyPrediction, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_dir)
        self.hidden_size = self.bert.config.hidden_size
        self.head_dropout = head_dropout
        self.dropout = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.MSELoss()

        self.config = {k: v for k, v in locals().items() if k not in ["self", "__class__"]}
        self.config['architecture'] = 'BertForPropertyPrediction'

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        item_embeds = outputs.last_hidden_state[:, 0, :]
        item_embeds = self.dropout(item_embeds)

        logits = self.sigmoid(self.classifier(item_embeds)).squeeze(1)
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels) if labels is not None else None
        return BertForPPOutput(
            loss=loss,
            logits=logits,
        )

    @classmethod
    def from_config(cls, config_path, **argv):
        with open(config_path, "r", encoding="utf-8") as rf:
            model_config = json.load(rf)
            model_config.update(argv)
            return cls(
                pretrained_model_dir=model_config['pretrained_model_dir'],
                head_dropout=model_config.get("head_dropout", 0.5)
            )

    # @classmethod
    # def from_pretrained(cls):
    #     NotImplementedError
    #     # 需要验证是否和huggingface的模型兼容


class BertForKnowledgePrediction(BaseModel):
    def __init__(self,
                 num_classes_list: List[int] = None,
                 num_total_classes: int = None,
                 pretrained_model_dir=None,
                 head_dropout=0.5,
                 flat_cls_weight=0.5,
                 attention_unit_size=256,
                 fc_hidden_size=512,
                 beta=0.5,
                 ):
        super(BertForKnowledgePrediction, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_dir)
        self.hidden_size = self.bert.config.hidden_size
        self.head_dropout = head_dropout
        self.dropout = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.MSELoss()
        self.flat_classifier = nn.Linear(self.hidden_size, num_total_classes)
        self.ham_classifier = HAM(
            num_classes_list=num_classes_list,
            num_total_classes=num_total_classes,
            sequence_model_hidden_size=self.bert.config.hidden_size,
            attention_unit_size=attention_unit_size,
            fc_hidden_size=fc_hidden_size,
            beta=beta,
            dropout_rate=head_dropout
        )
        self.flat_cls_weight = flat_cls_weight
        self.num_classes_list = num_classes_list
        self.num_total_classes = num_total_classes

        self.config = {k: v for k, v in locals().items() if k not in ["self", "__class__"]}
        self.config['architecture'] = 'BertForKnowledgePrediction'

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        item_embeds = outputs.last_hidden_state[:, 0, :]
        item_embeds = self.dropout(item_embeds)
        tokens_embeds = outputs.last_hidden_state
        tokens_embeds = self.dropout(tokens_embeds)
        flat_logits = self.sigmoid(self.flat_classifier(item_embeds))
        ham_outputs = self.ham_classifier(tokens_embeds)
        ham_logits = self.sigmoid(ham_outputs.scores)
        logits = self.flat_cls_weight * flat_logits + (1 - self.flat_cls_weight) * ham_logits
        loss = None
        if labels is not None:
            labels = torch.sum(torch.nn.functional.one_hot(labels, num_classes=self.num_total_classes), dim=1)
            labels = labels.float()
            loss = self.criterion(logits, labels) if labels is not None else None
        return BertForPPOutput(
            loss=loss,
            logits=logits,
        )

    @classmethod
    def from_config(cls, config_path, **argv):
        with open(config_path, "r", encoding="utf-8") as rf:
            model_config = json.load(rf)
            model_config.update(argv)
            return cls(
                pretrained_model_dir=model_config['pretrained_model_dir'],
                head_dropout=model_config.get("head_dropout", 0.5),
                num_classes_list=model_config.get('num_classes_list'),
                num_total_classes=model_config.get('num_total_classes'),
                flat_cls_weight=model_config.get('flat_cls_weight', 0.5),
                attention_unit_size=model_config.get('attention_unit_size', 256),
                fc_hidden_size=model_config.get('fc_hidden_size', 512),
                beta=model_config.get('beta', 0.5),
            )

    # @classmethod
    # def from_pretrained(cls):
    #     NotImplementedError
    #     # 需要验证是否和huggingface的模型兼容
