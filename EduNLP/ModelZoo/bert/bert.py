import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from baize.torch import load_net
import torch.nn.functional as F
import json
import os
from ..base_model import BaseModel
from transformers.modeling_outputs import ModelOutput
from transformers import PretrainedConfig, BertModel


class BertForPPOutput(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None


class BertForPropertyPrediction(BaseModel):
    def __init__(self, pretrained_model_dir=None, head_dropout=0.5):

        super(BertForPropertyPrediction, self).__init__()

        self.bert = BertModel.from_pretrained(pretrained_model_dir)
        
        hidden_size = self.bert.config.hidden_size

        self.head_dropout = head_dropout
        self.dropout = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss()

        config = {k: v for k, v in locals().items() if k not in ["self", "__class__", "pretrained_model_dir"]}
        config['architecture'] = 'BertForPropertyPrediction'
        self.config = PretrainedConfig.from_dict(config)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        item_embeds = outputs.last_hidden_state[:, 0, :]
        item_embeds = self.dropout(item_embeds)

        logits = self.sigmoid(self.classifier(item_embeds)).squeeze(1)
        loss = self.loss(logits, labels) if labels is not None else None
        return BertForPPOutput(
            loss = loss,
            logits = logits,
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

    @classmethod
    def from_pretrained(cls):
        NotImplementedError
        # 需要验证是否和huggingface的模型兼容
