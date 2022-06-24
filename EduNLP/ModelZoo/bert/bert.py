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
    def __init__(self, pretrained_model_dir=None, classifier_dropout=0.5):

        super(BertForPropertyPrediction, self).__init__()

        self.bert = BertModel.from_pretrained(pretrained_model_dir)
        
        hidden_size = self.bert.config.hidden_size
        # print(hidden_size)
        self.classifier_dropout = classifier_dropout
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
        config = {k: v for k, v in locals().items() if k != "self" and k != "__class__"}
        config['architecture'] = 'BertForPropertyPrediction'
        self.config = PretrainedConfig.from_dict(config)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None):
                # 固定使用labels, Trainer会对其做一些黑箱操作
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # print("[DEBUG]: outputs ", outputs)
        item_embed = outputs.last_hidden_state[:, 0, :]

        logits = self.sigmoid(self.classifier(item_embed)).squeeze(1)

        loss = F.mse_loss(logits, labels)

        return BertForPPOutput(
            loss = loss,
            logits = logits,
            # labels = labels # 多余的返回值，除loss外，会以元组合并在一起
        )
    
    @classmethod
    def from_config(cls, config_path):
         with open(config_path, "r", encoding="utf-8") as rf:
            model_config = json.load(rf)
            return cls(
                # vocab_size=model_config['vocab_size'],
                # embedding_dim=model_config['embedding_dim'],
                # hidden_size=model_config['hidden_size'],
                # dropout_rate=model_config['dropout_rate'],
                # batch_first=model_config['batch_first'],
                # classifier_dropout=model_config['classifier_dropout'], 
            )