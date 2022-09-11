import torch
import torch.nn as nn
from torch.autograd import Variable
from transformers.modeling_outputs import ModelOutput
import numpy as np

__all__ = ["HAM"]


class HAMOutput(ModelOutput):
    """
        Output type of [`HAM`]

        Parameters
        ----------
        scores: of shape (batch_size, sequence_length)
        first_att_weight: of shape
        second_visual: of shape (batch_size, num_classes)
        third_visual: of shape (batch_size, num_classes)
        first_logits: of shape (batch_size, num_classes)
        second_logits: of shape (batch_size, num_classes)
        third_logits: of shape (batch_size, num_classes)
        global_logits: of shape (batch_size, num_classes)
        first_scores: of shape (batch_size, num_classes)
        second_scores: of shape (batch_size, num_classes)
    """
    scores: torch.FloatTensor = None
    first_att_weight: torch.FloatTensor = None
    second_visual: torch.FloatTensor = None
    third_visual: torch.FloatTensor = None
    first_logits: torch.FloatTensor = None
    second_logits: torch.FloatTensor = None
    third_logits: torch.FloatTensor = None
    global_logits: torch.FloatTensor = None
    first_scores: torch.FloatTensor = None
    second_scores: torch.FloatTensor = None


def truncated_normal_(tensor, mean=0, std=0.1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


class AttentionLayer(nn.Module):
    def __init__(self, num_units, attention_unit_size, num_classes):
        super(AttentionLayer, self).__init__()
        self.fc1 = nn.Linear(num_units, attention_unit_size, bias=False)
        self.fc2 = nn.Linear(attention_unit_size, num_classes, bias=False)

    def forward(self, input_x):
        attention_matrix = self.fc2(torch.tanh(self.fc1(input_x))).transpose(1, 2)
        attention_weight = torch.softmax(attention_matrix, dim=-1)
        attention_out = torch.matmul(attention_weight, input_x)
        return attention_weight, torch.mean(attention_out, dim=1)


class LocalLayer(nn.Module):
    def __init__(self, num_units, num_classes):
        super(LocalLayer, self).__init__()
        self.fc = nn.Linear(num_units, num_classes)

    def forward(self, input_x, input_att_weight):
        logits = self.fc(input_x)
        scores = torch.sigmoid(logits)
        visual = torch.mul(input_att_weight, scores.unsqueeze(-1))
        visual = torch.softmax(visual, dim=-1)
        visual = torch.mean(visual, dim=1)
        return logits, scores, visual


class HAM(nn.Module):

    def __init__(
            self,
            num_classes_list,
            num_total_classes,
            lstm_hidden_size,
            attention_unit_size,
            fc_hidden_size,
            beta=0.0,
            dropout_rate=None):
        super(HAM, self).__init__()
        self.beta = beta

        # First Level
        self.first_attention = AttentionLayer(lstm_hidden_size * 2, attention_unit_size, num_classes_list[0])
        self.first_fc = nn.Linear(lstm_hidden_size * 4, fc_hidden_size)
        self.first_local = LocalLayer(fc_hidden_size, num_classes_list[0])

        # Second Level
        self.second_attention = AttentionLayer(lstm_hidden_size * 2, attention_unit_size, num_classes_list[1])
        self.second_fc = nn.Linear(lstm_hidden_size * 4, fc_hidden_size)
        self.second_local = LocalLayer(fc_hidden_size, num_classes_list[1])

        # Third Level
        self.third_attention = AttentionLayer(lstm_hidden_size * 2, attention_unit_size, num_classes_list[2])
        self.third_fc = nn.Linear(lstm_hidden_size * 4, fc_hidden_size)
        self.third_local = LocalLayer(fc_hidden_size, num_classes_list[2])

        # Fully Connected Layer
        self.fc = nn.Linear(fc_hidden_size * 4, fc_hidden_size)

        # Highway Layer
        self.highway_lin = nn.Linear(fc_hidden_size, fc_hidden_size)
        self.highway_gate = nn.Linear(fc_hidden_size, fc_hidden_size)

        # Add dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Global scores
        self.global_scores_fc = nn.Linear(fc_hidden_size, num_total_classes)

        for name, param in self.named_parameters():
            if 'embedding' not in name and 'weight' in name:
                truncated_normal_(param.data, mean=0, std=0.1)
            else:
                nn.init.constant_(param.data, 0.1)

    def forward(self, sequential_embeddings):

        sequential_embeddings_pool = torch.mean(sequential_embeddings, dim=1)

        # First Level
        first_att_weight, first_att_out = self.first_attention(sequential_embeddings)
        first_local_input = torch.cat((sequential_embeddings_pool, first_att_out), dim=1)
        first_local_fc_out = self.first_fc(first_local_input)
        first_logits, first_scores, first_visual = self.first_local(first_local_fc_out, first_att_weight)

        # Second Level
        second_att_input = torch.mul(sequential_embeddings, first_visual.unsqueeze(-1))
        second_att_weight, second_att_out = self.second_attention(second_att_input)
        second_local_input = torch.cat((sequential_embeddings_pool, second_att_out), dim=1)
        second_local_fc_out = self.second_fc(second_local_input)
        second_logits, second_scores, second_visual = self.second_local(second_local_fc_out, second_att_weight)

        # Third Level
        third_att_input = torch.mul(sequential_embeddings, second_visual.unsqueeze(-1))
        third_att_weight, third_att_out = self.third_attention(third_att_input)
        third_local_input = torch.cat((sequential_embeddings_pool, third_att_out), dim=1)
        third_local_fc_out = self.third_fc(third_local_input)
        third_logits, third_scores, third_visual = self.third_local(third_local_fc_out, third_att_weight)

        # Concat
        # shape of ham_out: [batch_size, fc_hidden_size * 3]
        ham_out = torch.cat((first_local_fc_out, second_local_fc_out,
                             third_local_fc_out), dim=1)

        # Fully Connected Layer
        fc_out = self.fc(ham_out)

        # Highway Layer and Dropout
        highway_g = torch.relu(self.highway_lin(fc_out))
        highway_t = torch.sigmoid(self.highway_gate(fc_out))
        highway_output = torch.mul(highway_g, highway_t) + torch.mul((1 - highway_t), fc_out)
        h_drop = self.dropout(highway_output)

        # Global scores
        global_logits = self.global_scores_fc(h_drop)
        global_scores = torch.sigmoid(global_logits)
        local_scores = torch.cat((first_scores, second_scores, third_scores), dim=1)
        scores = self.beta * global_scores + (1 - self.beta) * local_scores
        return HAMOutput(
            scores=scores,
            first_att_weight=first_att_weight,
            first_visual=first_visual,
            second_visual=second_visual,
            third_visual=third_visual,
            first_logits=first_logits,
            second_logits=second_logits,
            third_logits=third_logits,
            global_logits=global_logits,
            first_scores=first_scores,
            second_scores=second_scores
        )
