import torch
from transformers.modeling_outputs import ModelOutput


class PropertyPredictionOutput(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None


class KnowledgePredictionOutput(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
