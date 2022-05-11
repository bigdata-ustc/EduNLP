import pytest
import torch
from EduNLP.ModelZoo.utils import MLP, TextCNN


def test_modules():
    encoder = TextCNN(256, 128)

    input_embeds1 = torch.rand(4, 16, 256)
    hidden_embeds1 = encoder(input_embeds1)
    assert hidden_embeds1.shape == torch.Size([4, 128])
    input_embeds2 = torch.rand(4, 1, 256)
    hidden_embeds2 = encoder(input_embeds2)
    assert hidden_embeds2.shape == torch.Size([4, 128])

    classifier = MLP(128, 10, 64, 0.5, n_layers=4)
    logits = classifier(hidden_embeds1)
    assert logits.shape == torch.Size([4, 10])
