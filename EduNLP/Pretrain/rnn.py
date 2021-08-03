# coding: utf-8
# 2021/8/3 @ tongshiwei

import torch
import numpy as np
from longling import iterwrap
from longling.ML.PytorchHelper import light_module as lm
from longling.ML.PytorchHelper.utils import Configuration
from EduNLP.Vector import RNNModel, Embedding
from EduNLP.ModelZoo import Masker
from EduNLP.Tokenizer import get_tokenizer, Tokenizer


def form_batch(batch, indexer: Embedding, masker: Masker):
    batch_idx, batch_len = indexer.indexing(batch, padding=True)
    masked_batch_idx, masked = masker(batch_idx, batch_len)
    return torch.tensor(masked_batch_idx), torch.tensor(batch_idx), torch.tensor(masked)


@iterwrap()
def etl(items, tokenizer, indexer: Embedding, masker: Masker, params: Configuration):
    batch_size = params.batch_size
    batch = []
    for item in tokenizer(items):
        batch.append(item)
        if len(batch) == batch_size:
            yield form_batch(items, indexer, masker)
            batch = []
    if batch:
        yield form_batch(items, indexer, masker)


def train_rnn(items, tokenizer, rnn_type, w2v, vector_size, device=None, **kwargs):
    tokenizer = get_tokenizer(tokenizer)
    cfg = Configuration()
    rnn = RNNModel(rnn_type, w2v, vector_size, freeze_pretrained=True, device=device)

    lm.train(
        rnn,
        cfg,
    )
    pass
