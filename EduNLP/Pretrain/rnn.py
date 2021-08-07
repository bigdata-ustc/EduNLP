# coding: utf-8
# 2021/8/3 @ tongshiwei
# todo: finish this module

import torch
from torch import nn
import torch.nn.functional as F

from baize import iterwrap
from baize.torch import light_module as lm, Configuration
from baize.torch import fit_wrapper

from EduNLP.Vector import RNNModel, Embedding
from EduNLP.ModelZoo import Masker
from EduNLP.Tokenizer import get_tokenizer


def form_batch(batch, indexer: Embedding, masker: Masker):
    batch_idx, batch_len = indexer.indexing(batch, padding=True)
    masked_batch_idx, masked = masker(batch_idx, batch_len)
    return torch.tensor(masked_batch_idx), torch.tensor(batch_idx), torch.tensor(masked)


@iterwrap()
def etl(items, tokenizer, indexer: Embedding, masker: Masker, params: Configuration):
    batch_size = params.batch_size
    batch = []
    for item in tokenizer(items):
        batch.append(item[:20])
        if len(batch) == batch_size:
            yield form_batch(items, indexer, masker)
            batch = []
    if batch:
        yield form_batch(items, indexer, masker)


class MLMRNN(torch.nn.Module):
    def __init__(self, rnn_type, w2v, vector_size, *args, **kwargs):
        super(MLMRNN, self).__init__()
        self.rnn = RNNModel(rnn_type, w2v, vector_size, *args, **kwargs)
        self.pred_net = nn.Linear(vector_size, self.rnn.embedding.vocab_size)

    def __call__(self, x, *args, **kwargs):
        output, _ = self.rnn(x, *args, **kwargs)
        return F.log_softmax(self.pred_net(output), dim=-1)


@fit_wrapper
def fit_f(_net: RNNModel, batch_data, loss_function, *args, **kwargs):
    masked_seq, seq, mask = batch_data
    pred = _net(masked_seq, indexing=False, padding=False)
    return loss_function(pred, seq) * mask


def train_rnn(items, tokenizer, rnn_type, w2v, vector_size, **kwargs):
    tokenizer = get_tokenizer(tokenizer)
    cfg = Configuration(select="rnn(?!.*embedding)")
    mlm_rnn = MLMRNN(rnn_type, w2v, vector_size, freeze_pretrained=True)
    train_data = etl(items, tokenizer, mlm_rnn.rnn.embedding, Masker(min_mask=1), cfg)
    lm.train(
        mlm_rnn,
        cfg,
        fit_f=fit_f,
        trainer=None,
        loss_function=torch.nn.CrossEntropyLoss(),
        train_data=train_data,
        initial_net=True,
    )
