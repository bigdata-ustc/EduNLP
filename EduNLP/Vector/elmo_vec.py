from .rnn import ElmoBilm
from pathlib import PurePath
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as tud
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from EduNLP.Pretrain import ElmoVocab, ElmoDataset, elmo_collate_fn
from .meta import Vector


class ElmoModel(Vector):
    """

    Examples
    --------
    >>> from EduNLP.Pretrain import ElmoVocab
    >>> elmo_vocab=ElmoVocab()
    >>> items = ["有公式$\\FormFigureID{wrong1?}$，如图$\\FigureID{088f15ea-xxx}$,\
    ... 若$x,y$满足约束条件公式$\\FormFigureBase64{wrong2?}$,$\\SIFSep$，则$z=x+7 y$的最大值为$\\SIFBlank$"]
    >>> elmo_vocab.tokenize(items[0])
    ['公式', '如图', '[FIGURE]', 'x', ',', 'y', '约束条件', '公式', '[SEP]', 'z', '=', 'x', '+', '7', 'y', '最大值', '[MARK]']
    >>> elmo = ElmoModel(elmo_vocab=elmo_vocab)
    >>> inputs = ['如','图','所','示','，','有','公','式']
    >>> elmo.infer_tokens(inputs).shape
    torch.Size([8, 1024])
    >>> elmo.infer_vector(inputs).shape
    torch.Size([1024])
    """

    def __init__(self, path=None, elmo_vocab=None, emb_size: int = 512, hidden_size: int = 1024,
                 lr: float = 5e-4):
        if elmo_vocab is None:
            self.vocab = ElmoVocab()
        else:
            self.vocab = elmo_vocab
        id2t = {}
        if path is not None:
            self.load_vocab(path + '_elmo_vocab.json')
            self.Bilm = ElmoBilm(len(self.vocab.t2id), emb_size=emb_size, hidden_size=hidden_size, num_layers=2)
            self.load_weights(path + '_elmo_weights.bin')
        else:
            self.Bilm = ElmoBilm(len(self.vocab.t2id), emb_size=emb_size, hidden_size=hidden_size, num_layers=2)
        for t in self.vocab.t2id:
            id2t[self.vocab.t2id[t]] = t
        self.id2t = id2t
        self.adam = optim.Adam(self.Bilm.parameters(), lr=lr)
        self.loss_function = nn.BCELoss()

    def __call__(self, item):
        return self.infer_vector(item)

    def train(self, train_set: ElmoDataset, batch_size=16, shuffle=True, epochs=3):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Bilm.to(device)
        if torch.cuda.device_count() > 1:
            self.Bilm = torch.nn.DataParallel(self.Bilm)
        self.Bilm.train()
        global_step = 0
        self.loss_function.to(device)
        data_loader = tud.DataLoader(train_set, collate_fn=elmo_collate_fn, batch_size=batch_size, shuffle=shuffle)
        ids = -1
        for epoch in range(epochs):
            for step, sample in enumerate(data_loader):
                try:
                    mask = sample['mask'].to(device)
                    ids = sample['ids'].to(device)
                    y = F.one_hot(ids, num_classes=len(self.vocab)).to(device)
                    pred_forward, pred_backward, forward_hiddens, backward_hiddens = self.Bilm(ids)
                    pred_forward = pred_forward[mask]
                    pred_backward = pred_backward[torch.flip(mask, [1])]
                    y_rev = torch.flip(y, [1])[torch.flip(mask, [1])]
                    y = y[mask]
                    forward_loss = self.loss_function(pred_forward[:, :-1].double(), y[:, 1:].double())
                    # loss = self.loss_function(pred_forward[:, :-1].double(), y[:, 1:].double()) + self.loss_function(
                    #     pred_backward[:, :-1].double(), torch.flip(y, [1])[:, 1:].double())
                    backward_loss = self.loss_function(pred_backward[:, :-1].double(), y_rev[:, 1:].double())
                    forward_loss.backward()
                    backward_loss.backward()
                    self.adam.step()
                    self.adam.zero_grad()
                    global_step += 1
                    if global_step % 10 == 0:
                        print("[Global step %d, epoch %d, batch %d] Loss: %.10f" % (
                            global_step, epoch, step, forward_loss + backward_loss))
                except RuntimeError as e:
                    print("RuntimeError:", e)
                    print("[DEBUG]sample ids:", ids)

    def save_weights(self, path):
        torch.save(self.Bilm.state_dict(), path)
        return path

    def load_weights(self, path):
        self.Bilm.load_state_dict(torch.load(path))
        self.Bilm.eval()
        return path

    def save_vocab(self, path):
        self.vocab.save_vocab(path)
        return path

    def load_vocab(self, path):
        self.vocab.load_vocab(path)
        return path

    def get_contextual_emb(self, item_indices: list, token_idx: int, scale: int = 1):
        # get contextual embedding of a token, given a sentence containing it
        self.Bilm.to(torch.device("cpu"))
        self.Bilm.eval()
        pred_forward, pred_backward, forward_hiddens, backward_hiddens = self.Bilm.forward([item_indices])
        representations = torch.cat((forward_hiddens[0][0][token_idx], backward_hiddens[0][0][token_idx]),
                                    dim=0).unsqueeze(0)
        for i in range(self.Bilm.num_layers):
            representations = torch.cat((representations, torch.cat(
                (forward_hiddens[i + 1][0][token_idx], backward_hiddens[i + 1][0][token_idx]), 0).unsqueeze(
                0)), dim=0)
        return scale * torch.sum(representations, dim=0)

    def infer_vector(self, item, *args, **kwargs) -> torch.Tensor:
        item = [0 if token not in self.vocab.t2id else self.vocab.t2id[token] for token in item]
        pred_forward, pred_backward, forward_hiddens, backward_hiddens = self.Bilm([item])
        ret = torch.cat((forward_hiddens[-1, 0, -1], backward_hiddens[-1, 0, -1]), dim=-1)
        return ret

    def infer_tokens(self, item, *args, **kwargs) -> torch.Tensor:
        item = [0 if token not in self.vocab.t2id else self.vocab.t2id[token] for token in item]
        item_infer = [self.get_contextual_emb(item, i).tolist() for i in range(len(item))]
        return torch.Tensor(item_infer)

    @property
    def vector_size(self):
        return self.Bilm.hidden_size


def train_elmo(texts, filepath_prefix: str, pretrain_model=None, emb_dim=512, hid_dim=1024, batch_size=4,
               epochs=1):
    vocab = ElmoVocab()
    if pretrain_model is None:
        texts = [vocab.tokenize(text) for text in texts]  # This WILL append new token to vocabulary
        vocab.save_vocab(filepath_prefix + '/' + os.path.basename(filepath_prefix) + '_elmo_vocab.json')
    else:
        vocab.load_vocab(filepath_prefix + '/' + pretrain_model + '_elmo_vocab.json')
        texts = [vocab.pure_tokenizer(text) for text in texts]  # This will NOT append new token to vocabulary
    dataset = ElmoDataset(texts=texts, vocab=vocab)
    model = ElmoModel(elmo_vocab=vocab, emb_size=emb_dim, hidden_size=hid_dim)
    model.train(train_set=dataset, batch_size=batch_size, epochs=epochs)
    if pretrain_model is None:
        model.save_weights(filepath_prefix + '/' + os.path.basename(filepath_prefix) + '_elmo_weights.bin')
    else:
        model.save_weights(filepath_prefix + '/' + pretrain_model + '_elmo_weights.bin')
    return filepath_prefix
