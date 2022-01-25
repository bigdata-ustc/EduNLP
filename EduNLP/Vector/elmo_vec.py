from .rnn import ElmoBilm
from pathlib import PurePath
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
    # """

    # Examples
    # --------
    # >>> from EduNLP.Pretrain import ElmoVocab
    # >>> elmo_vocab=ElmoVocab()
    # >>> elmo_vocab.load_vocab('examples/test_model/data/elmo/vocab_wiki.json')
    # 'examples/test_model/data/elmo/vocab_wiki.json'
    # >>> elmo = ElmoModel(t2id=elmo_vocab.t2id, lr=1e-5)
    # >>> elmo.load_weights('examples/test_model/data/elmo/elmo_pretrain_weights_wiki.bin')
    # 'examples/test_model/data/elmo/elmo_pretrain_weights_wiki.bin'
    # >>> inputs = ['如','图','所','示','，','有','公','式']
    # >>> elmo.infer_tokens(inputs).shape
    # torch.Size([8, 1024])
    # >>> elmo.infer_vector(inputs).shape
    # torch.Size([1024])
    # """

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
        self.Bilm.cuda()
        if torch.cuda.device_count() > 1:
            self.Bilm = torch.nn.DataParallel(self.Bilm)
        self.Bilm.train()
        global_step = 0
        self.loss_function.cuda()
        data_loader = tud.DataLoader(train_set, collate_fn=elmo_collate_fn, batch_size=batch_size, shuffle=shuffle)
        for epoch in range(epochs):
            for step, sample in enumerate(data_loader):
                try:
                    mask = sample['mask'].cuda()
                    ids = sample['ids'].cuda()
                    y = F.one_hot(ids, num_classes=len(self.vocab)).cuda()
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

    def infer_vector(self, item) -> torch.Tensor:
        item = [0 if token not in self.vocab.t2id else self.vocab.t2id[token] for token in item]
        pred_forward, pred_backward, forward_hiddens, backward_hiddens = self.Bilm([item])
        ret = torch.cat((forward_hiddens[-1, 0, -1], backward_hiddens[-1, 0, -1]), dim=-1)
        return ret

    def infer_tokens(self, item) -> torch.Tensor:
        item = [0 if token not in self.vocab.t2id else self.vocab.t2id[token] for token in item]
        item_infer = [self.get_contextual_emb(item, i).tolist() for i in range(len(item))]
        return torch.Tensor(item_infer)
