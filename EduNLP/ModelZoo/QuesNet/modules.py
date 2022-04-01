import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, feat_size=512):
        super(FeatureExtractor, self).__init__()
        self.feat_size = feat_size

    def make_batch(self, data, pretrain=False):
        """Make batch from input data (python data / np arrays -> tensors)"""
        return torch.tensor(data)

    def load_emb(self, emb):
        pass

    def pretrain_loss(self, batch):
        """Returns pretraining loss on a batch of data"""
        raise NotImplementedError

    def forward(self, *input):
        raise NotImplementedError


class SP(nn.Module):
    def __init__(self, feat_size, wcnt, emb_size=50, seq_h_size=50,
                 n_layers=1, attn_k=10):
        super().__init__()
        self.wcnt = wcnt
        self.emb_size = emb_size
        self.ques_h_size = feat_size
        self.seq_h_size = seq_h_size
        self.n_layers = n_layers
        self.attn_k = attn_k

        self.seq_net = EERNNSeqNet(self.ques_h_size, seq_h_size,
                                   n_layers, attn_k)

    def forward(self, ques_h, score, hidden=None):
        s, h = self.seq_net(ques_h, score, hidden)

        if hidden is None:
            hidden = ques_h, h
        else:
            # concat all qs and hs for attention
            qs, hs = hidden
            qs = torch.cat([qs, ques_h])
            hs = torch.cat([hs, h])
            hidden = qs, hs

        return s, hidden


class EERNNSeqNet(nn.Module):
    def __init__(self, ques_size, seq_hidden_size, n_layers, attn_k):
        super(EERNNSeqNet, self).__init__()

        self.initial_h = nn.Parameter(torch.zeros(n_layers * seq_hidden_size))
        self.ques_size = ques_size  # exercise size
        self.seq_hidden_size = seq_hidden_size
        self.n_layers = n_layers
        self.attn_k = attn_k

        # initialize network
        self.seq_net = nn.GRU(ques_size * 2, seq_hidden_size, n_layers)
        self.score_net = nn.Linear(ques_size + seq_hidden_size, 1)

    def forward(self, question, score, hidden):
        if hidden is None:
            h = self.initial_h.view(self.n_layers, 1, self.seq_hidden_size)
            attn_h = self.initial_h
        else:
            questions, hs = hidden
            h = hs[-1:]
            alpha = torch.mm(questions, question.view(-1, 1)).view(-1)
            alpha, idx = alpha.topk(min(len(alpha), self.attn_k), sorted=False)
            alpha = nn.functional.softmax(alpha.view(1, -1), dim=-1)

            # flatten each h
            hs = hs.view(-1, self.n_layers * self.seq_hidden_size)
            attn_h = torch.mm(alpha, torch.index_select(hs, 0, idx)).view(-1)

        # prediction
        pred_v = torch.cat([question, attn_h]).view(1, -1)
        pred = self.score_net(pred_v)[0]

        if score is None:
            score = pred

        # update seq_net
        x = torch.cat([question * (score >= 0.5).float(),
                       question * (score < 0.5).float()])

        _, h_ = self.seq_net(x.view(1, 1, -1), h)
        return pred, h_


class AE(nn.Module):
    factor = 1

    def enc(self, item):
        return self.encoder(item)

    def dec(self, item):
        return self.decoder(item)

    def loss(self, item, emb=None):
        if emb is None:
            emb = self.enc(item)
            out = self.dec(emb)
        else:
            out = self.dec(emb)

        return self.recons_loss(out, item)

    def forward(self, item):
        return self.enc(item)


class ImageAE(AE):
    def __init__(self, emb_size):
        super().__init__()
        self.recons_loss = nn.MSELoss()
        self._encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(32, emb_size, 3, stride=2)
        )
        self._decoder = nn.Sequential(
            nn.ConvTranspose2d(emb_size // self.factor, 32, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encoder(self, item):
        return self._encoder(item).view(item.size(0), -1)

    def decoder(self, emb):
        return self._decoder(emb[:, :, None, None])


class MetaAE(AE):
    def __init__(self, meta_size, emb_size):
        super().__init__()
        self.emb_size = emb_size
        self.recons_loss = nn.BCEWithLogitsLoss()
        self.encoder = nn.Sequential(nn.Linear(meta_size, emb_size),
                                     nn.ReLU(True))
        # error: inplace
        # nn.Linear(emb_size, emb_size)
        self.decoder = nn.Sequential(nn.Linear(emb_size // self.factor,
                                               emb_size),
                                     nn.ReLU(True),
                                     nn.Linear(emb_size, meta_size))
