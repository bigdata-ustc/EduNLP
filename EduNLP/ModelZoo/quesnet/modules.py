import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, feat_size=512):
        super(FeatureExtractor, self).__init__()
        self.feat_size = feat_size

    def make_batch(self, data, device, pretrain=False):
        """Make batch from input data (python data / np arrays -> tensors)"""
        raise NotImplementedError

    def load_emb(self, emb):
        pass

    def pretrain_loss(self, batch):
        """Returns pretraining loss on a batch of data"""
        raise NotImplementedError

    def forward(self, *input):
        raise NotImplementedError


class AE(nn.Module):
    factor = 1

    def enc(self, item, *args, **kwargs):
        return self.encoder(item, *args, **kwargs)

    def dec(self, item, *args, **kwargs):
        return self.decoder(item, *args, **kwargs)

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
        self.emb_size = emb_size
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

    def encoder(self, item, detach_tensor=False):
        return self._encoder(item).detach().view(item.size(0), -1) if detach_tensor else self._encoder(item).view(
            item.size(0), -1)

    def decoder(self, emb, detach_tensor=False):
        return self._decoder(emb[:, :, None, None]).detach() if detach_tensor else self._decoder(emb[:, :, None, None])


class MetaAE(AE):
    def __init__(self, meta_size, emb_size):
        super().__init__()
        self.emb_size = emb_size
        self.meta_size = meta_size
        self.recons_loss = nn.BCEWithLogitsLoss()
        self.encoder = nn.Sequential(nn.Linear(meta_size, emb_size),
                                     nn.ReLU(True))
        # error: inplace
        # nn.Linear(emb_size, emb_size)
        self.decoder = nn.Sequential(nn.Linear(emb_size // self.factor,
                                               emb_size),
                                     nn.ReLU(True),
                                     nn.Linear(emb_size, meta_size))
