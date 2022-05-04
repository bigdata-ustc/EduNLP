import torch
import numpy as np
import torch.nn as nn
from .modules import FeatureExtractor, ImageAE, MetaAE
from .util import SeqBatch
import json
import os
from pathlib import Path
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
from torchvision.transforms.functional import to_tensor


class BiHRNN(FeatureExtractor):
    """Sequence-to-sequence feature extractor based on RNN. Supports different
    input forms and different RNN types (LSTM/GRU), """

    def __init__(self, _stoi, embs: np.ndarray = None, pretrained_image: nn.Module = None,
                 pretrained_meta: nn.Module = None,
                 meta='know_name',
                 emb_size=256,
                 rnn='LSTM',
                 lambda_input=None,
                 lambda_loss=None,
                 layers=4, **kwargs):
        super(BiHRNN, self).__init__(**kwargs)

        if lambda_loss is None:
            lambda_loss = [1., 1.]
        if lambda_input is None:
            lambda_input = [1., 1., 1.]
        self.config = {
            'meta': meta,
            'emb_size': emb_size,
            'rnn': rnn,
            'lambda_input': lambda_input,
            'lambda_loss': lambda_loss,
            'layers': layers,
            'feat_size': self.feat_size,
        }
        feat_size = self.feat_size
        rnn_size = self.rnn_size = feat_size // 2

        self.stoi = _stoi
        vocab_size = len(_stoi['word'])
        self.itos = {v: k for k, v in self.stoi['word'].items()}

        self.we = nn.Embedding(vocab_size, emb_size)
        # embs: word2vec embeddings, (vocab_size, emb_size)
        if embs is not None:
            self.load_emb(embs)

        self.ie = ImageAE(emb_size)
        self.meta = meta
        self.meta_size = len(_stoi[self.meta])
        self.me = MetaAE(self.meta_size, emb_size)
        if pretrained_image is not None:
            self.load_img(pretrained_image)
        if pretrained_meta is not None:
            self.load_meta(pretrained_meta)

        self.lambda_input = lambda_input
        self.lambda_loss = lambda_loss

        if rnn == 'GRU':
            self.rnn = nn.GRU(emb_size, rnn_size, layers,
                              bidirectional=True, batch_first=True)
            self.h0 = nn.Parameter(torch.rand(layers * 2, 1, rnn_size))
        elif rnn == 'LSTM':
            self.rnn = nn.LSTM(emb_size, rnn_size, layers,
                               bidirectional=True, batch_first=True)
            self.h0 = nn.Parameter(torch.rand(layers * 2, 1, rnn_size))
            self.c0 = nn.Parameter(torch.rand(layers * 2, 1, rnn_size))
        else:
            raise ValueError('quesnet only support GRU and LSTM now.')

        self.proj_q = nn.Linear(feat_size, feat_size)
        self.proj_k = nn.Linear(feat_size, feat_size)
        self.proj_v = nn.Linear(feat_size, feat_size)

        self.woutput = nn.Linear(feat_size, vocab_size)
        self.ioutput = nn.Linear(feat_size, emb_size)
        self.moutput = nn.Linear(feat_size, emb_size)

        self.lwoutput = nn.Linear(rnn_size, vocab_size)
        self.lioutput = nn.Linear(rnn_size, emb_size)
        self.lmoutput = nn.Linear(rnn_size, emb_size)

        self.rwoutput = nn.Linear(rnn_size, vocab_size)
        self.rioutput = nn.Linear(rnn_size, emb_size)
        self.rmoutput = nn.Linear(rnn_size, emb_size)

        self.ans_decode = nn.GRU(emb_size, feat_size, layers,
                                 batch_first=True)
        self.ans_output = nn.Linear(feat_size, vocab_size)
        self.ans_judge = nn.Linear(feat_size, 1)
        self.dropout = nn.Dropout(0.2)

    def load_emb(self, emb):
        self.we.weight.detach().copy_(torch.from_numpy(emb))

    def load_img(self, img_layer: nn.Module):
        if self.config['emb_size'] != img_layer.emb_size:
            raise ValueError("Unmatched pre-trained ImageAE and embedding size")
        else:
            self.ie.load_state_dict(img_layer.state_dict())

    def load_meta(self, meta_layer: nn.Module):
        if self.config['emb_size'] != meta_layer.emb_size or self.meta_size != meta_layer.meta_size:
            raise ValueError("Unmatched pre-trained MetaAE and embedding size or meta size")
        else:
            self.me.load_state_dict(meta_layer.state_dict())

    def make_batch(self, data, device, pretrain=False):
        """Returns embeddings"""
        lembs = []
        rembs = []
        embs = []
        gt = []
        ans_input = []
        ans_output = []
        false_options = [[] for i in range(3)]
        for q in data:
            meta = torch.zeros(len(self.stoi[self.meta])).to(device)
            meta[q.labels.get(self.meta) or []] = 1
            _lembs = [self.we(torch.tensor([0], device=device)),
                      self.we(torch.tensor([0], device=device)),
                      self.me.enc(meta.unsqueeze(0)) * self.lambda_input[2]]
            _rembs = [self.me.enc(meta.unsqueeze(0)) * self.lambda_input[2]]
            _embs = [self.we(torch.tensor([0], device=device)),
                     self.we(torch.tensor([0], device=device)),
                     self.me.enc(meta.unsqueeze(0)) * self.lambda_input[2]]
            _gt = [torch.tensor([0], device=device), meta]
            for w in q.content:
                if isinstance(w, int):
                    word = torch.tensor([w], device=device)
                    item = self.we(word) * self.lambda_input[0]
                    _lembs.append(item)
                    _rembs.append(item)
                    _embs.append(item)
                    _gt.append(word)
                else:
                    im = to_tensor(w).to(device)
                    item = self.ie.enc(im.unsqueeze(0), detach_tensor=True) * self.lambda_input[1]
                    _lembs.append(item)
                    _rembs.append(item)
                    _embs.append(item)
                    _gt.append(im)
            _gt.append(torch.tensor([0], device=device))
            _rembs.append(self.we(torch.tensor([0], device=device)))
            _rembs.append(self.we(torch.tensor([0], device=device)))
            _embs.append(self.we(torch.tensor([0], device=device)))
            _embs.append(self.we(torch.tensor([0], device=device)))

            lembs.append(torch.cat(_lembs, dim=0))
            rembs.append(torch.cat(_rembs, dim=0))
            embs.append(torch.cat(_embs, dim=0))
            gt.append(_gt)

            ans_input.append([0] + q.answer)
            ans_output.append(q.answer + [0])

            for i, fo in enumerate(q.false_options):
                false_options[i].append([0] + fo)

        lembs = SeqBatch(lembs, device=device)
        rembs = SeqBatch(rembs, device=device)
        embs = SeqBatch(embs, device=device)
        ans_input = SeqBatch(ans_input, device=device)
        ans_output = SeqBatch(ans_output, device=device)
        false_opt_input = [SeqBatch(foi, device=device) for foi in false_options]

        length = sum(lembs.lens)
        words = []
        ims = []
        metas = []
        # p = lembs.packed().data
        wmask = torch.zeros(length, device=device).byte()
        imask = torch.zeros(length, device=device).byte()
        mmask = torch.zeros(length, device=device).byte()

        for i, _gt in enumerate(gt):
            for j, v in enumerate(_gt):
                ind = lembs.index((j, i))
                if v.size() == torch.Size([1]):  # word
                    words.append((v, ind))
                    wmask[ind] = 1
                elif v.dim() == 1:  # meta
                    metas.append((v.unsqueeze(0), ind))
                    mmask[ind] = 1
                else:  # img
                    ims.append((v.unsqueeze(0), ind))
                    imask[ind] = 1
        words = [x[0] for x in sorted(words, key=lambda x: x[1])]
        ims = [x[0] for x in sorted(ims, key=lambda x: x[1])]
        metas = [x[0] for x in sorted(metas, key=lambda x: x[1])]

        words = torch.cat(words, dim=0) if words else None
        ims = torch.cat(ims, dim=0) if ims else None
        metas = torch.cat(metas, dim=0) if metas else None
        if pretrain:
            return (
                lembs, rembs, words, ims, metas, wmask, imask, mmask,
                embs, ans_input, ans_output, false_opt_input
            )
        else:
            return embs

    def forward(self, batch: SeqBatch):
        packed = batch.packed()
        h = self.init_h(packed.batch_sizes[0])
        # y: (batch_size, seq_len, 2*rnn_size)
        y, _ = self.rnn(packed, h)

        hs, lens = pad_packed_sequence(y, batch_first=True)
        mask = [[1] * lens[i].item() + [0] * (lens[0] - lens[i]).item()
                for i in range(len(lens))]
        mask = torch.tensor(mask).byte().to(batch.device)

        # hs: (B, S, D), mask: (B, S)
        q, k, v = self.proj_q(hs), self.proj_k(hs), self.proj_v(hs)

        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))  # (B, S, S)
        if mask is not None:
            mask = mask.float()
            scores -= 1e9 * (1.0 - mask.unsqueeze(1))
        scores = self.dropout(F.softmax(scores, dim=-1))  # (B, S, S)
        h = (scores @ v).max(1)[0]  # (B, D)

        return y, batch.invert(h, 0), hs

    def pretrain_loss(self, batch):
        left, right, words, ims, metas, wmask, imask, mmask, inputs, ans_input, ans_output, false_opt_input = batch

        # high-level loss
        _, h, _ = self(inputs)
        x = ans_input.packed()
        y, _ = self.ans_decode(PackedSequence(self.we(x.data), x.batch_sizes),
                               h.repeat(self.config['layers'], 1, 1))
        floss = F.cross_entropy(self.ans_output(y.data),
                                ans_output.packed().data)
        floss += F.binary_cross_entropy_with_logits(self.ans_judge(y.data),
                                                    torch.ones_like(self.ans_judge(y.data)))
        for false_opt in false_opt_input:
            x = false_opt.packed()
            y, _ = self.ans_decode(PackedSequence(self.we(x.data), x.batch_sizes),
                                   h.repeat(self.config['layers'], 1, 1))
            floss += F.binary_cross_entropy_with_logits(self.ans_judge(y.data),
                                                        torch.zeros_like(self.ans_judge(y.data)))
        # low-level loss
        left_hid = self(left)[0].data[:, :self.rnn_size]
        right_hid = self(right)[0].data[:, self.rnn_size:]

        wloss = iloss = mloss = None

        if words is not None:
            lwfea = torch.masked_select(left_hid, wmask.unsqueeze(1).bool()) \
                .view(-1, self.rnn_size)
            lout = self.lwoutput(lwfea)
            rwfea = torch.masked_select(right_hid, wmask.unsqueeze(1).bool()) \
                .view(-1, self.rnn_size)
            rout = self.rwoutput(rwfea)
            out = self.woutput(torch.cat([lwfea, rwfea], dim=1))
            wloss = (F.cross_entropy(out, words) + F.cross_entropy(lout, words) + F.
                     cross_entropy(rout, words)) * self.lambda_input[0] / 3
            wloss *= self.lambda_loss[0]

        if ims is not None:
            lifea = torch.masked_select(left_hid, imask.unsqueeze(1).bool()) \
                .view(-1, self.rnn_size)
            lout = self.lioutput(lifea)
            rifea = torch.masked_select(right_hid, imask.unsqueeze(1).bool()) \
                .view(-1, self.rnn_size)
            rout = self.rioutput(rifea)
            out = self.ioutput(torch.cat([lifea, rifea], dim=1))
            iloss = (self.ie.loss(ims, out) + self.ie.loss(ims, lout) + self.ie.
                     loss(ims, rout)) * self.lambda_input[1] / 3
            iloss *= self.lambda_loss[0]

        if metas is not None:
            lmfea = torch.masked_select(left_hid, mmask.unsqueeze(1).bool()) \
                .view(-1, self.rnn_size)
            lout = self.lmoutput(lmfea)
            rmfea = torch.masked_select(right_hid, mmask.unsqueeze(1).bool()) \
                .view(-1, self.rnn_size)
            rout = self.rmoutput(rmfea)
            out = self.moutput(torch.cat([lmfea, rmfea], dim=1))
            mloss = (self.me.loss(metas, out) + self.me.loss(metas, lout) + self.me.
                     loss(metas, rout)) * self.lambda_input[2] / 3
            mloss *= self.lambda_loss[0]

        return {
            'field_loss': floss * self.lambda_loss[1],
            'word_loss': wloss,
            'image_loss': iloss,
            'meta_loss': mloss
        }

    def init_h(self, batch_size):
        size = list(self.h0.size())
        size[1] = batch_size
        if self.config['rnn'] == 'GRU':
            return self.h0.expand(size).contiguous()
        else:
            return self.h0.expand(size).contiguous(), self.c0.expand(size).contiguous()

    @classmethod
    def from_pretrained(cls, pretrained_dir, tokenizer):
        config_path = os.path.join(pretrained_dir, "config.json")
        model_path = os.path.join(pretrained_dir, "model.pt")
        model = cls.from_config(config_path, tokenizer.stoi)
        model.load_state_dict(torch.load(model_path))
        return model

    @classmethod
    def from_config(cls, config_path, stoi):
        with open(config_path, "r", encoding="utf-8") as rf:
            model_config = json.load(rf)
            wv = torch.load(model_config["wv_path"],
                            map_location='cpu', mmap="r") if "wv_path" in model_config else None
            return cls(
                _stoi=stoi,
                embs=wv, meta=model_config["meta"], emb_size=model_config["emb_size"],
                rnn=model_config["rnn"], lambda_input=model_config["lambda_input"],
                lambda_loss=model_config["lambda_loss"], layers=model_config["layers"],
                feat_size=model_config["feat_size"])

    def save(self, path):
        """ Save model to path/model_name.pt and path/config.json

        Parameters
        ----------
        path : str
            directory to save model
        """
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, 'model.pt')
        model_path = Path(model_path)
        torch.save(self.state_dict(), model_path.open('wb'))
        config_path = os.path.join(path, "config.json")
        self.save_config(config_path)

    def save_config(self, config_path):
        with open(config_path, "w", encoding="utf-8") as wf:
            json.dump(self.config, wf, ensure_ascii=False, indent=2)
