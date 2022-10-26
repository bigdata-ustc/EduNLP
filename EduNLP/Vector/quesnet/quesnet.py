import torch
from typing import Union
from EduNLP.ModelZoo.quesnet import QuesNet
from EduNLP.Pretrain import Question, QuesNetTokenizer


class QuesNetModel(object):
    def __init__(self, pretrained_dir, img_dir=None, device="cpu", **argv):
        """
        Parameters
        ----------
        pretrained_dir: str
            the dirname to pretrained model
        device: str
            cpu or cuda, default is cpu
        img_dir: str
            image dir
        """
        self.device = torch.device(device)
        self.model = QuesNet.from_pretrained(pretrained_dir, img_dir=img_dir).to(device)
        self.model.eval()

    def __call__(self, items: dict, **kwargs):
        """ get question embedding with quesnet

        Parameters
        ----------
        items:
            encodes from tokenizer
        """
        qs = [Question("", items['seq_idx'][i],
                       [0], [[0], [0], [0]], items['meta_idx'][i]) for i in range(len(items['seq_idx']))]
        outputs = self.model(self.model.make_batch(qs, device=self.device))
        return outputs.hidden, outputs.embeded

    def infer_vector(self, items: Union[dict, list]) -> torch.Tensor:
        """ get question embedding with quesnet

        Parameters
        ----------
        items:
            encodes from tokenizer
        """
        return self(items)[0]

    def infer_tokens(self, items: Union[dict, list]) -> torch.Tensor:
        """ get token embeddings with quesnet

        Parameters
        ----------
        items:
            encodes from tokenizer
        Returns
        -------
        torch.Tensor
            word_embs + meta_emb
        """
        vector = self(items)[1]
        """ Please note that output vector is like 0 0 seq_idx(text with image) 0 meta_idx 0 0"""
        return vector[:, 2:-2, :]

    @property
    def vector_size(self):
        return self.model.feat_size
